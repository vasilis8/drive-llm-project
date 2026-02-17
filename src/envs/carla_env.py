"""
CARLA Gymnasium Environment Wrapper.

Wraps the CARLA simulator into a standard gymnasium.Env with:
- Dict observation space (LiDAR + command embedding + vehicle state)
- Continuous action space (steering + throttle/brake)
- Command-aware reward function

Requires a running CARLA server (typically via Docker on cloud GPU).
For local testing without CARLA, use DummyCarlaEnv instead.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

from src.envs.rewards import compute_reward
from src.envs.carla_config import (
    DEFAULT_CARLA_CONFIG,
    DEFAULT_LIDAR_CONFIG,
    DEFAULT_VEHICLE_CONFIG,
    get_lidar_attributes,
)
from src.utils.commands import CommandManager


class CarlaEnv(gym.Env):
    """
    CARLA-based racing environment with language-conditioned observations.

    This environment connects to a running CARLA server, spawns a vehicle
    with LiDAR sensor, and exposes a gymnasium-compatible interface.

    Observation space (Dict):
        - "lidar": (n_beams,) processed range data
        - "command": (384,)   frozen sentence embedding
        - "vehicle_state": (5,) [speed, accel, steering, yaw_rate, distance]

    Action space (Box):
        - [steering, throttle/brake] both in [-1, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        carla_config: Optional[Dict] = None,
        lidar_config: Optional[Dict] = None,
        vehicle_config: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
        command_manager: Optional[CommandManager] = None,
        max_episode_steps: int = 2000,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            carla_config: CARLA server connection settings.
            lidar_config: LiDAR sensor configuration.
            vehicle_config: Vehicle blueprint and limits.
            reward_config: Reward function weights.
            command_manager: CommandManager for sampling commands.
            max_episode_steps: Maximum steps before truncation.
            render_mode: Not used (CARLA handles rendering internally).
        """
        super().__init__()

        # Merge configs with defaults
        self.carla_cfg = {**DEFAULT_CARLA_CONFIG, **(carla_config or {})}
        self.lidar_cfg = {**DEFAULT_LIDAR_CONFIG, **(lidar_config or {})}
        self.vehicle_cfg = {**DEFAULT_VEHICLE_CONFIG, **(vehicle_config or {})}
        self.reward_cfg = reward_config or {}
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Command manager
        self.command_manager = command_manager or CommandManager(encoder=None)

        # Dimensions
        self.n_beams = self.lidar_cfg["n_beams"]
        self.command_dim = self.command_manager.embedding_dim
        self.vehicle_state_dim = 5
        self.max_speed = self.vehicle_cfg["max_speed"]

        # ── Spaces ────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(
                low=0.0,
                high=float(self.lidar_cfg["range"]),
                shape=(self.n_beams,),
                dtype=np.float32,
            ),
            "command": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.command_dim,),
                dtype=np.float32,
            ),
            "vehicle_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.vehicle_state_dim,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── CARLA handles (initialized on first reset) ────────
        self._client = None
        self._world = None
        self._vehicle = None
        self._lidar_sensor = None
        self._collision_sensor = None
        self._lidar_data = None
        self._collision_detected = False
        self._connected = False

        # ── Episode state ─────────────────────────────────────
        self._step_count = 0
        self._current_command = None
        self._prev_vehicle_state = None
        self._distance_traveled = 0.0
        self._prev_location = None

    def _connect_to_carla(self):
        """Establish connection to CARLA server."""
        import carla

        self._client = carla.Client(
            self.carla_cfg["host"],
            self.carla_cfg["port"],
        )
        self._client.set_timeout(self.carla_cfg["timeout"])
        self._world = self._client.load_world(self.carla_cfg["town"])

        # Set synchronous mode with fixed timestep
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla_cfg["fixed_delta_seconds"]
        self._world.apply_settings(settings)

        # Set weather
        weather_name = self.carla_cfg.get("weather", "ClearNoon")
        weather = getattr(carla.WeatherParameters, weather_name, None)
        if weather:
            self._world.set_weather(weather)

        self._connected = True

    def _spawn_vehicle(self):
        """Spawn the ego vehicle at a random spawn point."""
        import carla

        bp_library = self._world.get_blueprint_library()
        vehicle_bp = bp_library.find(self.vehicle_cfg["blueprint"])

        spawn_points = self._world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()

        self._vehicle = self._world.try_spawn_actor(vehicle_bp, spawn_point)
        if self._vehicle is None:
            # Try another spawn point
            for sp in spawn_points:
                self._vehicle = self._world.try_spawn_actor(vehicle_bp, sp)
                if self._vehicle is not None:
                    break

        if self._vehicle is None:
            raise RuntimeError("Failed to spawn vehicle in CARLA")

    def _setup_sensors(self):
        """Attach LiDAR and collision sensors to the vehicle."""
        import carla

        bp_library = self._world.get_blueprint_library()

        # ── LiDAR Sensor ──────────────────────────────────────
        lidar_bp = bp_library.find("sensor.lidar.ray_cast")
        lidar_attrs = get_lidar_attributes(self.lidar_cfg)
        for attr, value in lidar_attrs.items():
            lidar_bp.set_attribute(attr, value)

        lidar_pos = self.lidar_cfg.get("position", {})
        lidar_transform = carla.Transform(
            carla.Location(
                x=lidar_pos.get("x", 0.0),
                y=lidar_pos.get("y", 0.0),
                z=lidar_pos.get("z", 2.4),
            )
        )
        self._lidar_sensor = self._world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self._vehicle
        )
        self._lidar_data = None
        self._lidar_sensor.listen(self._on_lidar_data)

        # ── Collision Sensor ──────────────────────────────────
        collision_bp = bp_library.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self._vehicle,
        )
        self._collision_detected = False
        self._collision_sensor.listen(self._on_collision)

    def _on_lidar_data(self, data):
        """Callback: process raw LiDAR point cloud into range array."""
        # Raw data: array of (x, y, z, intensity) points
        raw_data = np.frombuffer(data.raw_data, dtype=np.float32)
        points = raw_data.reshape(-1, 4)[:, :3]  # (N, 3) XYZ

        # Convert to 2D range bins (angular bins around the vehicle)
        ranges = self._points_to_ranges(points)
        self._lidar_data = ranges

    def _points_to_ranges(self, points: np.ndarray) -> np.ndarray:
        """
        Convert 3D point cloud to 1D range array.

        Projects points onto the XY plane, bins them by angle,
        and takes the minimum range per bin.
        """
        if len(points) == 0:
            return np.full(self.n_beams, self.lidar_cfg["range"], dtype=np.float32)

        # Compute angles and distances in XY plane
        angles = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        # Bin into n_beams angular sectors
        bin_indices = np.floor(
            (angles + np.pi) / (2 * np.pi) * self.n_beams
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_beams - 1)

        # Take minimum distance per bin (vectorized — ~10x faster than Python loop)
        ranges = np.full(self.n_beams, self.lidar_cfg["range"], dtype=np.float32)
        np.minimum.at(ranges, bin_indices, distances.astype(np.float32))

        return ranges

    def _on_collision(self, event):
        """Callback: flag collision."""
        self._collision_detected = True

    def _get_vehicle_state(self) -> Dict[str, float]:
        """Extract vehicle state from CARLA."""
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        acceleration = self._vehicle.get_acceleration()
        accel_magnitude = math.sqrt(
            acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2
        )

        control = self._vehicle.get_control()
        steering = control.steer  # [-1, 1]

        angular_velocity = self._vehicle.get_angular_velocity()
        yaw_rate = angular_velocity.z  # rad/s

        # Update distance traveled
        location = self._vehicle.get_location()
        if self._prev_location is not None:
            dx = location.x - self._prev_location.x
            dy = location.y - self._prev_location.y
            self._distance_traveled += math.sqrt(dx ** 2 + dy ** 2)
        self._prev_location = location

        return {
            "speed": speed,
            "acceleration": accel_magnitude,
            "steering": steering,
            "yaw_rate": yaw_rate,
            "distance_traveled": self._distance_traveled,
        }

    def _apply_action(self, action: np.ndarray):
        """Apply control action to the CARLA vehicle.

        Action mapping:
            action[0] → steering in [-1, 1]
            action[1] → throttle, remapped from [-1, 1] to [0.3, 1.0]
                         The car ALWAYS drives forward (this is racing!)
        """
        import carla

        steering = float(np.clip(action[0], -1.0, 1.0))
        raw_throttle = float(np.clip(action[1], -1.0, 1.0))

        # Remap [-1, 1] → [0.3, 1.0]: car always moves forward
        # Agent controls speed via throttle modulation, not stopping
        min_throttle = 0.3
        max_throttle = 1.0
        throttle = min_throttle + (raw_throttle + 1.0) / 2.0 * (max_throttle - min_throttle)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        self._vehicle.apply_control(control)

    def _cleanup(self):
        """Destroy CARLA actors."""
        actors = [self._lidar_sensor, self._collision_sensor, self._vehicle]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self._lidar_sensor = None
        self._collision_sensor = None
        self._vehicle = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment: destroy old actors, spawn new vehicle, sample command."""
        super().reset(seed=seed)

        # Cleanup previous episode
        self._cleanup()

        # Connect if not already
        if not self._connected:
            self._connect_to_carla()

        # Spawn vehicle and sensors
        self._spawn_vehicle()
        self._setup_sensors()

        # Reset state
        self._step_count = 0
        self._collision_detected = False
        self._distance_traveled = 0.0
        self._prev_location = None
        self._prev_vehicle_state = None
        self._cached_vehicle_state = None
        self._lidar_data = None

        # Sample new command
        self._current_command = self.command_manager.sample()

        # Tick world to get initial sensor data
        self._world.tick()
        time.sleep(0.1)  # Wait for sensor callbacks

        # Build observation
        obs = self._build_observation()
        info = {
            "command_text": self._current_command.text,
            "command_category": self._current_command.category,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step."""
        self._step_count += 1

        # Apply action
        self._apply_action(action)

        # Tick simulation
        self._world.tick()

        # Get vehicle state (called ONCE, cached for _build_observation)
        vehicle_state = self._get_vehicle_state()
        self._cached_vehicle_state = vehicle_state

        # Compute reward
        reward = compute_reward(
            vehicle_state=vehicle_state,
            prev_vehicle_state=self._prev_vehicle_state,
            command_category=self._current_command.category,
            collision=self._collision_detected,
            reward_config=self.reward_cfg,
        )

        # Check termination
        terminated = self._collision_detected
        truncated = self._step_count >= self.max_episode_steps

        # Save state for next step
        self._prev_vehicle_state = vehicle_state

        # Build observation (uses cached state, no double-counting)
        obs = self._build_observation()

        info = {
            "command_text": self._current_command.text,
            "command_category": self._current_command.category,
            "speed": vehicle_state["speed"],
            "distance": self._distance_traveled,
            "collision": self._collision_detected,
            "step": self._step_count,
        }

        # Reset collision flag (in case we want to continue despite collision)
        self._collision_detected = False

        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build the Dict observation from current sensor/state data."""
        # LiDAR
        if self._lidar_data is not None:
            lidar = self._lidar_data.copy()
        else:
            lidar = np.full(
                self.n_beams, self.lidar_cfg["range"], dtype=np.float32
            )

        # Command embedding
        command = self._current_command.embedding.copy()

        # Vehicle state — use cached value to avoid double side-effects
        vs = getattr(self, '_cached_vehicle_state', None)
        if vs is not None:
            vehicle_state = np.array([
                vs["speed"],
                vs["acceleration"],
                vs["steering"],
                vs["yaw_rate"],
                vs["distance_traveled"],
            ], dtype=np.float32)
        elif self._vehicle is not None:
            # Fallback for reset() — called before any step()
            vs = self._get_vehicle_state()
            vehicle_state = np.array([
                vs["speed"],
                vs["acceleration"],
                vs["steering"],
                vs["yaw_rate"],
                vs["distance_traveled"],
            ], dtype=np.float32)
        else:
            vehicle_state = np.zeros(self.vehicle_state_dim, dtype=np.float32)

        return {
            "lidar": lidar,
            "command": command,
            "vehicle_state": vehicle_state,
        }

    @property
    def vehicle(self):
        """Public access to the CARLA vehicle actor (for external camera attachment etc)."""
        return self._vehicle

    def render(self):
        """Rendering handled by CARLA server."""
        pass

    def close(self):
        """Cleanup all CARLA resources."""
        self._cleanup()

        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)

        self._connected = False
