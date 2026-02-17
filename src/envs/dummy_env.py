"""
Dummy CARLA Environment for local development and testing.

Mimics the exact same observation/action spaces as the real CarlaEnv
but without requiring a CARLA server. Returns random but valid observations.
Useful for:
- Local pipeline testing on Mac M1
- Verifying policy network shapes
- Running SB3 check_env() locally
- Unit testing
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from src.utils.commands import CommandManager


class DummyCarlaEnv(gym.Env):
    """
    Lightweight environment with identical interface to CarlaEnv.

    Simulates a simple vehicle moving forward with random perturbations.
    No CARLA dependency required.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        n_lidar_beams: int = 1080,
        command_dim: int = 384,
        vehicle_state_dim: int = 5,
        max_episode_steps: int = 1000,
        command_manager: Optional[CommandManager] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            n_lidar_beams: Number of LiDAR range values.
            command_dim: Dimension of command embedding.
            vehicle_state_dim: Number of vehicle state features.
            max_episode_steps: Maximum steps before truncation.
            command_manager: CommandManager instance for sampling commands.
            render_mode: Render mode (not used in dummy env).
        """
        super().__init__()

        self.n_lidar_beams = n_lidar_beams
        self.command_dim = command_dim
        self.vehicle_state_dim = vehicle_state_dim
        self.max_episode_steps = max_episode_steps
        self.command_manager = command_manager or CommandManager(encoder=None)
        self.render_mode = render_mode

        # ── Observation Space ─────────────────────────────────────
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(
                low=0.0,
                high=50.0,
                shape=(n_lidar_beams,),
                dtype=np.float32,
            ),
            "command": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(command_dim,),
                dtype=np.float32,
            ),
            "vehicle_state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vehicle_state_dim,),
                dtype=np.float32,
            ),
        })

        # ── Action Space ──────────────────────────────────────────
        # [steering, throttle/brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── Internal State ────────────────────────────────────────
        self._step_count = 0
        self._current_command = None
        self._speed = 0.0
        self._steering = 0.0
        self._distance = 0.0
        self._done = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and sample a new command."""
        super().reset(seed=seed)

        self._step_count = 0
        self._speed = 0.0
        self._steering = 0.0
        self._distance = 0.0
        self._done = False

        # Sample a new command
        self._current_command = self.command_manager.sample()

        obs = self._get_observation()
        info = {
            "command_text": self._current_command.text,
            "command_category": self._current_command.category,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: [steering, throttle/brake] in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Simple physics simulation
        steering_input = float(np.clip(action[0], -1, 1))
        throttle_input = float(np.clip(action[1], -1, 1))

        self._steering = steering_input
        self._speed = np.clip(self._speed + throttle_input * 2.0, 0, 40.0)
        self._distance += self._speed * 0.05  # dt = 0.05s

        # Random collision (rare)
        collision = self.np_random.random() < 0.002

        # Compute reward
        reward = self._compute_simple_reward(collision)

        terminated = collision
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = {
            "command_text": self._current_command.text,
            "command_category": self._current_command.category,
            "speed": self._speed,
            "distance": self._distance,
            "collision": collision,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build the current observation dict."""
        # Simulated LiDAR: base range with some noise
        lidar = self.np_random.uniform(5.0, 50.0, size=(self.n_lidar_beams,)).astype(
            np.float32
        )

        # Command embedding
        command = self._current_command.embedding.copy()

        # Vehicle state: [speed, acceleration, steering, yaw_rate, distance]
        vehicle_state = np.array(
            [
                self._speed,
                0.0,  # acceleration placeholder
                self._steering,
                0.0,  # yaw_rate placeholder
                0.0,  # lane_deviation placeholder (was distance)
            ],
            dtype=np.float32,
        )

        return {
            "lidar": lidar,
            "command": command,
            "vehicle_state": vehicle_state,
        }

    def _compute_simple_reward(self, collision: bool) -> float:
        """Simple reward for dummy env: reward speed, penalize collision."""
        if collision:
            return -100.0
        return self._speed / 40.0  # Normalized speed reward

    def render(self):
        """Render not implemented for dummy env."""
        pass

    def close(self):
        """Cleanup."""
        pass
