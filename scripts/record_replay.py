"""
Replay & Record: Visualize a trained agent driving in CARLA.

Attaches a chase camera to the vehicle, overlays the active command,
and records MP4 videos for each command category.

Usage (on cloud with CARLA running):
    python -m scripts.record_replay --model checkpoints/drive_llm_final

This generates videos like:
    recordings/aggressive_push_hard.mp4
    recordings/conservative_conserve_tires.mp4
    ...
"""

import argparse
import os
import sys
import time
import numpy as np
from collections import deque
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def record_agent(
    model_path: str,
    config_path: str = "configs/default.yaml",
    output_dir: str = "recordings/",
    episodes_per_category: int = 1,
    max_steps: int = 1000,
    fps: int = 20,
    resolution: tuple = (1280, 720),
):
    """
    Record the trained agent driving in CARLA with a chase camera.

    Args:
        model_path: Path to saved PPO checkpoint.
        config_path: YAML config path.
        output_dir: Directory to save MP4 videos.
        episodes_per_category: Episodes to record per command type.
        max_steps: Max steps per episode.
        fps: Video frame rate.
        resolution: Camera resolution (width, height).
    """
    import yaml
    import cv2
    import carla
    from stable_baselines3 import PPO
    from src.models.instruction_encoder import InstructionEncoder
    from src.utils.commands import CommandManager, ALL_CATEGORIES
    from src.envs.carla_env import CarlaEnv

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Setup encoder & commands
    print("Loading encoder...")
    encoder = InstructionEncoder()
    command_manager = CommandManager(encoder=encoder)

    # Create CARLA environment
    print("Connecting to CARLA...")
    env = CarlaEnv(
        carla_config=config.get("carla", {}),
        lidar_config=config.get("sensors", {}).get("lidar", {}),
        vehicle_config=config.get("vehicle", {}),
        reward_config=config.get("rewards", {}),
        command_manager=command_manager,
        max_episode_steps=max_steps,
    )

    # Load trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)

    # Get CARLA client for camera setup
    client = carla.Client(
        config.get("carla", {}).get("host", "localhost"),
        config.get("carla", {}).get("port", 2000),
    )
    world = client.get_world()

    # ── Record each command category ──────────────────────────
    for category in ALL_CATEGORIES:
        commands = command_manager.get_category_commands(category)

        for cmd_idx in range(min(episodes_per_category, len(commands))):
            command = commands[cmd_idx]
            safe_name = command.text.lower().replace(" ", "_").replace("!", "")
            video_path = os.path.join(output_dir, f"{category}_{safe_name}.mp4")

            print(f"\n🎬 Recording: [{category}] \"{command.text}\"")
            print(f"   Output: {video_path}")

            # Reset env with specific command
            command_manager.allowed_categories = [category]
            obs, info = env.reset()

            # ── Attach chase camera ───────────────────────────
            camera_frames = deque(maxlen=2)
            camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(resolution[0]))
            camera_bp.set_attribute("image_size_y", str(resolution[1]))
            camera_bp.set_attribute("fov", "90")

            # Chase camera: behind and above the vehicle
            camera_transform = carla.Transform(
                carla.Location(x=-8.0, y=0.0, z=5.0),
                carla.Rotation(pitch=-15.0),
            )

            camera = world.spawn_actor(
                camera_bp, camera_transform, attach_to=env.vehicle
            )

            def camera_callback(image):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((resolution[1], resolution[0], 4))[:, :, :3]
                camera_frames.append(array)

            camera.listen(camera_callback)

            # ── Setup video writer ────────────────────────────
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                video_path, fourcc, fps, resolution
            )

            # ── Run episode ───────────────────────────────────
            total_reward = 0
            step = 0

            # Wait for first camera frame
            time.sleep(0.5)

            while step < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                # Write frame with overlay
                if camera_frames:
                    frame = camera_frames[-1].copy()
                    frame = _add_overlay(
                        frame,
                        command_text=command.text,
                        category=category,
                        speed=info.get("speed", 0),
                        steering=action[0],
                        throttle=action[1],
                        reward=total_reward,
                        step=step,
                    )
                    # OpenCV expects BGR
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if terminated or truncated:
                    break

            # Cleanup
            camera.stop()
            camera.destroy()
            writer.release()

            print(f"   ✅ Done! {step} steps, reward: {total_reward:.1f}")

    env.close()
    print(f"\n🎬 All recordings saved to {output_dir}/")


def _add_overlay(
    frame: np.ndarray,
    command_text: str,
    category: str,
    speed: float,
    steering: float,
    throttle: float,
    reward: float,
    step: int,
) -> np.ndarray:
    """Add HUD overlay with command, speed, and controls to video frame."""
    import cv2

    h, w = frame.shape[:2]

    # Category colors (BGR for OpenCV)
    colors = {
        "aggressive": (60, 76, 231),      # Red
        "conservative": (113, 204, 46),    # Green
        "defensive": (219, 152, 52),       # Blue
        "neutral": (182, 89, 155),         # Purple
    }
    color = colors.get(category, (200, 200, 200))

    # Semi-transparent overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Command text
    cv2.putText(
        frame,
        f'"{command_text}"',
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    # Category badge
    cv2.putText(
        frame,
        f"[{category.upper()}]",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    # Speed
    cv2.putText(
        frame,
        f"Speed: {speed:.1f} m/s",
        (w - 300, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Controls
    cv2.putText(
        frame,
        f"Steer: {steering:+.2f}  Thr: {throttle:+.2f}",
        (w - 380, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    # Bottom bar: reward and step
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        f"Step: {step}  |  Total Reward: {reward:.1f}",
        (20, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Steering indicator bar
    bar_center = w // 2
    bar_y = h - 30
    bar_width = 200
    steer_pos = int(bar_center + steering * bar_width / 2)
    cv2.line(frame, (bar_center - bar_width // 2, bar_y),
             (bar_center + bar_width // 2, bar_y), (100, 100, 100), 2)
    cv2.circle(frame, (steer_pos, bar_y), 8, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Record trained agent driving in CARLA"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    parser.add_argument(
        "--output", type=str, default="recordings/"
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per command category"
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--fps", type=int, default=20
    )
    parser.add_argument(
        "--width", type=int, default=1280
    )
    parser.add_argument(
        "--height", type=int, default=720
    )
    args = parser.parse_args()

    record_agent(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        episodes_per_category=args.episodes,
        max_steps=args.steps,
        fps=args.fps,
        resolution=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
