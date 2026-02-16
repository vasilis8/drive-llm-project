"""
Main training script for Language-Conditioned Racing Agent.

Usage (on cloud with CARLA running):
    python -m src.training.train --config configs/default.yaml

Usage (local testing with DummyEnv):
    python -m src.training.train --config configs/default.yaml --dummy
"""

import argparse
import os
import yaml
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.models.instruction_encoder import InstructionEncoder
from src.models.policy import create_ppo_agent
from src.utils.commands import CommandManager
from src.training.callbacks import CurriculumCallback, MetricsCallback


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_env(config: dict, use_dummy: bool = False, command_manager=None):
    """
    Create the training environment.

    Args:
        config: Full config dict.
        use_dummy: If True, use DummyCarlaEnv (no CARLA needed).
        command_manager: CommandManager instance.

    Returns:
        Gymnasium environment instance.
    """
    if use_dummy:
        from src.envs.dummy_env import DummyCarlaEnv

        obs_cfg = config.get("observation", {})
        return DummyCarlaEnv(
            n_lidar_beams=obs_cfg.get("lidar_dim", 1080),
            command_dim=obs_cfg.get("command_dim", 384),
            vehicle_state_dim=obs_cfg.get("vehicle_state_dim", 5),
            max_episode_steps=config.get("training", {}).get(
                "max_episode_steps", 2000
            ),
            command_manager=command_manager,
        )
    else:
        from src.envs.carla_env import CarlaEnv

        return CarlaEnv(
            carla_config=config.get("carla", {}),
            lidar_config=config.get("sensors", {}).get("lidar", {}),
            vehicle_config=config.get("vehicle", {}),
            reward_config=config.get("rewards", {}),
            command_manager=command_manager,
            max_episode_steps=config.get("training", {}).get(
                "max_episode_steps", 2000
            ),
        )


def train(config_path: str, use_dummy: bool = False, resume_from: str = None):
    """
    Main training loop.

    Args:
        config_path: Path to YAML config file.
        use_dummy: Use DummyCarlaEnv for local testing.
        resume_from: Path to model checkpoint to resume from.
    """
    config = load_config(config_path)
    training_cfg = config.get("training", {})

    # ── Setup directories ─────────────────────────────────────
    log_dir = training_cfg.get("log_dir", "logs/")
    checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Instruction Encoder ───────────────────────────────────
    print("Loading instruction encoder...")
    encoder = InstructionEncoder()
    command_manager = CommandManager(encoder=encoder)
    print(f"Loaded {command_manager.n_commands} commands with {encoder.embedding_dim}-dim embeddings")

    # ── Environment ───────────────────────────────────────────
    print(f"Creating environment (dummy={use_dummy})...")
    env = create_env(config, use_dummy=use_dummy, command_manager=command_manager)

    # ── PPO Agent ─────────────────────────────────────────────
    if resume_from:
        from stable_baselines3 import PPO

        print(f"Resuming from checkpoint: {resume_from}")
        agent = PPO.load(resume_from, env=env, tensorboard_log=log_dir)
    else:
        print("Creating new PPO agent...")
        agent = create_ppo_agent(env, config=config, tensorboard_log=log_dir)

    # ── Callbacks ─────────────────────────────────────────────
    curriculum = training_cfg.get("curriculum", {})
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=training_cfg.get("checkpoint_freq", 50_000),
            save_path=checkpoint_dir,
            name_prefix="drive_llm",
        ),
        CurriculumCallback(
            curriculum=curriculum,
            command_manager=command_manager,
        ),
        MetricsCallback(),
    ])

    # ── Training ──────────────────────────────────────────────
    total_timesteps = training_cfg.get("total_timesteps", 2_000_000)
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"  Logs: {log_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "drive_llm_final")
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Language-Conditioned Racing Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyCarlaEnv (no CARLA server needed)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        use_dummy=args.dummy,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
