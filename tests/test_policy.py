"""Tests for custom multimodal policy network."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.dummy_env import DummyCarlaEnv
from src.models.policy import MultimodalFeatureExtractor, create_ppo_agent
from src.utils.commands import CommandManager


class TestMultimodalFeatureExtractor:
    """Test the custom 1D-CNN + MLP feature extractor."""

    @pytest.fixture
    def env(self):
        mgr = CommandManager(encoder=None)
        env = DummyCarlaEnv(
            n_lidar_beams=1080,
            command_dim=384,
            max_episode_steps=100,
            command_manager=mgr,
        )
        yield env
        env.close()

    def test_extractor_output_dim(self, env):
        """Feature extractor should output the configured features_dim."""
        import torch

        extractor = MultimodalFeatureExtractor(
            observation_space=env.observation_space,
            features_dim=128,
        )

        # Create a dummy observation batch
        obs = {
            "lidar": torch.randn(4, 1080),
            "command": torch.randn(4, 384),
            "vehicle_state": torch.randn(4, 5),
        }

        features = extractor(obs)
        assert features.shape == (4, 128), f"Expected (4, 128), got {features.shape}"

    def test_extractor_different_features_dim(self, env):
        """Should work with different features_dim values."""
        import torch

        for dim in [64, 128, 256]:
            extractor = MultimodalFeatureExtractor(
                observation_space=env.observation_space,
                features_dim=dim,
            )
            obs = {
                "lidar": torch.randn(2, 1080),
                "command": torch.randn(2, 384),
                "vehicle_state": torch.randn(2, 5),
            }
            features = extractor(obs)
            assert features.shape == (2, dim)

    def test_extractor_single_sample(self, env):
        """Should work with batch_size=1."""
        import torch

        extractor = MultimodalFeatureExtractor(
            observation_space=env.observation_space,
            features_dim=128,
        )
        obs = {
            "lidar": torch.randn(1, 1080),
            "command": torch.randn(1, 384),
            "vehicle_state": torch.randn(1, 5),
        }
        features = extractor(obs)
        assert features.shape == (1, 128)


class TestPPOAgent:
    """Test PPO agent creation and short training."""

    @pytest.fixture
    def env(self):
        mgr = CommandManager(encoder=None)
        env = DummyCarlaEnv(
            n_lidar_beams=1080,
            command_dim=384,
            max_episode_steps=50,
            command_manager=mgr,
        )
        yield env
        env.close()

    def test_agent_creation(self, env):
        """PPO agent should be created with custom policy."""
        agent = create_ppo_agent(env)
        assert agent is not None
        assert agent.policy is not None

    def test_short_training(self, env):
        """Short training run (128 steps) should complete without error."""
        agent = create_ppo_agent(env, config={
            "training": {
                "n_steps": 64,
                "batch_size": 32,
            }
        })
        # Train for a tiny number of steps
        agent.learn(total_timesteps=128)

    def test_predict(self, env):
        """Agent should predict valid actions."""
        agent = create_ppo_agent(env)
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs, deterministic=True)
        assert env.action_space.contains(action), f"Invalid action: {action}"

    def test_save_load(self, env, tmp_path):
        """Agent should be saveable and loadable."""
        from stable_baselines3 import PPO

        agent = create_ppo_agent(env)
        save_path = str(tmp_path / "test_model")
        agent.save(save_path)

        # Load and verify
        loaded = PPO.load(save_path, env=env)
        obs, _ = env.reset(seed=42)
        action, _ = loaded.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
