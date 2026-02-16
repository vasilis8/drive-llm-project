"""Tests for environment wrappers (DummyCarlaEnv) and reward function."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.dummy_env import DummyCarlaEnv
from src.envs.rewards import compute_reward
from src.utils.commands import CommandManager


# ── DummyCarlaEnv Tests ───────────────────────────────────────────────


class TestDummyCarlaEnv:
    """Test the lightweight dummy environment."""

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

    def test_observation_space_structure(self, env):
        """Observation space should be a Dict with correct keys."""
        assert "lidar" in env.observation_space.spaces
        assert "command" in env.observation_space.spaces
        assert "vehicle_state" in env.observation_space.spaces

    def test_observation_shapes(self, env):
        """Observation shapes should match config."""
        assert env.observation_space["lidar"].shape == (1080,)
        assert env.observation_space["command"].shape == (384,)
        assert env.observation_space["vehicle_state"].shape == (5,)

    def test_action_space(self, env):
        """Action space should be Box with 2 dims in [-1, 1]."""
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, [-1, -1])
        assert np.allclose(env.action_space.high, [1, 1])

    def test_reset_returns_valid_obs(self, env):
        """Reset should return obs within observation space."""
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs), (
            f"Obs not in space. Shapes: "
            f"lidar={obs['lidar'].shape}, "
            f"command={obs['command'].shape}, "
            f"vehicle_state={obs['vehicle_state'].shape}"
        )
        assert "command_text" in info
        assert "command_category" in info

    def test_step_returns_valid(self, env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncation_at_max_steps(self, env):
        """Episode should truncate at max_episode_steps."""
        env.reset(seed=42)
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # If we reached 100 steps without crashing, should be truncated
        if not terminated:
            assert truncated

    def test_sb3_check_env(self, env):
        """SB3's check_env should pass."""
        from stable_baselines3.common.env_checker import check_env
        # check_env raises if something is wrong
        check_env(env, warn=True)

    def test_different_commands_on_reset(self, env):
        """Different resets should (eventually) give different commands."""
        commands = set()
        for i in range(20):
            obs, info = env.reset(seed=i)
            commands.add(info["command_text"])
        assert len(commands) > 1, "Expected different commands across resets"


# ── Reward Function Tests ─────────────────────────────────────────────


class TestRewardFunction:
    """Test the command-aware reward computation."""

    def _make_state(self, speed=10.0, steering=0.0, distance=100.0):
        return {
            "speed": speed,
            "acceleration": 0.0,
            "steering": steering,
            "yaw_rate": 0.0,
            "distance_traveled": distance,
        }

    def test_collision_penalty(self):
        """Collision should return large negative reward."""
        state = self._make_state()
        reward = compute_reward(
            vehicle_state=state,
            prev_vehicle_state=None,
            command_category="neutral",
            collision=True,
            reward_config={"collision_penalty": -100.0},
        )
        assert reward == -100.0

    def test_progress_reward(self):
        """Moving forward should give positive reward."""
        prev = self._make_state(distance=0.0)
        curr = self._make_state(distance=10.0)
        reward = compute_reward(
            vehicle_state=curr,
            prev_vehicle_state=prev,
            command_category="neutral",
            collision=False,
            reward_config={"progress_weight": 1.0, "speed_weight": 0.0,
                           "smoothness_weight": 0.0, "lane_deviation_weight": 0.0,
                           "max_speed": 40.0},
        )
        assert reward > 0

    def test_aggressive_rewards_high_speed(self):
        """Aggressive command should reward speed more than conservative."""
        state = self._make_state(speed=30.0)
        config = {
            "speed_weight": 0.5,
            "progress_weight": 0.0,
            "smoothness_weight": 0.0,
            "lane_deviation_weight": 0.0,
            "max_speed": 40.0,
            "command_modifiers": {
                "aggressive": {"speed_weight": 1.5},
                "conservative": {"speed_weight": 0.2},
            },
        }

        reward_aggressive = compute_reward(
            state, None, "aggressive", False, config
        )
        reward_conservative = compute_reward(
            state, None, "conservative", False, config
        )
        assert reward_aggressive > reward_conservative

    def test_conservative_penalizes_jerk(self):
        """Conservative command should penalize steering jerk more."""
        prev = self._make_state(steering=0.0)
        curr = self._make_state(steering=0.5)  # big jerk
        config = {
            "speed_weight": 0.0,
            "progress_weight": 0.0,
            "smoothness_weight": 0.3,
            "lane_deviation_weight": 0.0,
            "max_speed": 40.0,
            "command_modifiers": {
                "aggressive": {"smoothness_weight": 0.1},
                "conservative": {"smoothness_weight": 1.0},
            },
        }

        reward_aggressive = compute_reward(
            curr, prev, "aggressive", False, config
        )
        reward_conservative = compute_reward(
            curr, prev, "conservative", False, config
        )
        # Conservative should be punished more for the jerk
        assert reward_conservative < reward_aggressive
