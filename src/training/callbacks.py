"""
SB3 Training Callbacks for Language-Conditioned Racing Agent.

Provides callbacks for:
- Command curriculum progression
- Per-command-type metrics logging
- Model checkpointing
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Optional


class CurriculumCallback(BaseCallback):
    """
    Progressively introduces new command categories during training.

    Updates the environment's command manager to allow more categories
    as training progresses, following the schedule in the config.
    """

    def __init__(
        self,
        curriculum: Dict[str, int],
        command_manager=None,
        verbose: int = 1,
    ):
        """
        Args:
            curriculum: Dict mapping "category_from_step" -> step number.
            command_manager: CommandManager instance to update.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.curriculum = curriculum
        self.command_manager = command_manager
        self._active_categories: List[str] = []

    def _on_step(self) -> bool:
        if self.command_manager is None:
            return True

        current_step = self.num_timesteps
        new_categories = self.command_manager.get_curriculum_categories(
            current_step, self.curriculum
        )

        if set(new_categories) != set(self._active_categories):
            self._active_categories = new_categories
            # Restrict future sampling to these categories
            self.command_manager.allowed_categories = new_categories
            if self.verbose > 0:
                print(
                    f"[Curriculum] Step {current_step}: "
                    f"Active categories = {new_categories}"
                )

        return True


class MetricsCallback(BaseCallback):
    """
    Logs per-command-type metrics to TensorBoard.

    Tracks reward, speed, and episode length grouped by command category.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: Dict[str, List[float]] = {}
        self._episode_speeds: Dict[str, List[float]] = {}
        self._episode_lengths: Dict[str, List[int]] = {}
        self._current_rewards: Dict[int, float] = {}
        self._current_speeds: Dict[int, List[float]] = {}
        self._current_lengths: Dict[int, int] = {}
        self._current_categories: Dict[int, str] = {}

    def _on_step(self) -> bool:
        # Get info from all environments
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            env_id = i
            category = info.get("command_category", "unknown")
            speed = info.get("speed", 0.0)

            # Initialize tracking for this env
            if env_id not in self._current_rewards:
                self._current_rewards[env_id] = 0.0
                self._current_speeds[env_id] = []
                self._current_lengths[env_id] = 0
                self._current_categories[env_id] = category

            self._current_rewards[env_id] += rewards[i]
            self._current_speeds[env_id].append(speed)
            self._current_lengths[env_id] += 1

            # On episode end, record metrics
            if dones is not None and dones[i]:
                cat = self._current_categories[env_id]

                if cat not in self._episode_rewards:
                    self._episode_rewards[cat] = []
                    self._episode_speeds[cat] = []
                    self._episode_lengths[cat] = []

                self._episode_rewards[cat].append(self._current_rewards[env_id])
                self._episode_speeds[cat].append(
                    np.mean(self._current_speeds[env_id])
                    if self._current_speeds[env_id]
                    else 0.0
                )
                self._episode_lengths[cat].append(self._current_lengths[env_id])

                # Reset tracking
                self._current_rewards[env_id] = 0.0
                self._current_speeds[env_id] = []
                self._current_lengths[env_id] = 0
                self._current_categories[env_id] = category

        # Log aggregated metrics every 10000 steps
        if self.num_timesteps % 10000 == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        for cat, rewards in self._episode_rewards.items():
            if len(rewards) > 0:
                self.logger.record(
                    f"command/{cat}/mean_reward", np.mean(rewards[-20:])
                )
            speeds = self._episode_speeds.get(cat, [])
            if len(speeds) > 0:
                self.logger.record(
                    f"command/{cat}/mean_speed", np.mean(speeds[-20:])
                )
            lengths = self._episode_lengths.get(cat, [])
            if len(lengths) > 0:
                self.logger.record(
                    f"command/{cat}/mean_length", np.mean(lengths[-20:])
                )
