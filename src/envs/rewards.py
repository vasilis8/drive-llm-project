"""
Reward functions for Language-Conditioned Racing Agent.

Computes command-aware rewards that shape the agent's behavior based
on the current language command category (aggressive, conservative,
defensive, neutral).
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_reward(
    vehicle_state: Dict[str, float],
    prev_vehicle_state: Optional[Dict[str, float]],
    command_category: str,
    collision: bool,
    reward_config: Dict[str, Any],
) -> float:
    """
    Compute the total reward for a single timestep.

    Args:
        vehicle_state: Current vehicle state dict with keys:
            - speed: current speed in m/s
            - acceleration: current acceleration in m/s^2
            - steering: current steering angle [-1, 1]
            - yaw_rate: angular velocity in rad/s
            - distance_traveled: cumulative distance along track
        prev_vehicle_state: Previous timestep's vehicle state (None on first step).
        command_category: One of "aggressive", "conservative", "defensive", "neutral".
        collision: Whether a collision occurred this step.
        reward_config: Reward weights from config file.

    Returns:
        Total reward (float).
    """
    reward = 0.0

    # ── 1. Collision Penalty ──────────────────────────────────────
    if collision:
        return reward_config.get("collision_penalty", -100.0)

    # ── 2. Progress Reward ────────────────────────────────────────
    # Reward forward progress along the track
    if prev_vehicle_state is not None:
        progress = (
            vehicle_state["distance_traveled"]
            - prev_vehicle_state["distance_traveled"]
        )
        reward += reward_config.get("progress_weight", 1.0) * progress

    # ── 3. Speed Component ────────────────────────────────────────
    speed = vehicle_state["speed"]
    speed_weight = _get_modified_weight(
        "speed_weight", command_category, reward_config
    )
    # Normalize speed reward: positive for moving, scaled by speed
    reward += speed_weight * (speed / reward_config.get("max_speed", 40.0))

    # ── 4. Smoothness Component ───────────────────────────────────
    # Penalize jerky steering and throttle changes
    if prev_vehicle_state is not None:
        steering_jerk = abs(
            vehicle_state["steering"] - prev_vehicle_state["steering"]
        )
        accel_jerk = abs(
            vehicle_state["acceleration"] - prev_vehicle_state["acceleration"]
        )
        smoothness_penalty = steering_jerk + 0.5 * accel_jerk

        smoothness_weight = _get_modified_weight(
            "smoothness_weight", command_category, reward_config
        )
        reward -= smoothness_weight * smoothness_penalty

    # ── 5. Lane Deviation Penalty ─────────────────────────────────
    lane_dev = vehicle_state.get("lane_deviation", 0.0)
    lane_weight = _get_modified_weight(
        "lane_deviation_weight", command_category, reward_config
    )
    reward -= lane_weight * abs(lane_dev)

    return reward


def _get_modified_weight(
    weight_name: str,
    command_category: str,
    reward_config: Dict[str, Any],
) -> float:
    """
    Get a reward weight, potentially modified by command category.

    Looks up the base weight, then checks for a category-specific override
    in reward_config["command_modifiers"][category].
    """
    base_weight = reward_config.get(weight_name, 0.0)

    modifiers = reward_config.get("command_modifiers", {})
    category_mods = modifiers.get(command_category, {})

    return category_mods.get(weight_name, base_weight)
