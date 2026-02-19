"""
Reward functions for Language-Conditioned Racing Agent.

Computes command-aware rewards that shape the agent's behavior based
on the current language command category (aggressive, defensive,
neutral).
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
        command_category: One of "aggressive", "defensive", "neutral".
        collision: Whether a collision occurred this step.
        reward_config: Reward weights from config file.

    Returns:
        Total reward (float).
    """
    reward = 0.0
    max_speed = reward_config.get("max_speed", 40.0)

    # ── 1. Collision Penalty ──────────────────────────────────────
    if collision:
        return reward_config.get("collision_penalty", -10.0)

    # ── 2. Progress Reward ────────────────────────────────────────
    # Reward forward progress along the track
    if prev_vehicle_state is not None:
        progress = (
            vehicle_state["distance_traveled"]
            - prev_vehicle_state["distance_traveled"]
        )
        reward += reward_config.get("progress_weight", 2.0) * progress

    # ── 3. Speed Component (DOMINANT REWARD) ──────────────────────
    speed = vehicle_state["speed"]
    speed_weight = _get_modified_weight(
        "speed_weight", command_category, reward_config
    )
    # Quadratic speed reward — strongly incentivises high speeds
    speed_ratio = speed / max_speed
    reward += speed_weight * speed_ratio

    # ── 4. Standing-Still Penalty ─────────────────────────────────
    # Harsh penalty for not moving — prevents the "sit still" strategy
    min_speed = reward_config.get("min_speed_threshold", 2.0)  # m/s
    idle_penalty = reward_config.get("idle_penalty", -3.0)
    if speed < min_speed:
        reward += idle_penalty  # Strong penalty for being nearly stationary

    # ── 5. Speed Bonus — reward for reaching target speeds ────────
    # Extra reward for driving above 50% of max speed (>20 m/s = 72 km/h)
    speed_bonus_threshold = reward_config.get("speed_bonus_threshold", 0.5)
    speed_bonus_weight = reward_config.get("speed_bonus_weight", 2.0)
    if speed_ratio > speed_bonus_threshold:
        reward += speed_bonus_weight * (speed_ratio - speed_bonus_threshold)

    # ── 6. Alive Bonus ────────────────────────────────────────────
    # Reward surviving each step — makes long episodes at high speed
    # far more rewarding than short episodes
    alive_bonus = reward_config.get("alive_bonus", 0.0)
    reward += alive_bonus

    # ── 7. Smoothness Component ───────────────────────────────────
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

    # ── 8. Lane Deviation Penalty ─────────────────────────────────
    lane_dev = vehicle_state.get("lane_deviation", 0.0)
    lane_weight = _get_modified_weight(
        "lane_deviation_weight", command_category, reward_config
    )
    reward -= lane_weight * abs(lane_dev)

    # ── 9. Anti-Spin Penalty ──────────────────────────────────────
    yaw_rate = abs(vehicle_state["yaw_rate"])
    if yaw_rate > 1.5: # Threshold for spinning
        reward -= 2.0 * yaw_rate

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
