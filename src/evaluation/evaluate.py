"""
Evaluation module for Language-Conditioned Racing Agent.
"""

import argparse
import os
import json
import yaml
import numpy as np
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.models.instruction_encoder import InstructionEncoder
from src.utils.commands import CommandManager, ALL_CATEGORIES


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_agent(
    model_path: str,
    config_path: str = "configs/default.yaml",
    n_episodes: int = 50,
    use_dummy: bool = True,
    deterministic: bool = True,
    save_results: str = None,
) -> Dict:
    config = load_config(config_path)

    print("Loading encoder...")
    encoder = InstructionEncoder()
    command_manager = CommandManager(encoder=encoder)

    if use_dummy:
        from src.envs.dummy_env import DummyCarlaEnv
        obs_cfg = config.get("observation", {})
        env = DummyCarlaEnv(
            n_lidar_beams=obs_cfg.get("lidar_dim", 1080),
            command_dim=obs_cfg.get("command_dim", 384),
            max_episode_steps=1000,
            command_manager=command_manager,
        )
    else:
        from src.envs.carla_env import CarlaEnv
        env = CarlaEnv(
            carla_config=config.get("carla", {}),
            lidar_config=config.get("sensors", {}).get("lidar", {}),
            vehicle_config=config.get("vehicle", {}),
            reward_config=config.get("rewards", {}),
            command_manager=command_manager,
        )

    # ── THE FIX: Wrap in VecEnv and load Normalization Stats ──
    env = DummyVecEnv([lambda: env])
    
    # Locate the normalization stats saved during training
    model_dir = os.path.dirname(model_path)
    vec_norm_path = os.path.join(model_dir, "drive_llm_final_vecnormalize.pkl")
    
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}...")
        env = VecNormalize.load(vec_norm_path, env)
        # CRITICAL: Do not update stats or normalize rewards during evaluation
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No VecNormalize stats found! Agent may drive blindly.")

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)

    episodes_per_category = max(1, n_episodes // len(ALL_CATEGORIES))
    results = {"per_category": {}, "overall": {}}

    all_rewards, all_lengths, all_speeds, all_collisions = [], [], [], []

    for category in ALL_CATEGORIES:
        print(f"\nEvaluating category: {category}")
        cat_rewards, cat_speeds, cat_lengths, cat_steering_jerk = [], [], [], []
        cat_collisions = 0

        for ep in range(episodes_per_category):
            command_manager.allowed_categories = [category]
            obs = env.reset()

            episode_reward = 0.0
            episode_speeds = []
            episode_steerings = []
            step = 0

            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = env.step(action)
                
                # VecEnv returns lists, so we take index 0
                info = infos[0]

                episode_reward += rewards[0]
                episode_speeds.append(info.get("speed", 0.0))
                step += 1

                if "vehicle_state" in obs:
                    episode_steerings.append(obs["vehicle_state"][0][2])

                if dones[0]:
                    break

            cat_rewards.append(episode_reward)
            cat_speeds.append(np.mean(episode_speeds) if episode_speeds else 0.0)
            cat_lengths.append(step)
            if info.get("collision", False):
                cat_collisions += 1

            if len(episode_steerings) > 1:
                steerings = np.array(episode_steerings)
                jerk = np.mean(np.abs(np.diff(steerings)))
                cat_steering_jerk.append(jerk)

        results["per_category"][category] = {
            "mean_reward": float(np.mean(cat_rewards)),
            "std_reward": float(np.std(cat_rewards)),
            "mean_speed": float(np.mean(cat_speeds)),
            "mean_length": float(np.mean(cat_lengths)),
            "crash_rate": cat_collisions / episodes_per_category,
            "mean_steering_jerk": float(np.mean(cat_steering_jerk) if cat_steering_jerk else 0.0),
            "n_episodes": episodes_per_category,
        }

        all_rewards.extend(cat_rewards)
        all_speeds.extend(cat_speeds)
        all_lengths.extend(cat_lengths)
        all_collisions.append(cat_collisions)

        print(f"  Reward: {np.mean(cat_rewards):.2f} ± {np.std(cat_rewards):.2f}")
        print(f"  Speed:  {np.mean(cat_speeds):.2f} m/s")
        print(f"  Length: {np.mean(cat_lengths):.0f} steps")
        print(f"  Crash:  {cat_collisions}/{episodes_per_category}")

    results["overall"] = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_speed": float(np.mean(all_speeds)),
        "mean_length": float(np.mean(all_lengths)),
        "crash_rate": sum(all_collisions) / n_episodes,
        "total_episodes": n_episodes,
    }

    print(f"\n{'='*50}")
    print(f"Overall: Reward={results['overall']['mean_reward']:.2f}, "
          f"Crash Rate={results['overall']['crash_rate']:.1%}")

    if save_results:
        os.makedirs(os.path.dirname(save_results) or ".", exist_ok=True)
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_results}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--output", type=str, default="results/evaluation.json")
    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes,
        use_dummy=args.dummy,
        save_results=args.output,
    )


if __name__ == "__main__":
    main()