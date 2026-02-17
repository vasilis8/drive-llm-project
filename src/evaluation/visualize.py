"""
Visualization module for Language-Conditioned Racing Agent.

Generates plots and analysis charts from training logs and evaluation results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_evaluation_results(
    results_path: str,
    output_dir: str = "results/plots/",
):
    """
    Generate plots from evaluation results JSON.

    Args:
        results_path: Path to evaluation.json.
        output_dir: Directory to save plots.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    per_category = results["per_category"]
    categories = list(per_category.keys())

    # Color scheme
    colors = {
        "aggressive": "#e74c3c",
        "defensive": "#3498db",
        "neutral": "#9b59b6",
    }

    # ── 1. Mean Reward by Category ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = [per_category[c]["mean_reward"] for c in categories]
    stds = [per_category[c]["std_reward"] for c in categories]
    bars = ax.bar(
        categories,
        rewards,
        yerr=stds,
        color=[colors.get(c, "#95a5a6") for c in categories],
        capsize=5,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Reward by Command Category", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_by_category.png"), dpi=150)
    plt.close()

    # ── 2. Speed Profile by Category ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    speeds = [per_category[c]["mean_speed"] for c in categories]
    ax.bar(
        categories,
        speeds,
        color=[colors.get(c, "#95a5a6") for c in categories],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylabel("Mean Speed (m/s)", fontsize=12)
    ax.set_title("Speed Profile by Command", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_by_category.png"), dpi=150)
    plt.close()

    # ── 3. Smoothness (Steering Jerk) ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    jerks = [per_category[c]["mean_steering_jerk"] for c in categories]
    ax.bar(
        categories,
        jerks,
        color=[colors.get(c, "#95a5a6") for c in categories],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylabel("Mean Steering Jerk", fontsize=12)
    ax.set_title("Driving Smoothness by Command", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "smoothness_by_category.png"), dpi=150)
    plt.close()

    # ── 4. Crash Rate ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    crash_rates = [per_category[c]["crash_rate"] * 100 for c in categories]
    ax.bar(
        categories,
        crash_rates,
        color=[colors.get(c, "#95a5a6") for c in categories],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylabel("Crash Rate (%)", fontsize=12)
    ax.set_title("Crash Rate by Command Category", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crash_rate_by_category.png"), dpi=150)
    plt.close()

    # ── 5. Summary Dashboard ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Language-Conditioned Racing Agent — Evaluation Summary",
        fontsize=16,
        fontweight="bold",
    )

    metrics = [
        ("Mean Reward", [per_category[c]["mean_reward"] for c in categories]),
        ("Mean Speed (m/s)", speeds),
        ("Steering Jerk", jerks),
        ("Crash Rate (%)", crash_rates),
    ]

    for ax, (title, values) in zip(axes.flat, metrics):
        ax.bar(
            categories,
            values,
            color=[colors.get(c, "#95a5a6") for c in categories],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def plot_command_compliance_radar(
    results_path: str,
    output_dir: str = "results/plots/",
):
    """
    Generate a radar chart showing command compliance across dimensions.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    per_category = results["per_category"]
    categories = list(per_category.keys())

    # Normalize metrics to [0, 1] for radar chart
    metrics = ["mean_reward", "mean_speed", "mean_length", "mean_steering_jerk"]
    metric_labels = ["Reward", "Speed", "Survival", "Smoothness"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    colors = {
        "aggressive": "#e74c3c",
        # "conservative": "#2ecc71",
        "defensive": "#3498db",
        "neutral": "#9b59b6",
    }

    for cat in categories:
        values = []
        for metric in metrics:
            val = per_category[cat].get(metric, 0)
            # For jerk, lower is better → invert
            if metric == "mean_steering_jerk":
                val = max(0, 1 - val)
            values.append(val)

        # Normalize to [0, 1] based on max across categories
        max_vals = [
            max(per_category[c].get(m, 1) for c in categories) or 1
            for m in metrics
        ]
        # Re-handle jerk inversion for normalization
        normalized = []
        for i, (val, max_val) in enumerate(zip(values, max_vals)):
            if metrics[i] == "mean_steering_jerk":
                normalized.append(val)  # Already inverted
            else:
                normalized.append(val / max_val if max_val > 0 else 0)

        normalized += normalized[:1]

        ax.plot(angles, normalized, "o-", linewidth=2,
                color=colors.get(cat, "#95a5a6"), label=cat)
        ax.fill(angles, normalized, alpha=0.1, color=colors.get(cat, "#95a5a6"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_title(
        "Command Compliance Radar",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compliance_radar.png"), dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to evaluation.json")
    parser.add_argument("--output", type=str, default="results/plots/")
    args = parser.parse_args()

    plot_evaluation_results(args.results, args.output)
    plot_command_compliance_radar(args.results, args.output)


if __name__ == "__main__":
    main()
