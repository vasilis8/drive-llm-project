"""
Custom Policy Network for Language-Conditioned Racing Agent.

Implements a multimodal feature extractor that processes:
- LiDAR range data via 1D-CNN
- Command embeddings via MLP
- Vehicle state (pass-through)

Then fuses them into a single feature vector for PPO's actor-critic heads.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List


class MultimodalFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor for Dict observation space.

    Architecture:
        "lidar" (N,)        → 1D-CNN → flatten → (lidar_out_dim)
        "command" (384,)     → Linear+ReLU      → (cmd_hidden)
        "vehicle_state" (5,) → pass-through      → (5)

        CONCAT → MLP → features_dim
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        lidar_cnn_channels: List[int] = None,
        lidar_cnn_kernels: List[int] = None,
        lidar_cnn_strides: List[int] = None,
        command_hidden_dim: int = 128,
        fusion_hidden_dims: List[int] = None,
        features_dim: int = 128,
    ):
        """
        Args:
            observation_space: Dict space with "lidar", "command", "vehicle_state".
            lidar_cnn_channels: Output channels for each 1D-CNN layer.
            lidar_cnn_kernels: Kernel sizes for each layer.
            lidar_cnn_strides: Stride for each layer.
            command_hidden_dim: Hidden dim for command MLP.
            fusion_hidden_dims: Hidden dims for the fusion MLP.
            features_dim: Final output feature dimension.
        """
        # Must call super with the final features_dim
        super().__init__(observation_space, features_dim)

        # Defaults
        if lidar_cnn_channels is None:
            lidar_cnn_channels = [64, 128, 256]
        if lidar_cnn_kernels is None:
            lidar_cnn_kernels = [7, 5, 3]
        if lidar_cnn_strides is None:
            lidar_cnn_strides = [4, 2, 2]
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 128]

        # ── LiDAR Branch (1D-CNN) ─────────────────────────────
        lidar_shape = observation_space["lidar"].shape[0]
        cnn_layers = []
        in_channels = 1  # LiDAR as single-channel 1D signal

        for out_ch, kernel, stride in zip(
            lidar_cnn_channels, lidar_cnn_kernels, lidar_cnn_strides
        ):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_ch, kernel_size=kernel, stride=stride),
                nn.ReLU(),
            ])
            in_channels = out_ch

        self.lidar_cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_shape)
            cnn_out = self.lidar_cnn(dummy)
            lidar_out_dim = cnn_out.view(1, -1).shape[1]

        self.lidar_flatten = nn.Flatten()

        # ── Command Branch (MLP) ──────────────────────────────
        command_dim = observation_space["command"].shape[0]
        self.command_mlp = nn.Sequential(
            nn.Linear(command_dim, command_hidden_dim),
            nn.ReLU(),
        )

        # ── Vehicle State (pass-through) ──────────────────────
        vehicle_state_dim = observation_space["vehicle_state"].shape[0]

        # ── Fusion MLP ────────────────────────────────────────
        fusion_input_dim = lidar_out_dim + command_hidden_dim + vehicle_state_dim

        fusion_layers = []
        in_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        # Final projection to features_dim
        fusion_layers.append(nn.Linear(in_dim, features_dim))
        fusion_layers.append(nn.ReLU())

        self.fusion_mlp = nn.Sequential(*fusion_layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: process each modality and fuse.

        Args:
            observations: Dict with "lidar", "command", "vehicle_state" tensors.

        Returns:
            Feature vector of shape (batch_size, features_dim).
        """
        # LiDAR: (batch, N) → (batch, 1, N) → CNN → flatten
        lidar = observations["lidar"].unsqueeze(1)  # Add channel dim
        lidar_features = self.lidar_flatten(self.lidar_cnn(lidar))

        # Command: (batch, 384) → MLP → (batch, cmd_hidden)
        command_features = self.command_mlp(observations["command"])

        # Vehicle state: (batch, 5) pass-through
        vehicle_state = observations["vehicle_state"]

        # Concatenate and fuse
        fused = torch.cat([lidar_features, command_features, vehicle_state], dim=1)
        return self.fusion_mlp(fused)


def create_ppo_agent(
    env,
    config: dict = None,
    tensorboard_log: str = None,
):
    """
    Create a PPO agent with the custom multimodal policy.

    Args:
        env: Gymnasium environment with Dict observation space.
        config: Training config dict (from YAML).
        tensorboard_log: Path for TensorBoard logs.

    Returns:
        stable_baselines3.PPO instance.
    """
    from stable_baselines3 import PPO

    config = config or {}
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    # Policy kwargs for the custom feature extractor
    policy_kwargs = {
        "features_extractor_class": MultimodalFeatureExtractor,
        "features_extractor_kwargs": {
            "lidar_cnn_channels": model_cfg.get("lidar_cnn", {}).get(
                "channels", [64, 128, 256]
            ),
            "lidar_cnn_kernels": model_cfg.get("lidar_cnn", {}).get(
                "kernel_sizes", [7, 5, 3]
            ),
            "lidar_cnn_strides": model_cfg.get("lidar_cnn", {}).get(
                "strides", [4, 2, 2]
            ),
            "command_hidden_dim": model_cfg.get("command_mlp", {}).get(
                "hidden_dim", 128
            ),
            "fusion_hidden_dims": model_cfg.get("fusion_mlp", {}).get(
                "hidden_dims", [256, 128]
            ),
            "features_dim": model_cfg.get("features_dim", 128),
        },
    }

    agent = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        n_steps=training_cfg.get("n_steps", 2048),
        batch_size=training_cfg.get("batch_size", 64),
        n_epochs=training_cfg.get("n_epochs", 10),
        gamma=training_cfg.get("gamma", 0.99),
        gae_lambda=training_cfg.get("gae_lambda", 0.95),
        clip_range=training_cfg.get("clip_range", 0.2),
        ent_coef=training_cfg.get("ent_coef", 0.01),
        vf_coef=training_cfg.get("vf_coef", 0.5),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    return agent
