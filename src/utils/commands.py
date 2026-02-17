"""
Command definitions and sampling for Language-Conditioned Racing Agent.

Defines the command vocabulary organized by behavioral category,
pre-computes embeddings, and provides sampling strategies for
training curriculum.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import numpy as np


@dataclass
class Command:
    """A single racing command with its text and category."""
    text: str
    category: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ── Command Vocabulary ────────────────────────────────────────────────

COMMAND_CATEGORIES = {
    "aggressive": [
        "Push hard and go fast",
        "Full attack mode",
        "Overtake now",
        "Maximum speed on the straight",
        "Aggressive braking into the corner",
        "Close the gap quickly",
    ],
    "defensive": [
        "Defend the inside line",
        "Block the opponent behind",
        "Hold your position",
        "Cover the racing line",
        "Stay tight on the apex",
        "Do not let them pass",
    ],
    "neutral": [
        "Follow the racing line",
        "Maintain current gap",
        "Steady pace",
        "Drive normally",
        "Keep a consistent rhythm",
        "Standard driving",
    ],
}

ALL_CATEGORIES = list(COMMAND_CATEGORIES.keys())


class CommandManager:
    """
    Manages the command vocabulary, embeddings, and curriculum-based sampling.

    The manager pre-computes embeddings for all commands using the
    InstructionEncoder, and supports curriculum-based sampling where
    new command categories are introduced gradually during training.
    """

    def __init__(self, encoder=None):
        """
        Args:
            encoder: InstructionEncoder instance. If None, embeddings
                     will be zero vectors (useful for testing).
        """
        self.encoder = encoder
        self.commands: List[Command] = []
        self._category_commands: Dict[str, List[Command]] = {}
        self._embedding_dim = 384
        self.allowed_categories: Optional[List[str]] = None  # None = all

        # Build command list with embeddings
        for category, texts in COMMAND_CATEGORIES.items():
            self._category_commands[category] = []
            for text in texts:
                if encoder is not None:
                    embedding = encoder.encode(text)
                    self._embedding_dim = len(embedding)
                else:
                    embedding = np.zeros(self._embedding_dim, dtype=np.float32)

                cmd = Command(text=text, category=category, embedding=embedding)
                self.commands.append(cmd)
                self._category_commands[category].append(cmd)

    @property
    def embedding_dim(self) -> int:
        """Dimension of command embeddings."""
        return self._embedding_dim

    @property
    def n_commands(self) -> int:
        """Total number of commands."""
        return len(self.commands)

    def sample(
        self,
        allowed_categories: Optional[List[str]] = None,
    ) -> Command:
        """
        Sample a random command, optionally restricted to certain categories.

        Args:
            allowed_categories: List of category names to sample from.
                If None, samples from all categories.

        Returns:
            A Command instance with text, category, and embedding.
        """
        # Use the instance-level restriction if no explicit override
        if allowed_categories is None:
            allowed_categories = self.allowed_categories or ALL_CATEGORIES

        # Gather eligible commands
        eligible = []
        for cat in allowed_categories:
            eligible.extend(self._category_commands.get(cat, []))

        if not eligible:
            raise ValueError(
                f"No commands found for categories: {allowed_categories}"
            )

        return random.choice(eligible)

    def get_curriculum_categories(self, current_step: int, curriculum: dict) -> List[str]:
        """
        Determine which command categories are available at a given training step.

        Args:
            current_step: Current training timestep.
            curriculum: Dict mapping f"{category}_from_step" -> step number.

        Returns:
            List of active category names.
        """
        active = []
        for category in ALL_CATEGORIES:
            key = f"{category}_from_step"
            threshold = curriculum.get(key, 0)
            if current_step >= threshold:
                active.append(category)

        return active if active else ["neutral"]  # Always have at least neutral

    def get_by_text(self, text: str) -> Optional[Command]:
        """Look up a command by its exact text."""
        for cmd in self.commands:
            if cmd.text == text:
                return cmd
        return None

    def get_category_commands(self, category: str) -> List[Command]:
        """Get all commands in a specific category."""
        return list(self._category_commands.get(category, []))
