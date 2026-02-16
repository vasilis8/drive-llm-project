"""Tests for InstructionEncoder and CommandManager."""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.instruction_encoder import InstructionEncoder
from src.utils.commands import CommandManager, COMMAND_CATEGORIES, ALL_CATEGORIES


# ── InstructionEncoder Tests ──────────────────────────────────────────


class TestInstructionEncoder:
    """Test the sentence-transformer instruction encoder."""

    @pytest.fixture(scope="class")
    def encoder(self):
        """Create a shared encoder instance (model loading is slow)."""
        return InstructionEncoder(model_name="all-MiniLM-L6-v2")

    def test_embedding_shape(self, encoder):
        """Embedding should be a 384-dim float32 vector."""
        emb = encoder.encode("Push hard")
        assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"
        assert emb.dtype == np.float32

    def test_embedding_dimension_property(self, encoder):
        """Encoder should report its embedding dimension."""
        assert encoder.embedding_dim == 384

    def test_different_commands_differ(self, encoder):
        """Different commands should produce different embeddings."""
        emb1 = encoder.encode("Push hard and go fast")
        emb2 = encoder.encode("Conserve tires")
        # They should not be identical
        assert not np.allclose(emb1, emb2, atol=1e-3)

    def test_similar_commands_closer(self, encoder):
        """Semantically similar commands should have higher cosine similarity."""
        emb_push = encoder.encode("Push hard and go fast")
        emb_attack = encoder.encode("Full attack mode")
        emb_conserve = encoder.encode("Conserve tires")

        # Cosine similarity (embeddings are normalized)
        sim_similar = np.dot(emb_push, emb_attack)
        sim_different = np.dot(emb_push, emb_conserve)

        assert sim_similar > sim_different, (
            f"Similar commands should be closer: "
            f"sim(push, attack)={sim_similar:.3f} vs "
            f"sim(push, conserve)={sim_different:.3f}"
        )

    def test_caching(self, encoder):
        """Repeated encoding should use cache."""
        encoder.clear_cache()
        assert encoder.cache_size == 0

        encoder.encode("Test command")
        assert encoder.cache_size == 1

        # Second call should use cache (same result)
        emb1 = encoder.encode("Test command")
        emb2 = encoder.encode("Test command")
        assert np.array_equal(emb1, emb2)
        assert encoder.cache_size == 1

    def test_batch_encoding(self, encoder):
        """Batch encoding should return correct shape."""
        texts = ["Push hard", "Conserve tires", "Follow the racing line"]
        embeddings = encoder.encode_batch(texts)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32


# ── CommandManager Tests ──────────────────────────────────────────────


class TestCommandManager:
    """Test the command vocabulary manager."""

    def test_init_without_encoder(self):
        """Manager should work without an encoder (zero embeddings)."""
        mgr = CommandManager(encoder=None)
        assert mgr.n_commands == sum(len(v) for v in COMMAND_CATEGORIES.values())
        assert mgr.embedding_dim == 384

    def test_init_with_encoder(self):
        """Manager should pre-compute embeddings with an encoder."""
        encoder = InstructionEncoder()
        mgr = CommandManager(encoder=encoder)
        cmd = mgr.sample()
        assert cmd.embedding is not None
        assert cmd.embedding.shape == (384,)
        assert not np.allclose(cmd.embedding, 0)

    def test_sample_all_categories(self):
        """Sampling without restriction should cover all categories."""
        mgr = CommandManager(encoder=None)
        categories_seen = set()
        for _ in range(200):
            cmd = mgr.sample()
            categories_seen.add(cmd.category)
        assert categories_seen == set(ALL_CATEGORIES)

    def test_sample_specific_category(self):
        """Sampling with category restriction should only return that category."""
        mgr = CommandManager(encoder=None)
        for _ in range(50):
            cmd = mgr.sample(allowed_categories=["aggressive"])
            assert cmd.category == "aggressive"

    def test_curriculum(self):
        """Curriculum should progressively unlock categories."""
        mgr = CommandManager(encoder=None)
        curriculum = {
            "neutral_from_step": 0,
            "aggressive_from_step": 200_000,
            "defensive_from_step": 400_000,
            "conservative_from_step": 400_000,
        }

        # At step 0, only neutral
        cats = mgr.get_curriculum_categories(0, curriculum)
        assert cats == ["neutral"]

        # At step 200_000, neutral + aggressive
        cats = mgr.get_curriculum_categories(200_000, curriculum)
        assert set(cats) == {"neutral", "aggressive"}

        # At step 500_000, all categories
        cats = mgr.get_curriculum_categories(500_000, curriculum)
        assert set(cats) == set(ALL_CATEGORIES)

    def test_get_by_text(self):
        """Should find commands by exact text match."""
        mgr = CommandManager(encoder=None)
        cmd = mgr.get_by_text("Push hard and go fast")
        assert cmd is not None
        assert cmd.category == "aggressive"

        cmd = mgr.get_by_text("nonexistent command")
        assert cmd is None
