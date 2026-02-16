"""
Instruction Encoder for Language-Conditioned Racing Agent.

Wraps a frozen sentence-transformer model (all-MiniLM-L6-v2) to encode
natural language racing commands into fixed-size embedding vectors.
"""

import numpy as np
from typing import Dict, Optional


class InstructionEncoder:
    """
    Encodes race engineer text commands into dense embedding vectors
    using a pre-trained sentence-transformer model.

    The encoder is FROZEN — no gradient flow, no fine-tuning. It acts
    purely as a fixed feature extractor. Embeddings are cached to avoid
    redundant forward passes for known commands.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Sentence-transformer model to load from HuggingFace.
            device: Device to run inference on ("cpu", "cuda", "mps").
                    If None, auto-detects.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Cache: text -> embedding
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text command into an embedding vector.

        Args:
            text: Natural language command (e.g., "Push hard and go fast").

        Returns:
            numpy array of shape (embedding_dim,), dtype float32.
        """
        if text in self._cache:
            return self._cache[text]

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        self._cache[text] = embedding
        return embedding

    def encode_batch(self, texts: list) -> np.ndarray:
        """
        Encode a batch of text commands.

        Args:
            texts: List of command strings.

        Returns:
            numpy array of shape (len(texts), embedding_dim), dtype float32.
        """
        # Check cache for each text individually
        uncached_texts = [t for t in texts if t not in self._cache]

        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            ).astype(np.float32)

            for text, emb in zip(uncached_texts, embeddings):
                self._cache[text] = emb

        return np.array([self._cache[t] for t in texts], dtype=np.float32)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)
