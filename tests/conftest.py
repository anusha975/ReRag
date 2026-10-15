import hashlib
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pytest

from src.chunker import TextChunk

# Automatically redirect HuggingFace and temp caches to D:\hf_cache / D:\pip-temp to prevent C: drive exhaustion
if Path(r"D:\hf_cache").exists():
    os.environ["HF_HOME"] = r"D:\hf_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_cache\hub"
    os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache\hub"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\hf_cache"

if Path(r"D:\pip-temp").exists():
    os.environ["TEMP"] = r"D:\pip-temp"
    os.environ["TMP"] = r"D:\pip-temp"


class DeterministicMockEmbeddingManager:
    """Fast, deterministic, zero-memory embedding manager for offline unit/integration testing."""

    def __init__(self, embedding_dim: int = 384, normalize_embeddings: bool = True):
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.model_name = "mock-minilm"
        self.device = "cpu"
        self.batch_size = 32

    def _text_to_vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        stopwords = {"how", "does", "the", "is", "an", "for", "in", "and", "of", "to", "a", "with", "by", "at", "on"}
        words = re.findall(r"\w+", text.lower())
        meaningful = [w for w in words if w not in stopwords and len(w) > 2]
        tokens = meaningful if meaningful else words

        for w in tokens:
            stem = re.sub(r"(ing|tions|tion|ed|es|s)$", "", w)
            idx = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:6], 16) % self.embedding_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0 and self.normalize_embeddings:
            vec = vec / norm
        return vec

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        rows = [self._text_to_vec(str(t or "")) for t in texts]
        return np.vstack(rows)

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return self._text_to_vec(query)

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> np.ndarray:
        return self.embed_texts([c.text for c in chunks])


@pytest.fixture
def mock_embedder():
    """Deterministic mock embedding manager fixture for offline testing."""
    return DeterministicMockEmbeddingManager()
