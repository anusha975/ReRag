

"""Embedding generator module for ReRAG.

Wraps Sentence Transformers (default: 'all-MiniLM-L6-v2') to compute dense vector
representations for queries and document chunks, supporting both CPU and CUDA devices.
Implements model caching to prevent redundant weight reloading, vector normalization,
and safe empty batch handling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure HF and temporary caches default to D: drive if present to avoid C: drive space constraints
if Path(r"D:\hf_cache").exists():
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\hf_cache\hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\hub")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", r"D:\hf_cache")

if Path(r"D:\pip-temp").exists():
    os.environ.setdefault("TEMP", r"D:\pip-temp")
    os.environ.setdefault("TMP", r"D:\pip-temp")

import hashlib
import re

logger = logging.getLogger(__name__)

# Global in-process cache to avoid reloading model weights repeatedly
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Clear cached SentenceTransformer instances from memory."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
    logger.debug("Cleared embedding model cache.")


class FallbackEmbeddingModel:
    """Lightweight in-memory vectorizer used when system memory or pagefile limits prevent loading PyTorch weights."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        is_single = isinstance(sentences, str)
        seq = [sentences] if is_single else list(sentences)
        if not seq:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        stopwords = {"how", "does", "the", "is", "an", "for", "in", "and", "of", "to", "a", "with", "by", "at", "on"}
        out = []
        for text in seq:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            words = re.findall(r"\w+", str(text).lower())
            meaningful = [w for w in words if w not in stopwords and len(w) > 2]
            tokens = meaningful if meaningful else words

            for w in tokens:
                stem = re.sub(r"(ing|tions|tion|ed|es|s)$", "", w)
                idx = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:6], 16) % self.embedding_dim
                vec[idx] += 1.0

            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            out.append(vec)

        arr = np.vstack(out)
        return arr[0] if is_single else arr


class EmbeddingManager:
    """Manages SentenceTransformer model loading, caching, and batch vector encoding."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize embedding manager.

        Args:
            model_name: HuggingFace model identifier. Defaults to 'all-MiniLM-L6-v2'.
            device: Target compute device ('cpu', 'cuda', 'mps', etc.). Defaults to 'cpu'.
            batch_size: Encoding batch size. Defaults to 32.
            normalize_embeddings: If True, normalizes embeddings to unit L2 norm
                (essential for cosine similarity).
            config: Optional EmbeddingConfig dataclass instance to load parameters from.
        """
        cfg_model = getattr(config, "model_name", None) if config else None
        cfg_device = getattr(config, "device", None) if config else None
        cfg_batch = getattr(config, "batch_size", None) if config else None

        self.model_name = model_name or cfg_model or "all-MiniLM-L6-v2"
        self.device = (device or cfg_device or "cpu").lower()
        self.batch_size = max(1, batch_size or cfg_batch or 32)
        self.normalize_embeddings = normalize_embeddings
        self._dim: Optional[int] = None

    def _get_model(self) -> Any:
        """Retrieve model instance from global cache or instantiate if absent."""
        cache_key = (self.model_name, self.device)
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                # Ensure single-worker model loading on Windows to prevent
                # access violations in concurrent torch/transformers tensor materialization
                try:
                    import transformers.core_model_loading as _cml
                    _cml.GLOBAL_WORKERS = 1
                    _cml.spawn_materialize = lambda tp, t, d=None, dt=None: (lambda: _cml._materialize_copy(t, d, dt))
                except Exception:
                    pass

                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info(
                        "Loading SentenceTransformer model '%s' on device '%s'...",
                        self.model_name,
                        self.device,
                    )
                    target_model = self.model_name
                    local_snapshot_dir = Path(r"D:\hf_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
                    if self.model_name in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2") and local_snapshot_dir.exists():
                        target_model = str(local_snapshot_dir)

                    model = SentenceTransformer(target_model, device=self.device)
                except (OSError, RuntimeError, Exception) as exc:
                    logger.warning(
                        "Could not load neural embedding weights (%s). Falling back to resilient lightweight deterministic vectorizer.",
                        exc,
                    )
                    model = FallbackEmbeddingModel(embedding_dim=384)

                _MODEL_CACHE[cache_key] = model
            return _MODEL_CACHE[cache_key]

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the generated dense embedding vectors."""
        if self._dim is None:
            model = self._get_model()
            self._dim = int(model.get_sentence_embedding_dimension() or 384)
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate dense vector embeddings for a sequence of texts.

        Args:
            texts: Sequence of strings to encode.

        Returns:
            np.ndarray: Embedding matrix of shape (len(texts), embedding_dim).
        """
        if texts is None or len(texts) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Sanitize texts: replace empty/whitespace-only/None items with safe single space
        sanitized = [
            str(t).strip() if (t is not None and str(t).strip()) else " "
            for t in texts
        ]

        model = self._get_model()
        raw_embeddings = model.encode(
            sanitized,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        # Ensure 2D shape consistency even for single item sequences
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a dense embedding vector for a single query.

        Args:
            query: Query string.

        Returns:
            np.ndarray: 1D normalized embedding vector of shape (embedding_dim,).
        """
        if query is None:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        cleaned = str(query).strip()
        if not cleaned:
            # Safe representation for blank query: zero vector
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        model = self._get_model()
        emb = model.encode(
            cleaned,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(emb, dtype=np.float32).flatten()

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> np.ndarray:
        """Generate dense embeddings for a collection of TextChunk objects.

        Args:
            chunks: Sequence of TextChunk objects.

        Returns:
            np.ndarray: Embedding matrix of shape (len(chunks), embedding_dim).
        """
        if chunks is None or len(chunks) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

