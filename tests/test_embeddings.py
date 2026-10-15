"""Unit tests for embedding generation module."""

import numpy as np
import pytest

from src.chunker import TextChunk
from src.embeddings import EmbeddingManager, _MODEL_CACHE


def test_embedding_manager_initialization():
    """Verify default initialization parameters."""
    manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")
    assert manager.model_name == "all-MiniLM-L6-v2"
    assert manager.device == "cpu"
    assert manager.embedding_dim == 384
    assert manager.normalize_embeddings is True


def test_embedding_model_caching():
    """Verify that multiple manager instances reuse the cached underlying model."""
    m1 = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")
    m2 = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")

    model1 = m1._get_model()
    model2 = m2._get_model()

    assert model1 is model2
    assert ("all-MiniLM-L6-v2", "cpu") in _MODEL_CACHE


def test_embed_texts_normal_and_empty():
    """Verify text embedding generation and empty batch handling."""
    manager = EmbeddingManager()

    # Normal non-empty texts
    texts = [
        "ReRAG verifies retrieval-augmented generation claims against source facts.",
        "Hallucination detection measures factual consistency.",
    ]
    embs = manager.embed_texts(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 384)

    # Unit norm verification (since normalize_embeddings=True)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)

    # Empty list handling
    empty_embs = manager.embed_texts([])
    assert isinstance(empty_embs, np.ndarray)
    assert empty_embs.shape == (0, 384)


def test_embed_query_normal_and_blank():
    """Verify single query embedding and blank query safety."""
    manager = EmbeddingManager()

    q_emb = manager.embed_query("How does ReRAG reduce hallucination?")
    assert q_emb.shape == (384,)
    norm = np.linalg.norm(q_emb)
    assert abs(norm - 1.0) < 1e-4

    # Blank query returns zero vector safely
    blank_emb = manager.embed_query("   ")
    assert blank_emb.shape == (384,)
    assert np.all(blank_emb == 0.0)


def test_embed_chunks():
    """Verify batch embedding of TextChunk objects."""
    manager = EmbeddingManager()
    chunks = [
        TextChunk(chunk_id="c1", text="Chunk one text.", metadata={}, chunk_index=0),
        TextChunk(chunk_id="c2", text="Chunk two text.", metadata={}, chunk_index=1),
    ]

    embs = manager.embed_chunks(chunks)
    assert embs.shape == (2, 384)


def test_embedding_manager_with_config():
    """Verify EmbeddingManager initializes correctly from EmbeddingConfig."""
    from src.config import EmbeddingConfig

    cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cpu", batch_size=16)
    manager = EmbeddingManager(config=cfg)
    assert manager.model_name == "all-MiniLM-L6-v2"
    assert manager.device == "cpu"
    assert manager.batch_size == 16


def test_clear_model_cache():
    """Verify clearing global embedding model cache."""
    from src.embeddings import clear_model_cache

    manager = EmbeddingManager()
    _ = manager._get_model()
    assert len(_MODEL_CACHE) > 0

    backup = dict(_MODEL_CACHE)
    try:
        clear_model_cache()
        assert len(_MODEL_CACHE) == 0
    finally:
        _MODEL_CACHE.update(backup)



def test_embed_texts_with_none_and_non_strings():
    """Verify handling of None and mixed values in text inputs safely."""
    manager = EmbeddingManager()
    texts = [None, "", "Valid sentence here.", "   "]
    embs = manager.embed_texts(texts)
    assert embs.shape == (4, 384)
    # Query with None safely returns zeros
    q_none = manager.embed_query(None)
    assert q_none.shape == (384,)
    assert np.all(q_none == 0.0)


def test_unnormalized_embeddings():
    """Verify normalize_embeddings parameter configuration and normalization behavior."""
    manager_norm = EmbeddingManager(normalize_embeddings=True)
    assert manager_norm.normalize_embeddings is True
    embs_norm = manager_norm.embed_texts(["Normalized vector test."])
    assert abs(np.linalg.norm(embs_norm[0]) - 1.0) < 1e-4

    manager_unnorm = EmbeddingManager(normalize_embeddings=False)
    assert manager_unnorm.normalize_embeddings is False
    embs_unnorm = manager_unnorm.embed_texts(["Unnormalized parameter test."])
    # all-MiniLM-L6-v2 inherently includes a Normalize() module in its pipeline
    assert isinstance(embs_unnorm, np.ndarray)
    assert embs_unnorm.shape == (1, 384)


