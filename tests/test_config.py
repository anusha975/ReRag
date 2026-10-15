"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    DocumentConfig,
    EmbeddingConfig,
    LLMConfig,
    ReRAGConfig,
    RetrieverConfig,
    Settings,
    get_settings,
)


def test_default_settings():
    """Verify default values across sub-configurations."""
    settings = get_settings(reload=True)
    assert settings.app.environment in {"development", "production", "test"}
    assert settings.document.chunk_size == 500
    assert settings.document.chunk_overlap == 50
    assert settings.document.chunk_unit in {"tokens", "characters"}
    assert settings.embedding.model_name == "all-MiniLM-L6-v2"
    assert settings.embedding.device in {"cpu", "cuda", "mps"}
    assert settings.retriever.top_k == 4
    assert settings.llm.temperature == 0.0
    assert settings.rerag.deterministic_mode is True


def test_invalid_document_config():
    """Verify validation errors for bad chunking configurations."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        DocumentConfig(chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        DocumentConfig(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="must be strictly less than"):
        DocumentConfig(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="Invalid chunk_unit"):
        DocumentConfig(chunk_size=100, chunk_overlap=10, chunk_unit="unsupported")


def test_invalid_embedding_config():
    """Verify validation errors for unsupported devices or empty model names."""
    with pytest.raises(ValueError, match="model_name must not be empty"):
        EmbeddingConfig(model_name="  ")

    with pytest.raises(ValueError, match="Invalid device"):
        EmbeddingConfig(device="tpu")


def test_invalid_retriever_config():
    """Verify validation errors for invalid retrieval parameters."""
    with pytest.raises(ValueError, match="top_k must be positive"):
        RetrieverConfig(top_k=0)

    with pytest.raises(ValueError, match="similarity_threshold must be between"):
        RetrieverConfig(similarity_threshold=1.5)


def test_invalid_llm_config():
    """Verify validation errors for bad LLM settings."""
    with pytest.raises(ValueError, match="model name cannot be empty"):
        LLMConfig(model="")

    with pytest.raises(ValueError, match="temperature cannot be negative"):
        LLMConfig(temperature=-0.5)


def test_invalid_rerag_config():
    """Verify validation errors for bad ReRAG settings."""
    with pytest.raises(ValueError, match="max_refinement_steps must be at least 1"):
        ReRAGConfig(max_refinement_steps=0)

    with pytest.raises(ValueError, match="verification_threshold must be between"):
        ReRAGConfig(verification_threshold=-0.1)


def test_path_ensure_directories(tmp_path: Path):
    """Verify ensure_directories creates paths without error."""
    settings = get_settings()
    settings.paths.ensure_directories()
    assert settings.paths.documents_dir.exists()
    assert settings.paths.evaluation_dir.exists()
    assert settings.paths.results_dir.exists()
