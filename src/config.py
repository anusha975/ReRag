"""Configuration module for the ReRAG research platform.

Provides strongly typed, validated settings loaded from environment variables
or an optional .env file. Supports configurable chunking, embeddings, vector
storage, Ollama LLM integration, and deterministic evaluation settings with
CPU-friendly defaults.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# Attempt to load python-dotenv if installed
try:
    from dotenv import load_dotenv

    # Search for .env from current directory or project root
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
    else:
        load_dotenv()
except ImportError:  # pragma: no cover
    # python-dotenv not installed yet, rely on os.environ
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse boolean value from string or bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    """Parse JSON or comma-separated list of strings."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return [str(item) for item in value]
    val_str = str(value).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in val_str.split(",") if item.strip()]


@dataclass(frozen=True)
class AppConfig:
    """General application and runtime settings."""

    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )


@dataclass(frozen=True)
class DocumentConfig:
    """Document processing and text splitting configuration."""

    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    chunk_separators: List[str] = field(
        default_factory=lambda: _parse_list(
            os.getenv("CHUNK_SEPARATORS"), ["\n\n", "\n", " ", ""]
        )
    )

    chunk_unit: str = field(
        default_factory=lambda: os.getenv("CHUNK_UNIT", "tokens").lower()
    )

    def __post_init__(self) -> None:
        """Validate chunking parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if self.chunk_unit not in {"tokens", "characters"}:
            raise ValueError(
                f"Invalid chunk_unit '{self.chunk_unit}'. Supported units: 'tokens', 'characters'."
            )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model and device execution settings."""

    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cpu").lower()
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )

    def __post_init__(self) -> None:
        """Validate embedding configuration."""
        if not self.model_name.strip():
            raise ValueError("Embedding model_name must not be empty.")
        valid_devices = {"cpu", "cuda", "mps"}
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device '{self.device}'. Supported options: 'cpu', 'cuda', 'mps' or 'cuda:<idx>'."
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector database storage configuration for ChromaDB."""

    persist_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "rerag_collection")
    )

    def __post_init__(self) -> None:
        """Validate vector store parameters."""
        if not self.collection_name.strip():
            raise ValueError("Chroma collection_name must not be empty.")


@dataclass(frozen=True)
class RetrieverConfig:
    """Information retrieval settings."""

    top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.3"))
    )

    def __post_init__(self) -> None:
        """Validate retrieval parameters."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )


@dataclass(frozen=True)
class LLMConfig:
    """Ollama local language model configuration."""

    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )

    def __post_init__(self) -> None:
        """Validate LLM parameters."""
        if not self.model.strip():
            raise ValueError("Ollama model name cannot be empty.")
        if self.temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {self.temperature}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclass(frozen=True)
class ReRAGConfig:
    """ReRAG architecture verification and refinement parameters."""

    max_refinement_steps: int = field(
        default_factory=lambda: int(os.getenv("MAX_REFINEMENT_STEPS", "3"))
    )
    verification_threshold: float = field(
        default_factory=lambda: float(os.getenv("VERIFICATION_THRESHOLD", "0.7"))
    )
    deterministic_mode: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("DETERMINISTIC_MODE", "True"), default=True)
    )

    def __post_init__(self) -> None:
        """Validate ReRAG research parameters."""
        if self.max_refinement_steps < 1:
            raise ValueError(f"max_refinement_steps must be at least 1, got {self.max_refinement_steps}")
        if not (0.0 <= self.verification_threshold <= 1.0):
            raise ValueError(
                f"verification_threshold must be between 0.0 and 1.0, got {self.verification_threshold}"
            )


@dataclass(frozen=True)
class StoragePaths:
    """Filesystem storage paths for documents, evaluation, and experimental results."""

    root_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    )
    documents_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "data/documents")
    )
    evaluation_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("EVALUATION_DIR", "data/evaluation")
    )
    results_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")
    )

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist on disk."""
        for directory in [self.data_dir, self.documents_dir, self.evaluation_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Master configuration container for ReRAG application."""

    app: AppConfig = field(default_factory=AppConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rerag: ReRAGConfig = field(default_factory=ReRAGConfig)
    paths: StoragePaths = field(default_factory=StoragePaths)


_settings_instance: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Retrieve singleton configuration instance.

    Args:
        reload: If True, re-evaluates environment variables and instantiates fresh settings.

    Returns:
        Settings: The application settings instance.
    """
    global _settings_instance
    if _settings_instance is None or reload:
        _settings_instance = Settings()
    return _settings_instance


def setup_logging(level_name: Optional[str] = None) -> None:
    """Configure basic logging for application modules.

    Args:
        level_name: Optional logging level name (e.g. 'DEBUG', 'INFO').
            Defaults to settings.app.log_level.
    """
    settings = get_settings()
    log_level = getattr(logging, level_name or settings.app.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
