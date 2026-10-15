"""Local LLM client interface module using Ollama and extensible providers.

Provides robust, configurable communication with locally served Ollama models,
supporting deterministic inference via explicit seeds and zero-temperature parameters.
Implements provider abstraction, health checks, timeouts, and safe error handling.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM provider errors."""


class LLMConnectionError(LLMError):
    """Raised when the LLM service endpoint cannot be reached."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM generation or health request times out."""


class LLMResponseError(LLMError):
    """Raised when the LLM service returns an HTTP error or malformed payload."""


@dataclass
class LLMResponse:
    """Standardized response container from LLM providers."""

    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract base class separating LLM client providers from application logic."""

    @abstractmethod
    def check_health(self) -> bool:
        """Check whether the model provider service is reachable."""

    @abstractmethod
    def list_models(self) -> List[str]:
        """List models available on the provider instance."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from the configured LLM."""


class OllamaClient(BaseLLMClient):
    """Client for querying local Ollama LLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        temperature: float = 0.0,
        seed: int = 42,
        timeout: int = 60,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize Ollama client.

        Never assumes Ollama is running or hardcodes credentials.

        Args:
            base_url: Ollama API endpoint (e.g. 'http://localhost:11434').
            model: Model tag/identifier (e.g. 'llama3.2:latest', 'mistral:latest').
            temperature: Sampling temperature (0.0 for deterministic output).
            seed: Random seed for reproducibility.
            timeout: HTTP request timeout in seconds.
            config: Optional LLMConfig dataclass instance.
        """
        cfg_url = getattr(config, "base_url", None) if config else None
        cfg_model = getattr(config, "model", None) if config else None
        cfg_temp = getattr(config, "temperature", None) if config else None
        cfg_seed = getattr(config, "seed", None) if config else None
        cfg_timeout = getattr(config, "timeout", None) if config else None

        self.base_url = (cfg_url or base_url).rstrip("/")
        self.model = cfg_model or model
        self.temperature = cfg_temp if cfg_temp is not None else temperature
        self.seed = cfg_seed if cfg_seed is not None else seed
        self.timeout = max(1, cfg_timeout if cfg_timeout is not None else timeout)

    def check_health(self) -> bool:
        """Check whether the Ollama server is reachable without throwing unhandled exceptions.

        Returns:
            bool: True if Ollama server responds with HTTP 200, False otherwise.
        """
        health_url = f"{self.base_url}/api/version"
        try:
            resp = requests.get(health_url, timeout=min(5, self.timeout))
            return resp.status_code == 200
        except requests.RequestException as exc:
            logger.debug("Ollama health check failed for %s: %s", health_url, exc)
            return False

    def list_models(self) -> List[str]:
        """List models installed on the local Ollama instance.

        Returns:
            List[str]: List of model name tags.

        Raises:
            LLMConnectionError: If server cannot be reached.
            LLMTimeoutError: If the request times out.
            LLMResponseError: If server returns an error.
        """
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=self.timeout)
            if resp.status_code != 200:
                raise LLMResponseError(
                    f"Ollama server at '{self.base_url}' returned HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            models = data.get("models", [])
            return [str(m.get("name", "")) for m in models if m.get("name")]
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Connection to Ollama server at '{self.base_url}' timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to Ollama server at '{self.base_url}'. "
                f"Ensure Ollama is running via 'ollama serve'."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(f"Error querying Ollama models at '{tags_url}': {exc}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from the configured Ollama LLM.

        Args:
            prompt: User prompt content.
            system_prompt: Optional system instruction prompt.
            temperature: Optional override for sampling temperature.
            **kwargs: Additional parameters forwarded to Ollama options.

        Returns:
            LLMResponse: Generated content, token metrics, and metadata.

        Raises:
            LLMConnectionError: If Ollama server is unreachable.
            LLMTimeoutError: If generation request times out.
            LLMResponseError: If model is not found or server returns error.
        """
        cleaned_prompt = prompt.strip() if prompt else ""
        if not cleaned_prompt:
            return LLMResponse(
                content="",
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        generate_url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.temperature,
            "seed": self.seed,
        }
        options.update(kwargs)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": cleaned_prompt,
            "stream": False,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(generate_url, json=payload, timeout=self.timeout)

            if resp.status_code == 404:
                raise LLMResponseError(
                    f"Model '{self.model}' was not found in Ollama at '{self.base_url}'. "
                    f"Please pull it first using: 'ollama pull {self.model}'"
                )
            elif resp.status_code != 200:
                raise LLMResponseError(
                    f"Ollama server returned error status {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            content = data.get("response", "")
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")
            total_tokens = None
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            return LLMResponse(
                content=content,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=data,
            )

        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Request to Ollama model '{self.model}' at '{self.base_url}' timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to Ollama server at '{self.base_url}'. "
                f"Ensure Ollama is running (`ollama serve`). Details: {exc}"
            ) from exc

        except requests.RequestException as exc:
            raise LLMError(f"HTTP error communicating with Ollama: {exc}") from exc


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for offline unit and integration testing."""

    def __init__(
        self,
        default_response: str = "Mocked grounded answer.",
        model: str = "mock-model",
        response_map: Optional[Dict[str, str]] = None,
        is_healthy: bool = True,
    ) -> None:
        """Initialize mock LLM client.

        Args:
            default_response: Fallback response string.
            model: Mock model name identifier.
            response_map: Optional mapping of query substring -> canned response.
            is_healthy: Configured health status returned by check_health().
        """
        self.default_response = default_response
        self.model = model
        self.response_map = response_map or {}
        self.is_healthy = is_healthy
        self.call_history: List[Dict[str, Any]] = []

    def check_health(self) -> bool:
        """Return configured mock health status."""
        return self.is_healthy

    def list_models(self) -> List[str]:
        """Return mock models."""
        return [self.model, "mock-model-v2"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate deterministic mock response."""
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "kwargs": kwargs,
        })

        # Match against response_map if available
        matched_content = self.default_response
        prompt_lower = prompt.lower()

        # Prioritize matching against the isolated Question section if present
        query_text = prompt_lower
        if "question:" in prompt_lower:
            parts = prompt_lower.split("question:", 1)
            query_section = parts[1]
            if "grounded answer:" in query_section:
                query_section = query_section.split("grounded answer:", 1)[0]
            query_text = query_section.strip()

        # Check for query-level key match first
        matched = False
        for key, val in self.response_map.items():
            if key.lower() in query_text:
                matched_content = val
                matched = True
                break

        # Fall back to matching anywhere in the prompt if no question-level match
        if not matched:
            for key, val in self.response_map.items():
                if key.lower() in prompt_lower:
                    matched_content = val
                    break

        return LLMResponse(
            content=matched_content,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(matched_content.split()),
            total_tokens=len(prompt.split()) + len(matched_content.split()),
            raw_response={"mock": True},
        )

