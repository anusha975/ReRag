"""Unit tests for LLM client interface and mock inference."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import LLMConfig
from src.llm import (
    BaseLLMClient,
    LLMConnectionError,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    MockLLMClient,
    OllamaClient,
)


def test_mock_llm_client_basic():
    """Verify MockLLMClient satisfies BaseLLMClient interface and returns deterministic output."""
    mock_client = MockLLMClient(default_response="Quantum entanglement enables secure communication.")
    assert isinstance(mock_client, BaseLLMClient)
    assert mock_client.check_health() is True
    assert "mock-model" in mock_client.list_models()

    resp = mock_client.generate(prompt="What is entanglement?", system_prompt="Be concise.")
    assert isinstance(resp, LLMResponse)
    assert resp.content == "Quantum entanglement enables secure communication."
    assert resp.model == "mock-model"
    assert resp.prompt_tokens is not None
    assert resp.completion_tokens is not None
    assert len(mock_client.call_history) == 1


def test_mock_llm_client_response_map():
    """Verify MockLLMClient returns query-specific responses from response_map."""
    response_map = {
        "hallucination": "ReRAG detects factual hallucination via verification.",
        "orbit": "Satellites orbit in Low Earth Orbit at approximately 7.8 km/s.",
    }
    mock_client = MockLLMClient(default_response="Default answer.", response_map=response_map)

    r1 = mock_client.generate("Explain hallucination detection.")
    assert "ReRAG detects factual hallucination" in r1.content

    r2 = mock_client.generate("Tell me about orbit speed.")
    assert "Low Earth Orbit" in r2.content

    r3 = mock_client.generate("Something else entirely.")
    assert r3.content == "Default answer."


def test_ollama_client_init_and_config():
    """Verify OllamaClient initializes parameters from args and LLMConfig."""
    client1 = OllamaClient(
        base_url="http://127.0.0.1:11434/",
        model="mistral:latest",
        temperature=0.2,
        seed=100,
        timeout=45,
    )
    assert client1.base_url == "http://127.0.0.1:11434"
    assert client1.model == "mistral:latest"
    assert client1.temperature == 0.2
    assert client1.seed == 100
    assert client1.timeout == 45

    cfg = LLMConfig(
        base_url="http://localhost:11434",
        model="llama3.2:latest",
        temperature=0.0,
        seed=42,
        timeout=60,
    )
    client2 = OllamaClient(config=cfg)
    assert client2.base_url == "http://localhost:11434"
    assert client2.model == "llama3.2:latest"
    assert client2.temperature == 0.0
    assert client2.seed == 42


def test_ollama_client_check_health_mocked():
    """Verify check_health handles reachable and unreachable states."""
    client = OllamaClient()

    # Successful health check
    mock_ok = MagicMock()
    mock_ok.status_code = 200
    with patch("requests.get", return_value=mock_ok):
        assert client.check_health() is True

    # Failed health check (connection error)
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        assert client.check_health() is False


def test_ollama_client_list_models_mocked():
    """Verify list_models extracts available model tags."""
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:latest"}, {"name": "all-minilm:latest"}]
    }

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        assert models == ["llama3.2:latest", "all-minilm:latest"]


def test_ollama_client_generate_mocked_success():
    """Verify generate handles standard 200 response from Ollama API."""
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "ReRAG enforces factual consistency via verification.",
        "prompt_eval_count": 45,
        "eval_count": 12,
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = client.generate(
            prompt="How does ReRAG work?",
            system_prompt="Be factual.",
            temperature=0.0,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "ReRAG enforces factual consistency via verification."
        assert result.prompt_tokens == 45
        assert result.completion_tokens == 12
        assert result.total_tokens == 57

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "llama3.2:latest"
        assert call_kwargs["json"]["options"]["temperature"] == 0.0


def test_ollama_client_generate_connection_error():
    """Verify LLMConnectionError is raised when Ollama service is not running."""
    client = OllamaClient()
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(LLMConnectionError) as exc_info:
            client.generate("Any prompt")
        assert "Ensure Ollama is running" in str(exc_info.value)


def test_ollama_client_generate_timeout_error():
    """Verify LLMTimeoutError is raised when generation exceeds timeout."""
    client = OllamaClient(timeout=10)
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        with pytest.raises(LLMTimeoutError) as exc_info:
            client.generate("Any prompt")
        assert "timed out after 10s" in str(exc_info.value)


def test_ollama_client_generate_model_not_found():
    """Verify LLMResponseError provides actionable message on 404 model not found."""
    client = OllamaClient(model="nonexistent-model:latest")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model 'nonexistent-model:latest' not found"

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(LLMResponseError) as exc_info:
            client.generate("Any prompt")
        assert "ollama pull" in str(exc_info.value)


def test_ollama_client_empty_prompt():
    """Verify empty prompt returns empty LLMResponse without issuing network call."""
    client = OllamaClient()
    with patch("requests.post") as mock_post:
        resp = client.generate("   ")
        assert resp.content == ""
        assert resp.prompt_tokens == 0
        mock_post.assert_not_called()
