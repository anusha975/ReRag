"""Unit and integration tests for the ReRAG Streamlit UI modules.

Tests:
- Reusable UI presentation components (badges, cards, passage viewers, Altair charts, comparison views)
- Session state initialization and lazy resource handling
- Deterministic offline mock LLM provider selection
- Headless Streamlit AppTest navigation across all 5 operational sections
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src.config import Settings, get_settings
from src.llm import MockLLMClient, OllamaClient
from src.optimizer import CandidateTrial, ChunkConfig, OptimizationResult
from src.retriever import RetrievedContext
from ui.components import (
    inject_custom_styles,
    render_candidate_chart,
    render_candidate_table,
    render_comparison_view,
    render_empty_state,
    render_header,
    render_retrieved_passage,
    render_sidebar_config,
    render_status_badge,
    render_trace_viewer,
)
from ui.dashboard import get_llm_client, init_session_state


# ==============================================================================
# UNIT TESTS: UI COMPONENTS
# ==============================================================================


def test_render_status_badge():
    """Verify HTML generation for all candidate and system status badges."""
    # Accepted
    b_acc = render_status_badge("accepted")
    assert "rerag-badge-accepted" in b_acc
    assert "Accepted" in b_acc

    # Rejected
    b_rej = render_status_badge("rejected")
    assert "rerag-badge-rejected" in b_rej
    assert "Rejected" in b_rej

    # Invalid
    b_inv = render_status_badge("rejected_invalid")
    assert "rerag-badge-invalid" in b_inv
    assert "Invalid" in b_inv

    # Failed
    b_fail = render_status_badge("failed")
    assert "rerag-badge-failed" in b_fail
    assert "Failed" in b_fail

    # Baseline
    b_base = render_status_badge("baseline")
    assert "rerag-badge-baseline" in b_base
    assert "Baseline" in b_base


def test_render_candidate_table(monkeypatch):
    """Verify candidate table formats data into a pandas DataFrame."""
    trials = [
        CandidateTrial(
            trial_id=1,
            iteration=0,
            config=ChunkConfig(500, 50),
            step_size=100,
            metrics={"faithfulness_ratio": 0.8},
            objective_score=0.75,
            status="baseline",
            decision_reason="Baseline recorded",
            execution_time_seconds=1.2,
        ),
        CandidateTrial(
            trial_id=2,
            iteration=1,
            config=ChunkConfig(600, 50),
            step_size=100,
            metrics={"faithfulness_ratio": 0.9},
            objective_score=0.85,
            status="accepted",
            decision_reason="Improved score",
            execution_time_seconds=1.4,
        ),
    ]

    mock_df_call = MagicMock()
    monkeypatch.setattr("streamlit.dataframe", mock_df_call)

    render_candidate_table(trials)
    assert mock_df_call.called
    df_arg = mock_df_call.call_args[0][0]
    assert len(df_arg) == 2
    assert "Trial" in df_arg.columns
    assert "Score" in df_arg.columns


def test_render_candidate_chart(monkeypatch):
    """Verify candidate progression Altair chart generation."""
    trials = [
        CandidateTrial(
            trial_id=1,
            iteration=0,
            config=ChunkConfig(500, 50),
            step_size=100,
            metrics={},
            objective_score=0.60,
            status="baseline",
            decision_reason="Initial baseline",
            execution_time_seconds=0.5,
        ),
        CandidateTrial(
            trial_id=2,
            iteration=1,
            config=ChunkConfig(600, 50),
            step_size=100,
            metrics={},
            objective_score=0.72,
            status="accepted",
            decision_reason="Improved",
            execution_time_seconds=0.5,
        ),
    ]

    mock_chart_call = MagicMock()
    monkeypatch.setattr("streamlit.altair_chart", mock_chart_call)

    render_candidate_chart(trials, baseline_score=0.60)
    assert mock_chart_call.called


def test_render_comparison_view_metrics(monkeypatch):
    """Verify side-by-side metric comparison rendering."""
    mock_metric = MagicMock()
    mock_df = MagicMock()
    monkeypatch.setattr("streamlit.metric", mock_metric)
    monkeypatch.setattr("streamlit.dataframe", mock_df)

    b_metrics = {
        "objective_score": 0.65,
        "faithfulness_ratio": 0.70,
        "context_coverage": 0.60,
        "semantic_similarity": 0.65,
    }
    s_metrics = {
        "objective_score": 0.82,
        "faithfulness_ratio": 0.85,
        "context_coverage": 0.80,
        "semantic_similarity": 0.80,
    }

    render_comparison_view(
        baseline_metrics=b_metrics,
        selected_metrics=s_metrics,
        baseline_config=ChunkConfig(500, 50),
        selected_config=ChunkConfig(700, 50),
        held_out_evaluation={"baseline_score": 0.60, "selected_score": 0.75, "held_out_gain": 0.15, "test_samples_count": 2},
    )

    assert mock_metric.called
    assert mock_df.called


# ==============================================================================
# UNIT TESTS: CONTROLLERS & LOGIC
# ==============================================================================


def test_get_llm_client_offline_and_online():
    """Verify LLM client switching between offline demonstration and online Ollama."""
    settings = get_settings()
    # 1. Offline Mode
    client_offline = get_llm_client(settings, offline_mode=True)
    assert isinstance(client_offline, MockLLMClient)
    assert client_offline.check_health() is True
    # Test canned responses
    resp = client_offline.generate("What is the primary architectural purpose of ReRAG?")
    assert "reducing factual hallucinations" in resp.content.lower()

    # 2. Online Mode
    client_online = get_llm_client(settings, offline_mode=False)
    assert isinstance(client_online, OllamaClient)
    assert client_online.model == settings.llm.model


# ==============================================================================
# INTEGRATION TESTS: STREAMLIT APPTEST
# ==============================================================================


def test_streamlit_app_navigation():
    """Verify headless execution and section navigation across the entire dashboard."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()
    assert not at.exception, f"App execution raised exception: {at.exception}"

    # Verify Sidebar Navigation exists
    assert len(at.radio) >= 1
    nav_radio = at.radio[0]

    # Verify Section 1: Document Workspace renders
    assert "Document Workspace" in nav_radio.value

    # Navigate to Section 1B: QA Dataset Lab
    nav_radio.set_value("📚 QA Dataset Lab").run()
    assert not at.exception, f"QA Dataset Lab raised exception: {at.exception}"

    # Navigate to Section 2: RAG Chat
    nav_radio.set_value("💬 RAG Chat").run()
    assert not at.exception, f"RAG Chat raised exception: {at.exception}"

    # Navigate to Section 3: Optimization Lab
    nav_radio.set_value("🧪 Optimization Lab").run()
    assert not at.exception, f"Optimization Lab raised exception: {at.exception}"

    # Navigate to Section 4: Results and Comparison
    nav_radio.set_value("📊 Results and Comparison").run()
    assert not at.exception, f"Results and Comparison raised exception: {at.exception}"

    # Navigate to Section 5: Experiment History
    nav_radio.set_value("📜 Experiment History").run()
    assert not at.exception, f"Experiment History raised exception: {at.exception}"

    # Navigate to Section 5: Experiment History
    nav_radio.set_value("📜 Experiment History").run()
    assert not at.exception, f"Experiment History raised exception: {at.exception}"
