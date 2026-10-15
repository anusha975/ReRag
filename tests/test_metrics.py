"""Tests for evaluation metrics module."""

import pytest

from src.embeddings import EmbeddingManager
from src.metrics import (
    check_missing_answer,
    compute_aggregate_metrics,
    compute_context_coverage,
    compute_hallucination_rate,
    compute_page_coverage,
    compute_retrieval_precision_at_k,
    compute_retrieval_recall_at_k,
    compute_semantic_similarity,
    compute_source_match,
    compute_token_f1,
    detect_unsupported_claims_deterministic,
)


@pytest.fixture
def shared_embedder(mock_embedder):
    """Shared EmbeddingManager fixture."""
    return mock_embedder


def test_compute_token_f1_exact_and_partial():
    """Verify token F1 score matches exact, partial, and disjoint text."""
    assert compute_token_f1("hello world", "hello world") == 1.0
    assert compute_token_f1("hello world", "completely different") == 0.0
    f1 = compute_token_f1("the quick brown fox", "the fast brown fox")
    assert 0.0 < f1 < 1.0
    # Empty inputs
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1("some text", "") == 0.0


def test_semantic_similarity_exact_paraphrased_and_unrelated(shared_embedder):
    """Verify semantic similarity for exact matches, paraphrased assertions, and unrelated text."""
    ref = "ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation."

    # 1. Exact match -> 1.0
    sim_exact = compute_semantic_similarity(ref, ref, shared_embedder)
    assert abs(sim_exact - 1.0) < 1e-4

    # 2. Paraphrased statement -> high similarity
    paraphrase = "ReRAG minimizes factual errors and hallucinations in retrieval pipelines."
    sim_para = compute_semantic_similarity(paraphrase, ref, shared_embedder)
    min_sim = 0.40 if getattr(shared_embedder, "model_name", "") == "mock-minilm" else 0.65
    assert sim_para > min_sim

    # 3. Completely unrelated topic -> low similarity
    unrelated = "French culinary pastries require laminating butter dough into croissants."
    sim_unrelated = compute_semantic_similarity(unrelated, ref, shared_embedder)
    assert sim_unrelated < 0.35
    assert sim_para > sim_unrelated

    # 4. Empty inputs handling
    assert compute_semantic_similarity("", "", shared_embedder) == 1.0
    assert compute_semantic_similarity("some text", "", shared_embedder) == 0.0


def test_check_missing_answer_detection():
    """Verify identification of safe abstentions and missing answer indicators."""
    assert check_missing_answer("") is True
    assert check_missing_answer("   \n\t  ") is True
    assert check_missing_answer("The provided context does not contain sufficient information to answer this question.") is True
    assert check_missing_answer("I cannot answer this question based on the provided documents.") is True
    assert check_missing_answer("There is not enough information in the context to determine this.") is True

    # Valid factual answer is not a missing answer
    assert check_missing_answer("The James Webb telescope operates in infrared wavelengths.") is False


def test_detect_unsupported_claims_deterministic():
    """Verify deterministic claim extraction and evidence support detection."""
    context = [
        "Quantum computing relies on qubits exhibiting superposition and entanglement.",
        "Superconducting circuits operate at millikelvin temperatures.",
    ]

    # Grounded answer: all claims supported by context words
    grounded_answer = "Qubits exhibit quantum superposition and entanglement. Superconducting circuits operate at low millikelvin temperatures."
    check_grounded = detect_unsupported_claims_deterministic(grounded_answer, context)
    assert check_grounded["unsupported_sentences"] == 0
    assert check_grounded["faithfulness_ratio"] == 1.0
    assert len(check_grounded["unsupported_claims"]) == 0

    # Hallucinated / unsupported claim inserted
    hallucinated_answer = (
        "Qubits exhibit superposition and entanglement. "
        "Napoleon Bonaparte conquered Moscow in the winter of 1812."
    )
    check_hallucinated = detect_unsupported_claims_deterministic(hallucinated_answer, context)
    assert check_hallucinated["unsupported_sentences"] >= 1
    assert check_hallucinated["faithfulness_ratio"] < 1.0
    assert any("Napoleon" in c for c in check_hallucinated["unsupported_claims"])

    # Abstention answer
    check_abstain = detect_unsupported_claims_deterministic("Insufficient information to answer.", context)
    assert check_abstain["abstention"] is True
    assert check_abstain["faithfulness_ratio"] == 1.0


def test_retrieval_metrics_and_unavailable_handling():
    """Verify retrieval precision, recall, and page coverage with explicit unavailable handling."""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc2", "doc5"]

    # Normal recall and precision
    recall = compute_retrieval_recall_at_k(retrieved, relevant, k=2)
    assert recall == 0.5
    precision = compute_retrieval_precision_at_k(retrieved, relevant, k=2)
    assert precision == 0.5

    # When relevant_ids is None or empty: return None (unavailable, not 0.0)
    assert compute_retrieval_recall_at_k(retrieved, None, k=2) is None
    assert compute_retrieval_recall_at_k(retrieved, [], k=2) is None
    assert compute_retrieval_precision_at_k(retrieved, None, k=2) is None

    # Page coverage
    assert compute_page_coverage([1, 2, 3], [2, 4]) == 0.5
    assert compute_page_coverage([1, 2], [1, 2]) == 1.0
    assert compute_page_coverage([1, 2], None) is None
    assert compute_page_coverage([1, 2], []) is None

    # Source match
    assert compute_source_match(["doc_a", "doc_b"], "doc_a") == 1.0
    assert compute_source_match(["doc_a"], "doc_c") == 0.0
    assert compute_source_match(["doc_a"], None) is None


def test_context_coverage():
    """Verify lexical context coverage calculation."""
    contexts = ["Photosynthesis occurs in plant cells using chlorophyll and sunlight."]
    pred_covered = "Plant cells use sunlight and chlorophyll for photosynthesis."
    cov_high = compute_context_coverage(pred_covered, contexts)
    assert cov_high >= 0.8

    pred_uncovered = "Aerodynamic drag forces affect rocket ascent trajectories."
    cov_low = compute_context_coverage(pred_uncovered, contexts)
    assert cov_low <= 0.2


def test_hallucination_rate():
    """Verify hallucination rate computation."""
    assert compute_hallucination_rate(10, 2) == 0.2
    assert compute_hallucination_rate(0, 0) == 0.0
    assert compute_hallucination_rate(5, 5) == 1.0


def test_compute_aggregate_metrics():
    """Verify aggregate metric calculations properly handle unavailable entries and compute averages."""
    results = [
        {
            "semantic_similarity": 0.8,
            "token_f1": 0.6,
            "context_coverage": 0.7,
            "retrieval_recall": 1.0,
            "retrieval_precision": 0.5,
            "page_coverage": 1.0,
            "source_match": 1.0,
            "faithfulness_ratio": 1.0,
            "is_missing_answer": False,
        },
        {
            "semantic_similarity": 0.6,
            "token_f1": 0.4,
            "context_coverage": 0.5,
            "retrieval_recall": None,  # Not annotated in ground truth
            "retrieval_precision": None,
            "page_coverage": None,
            "source_match": None,
            "faithfulness_ratio": 0.5,
            "is_missing_answer": True,
        },
    ]

    aggregates = compute_aggregate_metrics(results)
    assert aggregates["total_samples"] == 2
    assert abs(aggregates["mean_semantic_similarity"] - 0.7) < 1e-4
    assert abs(aggregates["mean_token_f1"] - 0.5) < 1e-4
    assert abs(aggregates["mean_context_coverage"] - 0.6) < 1e-4
    # Retrieval recall averaged only over valid samples (sample 1: 1.0), NOT 0.5!
    assert aggregates["mean_retrieval_recall"] == 1.0
    assert aggregates["mean_retrieval_precision"] == 0.5
    assert aggregates["abstention_rate"] == 0.5

    # Empty dataset handling
    empty_aggs = compute_aggregate_metrics([])
    assert empty_aggs["total_samples"] == 0
    assert empty_aggs["mean_semantic_similarity"] is None
