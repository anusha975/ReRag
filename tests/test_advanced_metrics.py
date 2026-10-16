"""Automated tests for advanced ReRAG evaluation metrics.

Tests:
- Answer correctness: exact match, token precision/recall/F1, ROUGE-L
- Retrieval quality: MRR@k, Hit Rate@k, Noise Ratio
- Evidence grounding: citation precision/recall, grounded claim density
- Latency and Computational Cost tracking
- Metric aggregation
"""

import pytest

from src.metrics import (
    OptimizationCostTracker,
    compute_aggregate_metrics,
    compute_citation_precision_recall,
    compute_exact_match,
    compute_grounded_claim_density,
    compute_hit_rate_at_k,
    compute_latency_metrics,
    compute_noise_ratio,
    compute_reciprocal_rank,
    compute_rouge_l,
    compute_token_precision_recall,
)


def test_compute_exact_match():
    """Verify exact match normalization handling whitespace and punctuation."""
    assert compute_exact_match("Paris", "Paris") == 1.0
    assert compute_exact_match("paris.", "Paris") == 1.0
    assert compute_exact_match("The Paris", "Paris") == 1.0  # article stripped
    assert compute_exact_match("London", "Paris") == 0.0
    assert compute_exact_match("", "") == 1.0


def test_compute_token_precision_recall():
    """Verify token overlap precision, recall, and harmonic F1."""
    # Complete match
    prec, rec, f1 = compute_token_precision_recall("deep learning model", "deep learning model")
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0

    # Partial match
    prec, rec, f1 = compute_token_precision_recall("deep model", "deep learning model")
    assert prec == 1.0  # 2 of 2 pred tokens in reference
    assert pytest.approx(rec, 0.01) == 0.6667  # 2 of 3 ref tokens in prediction
    assert pytest.approx(f1, 0.01) == 0.80

    # Zero overlap
    prec, rec, f1 = compute_token_precision_recall("apples oranges", "deep learning")
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_compute_rouge_l():
    """Verify ROUGE-L longest common subsequence scoring."""
    # Identical
    f1_score = compute_rouge_l("the quick brown fox", "the quick brown fox")
    assert f1_score == 1.0

    # Subsequence with intervening word
    f1_partial = compute_rouge_l("the brown fox", "the quick brown fox")
    assert pytest.approx(f1_partial, 0.01) == 0.857

    # No overlap
    assert compute_rouge_l("quantum computing", "banana smoothie") == 0.0


def test_retrieval_metrics():
    """Verify MRR@k, Hit Rate@k, and Noise Ratio."""
    retrieved = ["doc_c", "doc_a", "doc_b", "doc_d"]
    expected = ["doc_a"]

    # MRR: doc_a is at rank 2 -> reciprocal rank is 1/2 = 0.5
    mrr = compute_reciprocal_rank(retrieved, expected, k=4)
    assert mrr == 0.5

    # Hit Rate @ k: found within top 2 -> 1.0, top 1 -> 0.0
    assert compute_hit_rate_at_k(retrieved, expected, k=1) == 0.0
    assert compute_hit_rate_at_k(retrieved, expected, k=2) == 1.0

    # Noise ratio: 1 relevant out of 4 retrieved -> 3/4 = 0.75 noise
    assert compute_noise_ratio(retrieved, expected, k=4) == 0.75

    # Empty retrieved
    assert compute_reciprocal_rank([], expected) == 0.0
    assert compute_hit_rate_at_k([], expected) == 0.0


def test_citation_precision_recall():
    """Verify evidence citation precision, recall, and hallucinatory citations."""
    citations = ["[Document, p. 1]", "[Document, p. 2]"]
    retrieved_pages = [1, 2]
    expected_pages = [1, 3]

    res = compute_citation_precision_recall(citations, retrieved_pages=retrieved_pages, expected_pages=expected_pages)
    assert res["citation_precision"] == 0.5  # 1 out of 2 cited pages (p1, p2) was in expected [1, 3]
    assert res["citation_recall"] == 0.5  # 1 out of 2 expected pages (p1, p3) was cited
    assert res["citation_f1"] == 0.5

    # Completely ungrounded/hallucinatory citation
    res_bad = compute_citation_precision_recall(["[Doc, p. 99]"], retrieved_pages=[1], expected_pages=[1])
    assert res_bad["citation_precision"] == 0.0
    assert res_bad["citation_recall"] == 0.0


def test_grounded_claim_density():
    """Verify claim density calculation against evidence contexts."""
    answer = "The chunk size was set to 500 tokens."
    contexts = ["The chunk size was set to 500 tokens in the evaluation setup."]

    density = compute_grounded_claim_density(answer, contexts)
    assert density > 0.0


def test_latency_metrics():
    """Verify latency summary statistics computation."""
    lat_res = compute_latency_metrics(retrieval_seconds=0.15, generation_seconds=0.35, total_seconds=0.50)

    assert lat_res["retrieval_latency_ms"] == 150.0
    assert lat_res["generation_latency_ms"] == 350.0
    assert lat_res["total_latency_ms"] == 500.0
    assert lat_res["retrieval_fraction"] == 0.3


def test_optimization_cost_tracker():
    """Verify OptimizationCostTracker tracks candidate trials and compute proxies."""
    tracker = OptimizationCostTracker()
    tracker.record_trial(
        trial_chunks=15,
        trial_queries=2,
        trial_llm_calls=1,
        trial_time_seconds=0.5,
        was_accepted=True,
        cache_hit=False,
    )

    summary = tracker.to_dict()
    assert summary["total_trials"] == 1
    assert summary["accepted_trials"] == 1
    assert summary["chunks_indexed"] == 15
    assert summary["llm_calls_count"] == 1
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == 1
    assert summary["estimated_cost_proxy"] > 0.0


def test_aggregate_metrics_includes_advanced_keys():
    """Verify that compute_aggregate_metrics produces all legacy and new advanced metric keys."""
    sample_reports = [
        {
            "faithfulness_ratio": 1.0,
            "context_coverage": 0.8,
            "semantic_similarity": 0.9,
            "exact_match": 1.0,
            "token_f1": 0.85,
            "token_precision": 0.90,
            "token_recall": 0.80,
            "rouge_l": 0.80,
            "reciprocal_rank": 1.0,
            "hit_rate": 1.0,
            "noise_ratio": 0.25,
            "grounded_claim_density": 12.5,
            "total_latency_ms": 250.0,
            "is_missing_answer": False,
        }
    ]

    agg = compute_aggregate_metrics(sample_reports)

    # Legacy keys
    assert "mean_faithfulness_ratio" in agg
    assert "mean_context_coverage" in agg
    assert "mean_semantic_similarity" in agg

    # Advanced keys
    assert "mean_exact_match" in agg
    assert "mean_rouge_l" in agg
    assert "mean_reciprocal_rank" in agg
    assert "mean_hit_rate" in agg
    assert "mean_noise_ratio" in agg
    assert "mean_token_precision" in agg
    assert "mean_token_recall" in agg
    assert "mean_grounded_claim_density" in agg
    assert "mean_latency_ms" in agg
