"""Tests for evaluator module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.document_loader import LoadedDocument, PageContent
from src.evaluator import (
    EvaluationReport,
    EvaluationSample,
    Evaluator,
    compute_statistical_significance,
    expand_dataset_from_document,
    load_dataset,
    split_train_test,
)
from src.llm import MockLLMClient
from src.prompts import INSUFFICIENT_INFORMATION_RESPONSE
from src.rag_pipeline import PipelineResult, RAGPipeline
from src.retriever import RetrievedContext, Retriever
from src.vector_store import VectorStore


def test_evaluator_sample_dataclass():
    """Verify EvaluationSample creation, serialization, and alias properties."""
    sample = EvaluationSample(
        sample_id="test_1",
        question="What is ReRAG?",
        reference_answer="ReRAG is an architecture for reducing hallucination.",
        expected_source="paper_rerag",
        expected_pages=[1, 2],
        document_id="paper_rerag",
        relevant_doc_ids=["paper_rerag"],
    )
    assert sample.sample_id == "test_1"
    assert sample.question.startswith("What is")
    assert sample.reference_answer == "ReRAG is an architecture for reducing hallucination."
    # Backwards compatibility alias
    assert sample.ground_truth_answer == sample.reference_answer
    assert sample.expected_pages == [1, 2]

    # to_dict and from_dict roundtrip
    d = sample.to_dict()
    reconstructed = EvaluationSample.from_dict(d)
    assert reconstructed.sample_id == sample.sample_id
    assert reconstructed.reference_answer == sample.reference_answer
    assert reconstructed.expected_pages == [1, 2]


def test_load_dataset_json_and_jsonl(tmp_path: Path):
    """Verify loading datasets from both JSON and JSONL formatted files."""
    dataset_json_path = Path("data/evaluation/synthetic_demonstration_dataset.json")
    if dataset_json_path.exists():
        samples_json = load_dataset(dataset_json_path)
        assert len(samples_json) >= 5
        assert any(s.sample_id == "synth_01_rerag_purpose" for s in samples_json)

    dataset_jsonl_path = Path("data/evaluation/synthetic_demonstration_dataset.jsonl")
    if dataset_jsonl_path.exists():
        samples_jsonl = load_dataset(dataset_jsonl_path)
        assert len(samples_jsonl) >= 3


def test_split_train_test():
    """Verify splitting dataset into training and held-out test sets."""
    samples = [
        EvaluationSample(sample_id=f"s_{i}", question=f"Q{i}", reference_answer=f"A{i}")
        for i in range(10)
    ]
    train, test = split_train_test(samples, test_ratio=0.2, seed=42)
    assert len(train) == 8
    assert len(test) == 2
    # Ensure sets are disjoint
    train_ids = {s.sample_id for s in train}
    test_ids = {s.sample_id for s in test}
    assert len(train_ids & test_ids) == 0


def test_evaluator_evaluate_with_pipeline(mock_embedder):
    """Verify Evaluator runs RAG pipeline across samples and calculates report metrics."""
    # Create mock pipeline
    mock_pipeline = MagicMock(spec=RAGPipeline)

    # Return valid pipeline result for sample 1
    ctx1 = RetrievedContext(
        chunk_id="c1",
        text="ReRAG is an architecture designed for reducing factual hallucinations.",
        score=0.88,
        distance=0.12,
        metadata={"document_id": "paper_rerag", "page_number": 1},
        page_number=1,
        document_id="paper_rerag",
    )
    res1 = PipelineResult(
        question="What is ReRAG?",
        final_answer="ReRAG is an architecture designed for reducing factual hallucinations in retrieval.",
        contexts=[ctx1],
        source_documents=["paper_rerag"],
        page_numbers=[1],
        retrieval_scores=[0.88],
        retrieval_distances=[0.12],
        citations=["[paper_rerag, p. 1]"],
        execution_time_seconds=0.15,
    )

    # Return abstention for sample 2
    res2 = PipelineResult(
        question="What is the pressure on Venus?",
        final_answer=INSUFFICIENT_INFORMATION_RESPONSE,
        contexts=[],
        source_documents=[],
        page_numbers=[],
        retrieval_scores=[],
        retrieval_distances=[],
        citations=[],
        execution_time_seconds=0.05,
    )

    mock_pipeline.run_baseline.side_effect = [res1, res2]

    evaluator = Evaluator(pipeline=mock_pipeline, embedding_manager=mock_embedder)

    samples = [
        EvaluationSample(
            sample_id="s1",
            question="What is ReRAG?",
            reference_answer="ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation.",
            expected_source="paper_rerag",
            expected_pages=[1],
            document_id="paper_rerag",
            relevant_doc_ids=["paper_rerag"],
        ),
        EvaluationSample(
            sample_id="s2",
            question="What is the pressure on Venus?",
            reference_answer="The provided context does not contain sufficient information to answer this question.",
            expected_source="paper_rerag",
            document_id="paper_rerag",
        ),
    ]

    report = evaluator.evaluate(samples)

    assert isinstance(report, EvaluationReport)
    assert report.total_samples == 2
    assert len(report.detailed_results) == 2

    # Check sample 1 results
    s1_res = report.detailed_results[0]
    assert s1_res["sample_id"] == "s1"
    assert s1_res["semantic_similarity"] > 0.8
    assert s1_res["retrieval_recall"] == 1.0
    assert s1_res["page_coverage"] == 1.0
    assert s1_res["source_match"] == 1.0

    # Check sample 2 results (abstention)
    s2_res = report.detailed_results[1]
    assert s2_res["correct_abstention"] is True

    # Check aggregate metrics
    aggs = report.aggregate_metrics
    assert aggs["total_samples"] == 2
    assert aggs["mean_semantic_similarity"] is not None
    assert aggs["abstention_rate"] == 0.5
    assert report.baseline_metrics == aggs


def test_evaluator_empty_dataset():
    """Verify evaluating empty dataset returns report safely without errors."""
    evaluator = Evaluator(pipeline=None)
    report = evaluator.evaluate([])
    assert report.total_samples == 0
    assert report.aggregate_metrics["total_samples"] == 0
    assert report.aggregate_metrics["mean_semantic_similarity"] is None
    assert report.detailed_results == []


def test_evaluator_report_save_to_file(tmp_path: Path):
    """Verify saving EvaluationReport to disk as JSON."""
    report = EvaluationReport(
        total_samples=1,
        aggregate_metrics={"mean_semantic_similarity": 0.95},
        detailed_results=[{"sample_id": "test", "score": 0.95}],
        config_snapshot={"model": "llama3.2"},
    )
    save_path = tmp_path / "reports" / "eval_report.json"
    report.save_to_file(save_path)
    assert save_path.exists()

    content = json.loads(save_path.read_text(encoding="utf-8"))
    assert content["total_samples"] == 1
    assert content["aggregate_metrics"]["mean_semantic_similarity"] == 0.95


def test_expand_dataset_from_document():
    """Verify synthetic benchmark dataset expansion from document pages."""
    p1 = PageContent(
        page_number=1,
        text="The Riverbend Community Energy Pilot was launched in March 2024. Its purpose is to test shared rooftop solar installations.",
        char_count=120,
    )
    doc = LoadedDocument(
        content="The Riverbend Community Energy Pilot was launched in March 2024.",
        metadata={"document_id": "riverbend_doc"},
        source_path=Path("riverbend.txt"),
        pages=[p1],
    )

    expanded = expand_dataset_from_document(document=doc, max_samples=5)
    assert len(expanded) >= 1
    sample = expanded[0]
    assert isinstance(sample, EvaluationSample)
    assert sample.expected_source == "riverbend_doc"
    assert sample.expected_pages == [1]
    assert len(sample.question) > 10
    assert len(sample.reference_answer) > 10


def test_compute_statistical_significance():
    """Verify paired t-stat, p-value, and confidence intervals computation."""
    baseline = [0.60, 0.65, 0.70, 0.55, 0.60]
    rerag = [0.85, 0.90, 0.88, 0.80, 0.85]

    stats = compute_statistical_significance(baseline_scores=baseline, rerag_scores=rerag)

    assert stats["status"] == "computed"
    assert stats["sample_size"] == 5
    assert stats["mean_improvement"] > 0.20
    assert stats["t_statistic"] > 5.0
    assert stats["p_value_approx"] < 0.05
    assert stats["is_statistically_significant_p05"] is True
    assert stats["ci_95_lower"] > 0


