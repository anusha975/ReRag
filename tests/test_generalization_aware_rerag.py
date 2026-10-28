"""Automated test suite for Generalization-Aware ReRAG extension.

Covers:
1. Dataset validation and duplicate / near-duplicate detection.
2. Dataset statistics and annotation status tracking (verified_expert vs. unverified_synthetic).
3. CSV and JSON serialization / export.
4. Leakage-resistant splitting (near-duplicate grouped, document-level, stratified).
5. Strict zero-leakage guarantee: verifying held-out test data is quarantined from optimizers.
6. Three-approach optimization execution:
   - Approach 1: Fixed-Configuration RAG (baseline static parameters, zero search).
   - Approach 2: Original ReRAG Baseline (greedy coordinate search).
   - Approach 3: Proposed Generalization-Aware ReRAG (validation stability selection & freezing).
7. Evaluation metrics registry, proxy vs. deterministic tagging, and missing-annotation handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.dataset_workflow import (
    DatasetSplits,
    DatasetStatistics,
    DatasetValidator,
    DatasetValidationResult,
    audit_split_leakage,
    compute_dataset_statistics,
    export_dataset_to_csv,
    export_dataset_to_json,
    load_dataset_from_csv,
    load_dataset_from_json,
    split_train_val_test_leakage_resistant,
)
from src.evaluator import EvaluationReport, EvaluationSample
from src.metrics import METRIC_METADATA_REGISTRY, format_metric_value
from src.optimizer import (
    AdaptiveChunkOptimizer,
    AdaptiveSearchConfig,
    CandidateTrial,
    ChunkConfig,
    ChunkOptimizerFactory,
    FixedConfigOptimizer,
    GeneralizationAwareConfig,
    GeneralizationAwareOptimizer,
    OptimizationObjective,
    OptimizationResult,
    SearchBounds,
)


@pytest.fixture
def sample_qa_records() -> List[EvaluationSample]:
    """Create a curated set of test QA samples with various topics, docs, and annotation statuses."""
    return [
        EvaluationSample(
            sample_id="qa_001",
            question="What is the primary architectural purpose of ReRAG?",
            reference_answer="ReRAG is designed to reduce factual hallucinations in retrieval-augmented generation.",
            document_id="doc_rerag_paper",
            topic="architecture",
            difficulty="easy",
            annotation_status="verified_expert",
        ),
        EvaluationSample(
            sample_id="qa_002",
            question="What is the primary architectural purpose of the ReRAG system?",  # Near duplicate of qa_001 (Jaccard = 0.80)
            reference_answer="It reduces hallucinations using iterative claim-level verification and chunk optimization.",
            document_id="doc_rerag_paper",
            topic="architecture",
            difficulty="medium",
            annotation_status="unverified_synthetic",
        ),
        EvaluationSample(
            sample_id="qa_003",
            question="What verification threshold does ReRAG employ for candidate claims?",
            reference_answer="The verification step employs an NLI consistency threshold of 0.70.",
            document_id="doc_rerag_paper",
            topic="verification",
            difficulty="medium",
            annotation_status="verified_expert",
        ),
        EvaluationSample(
            sample_id="qa_004",
            question="What wavelengths does the James Webb Space Telescope detect?",
            reference_answer="The JWST primarily detects infrared wavelengths to view distant astronomical bodies.",
            document_id="doc_astronomy_report",
            topic="astronomy",
            difficulty="easy",
            annotation_status="verified_expert",
        ),
        EvaluationSample(
            sample_id="qa_005",
            question="Which instruments on JWST operate in mid-infrared wavelengths?",
            reference_answer="MIRI (Mid-Infrared Instrument) operates in mid-infrared wavelengths.",
            document_id="doc_astronomy_report",
            topic="astronomy",
            difficulty="hard",
            annotation_status="unverified_synthetic",
        ),
        EvaluationSample(
            sample_id="qa_006",
            question="How do plant chloroplasts convert light into chemical energy?",
            reference_answer="Chloroplasts use chlorophyll pigments to drive the light reactions of photosynthesis.",
            document_id="doc_biology_textbook",
            topic="biology",
            difficulty="easy",
            annotation_status="reviewed",
        ),
        EvaluationSample(
            sample_id="qa_007",
            question="What is the role of ATP synthase in chloroplast thylakoid membranes?",
            reference_answer="ATP synthase synthesizes ATP from ADP using the proton gradient.",
            document_id="doc_biology_textbook",
            topic="biology",
            difficulty="hard",
            annotation_status="verified_expert",
        ),
    ]


# ==============================================================================
# 1. Dataset Validation & Statistics Tests
# ==============================================================================


def test_dataset_statistics(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify statistics computation across verified, synthetic, and reviewed samples."""
    stats = compute_dataset_statistics(sample_qa_records)
    assert stats.total_questions == 7
    assert stats.verified_expert_count == 4
    assert stats.unverified_synthetic_count == 2
    assert stats.reviewed_count == 1
    assert stats.unique_documents == 3
    assert len(stats.topics) == 4
    assert "architecture" in stats.topics
    assert "verification" in stats.topics
    assert "astronomy" in stats.topics
    assert "biology" in stats.topics


def test_near_duplicate_detection(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify near-duplicate question detection via Jaccard token overlap."""
    val_res = DatasetValidator.validate_samples(sample_qa_records)
    assert val_res.is_valid is True
    assert val_res.near_duplicate_count >= 1
    # qa_001 and qa_002 are near duplicates
    near_dups = [item for item in val_res.near_duplicate_questions if "qa_001" in item and "qa_002" in item]
    assert len(near_dups) >= 1


def test_malformed_record_validation() -> None:
    """Ensure validator flags missing answers, missing questions, and invalid document IDs."""
    invalid_samples = [
        EvaluationSample(
            sample_id="bad_001",
            question="",  # Missing question
            reference_answer="Some answer",
        ),
        EvaluationSample(
            sample_id="bad_002",
            question="Valid question?",
            reference_answer="",  # Missing reference answer
        ),
    ]
    val_res = DatasetValidator.validate_samples(invalid_samples)
    assert val_res.is_valid is False
    assert len(val_res.errors) >= 2


# ==============================================================================
# 2. CSV / JSON Import & Export Tests
# ==============================================================================


def test_csv_json_roundtrip(sample_qa_records: List[EvaluationSample], tmp_path: Path) -> None:
    """Verify dataset can be exported to JSON/CSV and reloaded without loss of fields."""
    json_path = tmp_path / "test_dataset.json"
    csv_path = tmp_path / "test_dataset.csv"

    # JSON export & load
    export_dataset_to_json(sample_qa_records, json_path)
    loaded_json, _ = load_dataset_from_json(json_path)
    assert len(loaded_json) == len(sample_qa_records)
    assert loaded_json[0].question == sample_qa_records[0].question
    assert loaded_json[0].annotation_status == "verified_expert"

    # CSV export & load
    export_dataset_to_csv(sample_qa_records, csv_path)
    loaded_csv = load_dataset_from_csv(csv_path)
    assert len(loaded_csv) == len(sample_qa_records)
    assert loaded_csv[1].question == sample_qa_records[1].question
    assert loaded_csv[1].annotation_status == "unverified_synthetic"


# ==============================================================================
# 3. Leakage-Resistant Dataset Splitting Tests
# ==============================================================================


def test_near_duplicate_grouped_split(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify that near-duplicate questions are assigned to the SAME partition."""
    splits = split_train_val_test_leakage_resistant(
        sample_qa_records,
        strategy="near_duplicate_grouped",
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
    )
    # Check that qa_001 and qa_002 ended up in the same partition
    q1_in_train = any(s.sample_id == "qa_001" for s in splits.train_samples)
    q2_in_train = any(s.sample_id == "qa_002" for s in splits.train_samples)
    q1_in_val = any(s.sample_id == "qa_001" for s in splits.val_samples)
    q2_in_val = any(s.sample_id == "qa_002" for s in splits.val_samples)
    q1_in_test = any(s.sample_id == "qa_001" for s in splits.test_samples)
    q2_in_test = any(s.sample_id == "qa_002" for s in splits.test_samples)

    assert (q1_in_train and q2_in_train) or (q1_in_val and q2_in_val) or (q1_in_test and q2_in_test)

    # Run audit
    audit = audit_split_leakage(splits)
    assert audit["exact_duplicate_leakage_count"] == 0
    assert audit["near_duplicate_leakage_count"] == 0
    assert audit["is_leakage_free"] is True


def test_document_level_split(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify document-level splitting prevents questions from same document crossing partitions."""
    splits = split_train_val_test_leakage_resistant(
        sample_qa_records,
        strategy="document_level",
        train_ratio=0.40,
        val_ratio=0.30,
        test_ratio=0.30,
        random_seed=42,
    )
    train_docs = {s.document_id for s in splits.train_samples if s.document_id}
    val_docs = {s.document_id for s in splits.val_samples if s.document_id}
    test_docs = {s.document_id for s in splits.test_samples if s.document_id}

    # Intersections must be empty
    assert len(train_docs.intersection(val_docs)) == 0
    assert len(train_docs.intersection(test_docs)) == 0
    assert len(val_docs.intersection(test_docs)) == 0


def test_split_reproducibility(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify split produces identical distributions with same seed."""
    splits1 = split_train_val_test_leakage_resistant(sample_qa_records, random_seed=123)
    splits2 = split_train_val_test_leakage_resistant(sample_qa_records, random_seed=123)
    assert [s.sample_id for s in splits1.train_samples] == [s.sample_id for s in splits2.train_samples]
    assert [s.sample_id for s in splits1.test_samples] == [s.sample_id for s in splits2.test_samples]


# ==============================================================================
# 4. Optimizer Tests (Approaches 1, 2, and 3)
# ==============================================================================


def test_approach_1_fixed_config_optimizer() -> None:
    """Verify Approach 1 (Fixed-Configuration RAG) evaluates only the baseline with zero search."""
    base_cfg = ChunkConfig(500, 100)
    opt = FixedConfigOptimizer(
        baseline_config=base_cfg,
        evaluate_fn=lambda cfg: {"mean_faithfulness_ratio": 0.85, "context_coverage": 0.70, "mean_semantic_similarity": 0.80},
    )
    result = opt.optimize()
    assert result.selected_config.chunk_size == 500
    assert result.selected_config.chunk_overlap == 100
    assert result.total_trials == 1
    assert result.stopping_reason == "Static fixed baseline evaluation (no search performed)"


def test_approach_2_greedy_rerag_optimizer() -> None:
    """Verify Approach 2 (Original ReRAG Baseline) executes greedy coordinate search."""
    base_cfg = ChunkConfig(400, 50)
    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=1000, min_chunk_overlap=0, max_chunk_overlap=200)

    # Simulated evaluation function where chunk_size=500 has higher score
    def mock_eval(cfg: ChunkConfig) -> Dict[str, Any]:
        bonus = 0.15 if cfg.chunk_size == 500 else 0.0
        return {
            "mean_faithfulness_ratio": 0.70 + bonus,
            "context_coverage": 0.60,
            "mean_semantic_similarity": 0.70,
        }

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=400,
        initial_chunk_overlap=50,
        initial_step_size=100,
        max_iterations=3,
        bounds=bounds,
        random_seed=42,
    )
    opt = AdaptiveChunkOptimizer(
        baseline_config=base_cfg,
        search_config=search_cfg,
        evaluate_fn=mock_eval,
    )
    result = opt.optimize()
    assert result.total_trials >= 2
    assert result.selected_config.chunk_size == 500


def test_approach_3_generalization_aware_optimizer(sample_qa_records: List[EvaluationSample]) -> None:
    """Verify Approach 3 explores on train, selects on val, and freezes configuration."""
    base_cfg = ChunkConfig(400, 50)
    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=800, min_chunk_overlap=0, max_chunk_overlap=200)

    train_samples = sample_qa_records[:4]
    val_samples = sample_qa_records[4:]

    # Evaluator mock
    mock_evaluator = MagicMock()

    def mock_evaluate(samples: Any, **kwargs: Any) -> EvaluationReport:
        cfg = kwargs.get("config_override", {})
        c_sz = cfg.get("chunk_size", 400)
        # 500 gives best score
        score = 0.85 if c_sz == 500 else 0.70
        rep = MagicMock(spec=EvaluationReport)
        rep.aggregate_metrics = {
            "mean_faithfulness_ratio": score,
            "context_coverage": score,
            "mean_semantic_similarity": score,
        }
        return rep

    mock_evaluator.evaluate.side_effect = mock_evaluate

    gen_cfg = GeneralizationAwareConfig(
        bounds=bounds,
        max_trials=6,
        top_k_candidates=2,
        stability_penalty_weight=0.10,
        overfit_penalty_weight=0.05,
        random_seed=42,
    )

    opt = GeneralizationAwareOptimizer(
        baseline_config=base_cfg,
        gen_config=gen_cfg,
        evaluator=mock_evaluator,
        optimization_samples=train_samples,
        validation_samples=val_samples,
    )

    result = opt.optimize()
    assert result.selected_config.chunk_size == 500
    assert result.metadata["strategy"] == "generalization_aware"
    assert "stability_selection" in result.metadata
    assert result.stopping_reason.startswith("Generalization stability selection complete")


def test_zero_leakage_guarantee(sample_qa_records: List[EvaluationSample]) -> None:
    """Strict test verifying that held-out test questions are NEVER passed to the optimizer."""
    splits = split_train_val_test_leakage_resistant(
        sample_qa_records,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
    )

    accessed_sample_ids: List[str] = []

    def tracking_evaluator(samples: Sequence[EvaluationSample], **kwargs: Any) -> EvaluationReport:
        for s in samples:
            accessed_sample_ids.append(s.sample_id)
        rep = MagicMock(spec=EvaluationReport)
        rep.aggregate_metrics = {
            "mean_faithfulness_ratio": 0.80,
            "context_coverage": 0.70,
            "mean_semantic_similarity": 0.75,
        }
        return rep

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.side_effect = tracking_evaluator

    gen_cfg = GeneralizationAwareConfig(max_trials=4, top_k_candidates=2)
    opt = GeneralizationAwareOptimizer(
        baseline_config=ChunkConfig(300, 50),
        gen_config=gen_cfg,
        evaluator=mock_evaluator,
        optimization_samples=splits.train_samples,
        validation_samples=splits.val_samples,
    )
    opt.optimize()

    # Verify none of the test sample IDs were accessed during optimization
    test_ids = {s.sample_id for s in splits.test_samples}
    accessed_set = set(accessed_sample_ids)
    overlap = accessed_set.intersection(test_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Optimizer accessed test samples: {overlap}"


# ==============================================================================
# 5. Metrics Registry & Formatting Tests
# ==============================================================================


def test_metric_metadata_registry() -> None:
    """Verify metric categorization across deterministic, neural_proxy, and llm_judge."""
    assert METRIC_METADATA_REGISTRY["mean_semantic_similarity"]["category"] == "neural_proxy"
    assert METRIC_METADATA_REGISTRY["mean_faithfulness_ratio"]["category"] == "deterministic"
    assert METRIC_METADATA_REGISTRY["mean_groundedness"]["category"] == "llm_judge"
    assert METRIC_METADATA_REGISTRY["mean_latency_seconds"]["category"] == "system_performance"


def test_format_metric_value() -> None:
    """Verify format_metric_value correctly handles None and missing references as 'Not available'."""
    assert format_metric_value(None) == "Not available"
    assert format_metric_value(float("nan")) == "Not available"
    assert format_metric_value(0.8524, 2) == "0.85"
