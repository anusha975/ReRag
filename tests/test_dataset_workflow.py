"""Automated unit and regression tests for Dataset Workflow and Zero-Leakage Splitting."""

import json
from pathlib import Path
import pytest

from src.dataset_workflow import (
    DatasetSplits,
    DatasetValidationResult,
    DatasetValidator,
    save_dataset_splits,
    split_train_val_test,
)
from src.evaluator import EvaluationSample, load_dataset


def test_dataset_validator_valid_sample():
    """Verify that correctly formatted samples pass validation."""
    samples = [
        EvaluationSample(
            sample_id="test_001",
            question="What is the capital of France?",
            ground_truth_answer="Paris",
            expected_passages=["Paris is the capital of France."],
            expected_citations=[1],
            difficulty="easy",
            is_negative=False,
        ),
        EvaluationSample(
            sample_id="test_002",
            question="What is the population of Mars?",
            ground_truth_answer="Insufficient information.",
            expected_passages=[],
            expected_citations=[],
            difficulty="hard",
            is_negative=True,
        ),
    ]

    result = DatasetValidator.validate_samples(samples)
    assert result.is_valid is True
    assert result.total_samples == 2
    assert result.negative_samples_count == 1
    assert result.positive_samples_count == 1
    assert len(result.errors) == 0


def test_dataset_validator_duplicate_ids():
    """Verify that duplicate sample IDs trigger validation errors."""
    samples = [
        EvaluationSample(
            sample_id="dup_id",
            question="Question 1",
            ground_truth_answer="Answer 1",
        ),
        EvaluationSample(
            sample_id="dup_id",
            question="Question 2",
            ground_truth_answer="Answer 2",
        ),
    ]

    result = DatasetValidator.validate_samples(samples)
    assert result.is_valid is False
    assert any("Duplicate sample_id" in err for err in result.errors)


def test_dataset_validator_invalid_page_number():
    """Verify non-positive page numbers (0 or negative) trigger validation errors."""
    samples = [
        EvaluationSample(
            sample_id="invalid_page",
            question="Question",
            ground_truth_answer="Answer",
            expected_citations=[0],  # 0 is invalid (1-indexed required)
        ),
    ]

    result = DatasetValidator.validate_samples(samples)
    assert result.is_valid is False
    assert any("positive integer" in err for err in result.errors)


def test_split_train_val_test_partition():
    """Verify 3-way split guarantees zero overlap and correct proportions."""
    samples = [
        EvaluationSample(sample_id=f"sample_{i}", question=f"Q{i}", ground_truth_answer=f"A{i}")
        for i in range(20)
    ]

    splits = split_train_val_test(samples, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, random_seed=42)

    assert len(splits.train_samples) == 10
    assert len(splits.val_samples) == 4
    assert len(splits.test_samples) == 6

    # Verify zero overlap between sets
    train_ids = {s.sample_id for s in splits.train_samples}
    val_ids = {s.sample_id for s in splits.val_samples}
    test_ids = {s.sample_id for s in splits.test_samples}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train_ids | val_ids | test_ids) == 20


def test_split_train_val_test_invalid_ratios():
    """Verify that ratios summing to greater than 1.0 or non-positive raise ValueError."""
    samples = [EvaluationSample(sample_id="s1", question="Q", ground_truth_answer="A")]

    with pytest.raises(ValueError, match="cannot exceed 1.0"):
        split_train_val_test(samples, train_ratio=0.6, val_ratio=0.3, test_ratio=0.2)

    with pytest.raises(ValueError, match="strictly positive"):
        split_train_val_test(samples, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.4)


def test_expert_qa_dataset_file_validation(tmp_path):
    """Verify the real data/evaluation/expert_qa_dataset.json and jsonl files pass full validation."""
    json_path = Path("data/evaluation/expert_qa_dataset.json")
    if json_path.exists():
        result = DatasetValidator.validate_file(json_path)
        assert result.is_valid is True, f"Validation errors: {result.errors}"
        assert result.total_samples >= 20
        assert result.negative_samples_count >= 4

    jsonl_path = Path("data/evaluation/expert_qa_dataset.jsonl")
    if jsonl_path.exists():
        result = DatasetValidator.validate_file(jsonl_path)
        assert result.is_valid is True, f"Validation errors: {result.errors}"
        assert result.total_samples >= 20


def test_save_and_load_splits(tmp_path):
    """Verify persistence and reloading of dataset splits."""
    samples = [
        EvaluationSample(sample_id=f"sample_{i}", question=f"Q{i}", ground_truth_answer=f"A{i}")
        for i in range(10)
    ]
    splits = split_train_val_test(samples, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, random_seed=123)

    saved_paths = save_dataset_splits(splits, output_dir=tmp_path, prefix="test_split")
    assert saved_paths["train"].exists()
    assert saved_paths["val"].exists()
    assert saved_paths["test"].exists()

    loaded_train = load_dataset(saved_paths["train"])
    assert len(loaded_train) == 5
    assert loaded_train[0].sample_id == splits.train_samples[0].sample_id
