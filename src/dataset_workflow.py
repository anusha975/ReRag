"""Expert Question-Answer Dataset Workflow and Partitioning Module for ReRAG.

Provides validation, integrity checks, and three-way dataset partitioning:
- Optimization set (training): Used exclusively for parameter search.
- Validation set: Used for intermediate validation, early stopping, and hyperparameter tuning.
- Held-out test set: Strictly sequestered; evaluated solely once post-selection to detect
  overfitting and compute the true empirical generalization gap.

Enforces zero-leakage guarantees: optimizers never receive or evaluate against held-out test data.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    """Summary report of dataset validation checks."""

    is_valid: bool
    sample_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    topics: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    abstention_count: int = 0
    duplicate_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "sample_count": self.sample_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "topics": self.topics,
            "difficulties": self.difficulties,
            "abstention_count": self.abstention_count,
            "duplicate_ids": self.duplicate_ids,
        }

    @property
    def total_samples(self) -> int:
        """Alias for sample_count."""
        return self.sample_count

    @property
    def negative_samples_count(self) -> int:
        """Alias for abstention_count."""
        return self.abstention_count

    @property
    def positive_samples_count(self) -> int:
        """Return count of positive (non-abstention) samples."""
        return max(0, self.sample_count - self.abstention_count)


class DatasetValidator:
    """Validates structure, provenance metadata, and integrity of evaluation datasets."""

    REQUIRED_FIELDS = ("sample_id", "question", "reference_answer")

    @classmethod
    def validate_samples(cls, samples: Sequence[Any]) -> DatasetValidationResult:
        """Validate a collection of EvaluationSample instances."""
        return cls.validate_dataset(samples)

    @classmethod
    def validate_file(cls, file_path: Union[Path, str]) -> DatasetValidationResult:
        """Validate a dataset file from disk."""
        return cls.validate_dataset(file_path)

    @classmethod
    def validate_sample(
        cls,
        sample: Any,
        index: int = 0,
    ) -> Tuple[List[str], List[str]]:
        """Validate a single evaluation sample or dictionary.

        Args:
            sample: EvaluationSample instance or dictionary.
            index: 0-indexed position in dataset for diagnostic reporting.

        Returns:
            Tuple[List[str], List[str]]: (errors, warnings)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Convert to dict if object
        if hasattr(sample, "to_dict"):
            data = sample.to_dict()
        elif hasattr(sample, "__dict__"):
            data = vars(sample)
        elif isinstance(sample, dict):
            data = sample
        else:
            errors.append(f"Sample at index {index} is not a valid dict or object: {type(sample)}")
            return errors, warnings

        # 1. Required fields
        for req in cls.REQUIRED_FIELDS:
            val = data.get(req)
            if val is None or str(val).strip() == "":
                errors.append(f"Sample at index {index} (ID: {data.get('sample_id', 'unknown')}) missing required field '{req}'")

        sample_id = str(data.get("sample_id", "")).strip()
        question = str(data.get("question", "")).strip()
        ref_answer = str(data.get("reference_answer", "")).strip()

        if question and len(question) < 5:
            warnings.append(f"Sample '{sample_id}' has suspiciously short question ({len(question)} chars): '{question}'")

        if ref_answer and len(ref_answer) < 5:
            warnings.append(f"Sample '{sample_id}' has suspiciously short reference answer: '{ref_answer}'")

        # 2. Page number validation
        pages = data.get("expected_pages")
        if pages is not None:
            if not isinstance(pages, (list, tuple)):
                errors.append(f"Sample '{sample_id}' expected_pages must be a list, got {type(pages)}")
            else:
                for p in pages:
                    if not isinstance(p, int) or p <= 0:
                        errors.append(f"Sample '{sample_id}' page number {p} is invalid; pages must be positive integers.")

        # 3. Provenance validation
        expected_src = data.get("expected_source")
        doc_id = data.get("document_id")
        rel_ids = data.get("relevant_doc_ids")
        if not expected_src and not doc_id and not rel_ids:
            warnings.append(f"Sample '{sample_id}' has no expected_source, document_id, or relevant_doc_ids annotated.")

        return errors, warnings

    @classmethod
    def validate_dataset(
        cls,
        samples_or_path: Union[Sequence[Any], Path, str],
    ) -> DatasetValidationResult:
        """Validate an entire collection of samples or a file path.

        Args:
            samples_or_path: Sequence of samples or path to JSON/JSONL file.

        Returns:
            DatasetValidationResult: Comprehensive validation report.
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []
        seen_ids: Set[str] = set()
        duplicate_ids: List[str] = []
        topic_counts: Counter[str] = Counter()
        difficulty_counts: Counter[str] = Counter()
        abstention_count = 0

        # Load if path provided
        if isinstance(samples_or_path, (Path, str)):
            from src.evaluator import load_dataset
            samples = load_dataset(samples_or_path)
        else:
            samples = list(samples_or_path)

        if not samples:
            return DatasetValidationResult(
                is_valid=False,
                sample_count=0,
                errors=["Dataset is completely empty."],
            )

        for idx, sample in enumerate(samples):
            errs, warns = cls.validate_sample(sample, index=idx)
            all_errors.extend(errs)
            all_warnings.extend(warns)

            sid = getattr(sample, "sample_id", None) or (sample.get("sample_id") if isinstance(sample, dict) else None)
            sid_str = str(sid).strip() if sid else f"sample_{idx}"
            if sid_str in seen_ids:
                duplicate_ids.append(sid_str)
                all_errors.append(f"Duplicate sample_id detected: '{sid_str}' at index {idx}")
            seen_ids.add(sid_str)

            # Metadata extraction
            meta = getattr(sample, "metadata", None) or (sample.get("metadata") if isinstance(sample, dict) else {})
            if isinstance(meta, dict):
                top = meta.get("topic") or "unclassified"
                topic_counts[str(top)] += 1
                diff = meta.get("difficulty") or "standard"
                difficulty_counts[str(diff)] += 1
                if meta.get("is_abstention_case") is True:
                    abstention_count += 1

            # Check reference answer for abstention phrases
            ref_ans = getattr(sample, "reference_answer", None) or (sample.get("reference_answer") if isinstance(sample, dict) else "")
            ref_str = str(ref_ans).lower()
            if "insufficient" in ref_str or "does not contain" in ref_str or "cannot answer" in ref_str:
                if isinstance(meta, dict) and not meta.get("is_abstention_case"):
                    abstention_count += 1

        is_valid = len(all_errors) == 0

        return DatasetValidationResult(
            is_valid=is_valid,
            sample_count=len(samples),
            errors=all_errors,
            warnings=all_warnings,
            topics=dict(topic_counts),
            difficulties=dict(difficulty_counts),
            abstention_count=abstention_count,
            duplicate_ids=duplicate_ids,
        )


@dataclass
class DatasetSplits:
    """Holds 3-way partitioned datasets with strict segregation."""

    optimization: List[Any]  # Train / Parameter search split
    validation: List[Any]    # Validation / Hyperparameter checkpointing split
    held_out_test: List[Any] # Held-out test split (strictly isolated during search)

    @property
    def train_samples(self) -> List[Any]:
        """Convenience alias for optimization split."""
        return self.optimization

    @property
    def val_samples(self) -> List[Any]:
        """Convenience alias for validation split."""
        return self.validation

    @property
    def test_samples(self) -> List[Any]:
        """Convenience alias for held_out_test split."""
        return self.held_out_test

    @property
    def total_count(self) -> int:
        return len(self.optimization) + len(self.validation) + len(self.held_out_test)

    def summary(self) -> Dict[str, int]:
        return {
            "optimization_count": len(self.optimization),
            "validation_count": len(self.validation),
            "held_out_test_count": len(self.held_out_test),
            "total_count": self.total_count,
        }

    def verify_no_overlap(self) -> bool:
        """Verify zero sample ID overlap across all three splits."""
        def _get_ids(split: Sequence[Any]) -> Set[str]:
            ids = set()
            for s in split:
                sid = getattr(s, "sample_id", None) or (s.get("sample_id") if isinstance(s, dict) else None)
                if sid:
                    ids.add(str(sid))
            return ids

        opt_ids = _get_ids(self.optimization)
        val_ids = _get_ids(self.validation)
        test_ids = _get_ids(self.held_out_test)

        return (
            len(opt_ids & val_ids) == 0
            and len(opt_ids & test_ids) == 0
            and len(val_ids & test_ids) == 0
        )


def split_train_val_test(
    samples: Sequence[Any],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    stratify_by_topic: bool = False,
    random_seed: Optional[int] = None,
) -> DatasetSplits:
    """Partition evaluation samples into three distinct, non-overlapping splits.

    Guarantees:
    1. Complete partition: len(optimization) + len(validation) + len(held_out_test) == len(samples).
    2. Strict isolation: No sample ID exists in more than one partition.
    3. Reproducibility: Deterministic split given identical seed.

    Args:
        samples: Sequence of EvaluationSample instances.
        train_ratio: Fraction for parameter optimization (search).
        val_ratio: Fraction for intermediate validation / early stopping.
        test_ratio: Fraction for final held-out generalization test.
        seed: Random seed for deterministic reproducibility.
        stratify_by_topic: If True, attempts balanced topic allocation across splits.
        random_seed: Optional alias for seed parameter.

    Returns:
        DatasetSplits: Named container with optimization, validation, and held_out_test lists.
    """
    effective_seed = random_seed if random_seed is not None else seed

    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be strictly positive, got {train_ratio}")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(f"val_ratio and test_ratio must be non-negative, got {val_ratio}, {test_ratio}")

    if not samples:
        return DatasetSplits(optimization=[], validation=[], held_out_test=[])

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio > 1.0 + 1e-4:
        raise ValueError(f"Ratios cannot exceed 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}")

    import random
    sample_list = list(samples)
    n = len(sample_list)

    if n == 1:
        return DatasetSplits(optimization=sample_list, validation=[], held_out_test=[])
    if n == 2:
        return DatasetSplits(optimization=[sample_list[0]], validation=[], held_out_test=[sample_list[1]])

    rng = random.Random(effective_seed)

    if not stratify_by_topic:
        shuffled = list(sample_list)
        rng.shuffle(shuffled)

        test_count = max(1 if test_ratio > 0 else 0, round(n * test_ratio))
        val_count = max(1 if val_ratio > 0 and (n - test_count) > 1 else 0, round(n * val_ratio))

        # Adjust bounds so optimization has at least 1 sample if possible
        if test_count + val_count >= n:
            test_count = max(1, n // 3)
            val_count = max(1, n // 3)

        held_out_test = shuffled[:test_count]
        validation = shuffled[test_count : test_count + val_count]
        optimization = shuffled[test_count + val_count :]
    else:
        # Group by topic
        by_topic: Dict[str, List[Any]] = {}
        for s in sample_list:
            meta = getattr(s, "metadata", {}) or {}
            top = meta.get("topic", "default") if isinstance(meta, dict) else "default"
            by_topic.setdefault(top, []).append(s)

        optimization = []
        validation = []
        held_out_test = []

        for top, topic_samples in by_topic.items():
            top_shuffled = list(topic_samples)
            rng.shuffle(top_shuffled)
            top_n = len(top_shuffled)
            top_test = max(1 if test_ratio > 0 and top_n >= 3 else 0, round(top_n * test_ratio))
            top_val = max(1 if val_ratio > 0 and top_n >= 3 else 0, round(top_n * val_ratio))

            if top_test + top_val >= top_n:
                top_test = max(0, (top_n - 1) // 2)
                top_val = max(0, top_n - 1 - top_test)

            held_out_test.extend(top_shuffled[:top_test])
            validation.extend(top_shuffled[top_test : top_test + top_val])
            optimization.extend(top_shuffled[top_test + top_val :])

    splits = DatasetSplits(
        optimization=optimization,
        validation=validation,
        held_out_test=held_out_test,
    )

    logger.info(
        "Partitioned dataset into 3 splits (Total: %d | Opt: %d, Val: %d, Held-out Test: %d)",
        n,
        len(splits.optimization),
        len(splits.validation),
        len(splits.held_out_test),
    )

    return splits


def save_dataset_splits(
    splits: DatasetSplits,
    output_dir: Union[Path, str],
    prefix: str = "dataset",
) -> Dict[str, Path]:
    """Persist the three dataset splits to JSON files for reproducibility.

    Args:
        splits: Partitioned DatasetSplits instance.
        output_dir: Directory where JSON files will be written.
        prefix: Filename prefix.

    Returns:
        Dict[str, Path]: Mapping of split name to written file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for split_name, sample_list in [
        ("optimization", splits.optimization),
        ("validation", splits.validation),
        ("held_out_test", splits.held_out_test),
    ]:
        file_path = out_path / f"{prefix}_{split_name}.json"
        data = {
            "metadata": {
                "split_name": split_name,
                "sample_count": len(sample_list),
            },
            "samples": [
                s.to_dict() if hasattr(s, "to_dict") else s for s in sample_list
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        paths[split_name] = file_path

    paths["train"] = paths["optimization"]
    paths["val"] = paths["validation"]
    paths["test"] = paths["held_out_test"]

    return paths
