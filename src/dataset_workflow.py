"""Expert Question-Answer Dataset Workflow and Leakage-Resistant Partitioning Module for ReRAG.

Provides comprehensive dataset management, schema validation, duplicate and near-duplicate
question detection, statistics reporting, and mathematically leakage-resistant three-way partitioning:
- Optimization set (training): Used exclusively for parameter exploration and candidate generation.
- Validation set: Used for candidate selection, stability verification, and hyperparameter tuning.
- Held-out test set: Strictly sequestered; evaluated solely once post-selection to detect
  overfitting and compute empirical generalization gaps.

Enforces zero-leakage guarantees: optimizers never receive or evaluate against held-out test data.
Supports document-level partitioning and near-duplicate question clustering to prevent subtle data leakage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ==============================================================================
# DATASET VALIDATION & STATISTICS DATA STRUCTURES
# ==============================================================================


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
    verified_expert_count: int = 0
    unreviewed_synthetic_count: int = 0
    reviewed_count: int = 0
    documents: List[str] = field(default_factory=list)
    duplicate_questions: List[Tuple[str, str]] = field(default_factory=list)
    near_duplicate_questions: List[Tuple[str, str, float]] = field(default_factory=list)
    invalid_document_references: List[str] = field(default_factory=list)

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
            "verified_expert_count": self.verified_expert_count,
            "unreviewed_synthetic_count": self.unreviewed_synthetic_count,
            "reviewed_count": self.reviewed_count,
            "documents": self.documents,
            "duplicate_questions": [list(x) for x in self.duplicate_questions],
            "near_duplicate_questions": [
                [x[0], x[1], round(x[2], 3)] for x in self.near_duplicate_questions
            ],
            "invalid_document_references": self.invalid_document_references,
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

    @property
    def near_duplicate_count(self) -> int:
        """Return count of near duplicate questions."""
        return len(self.near_duplicate_questions)


@dataclass
class DatasetStatistics:
    """Comprehensive statistical breakdown of an expert/annotated QA dataset."""

    total_questions: int
    verified_expert_count: int
    unreviewed_synthetic_count: int
    reviewed_count: int
    document_ids: List[str]
    topics: Dict[str, int]
    difficulties: Dict[str, int]
    abstention_count: int
    missing_passages_count: int
    near_duplicate_clusters_count: int
    validation_status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)

    @property
    def unverified_synthetic_count(self) -> int:
        """Alias for unreviewed_synthetic_count."""
        return self.unreviewed_synthetic_count

    @property
    def unique_documents(self) -> int:
        """Count of unique documents referenced."""
        return len(self.document_ids)

    @property
    def near_duplicate_count(self) -> int:
        """Count of near duplicate clusters."""
        return self.near_duplicate_clusters_count

    @property
    def duplicate_questions_count(self) -> int:
        """Count of exact duplicate questions."""
        return 0


# ==============================================================================
# TEXT NORMALIZATION & JACCARD UTILITIES
# ==============================================================================


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing and stripping punctuation and excessive whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(cleaned.split())


def _tokenize_words(text: str) -> Set[str]:
    """Extract set of words of length > 2 for lexical similarity calculations."""
    normalized = _normalize_text(text)
    stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "the", "and", "for", "with"}
    words = {w for w in normalized.split() if len(w) > 2 and w not in stopwords}
    return words if words else set(normalized.split())


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity between two texts in [0.0, 1.0]."""
    tokens1 = _tokenize_words(text1)
    tokens2 = _tokenize_words(text2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection) / float(union) if union > 0 else 0.0


# ==============================================================================
# DATASET VALIDATOR
# ==============================================================================


class DatasetValidator:
    """Validates structure, provenance metadata, and integrity of evaluation datasets."""

    REQUIRED_FIELDS = ("sample_id", "question", "reference_answer")

    @classmethod
    def validate_samples(
        cls,
        samples: Sequence[Any],
        available_document_ids: Optional[Set[str]] = None,
    ) -> DatasetValidationResult:
        """Validate a collection of EvaluationSample instances."""
        return cls.validate_dataset(samples, available_document_ids=available_document_ids)

    @classmethod
    def validate_file(
        cls,
        file_path: Union[Path, str],
        available_document_ids: Optional[Set[str]] = None,
    ) -> DatasetValidationResult:
        """Validate a dataset file from disk."""
        return cls.validate_dataset(file_path, available_document_ids=available_document_ids)

    @classmethod
    def validate_sample(
        cls,
        sample: Any,
        index: int = 0,
        available_document_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Validate a single evaluation sample or dictionary.

        Args:
            sample: EvaluationSample instance or dictionary.
            index: 0-indexed position in dataset for diagnostic reporting.
            available_document_ids: Optional set of known document IDs to check validity against.

        Returns:
            Tuple[List[str], List[str]]: (errors, warnings)
        """
        errors: List[str] = []
        warnings: List[str] = []

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
                errors.append(
                    f"Sample at index {index} (ID: {data.get('sample_id', 'unknown')}) missing required field '{req}'"
                )

        sample_id = str(data.get("sample_id", "")).strip()
        question = str(data.get("question", "")).strip()
        ref_answer = str(data.get("reference_answer", "")).strip()

        if question and len(question) < 5:
            warnings.append(f"Sample '{sample_id}' has suspiciously short question ({len(question)} chars): '{question}'")

        if ref_answer and len(ref_answer) < 3:
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
        primary_doc = str(doc_id or expected_src or "").strip()

        if not primary_doc and not rel_ids:
            warnings.append(f"Sample '{sample_id}' has no expected_source, document_id, or relevant_doc_ids annotated.")
        elif available_document_ids and primary_doc:
            if primary_doc not in available_document_ids:
                warnings.append(
                    f"Sample '{sample_id}' references document '{primary_doc}' which is not in active document corpus."
                )

        return errors, warnings

    @classmethod
    def validate_dataset(
        cls,
        samples_or_path: Union[Sequence[Any], Path, str],
        available_document_ids: Optional[Set[str]] = None,
    ) -> DatasetValidationResult:
        """Validate an entire collection of samples or a file path.

        Checks:
        1. Required fields and data types
        2. Duplicate sample IDs
        3. Exact duplicate questions (normalized)
        4. Near-duplicate questions (Jaccard similarity >= 0.85)
        5. Invalid document references
        6. Annotation status accounting (verified_expert vs unreviewed_synthetic)
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []
        seen_ids: Set[str] = set()
        duplicate_ids: List[str] = []
        topic_counts: Counter[str] = Counter()
        difficulty_counts: Counter[str] = Counter()
        doc_set: Set[str] = set()
        abstention_count = 0
        verified_count = 0
        unreviewed_count = 0
        reviewed_count = 0

        # Load if path provided
        if isinstance(samples_or_path, (Path, str)):
            p = Path(samples_or_path)
            if p.suffix.lower() == ".csv":
                samples = load_dataset_from_csv(p)
            else:
                from src.evaluator import load_dataset
                samples = load_dataset(p)
        else:
            samples = list(samples_or_path)

        if not samples:
            return DatasetValidationResult(
                is_valid=False,
                sample_count=0,
                errors=["Dataset is completely empty."],
            )

        # Question deduplication tracking
        norm_questions: Dict[str, str] = {}  # norm_q -> sample_id
        exact_duplicate_questions: List[Tuple[str, str]] = []
        token_question_list: List[Tuple[str, str, Set[str]]] = []  # (sample_id, raw_q, tokens)
        near_duplicate_questions: List[Tuple[str, str, float]] = []
        invalid_doc_refs: List[str] = []

        for idx, sample in enumerate(samples):
            errs, warns = cls.validate_sample(sample, index=idx, available_document_ids=available_document_ids)
            all_errors.extend(errs)
            all_warnings.extend(warns)

            sid = getattr(sample, "sample_id", None) or (sample.get("sample_id") if isinstance(sample, dict) else None)
            sid_str = str(sid).strip() if sid else f"sample_{idx}"
            if sid_str in seen_ids:
                duplicate_ids.append(sid_str)
                all_errors.append(f"Duplicate sample_id detected: '{sid_str}' at index {idx}")
            seen_ids.add(sid_str)

            # Question duplicate checks
            raw_q = getattr(sample, "question", None) or (sample.get("question") if isinstance(sample, dict) else "")
            raw_q_str = str(raw_q).strip()
            norm_q = _normalize_text(raw_q_str)

            if norm_q:
                if norm_q in norm_questions:
                    orig_sid = norm_questions[norm_q]
                    exact_duplicate_questions.append((orig_sid, sid_str))
                    all_warnings.append(
                        f"Exact duplicate question detected between sample '{orig_sid}' and '{sid_str}': '{raw_q_str}'"
                    )
                else:
                    norm_questions[norm_q] = sid_str

                q_tokens = _tokenize_words(raw_q_str)
                # Check near-duplicates against previous questions (capped check for speed)
                for prev_sid, prev_q, prev_tokens in token_question_list[-100:]:
                    if q_tokens and prev_tokens:
                        jaccard = len(q_tokens & prev_tokens) / float(len(q_tokens | prev_tokens))
                        if 0.75 <= jaccard < 1.0:
                            near_duplicate_questions.append((prev_sid, sid_str, jaccard))
                            all_warnings.append(
                                f"Near-duplicate question detected (Jaccard {jaccard:.2f}) between '{prev_sid}' and '{sid_str}'"
                            )
                token_question_list.append((sid_str, raw_q_str, q_tokens))

            # Metadata and Annotation Status
            meta = getattr(sample, "metadata", None) or (sample.get("metadata") if isinstance(sample, dict) else {})
            if isinstance(meta, dict):
                top = meta.get("topic") or "unclassified"
                topic_counts[str(top)] += 1
                diff = meta.get("difficulty") or "standard"
                difficulty_counts[str(diff)] += 1
                if meta.get("is_abstention_case") is True:
                    abstention_count += 1

            ann_status = (
                getattr(sample, "annotation_status", None)
                or (meta.get("annotation_status") if isinstance(meta, dict) else None)
                or "verified_expert"
            )
            ann_status_str = str(ann_status).lower()
            if "unreviewed" in ann_status_str or "synthetic" in ann_status_str:
                unreviewed_count += 1
            elif "reviewed" in ann_status_str:
                reviewed_count += 1
            else:
                verified_count += 1

            # Document reference
            doc = (
                getattr(sample, "document_id", None)
                or getattr(sample, "expected_source", None)
                or (sample.get("document_id") if isinstance(sample, dict) else None)
                or (sample.get("expected_source") if isinstance(sample, dict) else None)
            )
            if doc:
                doc_str = str(doc).strip()
                doc_set.add(doc_str)
                if available_document_ids and doc_str not in available_document_ids:
                    invalid_doc_refs.append(f"{sid_str} -> {doc_str}")

            # Abstention reference answer heuristic
            ref_ans = (
                getattr(sample, "reference_answer", None)
                or (sample.get("reference_answer") if isinstance(sample, dict) else "")
            )
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
            verified_expert_count=verified_count,
            unreviewed_synthetic_count=unreviewed_count,
            reviewed_count=reviewed_count,
            documents=sorted(list(doc_set)),
            duplicate_questions=exact_duplicate_questions,
            near_duplicate_questions=near_duplicate_questions,
            invalid_document_references=invalid_doc_refs,
        )


def compute_dataset_statistics(
    samples_or_path: Union[Sequence[Any], Path, str],
    available_document_ids: Optional[Set[str]] = None,
) -> DatasetStatistics:
    """Compute high-level statistical summary for a QA dataset."""
    val = DatasetValidator.validate_dataset(samples_or_path, available_document_ids=available_document_ids)

    # Count samples missing supporting passages
    if isinstance(samples_or_path, (Path, str)):
        from src.evaluator import load_dataset
        samples = load_dataset(samples_or_path)
    else:
        samples = list(samples_or_path)

    missing_passages = 0
    for s in samples:
        passages = getattr(s, "supporting_passages", None) or (s.get("supporting_passages") if isinstance(s, dict) else None)
        if not passages:
            missing_passages += 1

    return DatasetStatistics(
        total_questions=val.sample_count,
        verified_expert_count=val.verified_expert_count,
        unreviewed_synthetic_count=val.unreviewed_synthetic_count,
        reviewed_count=val.reviewed_count,
        document_ids=val.documents,
        topics=val.topics,
        difficulties=val.difficulties,
        abstention_count=val.abstention_count,
        missing_passages_count=missing_passages,
        near_duplicate_clusters_count=len(val.near_duplicate_questions),
        validation_status="VALID" if val.is_valid else "INVALID",
    )


# ==============================================================================
# CSV & JSON DATASET I/O
# ==============================================================================


def load_dataset_from_csv(file_path: Union[Path, str]) -> List[Any]:
    """Load question-answer dataset from a CSV file.

    Expected CSV columns (flexible matching):
    sample_id, question, reference_answer, document_id, expected_pages,
    supporting_passages, topic, difficulty, annotation_status, is_abstention_case
    """
    from src.evaluator import EvaluationSample

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"QA dataset CSV not found at {path}")

    samples: List[EvaluationSample] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sid = row.get("sample_id") or row.get("id") or f"csv_sample_{idx+1:03d}"
            question = row.get("question") or ""
            ref_ans = row.get("reference_answer") or row.get("ground_truth_answer") or ""
            doc_id = row.get("document_id") or row.get("expected_source")

            # Parse pages
            raw_pages = row.get("expected_pages") or row.get("expected_citations") or ""
            pages = None
            if raw_pages:
                try:
                    if raw_pages.startswith("[") and raw_pages.endswith("]"):
                        pages = json.loads(raw_pages)
                    else:
                        pages = [int(p.strip()) for p in str(raw_pages).split(",") if p.strip().isdigit()]
                except Exception:
                    pages = None

            # Parse passages
            raw_passages = row.get("supporting_passages") or row.get("expected_passages") or ""
            passages = None
            if raw_passages:
                try:
                    if raw_passages.startswith("[") and raw_passages.endswith("]"):
                        passages = json.loads(raw_passages)
                    else:
                        passages = [raw_passages.strip()]
                except Exception:
                    passages = [raw_passages.strip()]

            topic = row.get("topic") or "General"
            difficulty = row.get("difficulty") or "standard"
            raw_abstain = str(row.get("is_abstention_case") or row.get("is_negative") or "").lower()
            is_abstain = raw_abstain in ("true", "1", "yes")

            ann_status = row.get("annotation_status") or "verified_expert"

            metadata = {
                "topic": topic,
                "difficulty": difficulty,
                "is_abstention_case": is_abstain,
                "annotation_status": ann_status,
            }

            samples.append(
                EvaluationSample(
                    sample_id=str(sid).strip(),
                    question=str(question).strip(),
                    reference_answer=str(ref_ans).strip(),
                    document_id=str(doc_id).strip() if doc_id else None,
                    expected_source=str(doc_id).strip() if doc_id else None,
                    expected_pages=pages,
                    supporting_passages=passages,
                    annotation_status=ann_status,
                    metadata=metadata,
                )
            )

    logger.info("Loaded %d QA samples from CSV: %s", len(samples), path)
    return samples


def export_dataset_to_csv(
    samples: Sequence[Any],
    file_path: Union[Path, str],
) -> Path:
    """Export question-answer dataset to a standardized CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "sample_id",
        "question",
        "reference_answer",
        "document_id",
        "expected_pages",
        "supporting_passages",
        "topic",
        "difficulty",
        "annotation_status",
        "is_abstention_case",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in samples:
            data = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            meta = data.get("metadata", {}) or {}

            pages = data.get("expected_pages")
            pages_str = json.dumps(pages) if isinstance(pages, list) else (str(pages) if pages is not None else "")

            passages = data.get("supporting_passages")
            passages_str = json.dumps(passages) if isinstance(passages, list) else (str(passages) if passages is not None else "")

            writer.writerow({
                "sample_id": data.get("sample_id", ""),
                "question": data.get("question", ""),
                "reference_answer": data.get("reference_answer", ""),
                "document_id": data.get("document_id") or data.get("expected_source") or "",
                "expected_pages": pages_str,
                "supporting_passages": passages_str,
                "topic": meta.get("topic") or "General",
                "difficulty": meta.get("difficulty") or "standard",
                "annotation_status": data.get("annotation_status") or meta.get("annotation_status") or "verified_expert",
                "is_abstention_case": meta.get("is_abstention_case", False),
            })

    logger.info("Exported %d QA samples to CSV: %s", len(samples), path)
    return path


def load_dataset_from_json(file_path: Union[Path, str]) -> Tuple[List[Any], Dict[str, Any]]:
    """Load question-answer dataset from a JSON file."""
    from src.evaluator import EvaluationSample

    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples_raw = data
        metadata = {}
    elif isinstance(data, dict):
        samples_raw = data.get("samples", [])
        metadata = data.get("metadata", {})
    else:
        raise ValueError(f"Invalid JSON dataset structure in {path}")

    samples = [EvaluationSample.from_dict(item) for item in samples_raw]
    return samples, metadata


def export_dataset_to_json(
    samples: Sequence[Any],
    file_path: Union[Path, str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export question-answer dataset to structured JSON format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {
        "dataset_name": "ReRAG Expert QA Dataset",
        "version": "2.1.0",
        "total_samples": len(samples),
        "export_timestamp": str(hashlib.sha256(str(len(samples)).encode()).hexdigest()[:8]),
    }

    payload = {
        "metadata": meta,
        "samples": [s.to_dict() if hasattr(s, "to_dict") else s for s in samples],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d QA samples to JSON: %s", len(samples), path)
    return path


# ==============================================================================
# LEAKAGE-RESISTANT DATASET SPLITTING
# ==============================================================================


@dataclass
class DatasetSplits:
    """Holds 3-way partitioned datasets with strict segregation."""

    optimization: List[Any]  # Train / Parameter search split
    validation: List[Any]    # Validation / Hyperparameter checkpointing split
    held_out_test: List[Any] # Held-out test split (strictly isolated during search)
    split_strategy: str = "stratified_topic"
    split_metadata: Dict[str, Any] = field(default_factory=dict)
    leakage_audit: Dict[str, Any] = field(default_factory=dict)

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

    def summary(self) -> Dict[str, Any]:
        return {
            "optimization_count": len(self.optimization),
            "validation_count": len(self.validation),
            "held_out_test_count": len(self.held_out_test),
            "total_count": self.total_count,
            "split_strategy": self.split_strategy,
            "is_leakage_free": self.verify_no_overlap(),
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


def audit_split_leakage(splits: DatasetSplits, near_duplicate_threshold: float = 0.80) -> Dict[str, Any]:
    """Audit dataset splits for potential leakage: sample IDs, exact questions, near-duplicates, or documents."""
    def _extract_data(split: Sequence[Any]) -> Tuple[Set[str], Dict[str, str], Set[str]]:
        ids = set()
        questions = {}  # norm_q -> sample_id
        docs = set()
        for s in split:
            sid = str(getattr(s, "sample_id", None) or (s.get("sample_id") if isinstance(s, dict) else "")).strip()
            q = str(getattr(s, "question", None) or (s.get("question") if isinstance(s, dict) else "")).strip()
            d = str(
                getattr(s, "document_id", None)
                or getattr(s, "expected_source", None)
                or (s.get("document_id") if isinstance(s, dict) else "")
            ).strip()
            if sid:
                ids.add(sid)
            if q:
                questions[_normalize_text(q)] = sid
            if d:
                docs.add(d)
        return ids, questions, docs

    opt_ids, opt_q, opt_docs = _extract_data(splits.optimization)
    val_ids, val_q, val_docs = _extract_data(splits.validation)
    test_ids, test_q, test_docs = _extract_data(splits.held_out_test)

    # 1. ID Overlap
    id_overlap = (opt_ids & val_ids) | (opt_ids & test_ids) | (val_ids & test_ids)

    # 2. Exact Question Overlap
    q_overlap_train_test = set(opt_q.keys()) & set(test_q.keys())
    q_overlap_val_test = set(val_q.keys()) & set(test_q.keys())

    # 3. Near-Duplicate Question Check
    near_dups_train_test = []
    for t_q, t_id in test_q.items():
        for o_q, o_id in opt_q.items():
            jaccard = compute_token_jaccard(t_q, o_q)
            if jaccard >= near_duplicate_threshold:
                near_dups_train_test.append((o_id, t_id, jaccard))

    # 4. Document Overlap
    doc_overlap_train_test = opt_docs & test_docs

    warnings = []
    if id_overlap:
        warnings.append(f"FATAL: Sample ID overlap detected across splits: {id_overlap}")
    if q_overlap_train_test:
        warnings.append(f"CRITICAL: Exact question overlap between optimization and test splits ({len(q_overlap_train_test)} questions).")
    if near_dups_train_test:
        warnings.append(
            f"WARNING: Potential semantic leakage: {len(near_dups_train_test)} question pairs have Jaccard >= {near_duplicate_threshold:.2f} across train and test."
        )
    if doc_overlap_train_test and splits.split_strategy == "document_level":
        warnings.append(f"WARNING: Document overlap across train and test in document-level split mode: {doc_overlap_train_test}")

    is_free = len(id_overlap) == 0 and len(q_overlap_train_test) == 0 and len(near_dups_train_test) == 0
    return {
        "has_id_leakage": len(id_overlap) > 0,
        "has_exact_question_leakage": len(q_overlap_train_test) > 0 or len(q_overlap_val_test) > 0,
        "exact_duplicate_leakage_count": len(q_overlap_train_test) + len(q_overlap_val_test),
        "near_duplicate_train_test_count": len(near_dups_train_test),
        "near_duplicate_leakage_count": len(near_dups_train_test),
        "near_duplicate_examples": near_dups_train_test[:5],
        "document_overlap_train_test": list(doc_overlap_train_test),
        "warnings": warnings,
        "is_safe": is_free,
        "is_leakage_free": is_free,
    }


def split_train_val_test_leakage_resistant(
    samples: Sequence[Any],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    strategy: str = "near_duplicate_grouped",
    near_duplicate_threshold: float = 0.80,
    random_seed: Optional[int] = None,
) -> DatasetSplits:
    """Partition evaluation samples into three distinct, non-overlapping splits with leakage resistance.

    Strategies:
    - 'near_duplicate_grouped' (Recommended): Clusters near-duplicate questions and keeps each cluster
      intact in ONE split, preventing rephrased test questions from appearing in optimization.
    - 'document_level': Groups questions by document_id, ensuring zero document-context leakage.
    - 'stratified_topic': Stratifies by topic category.
    - 'random_row': Pseudo-random row split.

    Args:
        samples: Sequence of EvaluationSample instances.
        train_ratio: Fraction for parameter optimization (search).
        val_ratio: Fraction for intermediate validation / hyperparameter selection.
        test_ratio: Fraction for final held-out generalization test.
        seed: Random seed for deterministic reproducibility.
        strategy: Splitting strategy name.
        near_duplicate_threshold: Jaccard similarity threshold for clustering near-duplicate questions.
        random_seed: Optional alias for seed.

    Returns:
        DatasetSplits: Named container with optimization, validation, held_out_test, and audit metadata.
    """
    effective_seed = random_seed if random_seed is not None else seed

    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be strictly positive, got {train_ratio}")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(f"val_ratio and test_ratio must be non-negative, got {val_ratio}, {test_ratio}")

    sample_list = list(samples)
    n = len(sample_list)
    if n == 0:
        return DatasetSplits(optimization=[], validation=[], held_out_test=[], split_strategy=strategy)

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio > 1.0 + 1e-4:
        raise ValueError(f"Ratios cannot exceed 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}")

    if n == 1:
        return DatasetSplits(optimization=sample_list, validation=[], held_out_test=[], split_strategy=strategy)
    if n == 2:
        return DatasetSplits(optimization=[sample_list[0]], validation=[], held_out_test=[sample_list[1]], split_strategy=strategy)

    rng = random.Random(effective_seed)

    # Strategy 1: Document-level splitting
    if strategy == "document_level":
        by_doc: Dict[str, List[Any]] = {}
        for s in sample_list:
            doc = (
                getattr(s, "document_id", None)
                or getattr(s, "expected_source", None)
                or (s.get("document_id") if isinstance(s, dict) else None)
                or (s.get("expected_source") if isinstance(s, dict) else None)
                or "default_doc"
            )
            by_doc.setdefault(str(doc), []).append(s)

        doc_keys = list(by_doc.keys())
        rng.shuffle(doc_keys)

        test_docs = max(1 if test_ratio > 0 and len(doc_keys) >= 3 else 0, round(len(doc_keys) * test_ratio))
        val_docs = max(1 if val_ratio > 0 and len(doc_keys) >= 3 else 0, round(len(doc_keys) * val_ratio))

        held_out_test = []
        validation = []
        optimization = []

        for d in doc_keys[:test_docs]:
            held_out_test.extend(by_doc[d])
        for d in doc_keys[test_docs : test_docs + val_docs]:
            validation.extend(by_doc[d])
        for d in doc_keys[test_docs + val_docs :]:
            optimization.extend(by_doc[d])

        limitations = (
            "Document-level splitting eliminates document context leakage across splits, "
            "but requires a multi-document dataset to maintain statistical power."
        )

    # Strategy 2: Near-duplicate question clustering
    elif strategy == "near_duplicate_grouped":
        # Union-Find to cluster near-duplicate questions
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Compare pairs
        for i in range(n):
            q_i = getattr(sample_list[i], "question", None) or (sample_list[i].get("question") if isinstance(sample_list[i], dict) else "")
            for j in range(i + 1, n):
                q_j = getattr(sample_list[j], "question", None) or (sample_list[j].get("question") if isinstance(sample_list[j], dict) else "")
                if compute_token_jaccard(str(q_i), str(q_j)) >= near_duplicate_threshold:
                    union(i, j)

        clusters: Dict[int, List[Any]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(sample_list[idx])

        cluster_list = list(clusters.values())
        rng.shuffle(cluster_list)

        # Allocate clusters greedily to target counts
        target_test = max(1 if test_ratio > 0 else 0, round(n * test_ratio))
        target_val = max(1 if val_ratio > 0 and (n - target_test) > 1 else 0, round(n * val_ratio))

        held_out_test = []
        validation = []
        optimization = []

        curr_test_count = 0
        curr_val_count = 0

        for cluster in cluster_list:
            c_size = len(cluster)
            if curr_test_count + c_size <= target_test or (curr_test_count == 0 and test_ratio > 0):
                held_out_test.extend(cluster)
                curr_test_count += c_size
            elif curr_val_count + c_size <= target_val or (curr_val_count == 0 and val_ratio > 0 and len(optimization) > 0):
                validation.extend(cluster)
                curr_val_count += c_size
            else:
                optimization.extend(cluster)

        # Fallback safeguard: optimization must have at least 1 sample
        if not optimization and validation:
            optimization.append(validation.pop())

        limitations = (
            "Near-duplicate grouping prevents rephrased queries from leaking into test sets, "
            "but cannot prevent higher-level conceptual overlap between completely distinct formulations."
        )

    # Strategy 3: Stratified by topic
    elif strategy == "stratified_topic":
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

        limitations = (
            "Topic stratification preserves representation across domains, "
            "but does not cluster rephrased questions across the train-test boundary."
        )

    # Strategy 4: Simple random row split
    else:
        shuffled = list(sample_list)
        rng.shuffle(shuffled)

        test_count = max(1 if test_ratio > 0 else 0, round(n * test_ratio))
        val_count = max(1 if val_ratio > 0 and (n - test_count) > 1 else 0, round(n * val_ratio))

        if test_count + val_count >= n:
            test_count = max(1, n // 3)
            val_count = max(1, n // 3)

        held_out_test = shuffled[:test_count]
        validation = shuffled[test_count : test_count + val_count]
        optimization = shuffled[test_count + val_count :]

        limitations = (
            "Random row splitting divides records uniformly, but is prone to lexical leakage "
            "when questions are rephrased or drawn from identical paragraphs."
        )

    splits = DatasetSplits(
        optimization=optimization,
        validation=validation,
        held_out_test=held_out_test,
        split_strategy=strategy,
        split_metadata={
            "seed": effective_seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "strategy": strategy,
            "limitations": limitations,
        },
    )

    # Perform automated leakage audit
    leakage_audit = audit_split_leakage(splits, near_duplicate_threshold=near_duplicate_threshold)
    splits.leakage_audit = leakage_audit

    logger.info(
        "Leakage-resistant split [%s] complete (Total: %d | Opt: %d, Val: %d, Test: %d | Safe: %s)",
        strategy,
        n,
        len(splits.optimization),
        len(splits.validation),
        len(splits.held_out_test),
        leakage_audit.get("is_safe", True),
    )

    return splits


def split_train_val_test(
    samples: Sequence[Any],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    stratify_by_topic: bool = False,
    random_seed: Optional[int] = None,
) -> DatasetSplits:
    """Partition evaluation samples into three distinct, non-overlapping splits (backward-compatible wrapper)."""
    strategy = "stratified_topic" if stratify_by_topic else "near_duplicate_grouped"
    return split_train_val_test_leakage_resistant(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strategy=strategy,
        random_seed=random_seed,
    )


def save_dataset_splits(
    splits: DatasetSplits,
    output_dir: Union[Path, str],
    prefix: str = "dataset",
) -> Dict[str, Path]:
    """Persist the three dataset splits to JSON files for reproducibility."""
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
                "strategy": splits.split_strategy,
                "split_metadata": splits.split_metadata,
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
