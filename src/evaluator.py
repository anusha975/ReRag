"""Evaluation module for ReRAG benchmarking.

Executes quantitative evaluations comparing RAG pipeline outputs against
human-written reference answers, measuring semantic similarity, context coverage,
retrieval relevance, and supported claims across benchmark samples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.embeddings import EmbeddingManager
from src.llm import BaseLLMClient
from src.metrics import (
    check_missing_answer,
    compute_aggregate_metrics,
    compute_context_coverage,
    compute_page_coverage,
    compute_retrieval_precision_at_k,
    compute_retrieval_recall_at_k,
    compute_semantic_similarity,
    compute_source_match,
    compute_token_f1,
    detect_unsupported_claims_deterministic,
    llm_judge_faithfulness,
)
from src.rag_pipeline import PipelineResult, RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single benchmark question-answer test sample with expected provenance."""

    sample_id: str
    question: str
    reference_answer: str
    expected_source: Optional[str] = None
    expected_pages: Optional[List[int]] = None
    document_id: Optional[str] = None
    relevant_doc_ids: Optional[List[str]] = None
    supporting_passages: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ground_truth_answer(self) -> str:
        """Alias for reference_answer for backwards compatibility."""
        return self.reference_answer

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationSample:
        """Construct EvaluationSample from raw dictionary."""
        # Handle field aliases
        ref = data.get("reference_answer") or data.get("ground_truth_answer") or ""
        doc_id = data.get("document_id") or data.get("expected_source")
        rel_ids = data.get("relevant_doc_ids")
        if rel_ids is None and doc_id:
            rel_ids = [doc_id]

        pages = data.get("expected_pages")
        if isinstance(pages, int):
            pages = [pages]

        return cls(
            sample_id=str(data.get("sample_id") or data.get("id") or "sample"),
            question=str(data.get("question", "")).strip(),
            reference_answer=str(ref).strip(),
            expected_source=str(data["expected_source"]) if data.get("expected_source") else None,
            expected_pages=pages,
            document_id=str(doc_id) if doc_id else None,
            relevant_doc_ids=rel_ids,
            supporting_passages=data.get("supporting_passages"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "expected_source": self.expected_source,
            "expected_pages": self.expected_pages,
            "document_id": self.document_id,
            "relevant_doc_ids": self.relevant_doc_ids,
            "supporting_passages": self.supporting_passages,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationReport:
    """Summary and granular breakdown of a benchmark evaluation run."""

    total_samples: int
    aggregate_metrics: Dict[str, Any]
    detailed_results: List[Dict[str, Any]]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    rerag_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_metrics(self) -> Dict[str, Any]:
        """Alias for aggregate_metrics for backwards compatibility."""
        return self.aggregate_metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "total_samples": self.total_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "detailed_results": self.detailed_results,
            "config_snapshot": self.config_snapshot,
            "rerag_metrics": self.rerag_metrics,
        }

    def save_to_file(self, file_path: Union[Path, str]) -> None:
        """Persist evaluation report as JSON to disk."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved evaluation report to %s", target_path)


def load_dataset(dataset_path: Union[Path, str]) -> List[EvaluationSample]:
    """Load evaluation samples from a JSON or JSONL file.

    Supports:
    1. JSON file containing a list of sample objects: `[{...}, {...}]`
    2. JSON file containing an object with a 'samples' key: `{"samples": [...]}`
    3. JSONL file containing one JSON sample per line.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        List[EvaluationSample]: Parsed evaluation samples.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    samples: List[EvaluationSample] = []
    text = path.read_text(encoding="utf-8").strip()

    # Attempt parsing as JSON first
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        samples.append(EvaluationSample.from_dict(item))
                return samples
            elif isinstance(data, dict):
                sample_list = data.get("samples") or data.get("data") or []
                for item in sample_list:
                    if isinstance(item, dict):
                        samples.append(EvaluationSample.from_dict(item))
                return samples
        except json.JSONDecodeError:
            pass

    # Parse as JSONL (one record per line)
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue
        try:
            record = json.loads(line_clean)
            if isinstance(record, dict):
                samples.append(EvaluationSample.from_dict(record))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)

    logger.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def split_train_test(
    samples: Sequence[EvaluationSample],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[EvaluationSample], List[EvaluationSample]]:
    """Partition evaluation samples into training and held-out test sets.

    Ensures that the optimizer never evaluates against or overfits to the held-out test set.

    Args:
        samples: Sequence of all evaluation samples.
        test_ratio: Fraction of samples to reserve for held-out testing (e.g. 0.2).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple[List[EvaluationSample], List[EvaluationSample]]: (train_samples, held_out_test_samples).
    """
    if not samples:
        return [], []

    import random

    sample_list = list(samples)
    rng = random.Random(seed)
    rng.shuffle(sample_list)

    test_count = int(len(sample_list) * test_ratio)
    test_count = max(1 if test_ratio > 0 and len(sample_list) > 1 else 0, test_count)

    test_set = sample_list[:test_count]
    train_set = sample_list[test_count:]

    logger.info("Split dataset: %d train samples, %d held-out test samples.", len(train_set), len(test_set))
    return train_set, test_set


class Evaluator:
    """Evaluates RAG systems against benchmark test cases."""

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_judge: Optional[BaseLLMClient] = None,
    ) -> None:
        """Initialize evaluator with target RAG pipeline.

        Args:
            pipeline: Optional RAG pipeline instance.
            embedding_manager: Optional shared embedding manager for semantic similarity.
            llm_judge: Optional separate LLM client for verification judgements.
        """
        self.pipeline = pipeline
        self.embedding_manager = embedding_manager
        self.llm_judge = llm_judge

    def load_dataset(self, dataset_path: Union[Path, str]) -> List[EvaluationSample]:
        """Load benchmark dataset from JSON or JSONL file."""
        return load_dataset(dataset_path)

    def evaluate(
        self,
        samples: Sequence[EvaluationSample],
        deterministic: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> EvaluationReport:
        """Run quantitative evaluation across benchmark samples.

        For every question:
        1. Runs the complete RAG pipeline.
        2. Evaluates semantic similarity, token F1, context coverage, retrieval recall/precision.
        3. Identifies missing answers and deterministic unsupported claims.
        4. Calculates dataset-wide aggregate metrics.

        Args:
            samples: Benchmark evaluation samples.
            deterministic: Ensure zero temperature and static seeds.
            config_override: Optional configuration dictionary snapshot.
            collection_name: Optional collection identifier to evaluate against.

        Returns:
            EvaluationReport: Computed evaluation metrics and comparisons.
        """
        if not samples:
            return EvaluationReport(
                total_samples=0,
                aggregate_metrics=compute_aggregate_metrics([]),
                detailed_results=[],
                config_snapshot=config_override or {},
            )

        detailed_results: List[Dict[str, Any]] = []

        for sample in samples:
            logger.debug("Evaluating sample '%s': %s", sample.sample_id, sample.question)

            # 1. Run pipeline if available, otherwise record empty prediction
            if self.pipeline is not None:
                pipeline_result: PipelineResult = self.pipeline.run_baseline(
                    question=sample.question,
                    collection_name=sample.document_id or collection_name,
                )
                generated_answer = pipeline_result.final_answer
                retrieved_contexts = pipeline_result.contexts
                source_documents = pipeline_result.source_documents
                page_numbers = pipeline_result.page_numbers
                retrieval_scores = pipeline_result.retrieval_scores
                exec_time = pipeline_result.execution_time_seconds
            else:
                generated_answer = ""
                retrieved_contexts = []
                source_documents = []
                page_numbers = []
                retrieval_scores = []
                exec_time = 0.0

            # 2. Evaluate semantic similarity against reference answer
            sem_sim = compute_semantic_similarity(
                prediction=generated_answer,
                reference=sample.reference_answer,
                embedding_manager=self.embedding_manager,
            )

            # 3. Evaluate token-level lexical F1
            tok_f1 = compute_token_f1(
                prediction=generated_answer,
                reference=sample.reference_answer,
            )

            # 4. Context coverage (proportion of answer tokens found in retrieved context)
            ctx_cov = compute_context_coverage(
                prediction=generated_answer,
                contexts=retrieved_contexts,
            )

            # 5. Missing answer & safe abstention check
            is_missing = check_missing_answer(generated_answer)
            ref_is_missing = check_missing_answer(sample.reference_answer)
            # Correct abstention: when reference states info is insufficient and model agrees
            correct_abstention = is_missing and ref_is_missing

            # 6. Retrieval relevance against expected ground-truth sources and pages
            ret_recall = compute_retrieval_recall_at_k(
                retrieved_ids=source_documents,
                relevant_ids=sample.relevant_doc_ids,
                k=4,
            )
            ret_precision = compute_retrieval_precision_at_k(
                retrieved_ids=source_documents,
                relevant_ids=sample.relevant_doc_ids,
                k=4,
            )
            page_cov = compute_page_coverage(
                retrieved_pages=page_numbers,
                expected_pages=sample.expected_pages,
            )
            src_match = compute_source_match(
                retrieved_sources=source_documents,
                expected_source=sample.expected_source or sample.document_id,
            )

            # 7. Unsupported claim check (deterministic)
            claim_check = detect_unsupported_claims_deterministic(
                prediction=generated_answer,
                contexts=retrieved_contexts,
            )

            # 8. Optional LLM judge evaluation
            judge_res = None
            if self.llm_judge is not None:
                judge_res = llm_judge_faithfulness(
                    prediction=generated_answer,
                    contexts=retrieved_contexts,
                    llm_client=self.llm_judge,
                )

            # Compile per-sample result
            detailed_results.append({
                "sample_id": sample.sample_id,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "generated_answer": generated_answer,
                "semantic_similarity": sem_sim,
                "token_f1": tok_f1,
                "context_coverage": ctx_cov,
                "is_missing_answer": is_missing,
                "correct_abstention": correct_abstention,
                "retrieval_recall": ret_recall,
                "retrieval_precision": ret_precision,
                "page_coverage": page_cov,
                "source_match": src_match,
                "faithfulness_ratio": claim_check["faithfulness_ratio"],
                "unsupported_sentences": claim_check["unsupported_sentences"],
                "unsupported_claims": claim_check["unsupported_claims"],
                "llm_judge": judge_res,
                "retrieved_count": len(retrieved_contexts),
                "source_documents": source_documents,
                "page_numbers": page_numbers,
                "retrieval_scores": retrieval_scores,
                "execution_time_seconds": exec_time,
            })

        # Calculate dataset-wide aggregate metrics
        aggregates = compute_aggregate_metrics(detailed_results)

        config_snapshot = dict(config_override or {})
        config_snapshot["deterministic"] = deterministic

        return EvaluationReport(
            total_samples=len(samples),
            aggregate_metrics=aggregates,
            detailed_results=detailed_results,
            config_snapshot=config_snapshot,
        )

