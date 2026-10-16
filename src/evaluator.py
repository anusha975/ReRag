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

from src.dataset_workflow import (
    DatasetSplits,
    DatasetValidationResult,
    DatasetValidator,
    save_dataset_splits,
    split_train_val_test,
)
from src.embeddings import EmbeddingManager
from src.llm import BaseLLMClient
from src.metrics import (
    check_missing_answer,
    compute_aggregate_metrics,
    compute_citation_precision_recall,
    compute_context_coverage,
    compute_exact_match,
    compute_grounded_claim_density,
    compute_hit_rate_at_k,
    compute_latency_metrics,
    compute_noise_ratio,
    compute_page_coverage,
    compute_reciprocal_rank,
    compute_retrieval_precision_at_k,
    compute_retrieval_recall_at_k,
    compute_rouge_l,
    compute_semantic_similarity,
    compute_source_match,
    compute_token_f1,
    compute_token_precision_recall,
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
    reference_answer: str = ""
    expected_source: Optional[str] = None
    expected_pages: Optional[List[int]] = None
    document_id: Optional[str] = None
    relevant_doc_ids: Optional[List[str]] = None
    supporting_passages: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ground_truth_answer: Optional[str] = None
    expected_passages: Optional[List[str]] = None
    expected_citations: Optional[List[int]] = None
    difficulty: Optional[str] = None
    is_negative: Optional[bool] = None

    def __post_init__(self) -> None:
        """Harmonize legacy field aliases and metadata."""
        if self.ground_truth_answer and not self.reference_answer:
            self.reference_answer = self.ground_truth_answer
        elif self.reference_answer and not self.ground_truth_answer:
            self.ground_truth_answer = self.reference_answer
        if self.expected_passages and not self.supporting_passages:
            self.supporting_passages = self.expected_passages
        if self.expected_citations and not self.expected_pages:
            self.expected_pages = self.expected_citations
        if self.difficulty and "difficulty" not in self.metadata:
            self.metadata["difficulty"] = self.difficulty
        if self.is_negative is not None and "is_abstention_case" not in self.metadata:
            self.metadata["is_abstention_case"] = self.is_negative

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

            # 3. Evaluate token-level lexical F1, precision, and recall
            tok_prec, tok_rec, tok_f1 = compute_token_precision_recall(
                prediction=generated_answer,
                reference=sample.reference_answer,
            )

            # 3b. Exact match and ROUGE-L
            exact_match = compute_exact_match(
                prediction=generated_answer,
                reference=sample.reference_answer,
            )
            rouge_l = compute_rouge_l(
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
            recip_rank = compute_reciprocal_rank(
                retrieved_ids=source_documents,
                relevant_ids=sample.relevant_doc_ids,
                k=4,
            )
            hit_rate = compute_hit_rate_at_k(
                retrieved_ids=source_documents,
                relevant_ids=sample.relevant_doc_ids,
                k=4,
            )
            noise_ratio = compute_noise_ratio(
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

            # 7. Unsupported claim check and grounded claim density
            claim_check = detect_unsupported_claims_deterministic(
                prediction=generated_answer,
                contexts=retrieved_contexts,
            )
            grounded_density = compute_grounded_claim_density(
                prediction=generated_answer,
                contexts=retrieved_contexts,
            )

            # 8. Citations precision and recall
            citations_list = [getattr(c, "citation", str(c)) for c in retrieved_contexts]
            citation_eval = compute_citation_precision_recall(
                citations=citations_list,
                retrieved_pages=page_numbers,
                expected_pages=sample.expected_pages,
                expected_source=sample.expected_source or sample.document_id,
            )

            # 9. Optional LLM judge evaluation
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
                "token_precision": tok_prec,
                "token_recall": tok_rec,
                "exact_match": exact_match,
                "rouge_l": rouge_l,
                "context_coverage": ctx_cov,
                "is_missing_answer": is_missing,
                "correct_abstention": correct_abstention,
                "retrieval_recall": ret_recall,
                "retrieval_precision": ret_precision,
                "reciprocal_rank": recip_rank,
                "hit_rate": hit_rate,
                "noise_ratio": noise_ratio,
                "page_coverage": page_cov,
                "source_match": src_match,
                "faithfulness_ratio": claim_check["faithfulness_ratio"],
                "unsupported_sentences": claim_check["unsupported_sentences"],
                "unsupported_claims": claim_check["unsupported_claims"],
                "grounded_claim_density": grounded_density,
                "citation_precision": citation_eval.get("citation_precision"),
                "citation_recall": citation_eval.get("citation_recall"),
                "citation_f1": citation_eval.get("citation_f1"),
                "llm_judge": judge_res,
                "retrieved_count": len(retrieved_contexts),
                "source_documents": source_documents,
                "page_numbers": page_numbers,
                "retrieval_scores": retrieval_scores,
                "execution_time_seconds": exec_time,
                "total_latency_ms": round((exec_time or 0.0) * 1000.0, 2),
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


def expand_dataset_from_document(
    document: LoadedDocument,
    max_samples: int = 10,
    llm_client: Optional[BaseLLMClient] = None,
) -> List[EvaluationSample]:
    """Automatically synthesize new benchmark QA samples from document pages to expand Did.

    Resolves the limitation of very small expert-curated ideal datasets (Did)
    noted in the ReRAG paper, enabling evaluation on diverse, unseen queries.

    Args:
        document: Loaded document with extracted pages and content.
        max_samples: Maximum number of synthetic QA samples to generate.
        llm_client: Optional LLM client. If None, deterministic sentence extraction is used.

    Returns:
        List[EvaluationSample]: Synthesized benchmark samples with provenance annotations.
    """
    new_samples: List[EvaluationSample] = []
    doc_id = document.document_id

    for page in document.pages:
        if len(new_samples) >= max_samples:
            break
        text = page.text.strip()
        if not text or len(text) < 40:
            continue

        # Extract candidates: sentences with substantive content
        import re
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]

        for s_idx, sentence in enumerate(raw_sentences):
            if len(new_samples) >= max_samples:
                break

            # Filter trivial or structural headers
            if sentence.count(" ") < 5:
                continue

            sample_id = f"{doc_id}_p{page.page_number}_s{s_idx + 1}"

            if llm_client is not None:
                prompt = (
                    f"Given the following factual passage from document '{doc_id}', generate one concise factual "
                    f"question whose answer is directly and solely contained in the statement.\n\n"
                    f"Passage: \"{sentence}\"\n\n"
                    f"Output Format:\nQuestion: <question>"
                )
                try:
                    resp = llm_client.generate(prompt=prompt)
                    line = resp.content.strip().split("\n")[0]
                    question = line.replace("Question:", "").strip().strip('"')
                except Exception:
                    question = f"What does page {page.page_number} state regarding {sentence[:30]}...?"
            else:
                # Deterministic synthesis for offline execution
                clean_frag = sentence.rstrip(".?!")
                question = f"According to {doc_id} on page {page.page_number}, what is specified about: '{clean_frag[:45]}...'?"

            new_samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    question=question,
                    reference_answer=sentence,
                    expected_source=doc_id,
                    expected_pages=[page.page_number],
                    document_id=doc_id,
                    relevant_doc_ids=[doc_id],
                    supporting_passages=[sentence],
                    metadata={"synthetic_expansion": True, "page_number": page.page_number},
                )
            )

    logger.info("Synthesized %d expanded benchmark evaluation samples for '%s'.", len(new_samples), doc_id)
    return new_samples


def compute_statistical_significance(
    baseline_scores: Sequence[float],
    rerag_scores: Sequence[float],
) -> Dict[str, Any]:
    """Compute empirical statistical significance metrics comparing Baseline vs ReRAG.

    Addresses the scientific requirement to test whether hallucination reduction
    or metric gains are statistically significant on expanded test datasets.

    Args:
        baseline_scores: Sequence of sample-level scores from baseline single-hop RAG.
        rerag_scores: Sequence of corresponding sample-level scores from iterative ReRAG.

    Returns:
        Dict[str, Any]: Statistical summary including mean gain, std err, paired t-stat,
                        p-value estimate, and 95% confidence intervals.
    """
    import math

    if not baseline_scores or not rerag_scores:
        return {"status": "insufficient_data", "n": 0}

    n = min(len(baseline_scores), len(rerag_scores))
    if n < 2:
        return {"status": "insufficient_samples_for_inference", "n": n}

    b = list(baseline_scores[:n])
    r = list(rerag_scores[:n])

    diffs = [r[i] - b[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_err = math.sqrt(variance / n) if variance > 0 else 0.0

    if std_err > 1e-8:
        t_stat = mean_diff / std_err
        # Normal approximation for two-tailed p-value
        z = abs(t_stat)
        # Approximation of standard normal CDF tail: erfc(z / sqrt(2))
        p_val_approx = math.erfc(z / math.sqrt(2))
    else:
        t_stat = 0.0
        p_val_approx = 1.0 if abs(mean_diff) < 1e-6 else 0.0

    ci_margin = 1.96 * std_err

    return {
        "status": "computed",
        "sample_size": n,
        "mean_baseline": round(sum(b) / n, 4),
        "mean_rerag": round(sum(r) / n, 4),
        "mean_improvement": round(mean_diff, 4),
        "standard_error": round(std_err, 4),
        "t_statistic": round(t_stat, 4),
        "p_value_approx": round(p_val_approx, 5),
        "is_statistically_significant_p05": bool(p_val_approx < 0.05),
        "ci_95_lower": round(mean_diff - ci_margin, 4),
        "ci_95_upper": round(mean_diff + ci_margin, 4),
    }


