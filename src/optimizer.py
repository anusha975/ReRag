"""ReRAG adaptive chunk-size and chunk-overlap optimization module.

Implements greedy adaptive coordinate search to systematically explore and select
optimal chunking configurations (chunk_size and chunk_overlap) based on empirical
multi-metric objective scores on benchmark evaluation datasets.

Research Integrity Principles:
1. Grounding-First Objective: Default objective combines faithfulness, context coverage,
   and semantic similarity. Semantic similarity alone is NOT used as sole objective,
   preventing fluent hallucinations from being accepted as improvements.
2. Transparent Accounting: All candidate configurations (accepted, rejected, invalid, failed)
   are logged with explicit decision rationale and execution timing.
3. Separation of Concerns: Optimization candidates are evaluated solely on the training/tuning
   question set. Held-out test evaluation occurs only post-selection to evaluate true generalization.
4. No Fabricated Gains: A candidate is accepted if and only if its objective score exceeds
   the current best by at least the configured improvement tolerance.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from src.chunker import TextChunker
from src.document_loader import LoadedDocument
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkConfig:
    """Explicit chunking configuration parameters."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        """Validate parameter ranges and relational constraints."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def to_dict(self) -> Dict[str, int]:
        """Convert configuration to dictionary."""
        return {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    def is_valid_for_bounds(self, bounds: SearchBounds) -> Tuple[bool, str]:
        """Check whether configuration satisfies all search boundaries.

        Args:
            bounds: Search boundary constraints.

        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        if self.chunk_size < bounds.min_chunk_size:
            return False, f"chunk_size {self.chunk_size} < min_chunk_size {bounds.min_chunk_size}"
        if self.chunk_size > bounds.max_chunk_size:
            return False, f"chunk_size {self.chunk_size} > max_chunk_size {bounds.max_chunk_size}"
        if self.chunk_overlap < bounds.min_chunk_overlap:
            return False, f"chunk_overlap {self.chunk_overlap} < min_chunk_overlap {bounds.min_chunk_overlap}"
        if self.chunk_overlap > bounds.max_chunk_overlap:
            return False, f"chunk_overlap {self.chunk_overlap} > max_chunk_overlap {bounds.max_chunk_overlap}"
        if self.chunk_overlap >= self.chunk_size:
            return False, f"chunk_overlap {self.chunk_overlap} >= chunk_size {self.chunk_size}"
        return True, "Valid configuration"


class CandidateSpec(NamedTuple):
    """Specification of an adjacent candidate generated during search."""

    config: Optional[ChunkConfig]
    description: str
    invalid_reason: str
    attempted_coords: Tuple[int, int]


@dataclass(frozen=True)
class SearchBounds:
    """Boundary constraints for the chunking parameter search space."""

    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    min_chunk_overlap: int = 0
    max_chunk_overlap: int = 500
    min_step_size: int = 25
    min_overlap_step: int = 10

    def __post_init__(self) -> None:
        """Validate search boundary invariants and prevent negative or inverted limits."""
        if self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError(
                f"max_chunk_size ({self.max_chunk_size}) cannot be less than min_chunk_size ({self.min_chunk_size})"
            )
        if self.min_chunk_overlap < 0:
            raise ValueError(f"min_chunk_overlap cannot be negative, got {self.min_chunk_overlap}")
        if self.max_chunk_overlap < self.min_chunk_overlap:
            raise ValueError(
                f"max_chunk_overlap ({self.max_chunk_overlap}) cannot be less than min_chunk_overlap ({self.min_chunk_overlap})"
            )
        if self.min_step_size <= 0:
            raise ValueError(f"min_step_size must be positive, got {self.min_step_size}")
        if self.min_overlap_step <= 0:
            raise ValueError(f"min_overlap_step must be positive, got {self.min_overlap_step}")


@dataclass
class OptimizationObjective:
    """Configurable multi-metric objective for evaluating RAG candidates.

    The default objective is a composite score designed to balance semantic
    alignment with factual grounding and context retrieval support:
      - faithfulness_ratio (weight 0.40): Sentence-level support by context chunks
      - context_coverage (weight 0.30): Lexical grounding in retrieved context
      - semantic_similarity (weight 0.30): Semantic alignment with human reference

    Research Integrity Note:
      Optimizing semantic similarity alone produces fluent but potentially ungrounded
      answers. Incorporating faithfulness and context coverage penalizes hallucinated claims.
    """

    target_metric: str = "composite"
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "faithfulness_ratio": 0.40,
            "context_coverage": 0.30,
            "semantic_similarity": 0.30,
        }
    )
    minimize_metrics: List[str] = field(default_factory=lambda: ["hallucination_rate"])
    penalty_weight: float = 0.20

    def calculate_score(self, aggregate_metrics: Dict[str, Any]) -> float:
        """Calculate single scalar objective score from aggregate metrics.

        Args:
            aggregate_metrics: Dictionary of aggregate evaluation metrics.

        Returns:
            float: Scalar objective score in [0.0, 1.0].
        """
        if not aggregate_metrics:
            return 0.0

        if self.target_metric != "composite":
            val = aggregate_metrics.get(self.target_metric)
            if val is None:
                val = aggregate_metrics.get(f"mean_{self.target_metric}")
            if val is not None:
                return float(val)
            logger.warning("Target metric '%s' not found in aggregate metrics. Returning 0.0.", self.target_metric)
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name, weight in self.weights.items():
            metric_val = aggregate_metrics.get(metric_name)
            if metric_val is None:
                metric_val = aggregate_metrics.get(f"mean_{metric_name}")
            if metric_val is None and metric_name.startswith("mean_"):
                metric_val = aggregate_metrics.get(metric_name[5:])

            if metric_val is not None:
                try:
                    val_float = float(metric_val)
                    weighted_sum += weight * val_float
                    total_weight += weight
                except (ValueError, TypeError):
                    pass

        base_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

        # Apply penalty for minimize metrics (e.g., hallucination rate)
        penalty = 0.0
        for min_metric in self.minimize_metrics:
            pen_val = aggregate_metrics.get(min_metric)
            if pen_val is None:
                pen_val = aggregate_metrics.get(f"mean_{min_metric}")
            if pen_val is not None:
                try:
                    penalty += self.penalty_weight * float(pen_val)
                except (ValueError, TypeError):
                    pass

        score = max(0.0, min(1.0, base_score - penalty))
        return round(score, 4)


@dataclass
class AdaptiveSearchConfig:
    """Hyperparameters governing greedy adaptive coordinate search."""

    initial_chunk_size: int = 500
    initial_chunk_overlap: int = 50
    initial_step_size: int = 100
    initial_overlap_step: int = 25
    step_decay: float = 0.5
    improvement_tolerance: float = 0.005
    max_iterations: int = 10
    bounds: SearchBounds = field(default_factory=SearchBounds)
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
    exploration_directions: str = "coordinate"  # 'coordinate' (4 dirs) or 'full' (8 dirs)
    random_seed: int = 42


@dataclass
class CandidateTrial:
    """Record of an individual candidate evaluation trial."""

    trial_id: int
    iteration: int
    config: ChunkConfig
    step_size: int
    metrics: Dict[str, Any]
    objective_score: Optional[float]
    status: str  # 'accepted', 'rejected', 'failed', 'rejected_invalid', 'skipped_duplicate', 'baseline'
    decision_reason: str
    execution_time_seconds: float
    attempted_params: Optional[Dict[str, int]] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trial record to dictionary."""
        return {
            "trial_id": self.trial_id,
            "iteration": self.iteration,
            "config": self.config.to_dict(),
            "attempted_params": self.attempted_params or self.config.to_dict(),
            "step_size": self.step_size,
            "metrics": self.metrics,
            "objective_score": self.objective_score,
            "status": self.status,
            "decision_reason": self.decision_reason,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationResult:
    """Complete summary of an adaptive optimization experiment."""

    baseline_config: ChunkConfig
    baseline_metrics: Dict[str, Any]
    baseline_score: float
    selected_config: ChunkConfig
    selected_metrics: Dict[str, Any]
    selected_score: float
    improvement: float
    candidate_history: List[CandidateTrial]
    stopping_reason: str
    total_iterations: int
    total_trials: int
    accepted_trials: int
    total_execution_time_seconds: float
    optimization_dataset_id: str
    held_out_evaluation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment result to structured dictionary."""
        return {
            "baseline_config": self.baseline_config.to_dict(),
            "baseline_metrics": self.baseline_metrics,
            "baseline_score": self.baseline_score,
            "selected_config": self.selected_config.to_dict(),
            "selected_metrics": self.selected_metrics,
            "selected_score": self.selected_score,
            "improvement": round(self.improvement, 4),
            "stopping_reason": self.stopping_reason,
            "total_iterations": self.total_iterations,
            "total_trials": self.total_trials,
            "accepted_trials": self.accepted_trials,
            "total_execution_time_seconds": round(self.total_execution_time_seconds, 3),
            "optimization_dataset_id": self.optimization_dataset_id,
            "held_out_evaluation": self.held_out_evaluation,
            "metadata": self.metadata,
            "candidate_history": [t.to_dict() for t in self.candidate_history],
        }

    def save_to_json(self, file_path: Union[Path, str]) -> None:
        """Save complete result structure to JSON file."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved optimization result JSON to %s", target_path)

    def save_to_csv(self, file_path: Union[Path, str]) -> None:
        """Save candidate trial history table to CSV file."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_id",
                "iteration",
                "chunk_size",
                "chunk_overlap",
                "step_size",
                "status",
                "objective_score",
                "semantic_similarity",
                "faithfulness_ratio",
                "context_coverage",
                "token_f1",
                "decision_reason",
                "execution_time_seconds",
                "timestamp",
            ])
            for t in self.candidate_history:
                sz = t.attempted_params.get("chunk_size", t.config.chunk_size) if t.attempted_params else t.config.chunk_size
                ov = t.attempted_params.get("chunk_overlap", t.config.chunk_overlap) if t.attempted_params else t.config.chunk_overlap
                writer.writerow([
                    t.trial_id,
                    t.iteration,
                    sz,
                    ov,
                    t.step_size,
                    t.status,
                    t.objective_score if t.objective_score is not None else "",
                    t.metrics.get("semantic_similarity", ""),
                    t.metrics.get("faithfulness_ratio", ""),
                    t.metrics.get("context_coverage", ""),
                    t.metrics.get("token_f1", ""),
                    t.decision_reason,
                    round(t.execution_time_seconds, 3),
                    t.timestamp,
                ])
        logger.info("Saved optimization trial CSV to %s", target_path)

    @classmethod
    def load_from_json(cls, file_path: Union[Path, str]) -> OptimizationResult:
        """Load experiment result from saved JSON file."""
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Optimization result file not found: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history: List[CandidateTrial] = []
        for t_dict in data.get("candidate_history", []):
            cfg_dict = t_dict["config"]
            history.append(
                CandidateTrial(
                    trial_id=t_dict["trial_id"],
                    iteration=t_dict["iteration"],
                    config=ChunkConfig(cfg_dict["chunk_size"], cfg_dict["chunk_overlap"]),
                    step_size=t_dict["step_size"],
                    metrics=t_dict["metrics"],
                    objective_score=t_dict["objective_score"],
                    status=t_dict["status"],
                    decision_reason=t_dict["decision_reason"],
                    execution_time_seconds=t_dict["execution_time_seconds"],
                    attempted_params=t_dict.get("attempted_params"),
                    timestamp=t_dict.get("timestamp", ""),
                )
            )

        base_cfg = data["baseline_config"]
        sel_cfg = data["selected_config"]
        return cls(
            baseline_config=ChunkConfig(base_cfg["chunk_size"], base_cfg["chunk_overlap"]),
            baseline_metrics=data["baseline_metrics"],
            baseline_score=data["baseline_score"],
            selected_config=ChunkConfig(sel_cfg["chunk_size"], sel_cfg["chunk_overlap"]),
            selected_metrics=data["selected_metrics"],
            selected_score=data["selected_score"],
            improvement=data["improvement"],
            candidate_history=history,
            stopping_reason=data["stopping_reason"],
            total_iterations=data["total_iterations"],
            total_trials=data["total_trials"],
            accepted_trials=data["accepted_trials"],
            total_execution_time_seconds=data["total_execution_time_seconds"],
            optimization_dataset_id=data.get("optimization_dataset_id", "unknown"),
            held_out_evaluation=data.get("held_out_evaluation"),
            metadata=data.get("metadata", {}),
        )


class AdaptiveChunkOptimizer:
    """Executes greedy adaptive coordinate search to optimize chunking configurations."""

    def __init__(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        search_config: Optional[AdaptiveSearchConfig] = None,
        evaluate_fn: Optional[Callable[[ChunkConfig], Union[EvaluationReport, Dict[str, Any], float]]] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        optimization_samples: Optional[Sequence[EvaluationSample]] = None,
        collection_name: Optional[str] = None,
        dataset_id: str = "synthetic_demonstration_dataset",
        evaluation_cache: Optional[Dict[Tuple[int, int], Tuple[Dict[str, Any], float]]] = None,
    ) -> None:
        """Initialize optimizer.

        Can be initialized either with:
        1. An explicit `evaluate_fn` callable (mock evaluations, unit testing)
        2. Real pipeline components (`documents`, `pipeline`, `evaluator`, `optimization_samples`).

        Args:
            baseline_config: Initial chunking configuration.
            search_config: Adaptive search settings and boundaries.
            evaluate_fn: Optional custom evaluation callable.
            documents: Sequence of raw loaded documents to chunk and re-index.
            pipeline: Active RAG pipeline.
            evaluator: Evaluator instance.
            optimization_samples: Training evaluation question set.
            collection_name: Optional collection name for ChromaDB.
            dataset_id: Dataset identifier for audit tracing.
            evaluation_cache: Optional shared cache mapping (chunk_size, chunk_overlap) to (metrics, score).
        """
        self.search_config = search_config or AdaptiveSearchConfig()
        self.baseline_config = baseline_config or ChunkConfig(
            self.search_config.initial_chunk_size,
            self.search_config.initial_chunk_overlap,
        )
        self.dataset_id = dataset_id
        self.collection_name = collection_name or "rerag_opt_collection"
        self.evaluation_cache = evaluation_cache

        # Evaluation mechanism
        if evaluate_fn is not None:
            self._evaluate_fn = evaluate_fn
        elif documents is not None and pipeline is not None and evaluator is not None and optimization_samples is not None:
            self._evaluate_fn = self._create_pipeline_evaluator(
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                samples=optimization_samples,
                collection_name=self.collection_name,
            )
        else:
            raise ValueError(
                "AdaptiveChunkOptimizer requires either `evaluate_fn` or "
                "(`documents`, `pipeline`, `evaluator`, `optimization_samples`)."
            )

        # Seed pseudo-random generator
        random.seed(self.search_config.random_seed)

    def _create_pipeline_evaluator(
        self,
        documents: Sequence[LoadedDocument],
        pipeline: RAGPipeline,
        evaluator: Evaluator,
        samples: Sequence[EvaluationSample],
        collection_name: str,
    ) -> Callable[[ChunkConfig], EvaluationReport]:
        """Create evaluation function that rebuilds index and evaluates pipeline."""
        def _eval_config(config: ChunkConfig) -> EvaluationReport:
            # 1. Instantiate chunker with candidate parameters
            chunker = TextChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            # 2. Rebuild index from original document texts
            pipeline.index_documents(
                documents=documents,
                chunker=chunker,
                collection_name=collection_name,
            )
            # 3. Run evaluation across optimization questions
            report = evaluator.evaluate(
                samples=samples,
                deterministic=True,
                config_override=config.to_dict(),
                collection_name=collection_name,
            )
            return report

        return _eval_config

    def _run_single_evaluation(
        self,
        config: ChunkConfig,
    ) -> Tuple[Dict[str, Any], Optional[float], Optional[str]]:
        """Safely execute evaluation callable for a single configuration.

        Returns:
            Tuple[Dict[str, Any], Optional[float], Optional[str]]:
                (metrics_dict, objective_score, error_message_if_any)
        """
        cache_key = (config.chunk_size, config.chunk_overlap)
        if self.evaluation_cache is not None and cache_key in self.evaluation_cache:
            cached_metrics, cached_score = self.evaluation_cache[cache_key]
            return cached_metrics, cached_score, None

        try:
            eval_output = self._evaluate_fn(config)
            if isinstance(eval_output, EvaluationReport):
                metrics = eval_output.aggregate_metrics
                score = self.search_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, dict):
                metrics = eval_output
                score = self.search_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, (int, float)):
                score = round(float(eval_output), 4)
                metrics = {"objective_score": score}
            else:
                raise TypeError(f"Unsupported evaluation output type: {type(eval_output)}")

            if self.evaluation_cache is not None:
                self.evaluation_cache[cache_key] = (metrics, score)

            return metrics, score, None
        except Exception as e:
            logger.error("Evaluation failed for config %s: %s", config, e, exc_info=True)
            return {}, None, str(e)

    def generate_candidates(
        self,
        current_config: ChunkConfig,
        step_size: int,
        overlap_step: int,
    ) -> List[CandidateSpec]:
        """Generate candidate configurations by stepping chunk size and overlap in both directions.

        Returns:
            List[CandidateSpec]:
                List of CandidateSpec(config, description, invalid_reason, attempted_coords).
                Supports both tuple unpacking and attribute access.
        """
        bounds = self.search_config.bounds
        candidates_raw: List[Tuple[int, int, str]] = []

        # Coordinate exploration (4 directions)
        candidates_raw.append((current_config.chunk_size + step_size, current_config.chunk_overlap, "+size"))
        candidates_raw.append((current_config.chunk_size - step_size, current_config.chunk_overlap, "-size"))
        candidates_raw.append((current_config.chunk_size, current_config.chunk_overlap + overlap_step, "+overlap"))
        candidates_raw.append((current_config.chunk_size, current_config.chunk_overlap - overlap_step, "-overlap"))

        # Optional full diagonal exploration (additional 4 diagonal directions)
        if self.search_config.exploration_directions == "full":
            candidates_raw.append((current_config.chunk_size + step_size, current_config.chunk_overlap + overlap_step, "+size/+overlap"))
            candidates_raw.append((current_config.chunk_size + step_size, current_config.chunk_overlap - overlap_step, "+size/-overlap"))
            candidates_raw.append((current_config.chunk_size - step_size, current_config.chunk_overlap + overlap_step, "-size/+overlap"))
            candidates_raw.append((current_config.chunk_size - step_size, current_config.chunk_overlap - overlap_step, "-size/-overlap"))

        validated_candidates: List[CandidateSpec] = []

        for sz, ov, desc in candidates_raw:
            coords = (sz, ov)
            # Check for basic non-positives or overlap >= size
            if sz <= 0:
                validated_candidates.append(CandidateSpec(None, desc, f"Invalid chunk_size {sz} <= 0", coords))
                continue
            if ov < 0:
                validated_candidates.append(CandidateSpec(None, desc, f"Invalid chunk_overlap {ov} < 0", coords))
                continue
            if ov >= sz:
                validated_candidates.append(CandidateSpec(None, desc, f"Invalid: overlap {ov} >= chunk_size {sz}", coords))
                continue

            cfg = ChunkConfig(chunk_size=sz, chunk_overlap=ov)
            is_valid, reason = cfg.is_valid_for_bounds(bounds)
            if not is_valid:
                validated_candidates.append(CandidateSpec(None, desc, f"Bounds violation: {reason}", coords))
            else:
                validated_candidates.append(CandidateSpec(cfg, desc, "", coords))

        return validated_candidates

    def optimize(
        self,
        progress_callback: Optional[Callable[[int, int, CandidateTrial], None]] = None,
    ) -> OptimizationResult:
        """Run greedy adaptive coordinate search to find optimal chunking parameters.

        Workflow:
        1. Evaluate baseline configuration and record metrics.
        2. In each iteration, generate candidate configurations by adjusting chunk size and overlap.
        3. Reject invalid candidates or boundary violations.
        4. Rebuild index and evaluate each valid candidate on optimization dataset.
        5. Select the best candidate that improves over the current best by at least `improvement_tolerance`.
        6. If no candidate improves, reduce step size and search more finely.
        7. Stop when minimum step size is reached, max iterations reached, or search space exhausted.

        Args:
            progress_callback: Optional callback receiving (iteration, max_iterations, CandidateTrial).

        Returns:
            OptimizationResult: Detailed experimental record with history and decisions.
        """
        start_time = time.perf_counter()
        bounds = self.search_config.bounds
        tolerance = self.search_config.improvement_tolerance
        max_iters = self.search_config.max_iterations

        candidate_history: List[CandidateTrial] = []
        visited_configs: Dict[Tuple[int, int], CandidateTrial] = {}
        trial_id = 1
        accepted_count = 0

        # 1. Evaluate baseline configuration
        logger.info("Evaluating baseline configuration: %s", self.baseline_config)
        base_t0 = time.perf_counter()
        base_metrics, base_score, base_err = self._run_single_evaluation(self.baseline_config)
        base_elapsed = time.perf_counter() - base_t0

        if base_err or base_score is None:
            baseline_status = "failed"
            baseline_score = 0.0
            baseline_reason = f"Baseline evaluation failed: {base_err}"
        else:
            baseline_status = "baseline"
            baseline_score = base_score
            baseline_reason = f"Baseline evaluation established (score: {baseline_score:.4f})"

        baseline_trial = CandidateTrial(
            trial_id=trial_id,
            iteration=0,
            config=self.baseline_config,
            step_size=self.search_config.initial_step_size,
            metrics=base_metrics,
            objective_score=baseline_score,
            status=baseline_status,
            decision_reason=baseline_reason,
            execution_time_seconds=base_elapsed,
            attempted_params={"chunk_size": self.baseline_config.chunk_size, "chunk_overlap": self.baseline_config.chunk_overlap},
        )
        def record_trial(trial: CandidateTrial, iter_num: int) -> None:
            candidate_history.append(trial)
            if progress_callback is not None:
                try:
                    progress_callback(iter_num, max_iters, trial)
                except Exception as cb_err:
                    logger.debug("Progress callback error: %s", cb_err)

        record_trial(baseline_trial, 0)
        visited_configs[(self.baseline_config.chunk_size, self.baseline_config.chunk_overlap)] = baseline_trial

        current_best_config = self.baseline_config
        current_best_score = baseline_score
        current_best_metrics = base_metrics

        current_step_size = self.search_config.initial_step_size
        current_overlap_step = self.search_config.initial_overlap_step
        stopping_reason = "Completed all iterations"
        total_iterations_run = 0

        # 2. Main adaptive search loop
        for iteration in range(1, max_iters + 1):
            total_iterations_run = iteration
            logger.info(
                "--- Optimization Iteration %d (step_size=%d, overlap_step=%d, current_best=%.4f @ %s) ---",
                iteration,
                current_step_size,
                current_overlap_step,
                current_best_score,
                current_best_config,
            )

            # Generate candidates
            raw_candidates = self.generate_candidates(
                current_config=current_best_config,
                step_size=current_step_size,
                overlap_step=current_overlap_step,
            )

            evaluated_in_iteration: List[Tuple[ChunkConfig, float, Dict[str, Any], float, int]] = []
            valid_candidates_count = 0

            for cand_spec in raw_candidates:
                cfg_opt = cand_spec.config
                desc = cand_spec.description
                invalid_reason = cand_spec.invalid_reason
                coords = cand_spec.attempted_coords

                trial_id += 1

                # Handle invalid candidate
                if cfg_opt is None:
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=current_best_config,  # Fallback anchor
                        step_size=current_step_size,
                        metrics={},
                        objective_score=None,
                        status="rejected_invalid",
                        decision_reason=f"Candidate [{desc}] rejected: {invalid_reason}",
                        execution_time_seconds=0.0,
                        attempted_params={"chunk_size": coords[0], "chunk_overlap": coords[1]},
                    )
                    record_trial(rec_trial, iteration)
                    continue

                # Check duplicate
                cfg_key = (cfg_opt.chunk_size, cfg_opt.chunk_overlap)
                if cfg_key in visited_configs:
                    prev_trial = visited_configs[cfg_key]
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=cfg_opt,
                        step_size=current_step_size,
                        metrics=prev_trial.metrics,
                        objective_score=prev_trial.objective_score,
                        status="skipped_duplicate",
                        decision_reason=f"Candidate [{desc}] already evaluated in trial #{prev_trial.trial_id}",
                        execution_time_seconds=0.0,
                        attempted_params={"chunk_size": cfg_opt.chunk_size, "chunk_overlap": cfg_opt.chunk_overlap},
                    )
                    record_trial(rec_trial, iteration)
                    continue

                valid_candidates_count += 1
                # Evaluate valid candidate
                t0 = time.perf_counter()
                cand_metrics, cand_score, cand_err = self._run_single_evaluation(cfg_opt)
                elapsed = time.perf_counter() - t0

                if cand_err or cand_score is None:
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=cfg_opt,
                        step_size=current_step_size,
                        metrics={},
                        objective_score=None,
                        status="failed",
                        decision_reason=f"Evaluation failed: {cand_err}",
                        execution_time_seconds=elapsed,
                        attempted_params={"chunk_size": cfg_opt.chunk_size, "chunk_overlap": cfg_opt.chunk_overlap},
                    )
                    record_trial(rec_trial, iteration)
                    visited_configs[cfg_key] = rec_trial
                else:
                    evaluated_in_iteration.append((cfg_opt, cand_score, cand_metrics, elapsed, trial_id))

            # Select best candidate in current iteration
            if evaluated_in_iteration:
                evaluated_in_iteration.sort(key=lambda x: x[1], reverse=True)
                top_cfg, top_score, top_metrics, top_elapsed, top_tid = evaluated_in_iteration[0]

                gain = top_score - current_best_score

                if gain >= tolerance:
                    # ACCEPT top candidate!
                    accepted_count += 1
                    logger.info(
                        "Accepted candidate %s with score %.4f (gain: +%.4f >= tol %.4f)",
                        top_cfg,
                        top_score,
                        gain,
                        tolerance,
                    )
                    # Record accepted candidate trial
                    acc_trial = CandidateTrial(
                        trial_id=top_tid,
                        iteration=iteration,
                        config=top_cfg,
                        step_size=current_step_size,
                        metrics=top_metrics,
                        objective_score=top_score,
                        status="accepted",
                        decision_reason=f"Improved objective from {current_best_score:.4f} to {top_score:.4f} (+{gain:.4f} >= tol {tolerance:.4f})",
                        execution_time_seconds=top_elapsed,
                        attempted_params={"chunk_size": top_cfg.chunk_size, "chunk_overlap": top_cfg.chunk_overlap},
                    )
                    record_trial(acc_trial, iteration)
                    visited_configs[(top_cfg.chunk_size, top_cfg.chunk_overlap)] = acc_trial

                    # Update current best
                    current_best_config = top_cfg
                    current_best_score = top_score
                    current_best_metrics = top_metrics

                    # Record other candidates in this iteration as rejected
                    for r_cfg, r_score, r_metrics, r_elapsed, r_tid in evaluated_in_iteration[1:]:
                        rej_trial = CandidateTrial(
                            trial_id=r_tid,
                            iteration=iteration,
                            config=r_cfg,
                            step_size=current_step_size,
                            metrics=r_metrics,
                            objective_score=r_score,
                            status="rejected",
                            decision_reason=f"Score {r_score:.4f} did not beat top candidate {top_score:.4f}",
                            execution_time_seconds=r_elapsed,
                            attempted_params={"chunk_size": r_cfg.chunk_size, "chunk_overlap": r_cfg.chunk_overlap},
                        )
                        record_trial(rej_trial, iteration)
                        visited_configs[(r_cfg.chunk_size, r_cfg.chunk_overlap)] = rej_trial

                else:
                    # No candidate beat current best by tolerance: REJECT all
                    logger.info(
                        "No candidate in iteration %d improved objective by tolerance %.4f (best candidate: %.4f vs current: %.4f).",
                        iteration,
                        tolerance,
                        top_score,
                        current_best_score,
                    )
                    for r_cfg, r_score, r_metrics, r_elapsed, r_tid in evaluated_in_iteration:
                        rej_trial = CandidateTrial(
                            trial_id=r_tid,
                            iteration=iteration,
                            config=r_cfg,
                            step_size=current_step_size,
                            metrics=r_metrics,
                            objective_score=r_score,
                            status="rejected",
                            decision_reason=f"Score {r_score:.4f} did not exceed best {current_best_score:.4f} + tol {tolerance:.4f}",
                            execution_time_seconds=r_elapsed,
                            attempted_params={"chunk_size": r_cfg.chunk_size, "chunk_overlap": r_cfg.chunk_overlap},
                        )
                        record_trial(rej_trial, iteration)
                        visited_configs[(r_cfg.chunk_size, r_cfg.chunk_overlap)] = rej_trial

                    # Reduce step size
                    new_step = int(current_step_size * self.search_config.step_decay)
                    new_ov_step = max(bounds.min_overlap_step, int(current_overlap_step * self.search_config.step_decay))

                    if new_step < bounds.min_step_size:
                        stopping_reason = f"Step size reduced below minimum ({new_step} < {bounds.min_step_size})"
                        logger.info("Stopping search: %s", stopping_reason)
                        break

                    current_step_size = new_step
                    current_overlap_step = new_ov_step
            else:
                # No valid unvisited candidates found
                stopping_reason = "No valid unvisited candidate configurations generated"
                logger.info("Stopping search: %s", stopping_reason)
                break

        total_exec_time = time.perf_counter() - start_time
        overall_improvement = round(current_best_score - baseline_score, 4)

        logger.info(
            "=== Optimization Complete: %s ===\nBaseline: %s (score: %.4f) -> Selected: %s (score: %.4f, gain: %+.4f)\nTotal Trials: %d, Accepted: %d, Time: %.2fs",
            stopping_reason,
            self.baseline_config,
            baseline_score,
            current_best_config,
            current_best_score,
            overall_improvement,
            len(candidate_history),
            accepted_count,
            total_exec_time,
        )

        return OptimizationResult(
            baseline_config=self.baseline_config,
            baseline_metrics=base_metrics,
            baseline_score=baseline_score,
            selected_config=current_best_config,
            selected_metrics=current_best_metrics,
            selected_score=current_best_score,
            improvement=overall_improvement,
            candidate_history=candidate_history,
            stopping_reason=stopping_reason,
            total_iterations=total_iterations_run,
            total_trials=len(candidate_history),
            accepted_trials=accepted_count,
            total_execution_time_seconds=total_exec_time,
            optimization_dataset_id=self.dataset_id,
            metadata={
                "random_seed": self.search_config.random_seed,
                "improvement_tolerance": tolerance,
                "target_metric": self.search_config.objective.target_metric,
                "objective_weights": self.search_config.objective.weights,
                "bounds": asdict(bounds),
            },
        )

    def resume_from_result(
        self,
        previous_result: Union[OptimizationResult, Path, str],
        additional_iterations: Optional[int] = None,
        reset_step_size: bool = False,
    ) -> OptimizationResult:
        """Resume optimization search starting from a previously saved result or checkpoint.

        Args:
            previous_result: Previous OptimizationResult instance or path to result JSON.
            additional_iterations: Additional search iterations to execute beyond previous total.
            reset_step_size: If True, resets search step size to initial values.

        Returns:
            OptimizationResult: Updated cumulative result containing full candidate history.
        """
        if isinstance(previous_result, (str, Path)):
            prev = OptimizationResult.load_from_json(previous_result)
        else:
            prev = previous_result

        visited_configs: Dict[Tuple[int, int], CandidateTrial] = {}
        candidate_history: List[CandidateTrial] = list(prev.candidate_history)

        for trial in candidate_history:
            sz = (
                trial.attempted_params.get("chunk_size", trial.config.chunk_size)
                if trial.attempted_params
                else trial.config.chunk_size
            )
            ov = (
                trial.attempted_params.get("chunk_overlap", trial.config.chunk_overlap)
                if trial.attempted_params
                else trial.config.chunk_overlap
            )
            if trial.status not in ("rejected_invalid", "failed"):
                visited_configs[(sz, ov)] = trial

        current_best_config = prev.selected_config
        current_best_score = prev.selected_score
        current_best_metrics = prev.selected_metrics
        baseline_config = prev.baseline_config
        baseline_score = prev.baseline_score
        baseline_metrics = prev.baseline_metrics

        current_step_size = (
            self.search_config.initial_step_size
            if reset_step_size
            else max(
                self.search_config.bounds.min_step_size,
                int(self.search_config.initial_step_size * (self.search_config.step_decay ** max(0, prev.total_iterations - 1))),
            )
        )
        current_overlap_step = (
            self.search_config.initial_overlap_step
            if reset_step_size
            else max(
                self.search_config.bounds.min_overlap_step,
                int(self.search_config.initial_overlap_step * (self.search_config.step_decay ** max(0, prev.total_iterations - 1))),
            )
        )

        trial_id = len(candidate_history)
        accepted_count = prev.accepted_trials
        start_time = time.perf_counter()
        bounds = self.search_config.bounds
        tolerance = self.search_config.improvement_tolerance
        iters_to_run = additional_iterations or self.search_config.max_iterations

        stopping_reason = f"Completed resumed iterations ({iters_to_run})"
        iterations_executed = 0

        for it in range(1, iters_to_run + 1):
            iteration = prev.total_iterations + it
            iterations_executed = it
            logger.info(
                "--- Resumed Optimization Iteration %d (step_size=%d, overlap_step=%d, current_best=%.4f @ %s) ---",
                iteration,
                current_step_size,
                current_overlap_step,
                current_best_score,
                current_best_config,
            )

            raw_candidates = self.generate_candidates(
                current_config=current_best_config,
                step_size=current_step_size,
                overlap_step=current_overlap_step,
            )

            evaluated_in_iteration: List[Tuple[ChunkConfig, float, Dict[str, Any], float, int]] = []

            for cand_spec in raw_candidates:
                cfg_opt = cand_spec.config
                desc = cand_spec.description
                invalid_reason = cand_spec.invalid_reason
                coords = cand_spec.attempted_coords

                trial_id += 1

                if cfg_opt is None:
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=current_best_config,
                        step_size=current_step_size,
                        metrics={},
                        objective_score=None,
                        status="rejected_invalid",
                        decision_reason=f"Candidate [{desc}] rejected: {invalid_reason}",
                        execution_time_seconds=0.0,
                        attempted_params={"chunk_size": coords[0], "chunk_overlap": coords[1]},
                    )
                    candidate_history.append(rec_trial)
                    continue

                cfg_key = (cfg_opt.chunk_size, cfg_opt.chunk_overlap)
                if cfg_key in visited_configs:
                    prev_trial = visited_configs[cfg_key]
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=cfg_opt,
                        step_size=current_step_size,
                        metrics=prev_trial.metrics,
                        objective_score=prev_trial.objective_score,
                        status="skipped_duplicate",
                        decision_reason=f"Candidate [{desc}] already evaluated in trial #{prev_trial.trial_id}",
                        execution_time_seconds=0.0,
                        attempted_params={"chunk_size": cfg_opt.chunk_size, "chunk_overlap": cfg_opt.chunk_overlap},
                    )
                    candidate_history.append(rec_trial)
                    continue

                t0 = time.perf_counter()
                cand_metrics, cand_score, cand_err = self._run_single_evaluation(cfg_opt)
                elapsed = time.perf_counter() - t0

                if cand_err or cand_score is None:
                    rec_trial = CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=cfg_opt,
                        step_size=current_step_size,
                        metrics={},
                        objective_score=None,
                        status="failed",
                        decision_reason=f"Evaluation failed: {cand_err}",
                        execution_time_seconds=elapsed,
                        attempted_params={"chunk_size": cfg_opt.chunk_size, "chunk_overlap": cfg_opt.chunk_overlap},
                    )
                    candidate_history.append(rec_trial)
                    visited_configs[cfg_key] = rec_trial
                else:
                    evaluated_in_iteration.append((cfg_opt, cand_score, cand_metrics, elapsed, trial_id))

            if evaluated_in_iteration:
                evaluated_in_iteration.sort(key=lambda x: x[1], reverse=True)
                top_cfg, top_score, top_metrics, top_elapsed, top_tid = evaluated_in_iteration[0]
                gain = top_score - current_best_score

                if gain >= tolerance:
                    accepted_count += 1
                    logger.info(
                        "Accepted candidate %s with score %.4f (gain: +%.4f >= tol %.4f)",
                        top_cfg,
                        top_score,
                        gain,
                        tolerance,
                    )
                    acc_trial = CandidateTrial(
                        trial_id=top_tid,
                        iteration=iteration,
                        config=top_cfg,
                        step_size=current_step_size,
                        metrics=top_metrics,
                        objective_score=top_score,
                        status="accepted",
                        decision_reason=f"Improved objective from {current_best_score:.4f} to {top_score:.4f} (+{gain:.4f} >= tol {tolerance:.4f})",
                        execution_time_seconds=top_elapsed,
                        attempted_params={"chunk_size": top_cfg.chunk_size, "chunk_overlap": top_cfg.chunk_overlap},
                    )
                    candidate_history.append(acc_trial)
                    visited_configs[(top_cfg.chunk_size, top_cfg.chunk_overlap)] = acc_trial

                    current_best_config = top_cfg
                    current_best_score = top_score
                    current_best_metrics = top_metrics

                    for r_cfg, r_score, r_metrics, r_elapsed, r_tid in evaluated_in_iteration[1:]:
                        rej_trial = CandidateTrial(
                            trial_id=r_tid,
                            iteration=iteration,
                            config=r_cfg,
                            step_size=current_step_size,
                            metrics=r_metrics,
                            objective_score=r_score,
                            status="rejected",
                            decision_reason=f"Score {r_score:.4f} did not beat top candidate {top_score:.4f}",
                            execution_time_seconds=r_elapsed,
                            attempted_params={"chunk_size": r_cfg.chunk_size, "chunk_overlap": r_cfg.chunk_overlap},
                        )
                        candidate_history.append(rej_trial)
                        visited_configs[(r_cfg.chunk_size, r_cfg.chunk_overlap)] = rej_trial
                else:
                    for r_cfg, r_score, r_metrics, r_elapsed, r_tid in evaluated_in_iteration:
                        rej_trial = CandidateTrial(
                            trial_id=r_tid,
                            iteration=iteration,
                            config=r_cfg,
                            step_size=current_step_size,
                            metrics=r_metrics,
                            objective_score=r_score,
                            status="rejected",
                            decision_reason=f"Score {r_score:.4f} did not exceed best {current_best_score:.4f} + tol {tolerance:.4f}",
                            execution_time_seconds=r_elapsed,
                            attempted_params={"chunk_size": r_cfg.chunk_size, "chunk_overlap": r_cfg.chunk_overlap},
                        )
                        candidate_history.append(rej_trial)
                        visited_configs[(r_cfg.chunk_size, r_cfg.chunk_overlap)] = rej_trial

                    new_step = int(current_step_size * self.search_config.step_decay)
                    new_ov_step = max(bounds.min_overlap_step, int(current_overlap_step * self.search_config.step_decay))

                    if new_step < bounds.min_step_size:
                        stopping_reason = f"Step size reduced below minimum ({new_step} < {bounds.min_step_size})"
                        logger.info("Stopping search: %s", stopping_reason)
                        break

                    current_step_size = new_step
                    current_overlap_step = new_ov_step
            else:
                stopping_reason = "No valid unvisited candidate configurations generated"
                logger.info("Stopping search: %s", stopping_reason)
                break

        total_exec_time = prev.total_execution_time_seconds + (time.perf_counter() - start_time)
        overall_improvement = round(current_best_score - baseline_score, 4)

        return OptimizationResult(
            baseline_config=baseline_config,
            baseline_metrics=baseline_metrics,
            baseline_score=baseline_score,
            selected_config=current_best_config,
            selected_metrics=current_best_metrics,
            selected_score=current_best_score,
            improvement=overall_improvement,
            candidate_history=candidate_history,
            stopping_reason=stopping_reason,
            total_iterations=prev.total_iterations + iterations_executed,
            total_trials=len(candidate_history),
            accepted_trials=accepted_count,
            total_execution_time_seconds=total_exec_time,
            optimization_dataset_id=self.dataset_id,
            metadata={
                "random_seed": self.search_config.random_seed,
                "improvement_tolerance": tolerance,
                "target_metric": self.search_config.objective.target_metric,
                "objective_weights": self.search_config.objective.weights,
                "bounds": asdict(bounds),
                "resumed_from_trials": prev.total_trials,
            },
        )


# ==============================================================================
# SIMULATED ANNEALING HYPERPARAMETER OPTIMIZATION
# ==============================================================================


@dataclass
class SimulatedAnnealingConfig:
    """Hyperparameters governing simulated annealing metaheuristic optimization."""

    initial_temperature: float = 1.0
    min_temperature: float = 0.05
    cooling_rate: float = 0.80  # T_{k+1} = T_k * cooling_rate
    steps_per_temperature: int = 2
    max_trials: int = 25
    chunk_step: int = 100
    overlap_step: int = 25
    bounds: SearchBounds = field(default_factory=SearchBounds)
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
    random_seed: int = 42


class SimulatedAnnealingOptimizer:
    """Metaheuristic Simulated Annealing optimizer for RAG chunking hyperparameters.

    Implements probabilistic exploration with Boltzmann/Metropolis acceptance:
        P(accept) = 1.0 if Delta E >= 0 else exp(Delta E / T)
    where Delta E = score(candidate) - score(current).

    This directly resolves the limitation noted in the ReRAG paper by enabling the
    optimizer to escape local optima in the non-convex chunk size / overlap landscape.
    """

    def __init__(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        sa_config: Optional[SimulatedAnnealingConfig] = None,
        evaluate_fn: Optional[Callable[[ChunkConfig], Union[EvaluationReport, Dict[str, Any], float]]] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        optimization_samples: Optional[Sequence[EvaluationSample]] = None,
        collection_name: Optional[str] = None,
        dataset_id: str = "synthetic_demonstration_dataset",
        evaluation_cache: Optional[Dict[Tuple[int, int], Tuple[Dict[str, Any], float]]] = None,
    ) -> None:
        """Initialize simulated annealing optimizer.

        Args:
            baseline_config: Starting chunk configuration.
            sa_config: Simulated annealing settings.
            evaluate_fn: Optional custom evaluation callable.
            documents: Sequence of LoadedDocument objects to chunk and index.
            pipeline: Active RAGPipeline.
            evaluator: Evaluator instance.
            optimization_samples: Training evaluation question set.
            collection_name: Optional ChromaDB collection name.
            dataset_id: Dataset identifier for audit tracing.
            evaluation_cache: Optional shared cache mapping (chunk_size, chunk_overlap) to (metrics, score).
        """
        self.sa_config = sa_config or SimulatedAnnealingConfig()
        self.baseline_config = baseline_config or ChunkConfig(500, 50)
        self.dataset_id = dataset_id
        self.collection_name = collection_name or "rerag_sa_collection"
        self.evaluation_cache = evaluation_cache

        if evaluate_fn is not None:
            self._evaluate_fn = evaluate_fn
        elif documents is not None and pipeline is not None and evaluator is not None and optimization_samples is not None:
            self._evaluate_fn = self._create_pipeline_evaluator(
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                samples=optimization_samples,
                collection_name=self.collection_name,
            )
        else:
            raise ValueError(
                "SimulatedAnnealingOptimizer requires either `evaluate_fn` or "
                "(`documents`, `pipeline`, `evaluator`, `optimization_samples`)."
            )

        random.seed(self.sa_config.random_seed)

    def _create_pipeline_evaluator(
        self,
        documents: Sequence[LoadedDocument],
        pipeline: RAGPipeline,
        evaluator: Evaluator,
        samples: Sequence[EvaluationSample],
        collection_name: str,
    ) -> Callable[[ChunkConfig], EvaluationReport]:
        """Create evaluation function that rebuilds index and evaluates pipeline."""
        def _eval_config(config: ChunkConfig) -> EvaluationReport:
            chunker = TextChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            pipeline.index_documents(
                documents=documents,
                chunker=chunker,
                collection_name=collection_name,
            )
            return evaluator.evaluate(
                samples=samples,
                deterministic=True,
                config_override=config.to_dict(),
                collection_name=collection_name,
            )

        return _eval_config

    def _run_single_evaluation(
        self,
        config: ChunkConfig,
    ) -> Tuple[Dict[str, Any], Optional[float], Optional[str]]:
        """Safely execute evaluation callable for a single configuration."""
        cache_key = (config.chunk_size, config.chunk_overlap)
        if self.evaluation_cache is not None and cache_key in self.evaluation_cache:
            cached_metrics, cached_score = self.evaluation_cache[cache_key]
            return cached_metrics, cached_score, None

        try:
            eval_output = self._evaluate_fn(config)
            if isinstance(eval_output, EvaluationReport):
                metrics = eval_output.aggregate_metrics
                score = self.sa_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, dict):
                metrics = eval_output
                score = self.sa_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, (int, float)):
                score = round(float(eval_output), 4)
                metrics = {"objective_score": score}
            else:
                raise TypeError(f"Unsupported evaluation output type: {type(eval_output)}")

            if self.evaluation_cache is not None:
                self.evaluation_cache[cache_key] = (metrics, score)

            return metrics, score, None
        except Exception as e:
            logger.error("SA evaluation failed for config %s: %s", config, e, exc_info=True)
            return {}, None, str(e)

    def _generate_neighbor(self, current: ChunkConfig) -> ChunkConfig:
        """Generate a random perturbed valid neighboring configuration within search bounds."""
        bounds = self.sa_config.bounds
        for _ in range(50):
            # Sample discrete perturbation
            d_size = random.choice([-self.sa_config.chunk_step, 0, self.sa_config.chunk_step])
            d_overlap = random.choice([-self.sa_config.overlap_step, 0, self.sa_config.overlap_step])
            if d_size == 0 and d_overlap == 0:
                d_size = random.choice([-self.sa_config.chunk_step, self.sa_config.chunk_step])

            new_size = max(bounds.min_chunk_size, min(bounds.max_chunk_size, current.chunk_size + d_size))
            new_overlap = max(bounds.min_chunk_overlap, min(bounds.max_chunk_overlap, current.chunk_overlap + d_overlap))

            # Enforce valid overlap constraint
            if new_overlap < new_size and (new_size != current.chunk_size or new_overlap != current.chunk_overlap):
                return ChunkConfig(chunk_size=new_size, chunk_overlap=new_overlap)

        # Fallback safe tweak
        fallback_size = min(bounds.max_chunk_size, current.chunk_size + self.sa_config.chunk_step)
        fallback_overlap = min(fallback_size - 10, current.chunk_overlap)
        return ChunkConfig(chunk_size=fallback_size, chunk_overlap=fallback_overlap)

    def optimize(
        self,
        held_out_samples: Optional[Sequence[EvaluationSample]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> OptimizationResult:
        """Execute Simulated Annealing optimization.

        Args:
            held_out_samples: Optional held-out test set for generalization testing.
            progress_callback: Optional callback for UI / progress reporting.

        Returns:
            OptimizationResult: Final summary with complete trial history.
        """
        start_time = time.perf_counter()
        history: List[CandidateTrial] = []
        trial_id = 1
        iteration = 1

        # 1. Evaluate baseline
        base_t0 = time.perf_counter()
        base_metrics, base_score, base_err = self._run_single_evaluation(self.baseline_config)
        base_time = time.perf_counter() - base_t0

        if base_score is None:
            raise RuntimeError(f"Baseline configuration evaluation failed: {base_err}")

        current_config = self.baseline_config
        current_score = base_score
        current_metrics = base_metrics

        best_config = self.baseline_config
        best_score = base_score
        best_metrics = base_metrics

        history.append(
            CandidateTrial(
                trial_id=trial_id,
                iteration=iteration,
                config=self.baseline_config,
                step_size=self.sa_config.chunk_step,
                metrics=base_metrics,
                objective_score=base_score,
                status="baseline",
                decision_reason="Initial baseline configuration",
                execution_time_seconds=base_time,
            )
        )

        temperature = self.sa_config.initial_temperature
        accepted_count = 0
        stopping_reason = "max_trials_reached"

        while temperature >= self.sa_config.min_temperature and trial_id < self.sa_config.max_trials:
            iteration += 1
            for _ in range(self.sa_config.steps_per_temperature):
                trial_id += 1
                if trial_id > self.sa_config.max_trials:
                    break

                candidate = self._generate_neighbor(current_config)

                cand_t0 = time.perf_counter()
                cand_metrics, cand_score, cand_err = self._run_single_evaluation(candidate)
                cand_time = time.perf_counter() - cand_t0

                if cand_score is None:
                    history.append(
                        CandidateTrial(
                            trial_id=trial_id,
                            iteration=iteration,
                            config=candidate,
                            step_size=self.sa_config.chunk_step,
                            metrics={},
                            objective_score=None,
                            status="failed",
                            decision_reason=f"Evaluation error: {cand_err}",
                            execution_time_seconds=cand_time,
                        )
                    )
                    continue

                delta_e = cand_score - current_score

                if delta_e >= 0:
                    # Improvement: always accept
                    accept = True
                    decision_reason = f"Metropolis improvement (ΔE = +{delta_e:.4f}, T={temperature:.3f})"
                else:
                    # Probabilistic acceptance of worse state to escape local minima
                    prob = math.exp(delta_e / max(1e-6, temperature))
                    accept = random.random() < prob
                    if accept:
                        decision_reason = f"Metropolis accepted worse move (ΔE = {delta_e:.4f}, P={prob:.3f}, T={temperature:.3f})"
                    else:
                        decision_reason = f"Metropolis rejected worse move (ΔE = {delta_e:.4f}, P={prob:.3f}, T={temperature:.3f})"

                if accept:
                    current_config = candidate
                    current_score = cand_score
                    current_metrics = cand_metrics
                    accepted_count += 1
                    status = "accepted"

                    if cand_score > best_score:
                        best_config = candidate
                        best_score = cand_score
                        best_metrics = cand_metrics
                else:
                    status = "rejected"

                history.append(
                    CandidateTrial(
                        trial_id=trial_id,
                        iteration=iteration,
                        config=candidate,
                        step_size=self.sa_config.chunk_step,
                        metrics=cand_metrics,
                        objective_score=cand_score,
                        status=status,
                        decision_reason=decision_reason,
                        execution_time_seconds=cand_time,
                    )
                )

                if progress_callback:
                    progress_callback({
                        "trial_id": trial_id,
                        "iteration": iteration,
                        "temperature": temperature,
                        "current_config": current_config.to_dict(),
                        "best_config": best_config.to_dict(),
                        "best_score": best_score,
                    })

            # Cooling schedule: geometric decay
            temperature *= self.sa_config.cooling_rate

        if temperature < self.sa_config.min_temperature:
            stopping_reason = f"Temperature cooled below minimum threshold ({self.sa_config.min_temperature})"

        total_time = time.perf_counter() - start_time

        # Held-out generalization evaluation
        held_out_eval = None
        if held_out_samples:
            try:
                ho_eval = self._evaluate_fn(best_config)
                if isinstance(ho_eval, EvaluationReport):
                    held_out_eval = ho_eval.aggregate_metrics
                elif isinstance(ho_eval, dict):
                    held_out_eval = ho_eval
            except Exception as exc:
                logger.warning("Held-out evaluation failed: %s", exc)

        return OptimizationResult(
            baseline_config=self.baseline_config,
            baseline_metrics=base_metrics,
            baseline_score=base_score,
            selected_config=best_config,
            selected_metrics=best_metrics,
            selected_score=best_score,
            improvement=round(best_score - base_score, 4),
            candidate_history=history,
            stopping_reason=stopping_reason,
            total_iterations=iteration,
            total_trials=len(history),
            accepted_trials=accepted_count,
            total_execution_time_seconds=total_time,
            optimization_dataset_id=self.dataset_id,
            held_out_evaluation=held_out_eval,
            metadata={
                "algorithm": "simulated_annealing",
                "initial_temperature": self.sa_config.initial_temperature,
                "final_temperature": round(temperature, 4),
                "cooling_rate": self.sa_config.cooling_rate,
            },
        )


# ==============================================================================
# RANDOM SEARCH HYPERPARAMETER OPTIMIZATION
# ==============================================================================


@dataclass
class RandomSearchConfig:
    """Hyperparameters governing random chunking parameter search."""

    num_trials: int = 15
    bounds: SearchBounds = field(default_factory=SearchBounds)
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
    random_seed: int = 42
    step_multiple: int = 25  # Snap samples to discrete step multiples for realistic chunking
    max_trials: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate trial budget and support max_trials alias."""
        if self.max_trials is not None:
            self.num_trials = self.max_trials
        if self.num_trials <= 0:
            raise ValueError(f"num_trials must be at least 1, got {self.num_trials}")


class RandomSearchOptimizer:
    """Uniform random search optimizer for chunk size and overlap hyperparameter exploration.

    Samples uniformly from valid parameter space bounded by SearchBounds while enforcing
    the relational constraint chunk_overlap < chunk_size. Evaluates candidate configurations
    strictly on the optimization training questions without accessing held-out test data.
    """

    def __init__(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        random_config: Optional[RandomSearchConfig] = None,
        config: Optional[RandomSearchConfig] = None,
        evaluate_fn: Optional[Callable[[ChunkConfig], Union[EvaluationReport, Dict[str, Any], float]]] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        optimization_samples: Optional[Sequence[EvaluationSample]] = None,
        collection_name: Optional[str] = None,
        dataset_id: str = "synthetic_demonstration_dataset",
        evaluation_cache: Optional[Dict[Tuple[int, int], Tuple[Dict[str, Any], float]]] = None,
    ) -> None:
        """Initialize random search optimizer."""
        self.random_config = config or random_config or RandomSearchConfig()
        self.baseline_config = baseline_config or ChunkConfig(500, 50)
        self.dataset_id = dataset_id
        self.collection_name = collection_name or "rerag_rand_collection"
        self.evaluation_cache = evaluation_cache

        if evaluate_fn is not None:
            self._evaluate_fn = evaluate_fn
        elif documents is not None and pipeline is not None and evaluator is not None and optimization_samples is not None:
            self._evaluate_fn = self._create_pipeline_evaluator(
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                samples=optimization_samples,
                collection_name=self.collection_name,
            )
        else:
            raise ValueError(
                "RandomSearchOptimizer requires either `evaluate_fn` or "
                "(`documents`, `pipeline`, `evaluator`, `optimization_samples`)."
            )

        self._rng = random.Random(self.random_config.random_seed)

    def _create_pipeline_evaluator(
        self,
        documents: Sequence[LoadedDocument],
        pipeline: RAGPipeline,
        evaluator: Evaluator,
        samples: Sequence[EvaluationSample],
        collection_name: str,
    ) -> Callable[[ChunkConfig], EvaluationReport]:
        """Create evaluation function that rebuilds index and evaluates pipeline."""
        def _eval_config(config: ChunkConfig) -> EvaluationReport:
            chunker = TextChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            pipeline.index_documents(
                documents=documents,
                chunker=chunker,
                collection_name=collection_name,
            )
            return evaluator.evaluate(
                samples=samples,
                deterministic=True,
                config_override=config.to_dict(),
                collection_name=collection_name,
            )

        return _eval_config

    def _run_single_evaluation(
        self,
        config: ChunkConfig,
    ) -> Tuple[Dict[str, Any], Optional[float], Optional[str]]:
        """Safely execute evaluation callable for a single configuration."""
        cache_key = (config.chunk_size, config.chunk_overlap)
        if self.evaluation_cache is not None and cache_key in self.evaluation_cache:
            cached_metrics, cached_score = self.evaluation_cache[cache_key]
            return cached_metrics, cached_score, None

        try:
            eval_output = self._evaluate_fn(config)
            if isinstance(eval_output, EvaluationReport):
                metrics = eval_output.aggregate_metrics
                score = self.random_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, tuple) and len(eval_output) == 2:
                metrics, score = eval_output
                if isinstance(score, (int, float)):
                    score = round(float(score), 4)
                if not isinstance(metrics, dict):
                    metrics = {"objective_score": score}
            elif isinstance(eval_output, dict):
                metrics = eval_output
                score = self.random_config.objective.calculate_score(metrics)
            elif isinstance(eval_output, (int, float)):
                score = round(float(eval_output), 4)
                metrics = {"objective_score": score}
            else:
                raise TypeError(f"Unsupported evaluation output type: {type(eval_output)}")

            if self.evaluation_cache is not None:
                self.evaluation_cache[cache_key] = (metrics, score)

            return metrics, score, None
        except Exception as e:
            logger.error("Random search evaluation failed for config %s: %s", config, e, exc_info=True)
            return {}, None, str(e)

    def _sample_candidate(self, existing_configs: Set[Tuple[int, int]]) -> ChunkConfig:
        """Sample a valid, non-redundant chunk configuration within search bounds."""
        bounds = self.random_config.bounds
        mult = max(1, self.random_config.step_multiple)

        min_sz_step = math.ceil(bounds.min_chunk_size / mult)
        max_sz_step = math.floor(bounds.max_chunk_size / mult)

        for _ in range(200):
            sz = self._rng.randint(min_sz_step, max_sz_step) * mult
            sz = max(bounds.min_chunk_size, min(bounds.max_chunk_size, sz))

            max_ov = min(bounds.max_chunk_overlap, sz - 1)
            min_ov = bounds.min_chunk_overlap

            if max_ov < min_ov:
                continue

            min_ov_step = math.ceil(min_ov / mult)
            max_ov_step = math.floor(max_ov / mult)

            if max_ov_step >= min_ov_step:
                ov = self._rng.randint(min_ov_step, max_ov_step) * mult
            else:
                ov = min_ov

            ov = max(min_ov, min(max_ov, ov))

            if ov < sz and (sz, ov) not in existing_configs:
                return ChunkConfig(chunk_size=sz, chunk_overlap=ov)

        fallback_sz = max(bounds.min_chunk_size, min(bounds.max_chunk_size, self.baseline_config.chunk_size + mult))
        fallback_ov = min(fallback_sz - 1, self.baseline_config.chunk_overlap)
        return ChunkConfig(chunk_size=fallback_sz, chunk_overlap=fallback_ov)

    def optimize(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        held_out_samples: Optional[Sequence[EvaluationSample]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> OptimizationResult:
        """Run uniform random hyperparameter exploration over chunk size and overlap."""
        if baseline_config is not None:
            self.baseline_config = baseline_config
        start_time = time.perf_counter()
        history: List[CandidateTrial] = []
        seen_configs: Set[Tuple[int, int]] = set()

        # 1. Baseline trial
        base_t0 = time.perf_counter()
        base_metrics, base_score, base_err = self._run_single_evaluation(self.baseline_config)
        base_time = time.perf_counter() - base_t0

        if base_score is None:
            raise RuntimeError(f"Baseline configuration evaluation failed: {base_err}")

        best_config = self.baseline_config
        best_score = base_score
        best_metrics = base_metrics
        accepted_count = 0

        seen_configs.add((self.baseline_config.chunk_size, self.baseline_config.chunk_overlap))

        history.append(
            CandidateTrial(
                trial_id=1,
                iteration=1,
                config=self.baseline_config,
                step_size=self.random_config.step_multiple,
                metrics=base_metrics,
                objective_score=base_score,
                status="baseline",
                decision_reason="Initial baseline configuration",
                execution_time_seconds=base_time,
            )
        )

        num_trials = max(1, self.random_config.num_trials)

        for trial_id in range(2, num_trials + 1):
            candidate = self._sample_candidate(seen_configs)
            seen_configs.add((candidate.chunk_size, candidate.chunk_overlap))

            cand_t0 = time.perf_counter()
            cand_metrics, cand_score, cand_err = self._run_single_evaluation(candidate)
            cand_time = time.perf_counter() - cand_t0

            if cand_score is None:
                history.append(
                    CandidateTrial(
                        trial_id=trial_id,
                        iteration=trial_id,
                        config=candidate,
                        step_size=self.random_config.step_multiple,
                        metrics={},
                        objective_score=None,
                        status="failed",
                        decision_reason=f"Evaluation failed: {cand_err}",
                        execution_time_seconds=cand_time,
                    )
                )
                continue

            delta = cand_score - best_score
            if delta > 0.0:
                best_config = candidate
                best_score = cand_score
                best_metrics = cand_metrics
                accepted_count += 1
                status = "accepted"
                decision_reason = f"Random search improvement over previous best (Δ = +{delta:.4f})"
            else:
                status = "rejected"
                decision_reason = f"Random search sub-optimal score ({cand_score:.4f} <= {best_score:.4f})"

            history.append(
                CandidateTrial(
                    trial_id=trial_id,
                    iteration=trial_id,
                    config=candidate,
                    step_size=self.random_config.step_multiple,
                    metrics=cand_metrics,
                    objective_score=cand_score,
                    status=status,
                    decision_reason=decision_reason,
                    execution_time_seconds=cand_time,
                )
            )

            if progress_callback:
                progress_callback({
                    "trial_id": trial_id,
                    "iteration": trial_id,
                    "candidate_config": candidate.to_dict(),
                    "candidate_score": cand_score,
                    "best_config": best_config.to_dict(),
                    "best_score": best_score,
                    "status": status,
                })

        total_time = time.perf_counter() - start_time

        # Held-out evaluation if requested
        held_out_eval = None
        if held_out_samples:
            try:
                ho_eval = self._evaluate_fn(best_config)
                if isinstance(ho_eval, EvaluationReport):
                    held_out_eval = ho_eval.aggregate_metrics
                elif isinstance(ho_eval, dict):
                    held_out_eval = ho_eval
            except Exception as exc:
                logger.warning("Held-out evaluation failed: %s", exc)

        return OptimizationResult(
            baseline_config=self.baseline_config,
            baseline_metrics=base_metrics,
            baseline_score=base_score,
            selected_config=best_config,
            selected_metrics=best_metrics,
            selected_score=best_score,
            improvement=round(best_score - base_score, 4),
            candidate_history=history,
            stopping_reason=f"Completed configured {num_trials} random trials",
            total_iterations=num_trials,
            total_trials=len(history),
            accepted_trials=accepted_count,
            total_execution_time_seconds=total_time,
            optimization_dataset_id=self.dataset_id,
            held_out_evaluation=held_out_eval,
            metadata={
                "algorithm": "random_search",
                "num_trials": num_trials,
                "step_multiple": self.random_config.step_multiple,
                "random_seed": self.random_config.random_seed,
            },
        )


class ChunkOptimizerFactory:
    """Factory creating configured chunk optimizers by strategy name."""

    @staticmethod
    def create(
        strategy: str,
        baseline_config: Optional[ChunkConfig] = None,
        search_bounds: Optional[SearchBounds] = None,
        objective: Optional[OptimizationObjective] = None,
        random_seed: int = 42,
        max_trials: int = 15,
        evaluate_fn: Optional[Callable[[ChunkConfig], Union[EvaluationReport, Dict[str, Any], float]]] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        optimization_samples: Optional[Sequence[EvaluationSample]] = None,
        collection_name: Optional[str] = None,
        dataset_id: str = "synthetic_demonstration_dataset",
        evaluation_cache: Optional[Dict[Tuple[int, int], Tuple[Dict[str, Any], float]]] = None,
        **kwargs: Any,
    ) -> Union[AdaptiveChunkOptimizer, RandomSearchOptimizer, SimulatedAnnealingOptimizer]:
        """Instantiate optimizer for the requested strategy."""
        strat = strategy.lower().strip()
        bounds = search_bounds or SearchBounds()
        obj = objective or OptimizationObjective()

        if strat in ("greedy", "adaptive", "coordinate", "original"):
            cfg = AdaptiveSearchConfig(
                bounds=bounds,
                objective=obj,
                random_seed=random_seed,
                max_iterations=kwargs.get("max_iterations", max_trials),
                initial_step_size=kwargs.get("initial_step_size", 100),
                initial_overlap_step=kwargs.get("initial_overlap_step", 25),
                step_decay=kwargs.get("step_decay", 0.5),
                improvement_tolerance=kwargs.get("improvement_tolerance", 0.005),
            )
            return AdaptiveChunkOptimizer(
                baseline_config=baseline_config,
                search_config=cfg,
                evaluate_fn=evaluate_fn,
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                optimization_samples=optimization_samples,
                collection_name=collection_name,
                dataset_id=dataset_id,
                evaluation_cache=evaluation_cache,
            )
        elif strat in ("random", "random_search"):
            r_cfg = RandomSearchConfig(
                num_trials=max_trials,
                bounds=bounds,
                objective=obj,
                random_seed=random_seed,
                step_multiple=kwargs.get("step_multiple", 25),
            )
            return RandomSearchOptimizer(
                baseline_config=baseline_config,
                random_config=r_cfg,
                evaluate_fn=evaluate_fn,
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                optimization_samples=optimization_samples,
                collection_name=collection_name,
                dataset_id=dataset_id,
                evaluation_cache=evaluation_cache,
            )
        elif strat in ("simulated_annealing", "sa", "annealing"):
            sa_cfg = SimulatedAnnealingConfig(
                initial_temperature=kwargs.get("initial_temperature", 1.0),
                min_temperature=kwargs.get("min_temperature", 0.05),
                cooling_rate=kwargs.get("cooling_rate", 0.80),
                steps_per_temperature=kwargs.get("steps_per_temperature", 2),
                max_trials=max_trials,
                chunk_step=kwargs.get("chunk_step", 100),
                overlap_step=kwargs.get("overlap_step", 25),
                bounds=bounds,
                objective=obj,
                random_seed=random_seed,
            )
            return SimulatedAnnealingOptimizer(
                baseline_config=baseline_config,
                sa_config=sa_cfg,
                evaluate_fn=evaluate_fn,
                documents=documents,
                pipeline=pipeline,
                evaluator=evaluator,
                optimization_samples=optimization_samples,
                collection_name=collection_name,
                dataset_id=dataset_id,
                evaluation_cache=evaluation_cache,
            )
        else:
            raise ValueError(
                f"Unknown optimization strategy: '{strategy}'. Supported strategies: 'greedy', 'random', 'simulated_annealing'"
            )


def get_optimizer(strategy: str = "greedy", **kwargs: Any) -> Any:
    """Convenience helper to obtain a chunk optimizer by strategy name."""
    return ChunkOptimizerFactory.create(strategy=strategy, **kwargs)


# Backward compatibility classes
@dataclass
class ParameterGrid:
    """Defines search space for hyperparameter optimization (for backwards compatibility)."""

    chunk_sizes: Sequence[int] = (250, 500, 1000)
    chunk_overlaps: Sequence[int] = (25, 50, 100)
    top_ks: Sequence[int] = (2, 4, 8)
    verification_thresholds: Sequence[float] = (0.5, 0.7, 0.85)


class ReRAGOptimizer:
    """Wrapper class providing grid search interface for backwards compatibility."""

    def __init__(
        self,
        evaluator: Evaluator,
        parameter_grid: Optional[ParameterGrid] = None,
        target_metric: str = "faithfulness_ratio",
    ) -> None:
        """Initialize optimizer."""
        self.evaluator = evaluator
        self.parameter_grid = parameter_grid or ParameterGrid()
        self.target_metric = target_metric

    def run_grid_search(
        self,
        samples: List[EvaluationSample],
    ) -> OptimizationResult:
        """Execute grid search across parameter grid."""
        raise NotImplementedError("Use AdaptiveChunkOptimizer or SimulatedAnnealingOptimizer for parameter optimization.")

