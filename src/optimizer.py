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
        """
        self.search_config = search_config or AdaptiveSearchConfig()
        self.baseline_config = baseline_config or ChunkConfig(
            self.search_config.initial_chunk_size,
            self.search_config.initial_chunk_overlap,
        )
        self.dataset_id = dataset_id
        self.collection_name = collection_name or "rerag_opt_collection"

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
        raise NotImplementedError("Use AdaptiveChunkOptimizer for adaptive greedy parameter optimization.")
