"""Reproducible Multi-Method Comparison Runner for ReRAG Parameter Optimization.

Orchestrates a controlled empirical comparison across:
1. Fixed-Parameter Baseline RAG (static chunking)
2. Original ReRAG Greedy Coordinate Search (AdaptiveChunkOptimizer)
3. Random Search (RandomSearchOptimizer)
4. Simulated Annealing (SimulatedAnnealingOptimizer)

Guarantees:
- Controlled Protocol: Same document corpus, same train/val/test splits, same random seeds.
- Strict Data Isolation: Optimizers ONLY have access to the optimization split.
  Held-out test answers are never accessed during parameter selection.
- Shared Evaluation Cache: Reuses identical candidate evaluations to save computation.
- Diagnostics & Persistence: Records complete trial histories, convergence trajectories,
  generalization gaps, latency breakdowns, and computational cost proxies to JSON and CSV.
- No Fabricated Results: Reports exact empirical outcomes for all methods.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.chunker import TextChunker
from src.config import Settings, get_settings
from src.dataset_workflow import DatasetSplits, split_train_val_test
from src.document_loader import LoadedDocument
from src.embeddings import EmbeddingManager
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator, load_dataset
from src.llm import BaseLLMClient, OllamaClient
from src.metrics import (
    OptimizationCostTracker,
    compute_aggregate_metrics,
    compute_latency_metrics,
)
from src.optimizer import (
    AdaptiveChunkOptimizer,
    AdaptiveSearchConfig,
    CandidateTrial,
    ChunkConfig,
    ChunkOptimizerFactory,
    OptimizationObjective,
    OptimizationResult,
    RandomSearchConfig,
    RandomSearchOptimizer,
    SearchBounds,
    SimulatedAnnealingConfig,
    SimulatedAnnealingOptimizer,
)
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMethodResult:
    """Detailed results and diagnostics for a single optimization strategy."""

    method_id: str
    display_name: str
    selected_config: ChunkConfig
    baseline_config: ChunkConfig
    train_score: float
    train_metrics: Dict[str, Any]
    test_score: float
    test_metrics: Dict[str, Any]
    generalization_gap: float
    total_trials: int
    accepted_trials: int
    execution_time_seconds: float
    stopping_reason: str
    candidate_history: List[CandidateTrial] = field(default_factory=list)
    validation_score: Optional[float] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    cost_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert method result to dictionary."""
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "selected_config": self.selected_config.to_dict(),
            "baseline_config": self.baseline_config.to_dict(),
            "train_score": round(self.train_score, 4),
            "train_metrics": self.train_metrics,
            "validation_score": round(self.validation_score, 4) if self.validation_score is not None else None,
            "validation_metrics": self.validation_metrics,
            "test_score": round(self.test_score, 4),
            "test_metrics": self.test_metrics,
            "generalization_gap": round(self.generalization_gap, 4),
            "total_trials": self.total_trials,
            "accepted_trials": self.accepted_trials,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "stopping_reason": self.stopping_reason,
            "cost_metrics": self.cost_metrics,
            "metadata": self.metadata,
            "candidate_history": [t.to_dict() for t in self.candidate_history],
        }


@dataclass
class ComparisonResult:
    """Unified container for multi-method comparison experiments."""

    experiment_id: str
    timestamp: str
    dataset_id: str
    methods: Dict[str, ComparisonMethodResult]
    summary_records: List[Dict[str, Any]]
    best_train_method: str
    best_test_method: str
    baseline_config: ChunkConfig
    search_bounds: SearchBounds
    protocol_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete comparison result to serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "dataset_id": self.dataset_id,
            "best_train_method": self.best_train_method,
            "best_test_method": self.best_test_method,
            "baseline_config": self.baseline_config.to_dict(),
            "search_bounds": asdict(self.search_bounds),
            "protocol_metadata": self.protocol_metadata,
            "summary_records": self.summary_records,
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
        }

    def save_to_json(self, file_path: Union[Path, str]) -> None:
        """Persist comparison result to JSON."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved comparison result JSON to %s", target)

    def save_summary_csv(self, file_path: Union[Path, str]) -> None:
        """Persist comparative summary table to CSV."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.summary_records:
            return

        headers = list(self.summary_records[0].keys())
        with open(target, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in self.summary_records:
                writer.writerow(row)
        logger.info("Saved comparison summary CSV to %s", target)

    def save_trials_csv(self, file_path: Union[Path, str]) -> None:
        """Persist combined candidate trials across all methods to CSV."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        for m_id, m_res in self.methods.items():
            for t in m_res.candidate_history:
                rows.append({
                    "method": m_id,
                    "trial_id": t.trial_id,
                    "iteration": t.iteration,
                    "chunk_size": t.config.chunk_size,
                    "chunk_overlap": t.config.chunk_overlap,
                    "status": t.status,
                    "objective_score": t.objective_score,
                    "decision_reason": t.decision_reason,
                    "execution_time_seconds": round(t.execution_time_seconds, 3),
                    "timestamp": t.timestamp,
                })

        if not rows:
            return

        headers = list(rows[0].keys())
        with open(target, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logger.info("Saved combined trials CSV to %s", target)

    @classmethod
    def load_from_json(cls, file_path: Union[Path, str]) -> ComparisonResult:
        """Load comparative experiment result from JSON."""
        target = Path(file_path)
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        methods_dict: Dict[str, ComparisonMethodResult] = {}
        for m_id, m_data in data.get("methods", {}).items():
            sel_cfg = ChunkConfig(m_data["selected_config"]["chunk_size"], m_data["selected_config"]["chunk_overlap"])
            base_cfg = ChunkConfig(m_data["baseline_config"]["chunk_size"], m_data["baseline_config"]["chunk_overlap"])

            history = []
            for t in m_data.get("candidate_history", []):
                cfg = ChunkConfig(t["config"]["chunk_size"], t["config"]["chunk_overlap"])
                history.append(
                    CandidateTrial(
                        trial_id=t["trial_id"],
                        iteration=t["iteration"],
                        config=cfg,
                        step_size=t["step_size"],
                        metrics=t["metrics"],
                        objective_score=t["objective_score"],
                        status=t["status"],
                        decision_reason=t["decision_reason"],
                        execution_time_seconds=t["execution_time_seconds"],
                        timestamp=t.get("timestamp", ""),
                    )
                )

            methods_dict[m_id] = ComparisonMethodResult(
                method_id=m_data["method_id"],
                display_name=m_data["display_name"],
                selected_config=sel_cfg,
                baseline_config=base_cfg,
                train_score=m_data["train_score"],
                train_metrics=m_data["train_metrics"],
                validation_score=m_data.get("validation_score"),
                validation_metrics=m_data.get("validation_metrics"),
                test_score=m_data["test_score"],
                test_metrics=m_data["test_metrics"],
                generalization_gap=m_data["generalization_gap"],
                total_trials=m_data["total_trials"],
                accepted_trials=m_data["accepted_trials"],
                execution_time_seconds=m_data["execution_time_seconds"],
                stopping_reason=m_data["stopping_reason"],
                candidate_history=history,
                cost_metrics=m_data.get("cost_metrics", {}),
                metadata=m_data.get("metadata", {}),
            )

        base_raw = data["baseline_config"]
        bounds_raw = data.get("search_bounds", {})

        return cls(
            experiment_id=data["experiment_id"],
            timestamp=data["timestamp"],
            dataset_id=data.get("dataset_id", "unknown"),
            methods=methods_dict,
            summary_records=data.get("summary_records", []),
            best_train_method=data.get("best_train_method", ""),
            best_test_method=data.get("best_test_method", ""),
            baseline_config=ChunkConfig(base_raw["chunk_size"], base_raw["chunk_overlap"]),
            search_bounds=SearchBounds(**bounds_raw) if bounds_raw else SearchBounds(),
            protocol_metadata=data.get("protocol_metadata", {}),
        )


class ComparisonRunner:
    """Executes controlled multi-method hyperparameter optimization benchmarks."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        samples: Optional[Sequence[EvaluationSample]] = None,
        dataset_path: Optional[Union[Path, str]] = None,
        train_samples: Optional[List[EvaluationSample]] = None,
        val_samples: Optional[List[EvaluationSample]] = None,
        test_samples: Optional[List[EvaluationSample]] = None,
        test_ratio: float = 0.20,
        val_ratio: float = 0.20,
        train_ratio: float = 0.60,
        random_seed: int = 42,
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_client: Optional[Union[BaseLLMClient, OllamaClient]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        collection_name: Optional[str] = None,
        results_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initialize comparison runner with controlled environment settings."""
        self.settings = settings or get_settings()
        self.random_seed = random_seed
        self.collection_name = collection_name or f"cmp_{int(time.time())}"
        self.results_dir = Path(results_dir or self.settings.paths.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.documents = list(documents or [])

        # Partition dataset
        if train_samples is not None and test_samples is not None:
            self.train_samples = list(train_samples)
            self.val_samples = list(val_samples or [])
            self.test_samples = list(test_samples)
            self.dataset_id = "prepartitioned"
        else:
            all_samples: List[EvaluationSample] = []
            if samples:
                all_samples = list(samples)
                self.dataset_id = "provided_samples"
            elif dataset_path:
                all_samples = load_dataset(dataset_path)
                self.dataset_id = Path(dataset_path).stem
            else:
                expert_path = self.settings.paths.evaluation_dir / "expert_qa_dataset.json"
                demo_path = self.settings.paths.evaluation_dir / "synthetic_demonstration_dataset.json"
                target_path = expert_path if expert_path.exists() else demo_path
                all_samples = load_dataset(target_path)
                self.dataset_id = target_path.stem

            splits: DatasetSplits = split_train_val_test(
                samples=all_samples,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=self.random_seed,
            )
            self.train_samples = splits.optimization
            self.val_samples = splits.validation
            self.test_samples = splits.held_out_test

        # Resolve pipeline components
        self.embedding_manager = embedding_manager or EmbeddingManager(
            model_name=self.settings.embedding.model_name,
            device=self.settings.embedding.device,
            batch_size=self.settings.embedding.batch_size,
        )

        self.llm_client = llm_client or OllamaClient(
            base_url=self.settings.llm.base_url,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
            temperature=self.settings.llm.temperature,
            seed=self.random_seed,
        )

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            vector_store = VectorStore(
                persist_dir=self.settings.vector_store.persist_dir,
                collection_name=self.collection_name,
                embedding_manager=self.embedding_manager,
            )
            retriever = Retriever(
                vector_store=vector_store,
                top_k=self.settings.retriever.top_k,
                similarity_threshold=self.settings.retriever.similarity_threshold,
            )
            self.pipeline = RAGPipeline(
                retriever=retriever,
                llm_client=self.llm_client,
                max_refinement_steps=self.settings.rerag.max_refinement_steps,
                verification_threshold=self.settings.rerag.verification_threshold,
            )

        self.evaluator = evaluator or Evaluator(
            pipeline=self.pipeline,
            embedding_manager=self.embedding_manager,
        )

        # Shared evaluation cache across strategies: (size, overlap) -> (metrics, score)
        self.evaluation_cache: Dict[Tuple[int, int], Tuple[Dict[str, Any], float]] = {}

    def _evaluate_config_on_samples(
        self,
        config: ChunkConfig,
        samples: Sequence[EvaluationSample],
        objective: OptimizationObjective,
    ) -> Tuple[Dict[str, Any], float]:
        """Re-index documents with candidate config and evaluate on specified sample split."""
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.pipeline.index_documents(
            documents=self.documents,
            chunker=chunker,
            collection_name=self.collection_name,
        )
        report = self.evaluator.evaluate(
            samples=samples,
            deterministic=True,
            config_override=config.to_dict(),
            collection_name=self.collection_name,
        )
        metrics = report.aggregate_metrics
        score = objective.calculate_score(metrics)
        return metrics, score

    def run(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        bounds: Optional[SearchBounds] = None,
        search_bounds: Optional[SearchBounds] = None,
        objective: Optional[OptimizationObjective] = None,
        methods: Sequence[str] = ("fixed", "greedy", "random", "simulated_annealing"),
        matched_budget: int = 10,
        save_results: bool = True,
        experiment_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> ComparisonResult:
        """Convenience alias for run_comparison with flexible keyword arguments."""
        effective_bounds = bounds or search_bounds
        return self.run_comparison(
            methods=methods,
            matched_budget=matched_budget,
            baseline_config=baseline_config,
            search_bounds=effective_bounds,
            objective=objective,
            save_results=save_results,
            experiment_name=experiment_name,
            progress_callback=progress_callback,
        )

    def run_comparison(
        self,
        methods: Sequence[str] = ("fixed", "greedy", "random", "simulated_annealing"),
        matched_budget: int = 10,
        baseline_config: Optional[ChunkConfig] = None,
        search_bounds: Optional[SearchBounds] = None,
        objective: Optional[OptimizationObjective] = None,
        save_results: bool = True,
        experiment_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
    ) -> ComparisonResult:
        """Run controlled benchmark comparison across optimization strategies.

        Args:
            methods: Sequence of method identifiers to compare.
            matched_budget: Number of candidate evaluations allocated per strategy.
            baseline_config: Reference starting configuration.
            search_bounds: Search boundary limits.
            objective: Multi-metric objective.
            save_results: Whether to persist outputs to disk.
            experiment_name: Optional custom prefix for saved artifacts.
            progress_callback: Optional progress reporter (method_name, current_step, total_steps, data).

        Returns:
            ComparisonResult: Comprehensive comparative report.
        """
        start_time = time.perf_counter()
        bounds = search_bounds or SearchBounds()
        obj = objective or OptimizationObjective()
        base_cfg = baseline_config or ChunkConfig(500, 50)

        timestamp_str = datetime.now(timezone.utc).isoformat()
        exp_id = experiment_name or f"comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            "Starting controlled multi-method comparison (Train: %d, Val: %d, Held-out Test: %d | Budget: %d)",
            len(self.train_samples),
            len(self.val_samples),
            len(self.test_samples),
            matched_budget,
        )

        method_results: Dict[str, ComparisonMethodResult] = {}
        total_methods = len(methods)

        for m_idx, method_key in enumerate(methods, start=1):
            m_key = method_key.lower().strip()
            logger.info("Executing method [%d/%d]: %s", m_idx, total_methods, m_key)

            if m_key in ("fixed", "baseline", "fixed_rag"):
                display_name = "Fixed Parameter RAG (Baseline)"
                t0 = time.perf_counter()

                # Evaluate baseline on train split
                train_metrics, train_score = self._evaluate_config_on_samples(base_cfg, self.train_samples, obj)
                # Evaluate baseline on val split if present
                val_metrics, val_score = (None, None)
                if self.val_samples:
                    val_metrics, val_score = self._evaluate_config_on_samples(base_cfg, self.val_samples, obj)
                # Evaluate baseline on held-out test split
                test_metrics, test_score = self._evaluate_config_on_samples(base_cfg, self.test_samples, obj)

                exec_time = time.perf_counter() - t0
                gen_gap = round(test_score - train_score, 4)

                trial_baseline = CandidateTrial(
                    trial_id=1,
                    iteration=1,
                    config=base_cfg,
                    step_size=0,
                    metrics=train_metrics,
                    objective_score=train_score,
                    status="baseline",
                    decision_reason="Static fixed configuration",
                    execution_time_seconds=exec_time,
                )

                method_results["fixed"] = ComparisonMethodResult(
                    method_id="fixed",
                    display_name=display_name,
                    selected_config=base_cfg,
                    baseline_config=base_cfg,
                    train_score=train_score,
                    train_metrics=train_metrics,
                    validation_score=val_score,
                    validation_metrics=val_metrics,
                    test_score=test_score,
                    test_metrics=test_metrics,
                    generalization_gap=gen_gap,
                    total_trials=1,
                    accepted_trials=1,
                    execution_time_seconds=exec_time,
                    stopping_reason="Static baseline evaluation",
                    candidate_history=[trial_baseline],
                    cost_metrics={
                        "chunks_indexed": len(self.documents),
                        "llm_calls": len(self.train_samples) + len(self.test_samples),
                        "cost_proxy": round(0.01 * exec_time, 4),
                    },
                    metadata={"strategy": "fixed"},
                )

            elif m_key in ("greedy", "adaptive", "coordinate", "original_rerag"):
                display_name = "Original ReRAG Greedy Coordinate Search"
                t0 = time.perf_counter()

                # Configure greedy search
                opt_greedy = ChunkOptimizerFactory.create(
                    strategy="greedy",
                    baseline_config=base_cfg,
                    search_bounds=bounds,
                    objective=obj,
                    random_seed=self.random_seed,
                    max_trials=matched_budget,
                    documents=self.documents,
                    pipeline=self.pipeline,
                    evaluator=self.evaluator,
                    optimization_samples=self.train_samples,
                    collection_name=self.collection_name,
                    dataset_id=f"train_n{len(self.train_samples)}",
                    evaluation_cache=self.evaluation_cache,
                )

                greedy_res = opt_greedy.optimize()
                exec_time = time.perf_counter() - t0

                # Post-selection held-out test evaluation (Zero leakage guarantee)
                test_metrics, test_score = self._evaluate_config_on_samples(greedy_res.selected_config, self.test_samples, obj)
                val_metrics, val_score = (None, None)
                if self.val_samples:
                    val_metrics, val_score = self._evaluate_config_on_samples(greedy_res.selected_config, self.val_samples, obj)

                gen_gap = round(test_score - greedy_res.selected_score, 4)

                method_results["greedy"] = ComparisonMethodResult(
                    method_id="greedy",
                    display_name=display_name,
                    selected_config=greedy_res.selected_config,
                    baseline_config=base_cfg,
                    train_score=greedy_res.selected_score,
                    train_metrics=greedy_res.selected_metrics,
                    validation_score=val_score,
                    validation_metrics=val_metrics,
                    test_score=test_score,
                    test_metrics=test_metrics,
                    generalization_gap=gen_gap,
                    total_trials=greedy_res.total_trials,
                    accepted_trials=greedy_res.accepted_trials,
                    execution_time_seconds=exec_time,
                    stopping_reason=greedy_res.stopping_reason,
                    candidate_history=greedy_res.candidate_history,
                    cost_metrics={
                        "chunks_indexed": greedy_res.total_trials * len(self.documents),
                        "llm_calls": greedy_res.total_trials * len(self.train_samples),
                        "cost_proxy": round(0.01 * exec_time + 0.005 * greedy_res.total_trials, 4),
                    },
                    metadata={"strategy": "greedy", "step_decay": 0.5},
                )

            elif m_key in ("random", "random_search"):
                display_name = "Random Parameter Search"
                t0 = time.perf_counter()

                opt_random = ChunkOptimizerFactory.create(
                    strategy="random",
                    baseline_config=base_cfg,
                    search_bounds=bounds,
                    objective=obj,
                    random_seed=self.random_seed,
                    max_trials=matched_budget,
                    documents=self.documents,
                    pipeline=self.pipeline,
                    evaluator=self.evaluator,
                    optimization_samples=self.train_samples,
                    collection_name=self.collection_name,
                    dataset_id=f"train_n{len(self.train_samples)}",
                    evaluation_cache=self.evaluation_cache,
                )

                rand_res = opt_random.optimize()
                exec_time = time.perf_counter() - t0

                # Post-selection held-out test evaluation
                test_metrics, test_score = self._evaluate_config_on_samples(rand_res.selected_config, self.test_samples, obj)
                val_metrics, val_score = (None, None)
                if self.val_samples:
                    val_metrics, val_score = self._evaluate_config_on_samples(rand_res.selected_config, self.val_samples, obj)

                gen_gap = round(test_score - rand_res.selected_score, 4)

                method_results["random"] = ComparisonMethodResult(
                    method_id="random",
                    display_name=display_name,
                    selected_config=rand_res.selected_config,
                    baseline_config=base_cfg,
                    train_score=rand_res.selected_score,
                    train_metrics=rand_res.selected_metrics,
                    validation_score=val_score,
                    validation_metrics=val_metrics,
                    test_score=test_score,
                    test_metrics=test_metrics,
                    generalization_gap=gen_gap,
                    total_trials=rand_res.total_trials,
                    accepted_trials=rand_res.accepted_trials,
                    execution_time_seconds=exec_time,
                    stopping_reason=rand_res.stopping_reason,
                    candidate_history=rand_res.candidate_history,
                    cost_metrics={
                        "chunks_indexed": rand_res.total_trials * len(self.documents),
                        "llm_calls": rand_res.total_trials * len(self.train_samples),
                        "cost_proxy": round(0.01 * exec_time + 0.005 * rand_res.total_trials, 4),
                    },
                    metadata={"strategy": "random"},
                )

            elif m_key in ("simulated_annealing", "sa", "annealing"):
                display_name = "Simulated Annealing"
                t0 = time.perf_counter()

                opt_sa = ChunkOptimizerFactory.create(
                    strategy="simulated_annealing",
                    baseline_config=base_cfg,
                    search_bounds=bounds,
                    objective=obj,
                    random_seed=self.random_seed,
                    max_trials=matched_budget,
                    documents=self.documents,
                    pipeline=self.pipeline,
                    evaluator=self.evaluator,
                    optimization_samples=self.train_samples,
                    collection_name=self.collection_name,
                    dataset_id=f"train_n{len(self.train_samples)}",
                    evaluation_cache=self.evaluation_cache,
                )

                sa_res = opt_sa.optimize()
                exec_time = time.perf_counter() - t0

                # Post-selection held-out test evaluation
                test_metrics, test_score = self._evaluate_config_on_samples(sa_res.selected_config, self.test_samples, obj)
                val_metrics, val_score = (None, None)
                if self.val_samples:
                    val_metrics, val_score = self._evaluate_config_on_samples(sa_res.selected_config, self.val_samples, obj)

                gen_gap = round(test_score - sa_res.selected_score, 4)

                method_results["simulated_annealing"] = ComparisonMethodResult(
                    method_id="simulated_annealing",
                    display_name=display_name,
                    selected_config=sa_res.selected_config,
                    baseline_config=base_cfg,
                    train_score=sa_res.selected_score,
                    train_metrics=sa_res.selected_metrics,
                    validation_score=val_score,
                    validation_metrics=val_metrics,
                    test_score=test_score,
                    test_metrics=test_metrics,
                    generalization_gap=gen_gap,
                    total_trials=sa_res.total_trials,
                    accepted_trials=sa_res.accepted_trials,
                    execution_time_seconds=exec_time,
                    stopping_reason=sa_res.stopping_reason,
                    candidate_history=sa_res.candidate_history,
                    cost_metrics={
                        "chunks_indexed": sa_res.total_trials * len(self.documents),
                        "llm_calls": sa_res.total_trials * len(self.train_samples),
                        "cost_proxy": round(0.01 * exec_time + 0.005 * sa_res.total_trials, 4),
                    },
                    metadata={"strategy": "simulated_annealing"},
                )

            if progress_callback:
                progress_callback(m_key, m_idx, total_methods, {"completed": m_key})

        # Compile comparative summary table
        summary_records: List[Dict[str, Any]] = []
        for m_id, m_res in method_results.items():
            t_met = m_res.test_metrics
            summary_records.append({
                "Method": m_res.display_name,
                "Method ID": m_id,
                "Chunk Size": m_res.selected_config.chunk_size,
                "Chunk Overlap": m_res.selected_config.chunk_overlap,
                "Train Objective": round(m_res.train_score, 4),
                "Held-Out Test Objective": round(m_res.test_score, 4),
                "Generalization Gap": round(m_res.generalization_gap, 4),
                "Test Faithfulness": round(t_met.get("mean_faithfulness_ratio") or 0.0, 4),
                "Test Context Coverage": round(t_met.get("mean_context_coverage") or 0.0, 4),
                "Test Semantic Sim": round(t_met.get("mean_semantic_similarity") or 0.0, 4),
                "Test Token F1": round(t_met.get("mean_token_f1") or 0.0, 4),
                "Total Trials": m_res.total_trials,
                "Accepted Trials": m_res.accepted_trials,
                "Execution Time (s)": round(m_res.execution_time_seconds, 2),
                "Cost Proxy": m_res.cost_metrics.get("cost_proxy", 0.0),
            })

        # Identify best methods
        best_train_m = max(method_results.items(), key=lambda item: item[1].train_score)[0] if method_results else ""
        best_test_m = max(method_results.items(), key=lambda item: item[1].test_score)[0] if method_results else ""

        comparison_output = ComparisonResult(
            experiment_id=exp_id,
            timestamp=timestamp_str,
            dataset_id=self.dataset_id,
            methods=method_results,
            summary_records=summary_records,
            best_train_method=best_train_m,
            best_test_method=best_test_m,
            baseline_config=base_cfg,
            search_bounds=bounds,
            protocol_metadata={
                "train_samples_count": len(self.train_samples),
                "val_samples_count": len(self.val_samples),
                "test_samples_count": len(self.test_samples),
                "documents_count": len(self.documents),
                "random_seed": self.random_seed,
                "matched_budget": matched_budget,
                "embedding_model": self.embedding_manager.model_name,
                "llm_model": getattr(self.llm_client, "model", "mock"),
                "total_wall_clock_seconds": round(time.perf_counter() - start_time, 2),
                "cached_configurations_count": len(self.evaluation_cache),
            },
        )

        # Persist results
        if save_results:
            json_file = self.results_dir / f"{exp_id}.json"
            summary_csv_file = self.results_dir / f"{exp_id}_summary.csv"
            trials_csv_file = self.results_dir / f"{exp_id}_trials.csv"

            comparison_output.save_to_json(json_file)
            comparison_output.save_summary_csv(summary_csv_file)
            comparison_output.save_trials_csv(trials_csv_file)

        return comparison_output
