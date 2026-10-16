"""End-to-end experiment runner for ReRAG adaptive parameter optimization.

Coordinates document loading, benchmark dataset splitting, pipeline instantiation,
adaptive parameter search, held-out generalization testing, and result persistence.

Key Research Constraints Enforced:
1. Strict Dataset Isolation: Optimization questions (training) are strictly partitioned
   from held-out test questions. Candidate selection NEVER accesses held-out test data.
2. Independent Generalization Testing: After search completes, both the baseline configuration
   and selected configuration are evaluated on held-out test data to verify whether optimization
   gains transfer beyond the training distribution.
3. Persistent Reproducibility: All random seeds, candidate trials, execution times,
   and decision logs are written to versioned JSON and CSV artifacts.
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.chunker import TextChunker
from src.config import Settings, get_settings
from src.document_loader import DocumentLoader, LoadedDocument
from src.embeddings import EmbeddingManager
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator, load_dataset, split_train_test
from src.llm import BaseLLMClient, OllamaClient
from src.optimizer import (
    AdaptiveChunkOptimizer,
    AdaptiveSearchConfig,
    CandidateTrial,
    ChunkConfig,
    OptimizationObjective,
    OptimizationResult,
    RandomSearchConfig,
    RandomSearchOptimizer,
    SearchBounds,
    SimulatedAnnealingConfig,
    SimulatedAnnealingOptimizer,
)
from src.comparison_runner import (
    ComparisonMethodResult,
    ComparisonResult,
    ComparisonRunner,
)
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore

# Global HF cache redirect for environments with limited C: drive capacity
if "HF_HOME" not in os.environ and Path(r"D:\hf_cache").exists():
    os.environ["HF_HOME"] = r"D:\hf_cache"

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages end-to-end adaptive parameter optimization experiments."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        documents: Optional[Sequence[LoadedDocument]] = None,
        samples: Optional[Sequence[EvaluationSample]] = None,
        dataset_path: Optional[Union[Path, str]] = None,
        train_samples: Optional[List[EvaluationSample]] = None,
        test_samples: Optional[List[EvaluationSample]] = None,
        test_ratio: float = 0.33,
        random_seed: int = 42,
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_client: Optional[Union[BaseLLMClient, OllamaClient]] = None,
        pipeline: Optional[RAGPipeline] = None,
        evaluator: Optional[Evaluator] = None,
        collection_name: Optional[str] = None,
        results_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initialize experiment runner.

        Args:
            settings: Optional system configuration settings.
            documents: Sequence of loaded raw documents.
            samples: Full evaluation dataset (will be split if train/test not provided).
            dataset_path: Path to load evaluation dataset if samples not provided.
            train_samples: Pre-partitioned training evaluation samples.
            test_samples: Pre-partitioned held-out test evaluation samples.
            test_ratio: Ratio for train/test split if partitioning automatically.
            random_seed: Fixed seed for reproducible splits and operations.
            embedding_manager: Shared embedding manager instance.
            llm_client: Configured LLM client.
            pipeline: Pre-configured RAG pipeline instance.
            evaluator: Pre-configured Evaluator instance.
            collection_name: Unique collection identifier for vector storage.
            results_dir: Directory to save experimental JSON and CSV outputs.
        """
        self.settings = settings or get_settings()
        self.random_seed = random_seed
        self.collection_name = collection_name or f"exp_{int(time.time())}"
        self.results_dir = Path(results_dir or self.settings.paths.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set deterministic seeds
        random.seed(self.random_seed)

        # 1. Resolve documents
        self.documents: List[LoadedDocument] = list(documents or [])

        # 2. Resolve dataset & splits
        if train_samples is not None:
            self.train_samples = list(train_samples)
            self.test_samples = list(test_samples or [])
        else:
            all_samples: List[EvaluationSample] = []
            if samples:
                all_samples = list(samples)
            elif dataset_path:
                all_samples = load_dataset(dataset_path)
            else:
                demo_path = self.settings.paths.evaluation_dir / "synthetic_demonstration_dataset.json"
                if demo_path.exists():
                    all_samples = load_dataset(demo_path)

            if all_samples and test_ratio > 0.0:
                self.train_samples, self.test_samples = split_train_test(
                    all_samples, test_ratio=test_ratio, seed=self.random_seed
                )
            else:
                self.train_samples = all_samples
                self.test_samples = []

        # 3. Resolve pipeline components
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

    def run_experiment(
        self,
        baseline_config: Optional[ChunkConfig] = None,
        search_config: Optional[AdaptiveSearchConfig] = None,
        evaluate_held_out: bool = True,
        save_results: bool = True,
        experiment_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, CandidateTrial], None]] = None,
    ) -> OptimizationResult:
        """Execute end-to-end adaptive parameter optimization experiment.

        Workflow:
        1. Initialize optimizer with training samples only.
        2. Run adaptive coordinate search across chunk size and overlap.
        3. Evaluate selected best configuration and baseline on held-out test data.
        4. Measure generalization gap (training improvement vs held-out improvement).
        5. Persist comprehensive experiment report to JSON and CSV.

        Args:
            baseline_config: Initial chunking configuration.
            search_config: Search boundaries and parameters.
            evaluate_held_out: Whether to test baseline and selected config on held-out data.
            save_results: Whether to persist results to disk.
            experiment_name: Optional custom prefix for output files.

        Returns:
            OptimizationResult: Complete optimization outcome and candidate history.
        """
        logger.info(
            "Starting ReRAG optimization experiment (Train samples: %d, Held-out test samples: %d)",
            len(self.train_samples),
            len(self.test_samples),
        )

        effective_search_cfg = search_config or AdaptiveSearchConfig(random_seed=self.random_seed)
        effective_base_cfg = baseline_config or ChunkConfig(
            effective_search_cfg.initial_chunk_size,
            effective_search_cfg.initial_chunk_overlap,
        )

        # 1. Instantiate optimizer restricted strictly to training samples
        optimizer = AdaptiveChunkOptimizer(
            baseline_config=effective_base_cfg,
            search_config=effective_search_cfg,
            documents=self.documents,
            pipeline=self.pipeline,
            evaluator=self.evaluator,
            optimization_samples=self.train_samples,
            collection_name=self.collection_name,
            dataset_id=f"train_split_n{len(self.train_samples)}",
        )

        # 2. Execute optimization sweep
        result = optimizer.optimize(progress_callback=progress_callback)

        # 3. Post-selection held-out evaluation (if configured and test data available)
        if evaluate_held_out and self.test_samples and self.documents:
            logger.info("Evaluating generalization on held-out test set (N=%d)...", len(self.test_samples))
            held_out_results = self._evaluate_generalization_on_held_out(
                baseline_config=result.baseline_config,
                selected_config=result.selected_config,
                objective=effective_search_cfg.objective,
            )
            result.held_out_evaluation = held_out_results

        # 4. Persist results
        if save_results:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            prefix = experiment_name or f"rerag_opt_{timestamp}"
            json_path = self.results_dir / f"{prefix}.json"
            csv_path = self.results_dir / f"{prefix}.csv"

            result.save_to_json(json_path)
            result.save_to_csv(csv_path)

        return result

    def resume_experiment(
        self,
        checkpoint_path: Union[Path, str],
        additional_iterations: int = 5,
        evaluate_held_out: bool = True,
        save_results: bool = True,
        experiment_name: Optional[str] = None,
    ) -> OptimizationResult:
        """Resume an existing experiment from a checkpoint JSON file.

        Args:
            checkpoint_path: Path to previous experiment JSON result.
            additional_iterations: Extra iterations to run beyond original count.
            evaluate_held_out: Whether to evaluate generalization on held-out test data post-selection.
            save_results: Whether to persist updated results.
            experiment_name: Optional name prefix for saved files.

        Returns:
            OptimizationResult: Updated cumulative result.
        """
        prev_result = OptimizationResult.load_from_json(checkpoint_path)
        effective_search_cfg = AdaptiveSearchConfig(random_seed=self.random_seed)

        optimizer = AdaptiveChunkOptimizer(
            baseline_config=prev_result.baseline_config,
            search_config=effective_search_cfg,
            documents=self.documents,
            pipeline=self.pipeline,
            evaluator=self.evaluator,
            optimization_samples=self.train_samples,
            collection_name=self.collection_name,
            dataset_id=f"train_split_n{len(self.train_samples)}",
        )

        result = optimizer.resume_from_result(
            previous_result=prev_result,
            additional_iterations=additional_iterations,
        )

        if evaluate_held_out and self.test_samples and self.documents:
            logger.info("Evaluating generalization on held-out test set (N=%d)...", len(self.test_samples))
            held_out_results = self._evaluate_generalization_on_held_out(
                baseline_config=result.baseline_config,
                selected_config=result.selected_config,
                objective=effective_search_cfg.objective,
            )
            result.held_out_evaluation = held_out_results

        if save_results:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            prefix = experiment_name or f"rerag_opt_resumed_{timestamp}"
            json_path = self.results_dir / f"{prefix}.json"
            csv_path = self.results_dir / f"{prefix}.csv"

            result.save_to_json(json_path)
            result.save_to_csv(csv_path)

        return result

    def _evaluate_generalization_on_held_out(
        self,
        baseline_config: ChunkConfig,
        selected_config: ChunkConfig,
        objective: OptimizationObjective,
    ) -> Dict[str, Any]:
        """Evaluate baseline and selected configuration on held-out test samples.

        Ensures clean vector store re-indexing between evaluations.
        """
        # A. Baseline on held-out test
        logger.info("Evaluating baseline on held-out test set: %s", baseline_config)
        chunker_base = TextChunker(
            chunk_size=baseline_config.chunk_size,
            chunk_overlap=baseline_config.chunk_overlap,
        )
        self.pipeline.index_documents(self.documents, chunker_base, self.collection_name)
        report_base = self.evaluator.evaluate(
            samples=self.test_samples,
            deterministic=True,
            config_override=baseline_config.to_dict(),
            collection_name=self.collection_name,
        )
        base_score = objective.calculate_score(report_base.aggregate_metrics)

        # B. Selected on held-out test
        logger.info("Evaluating selected config on held-out test set: %s", selected_config)
        chunker_sel = TextChunker(
            chunk_size=selected_config.chunk_size,
            chunk_overlap=selected_config.chunk_overlap,
        )
        self.pipeline.index_documents(self.documents, chunker_sel, self.collection_name)
        report_sel = self.evaluator.evaluate(
            samples=self.test_samples,
            deterministic=True,
            config_override=selected_config.to_dict(),
            collection_name=self.collection_name,
        )
        sel_score = objective.calculate_score(report_sel.aggregate_metrics)

        held_out_gain = round(sel_score - base_score, 4)

        logger.info(
            "Held-out Generalization: Baseline Score=%.4f -> Selected Score=%.4f (Gain: %+.4f)",
            base_score,
            sel_score,
            held_out_gain,
        )

        return {
            "test_samples_count": len(self.test_samples),
            "baseline_config": baseline_config.to_dict(),
            "baseline_metrics": report_base.aggregate_metrics,
            "baseline_score": base_score,
            "selected_config": selected_config.to_dict(),
            "selected_metrics": report_sel.aggregate_metrics,
            "selected_score": sel_score,
            "held_out_gain": held_out_gain,
        }

    def run_comparative_experiment(
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

        Executes fixed-parameter baseline, original ReRAG greedy search, random search,
        and simulated annealing under identical experimental conditions and evaluates
        each selected configuration on held-out test data.

        Returns:
            ComparisonResult: Multi-method comparison outcome and summary tables.
        """
        runner = ComparisonRunner(
            settings=self.settings,
            documents=self.documents,
            train_samples=self.train_samples,
            test_samples=self.test_samples,
            random_seed=self.random_seed,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            pipeline=self.pipeline,
            evaluator=self.evaluator,
            collection_name=self.collection_name,
            results_dir=self.results_dir,
        )
        return runner.run_comparison(
            methods=methods,
            matched_budget=matched_budget,
            baseline_config=baseline_config,
            search_bounds=search_bounds,
            objective=objective,
            save_results=save_results,
            experiment_name=experiment_name,
            progress_callback=progress_callback,
        )

