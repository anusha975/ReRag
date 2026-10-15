"""Comprehensive unit and integration tests for ReRAG adaptive parameter optimization.

Tests cover:
1. Parameter validation, bounds constraints, and relational invariants (overlap < chunk_size).
2. Optimization objective calculation (composite weighting, single metric targeting, penalties).
3. Candidate generation and immediate rejection of invalid/boundary-violating configurations.
4. Candidate improvement acceptance (gain >= tolerance).
5. Candidate rejection and step size decay.
6. Stopping conditions (minimum step size reached, maximum iterations reached).
7. Graceful error handling for failed candidate evaluations without crashing the search.
8. Reproducibility across identical seeds.
9. Result serialization to JSON and CSV and roundtrip deserialization.
10. Integration with real RAG pipeline, document re-indexing, and evaluator.
11. Held-out test set isolation in ExperimentRunner.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.chunker import TextChunker
from src.document_loader import LoadedDocument, PageContent
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator
from src.experiment_runner import ExperimentRunner
from src.llm import MockLLMClient
from src.optimizer import (
    AdaptiveChunkOptimizer,
    AdaptiveSearchConfig,
    CandidateTrial,
    ChunkConfig,
    OptimizationObjective,
    OptimizationResult,
    ParameterGrid,
    ReRAGOptimizer,
    SearchBounds,
)
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore


def test_parameter_grid_defaults():
    """Verify default search space values in ParameterGrid (backwards compatibility)."""
    grid = ParameterGrid()
    assert 500 in grid.chunk_sizes
    assert 50 in grid.chunk_overlaps
    assert 4 in grid.top_ks
    assert 0.7 in grid.verification_thresholds


def test_rerag_optimizer_not_implemented():
    """Verify ReRAGOptimizer raises NotImplementedError on unbuilt methods."""
    optimizer = ReRAGOptimizer(evaluator=object(), parameter_grid=ParameterGrid())
    with pytest.raises(NotImplementedError):
        optimizer.run_grid_search([])


def test_chunk_config_validation_and_bounds():
    """Verify ChunkConfig rejects invalid dimensions, negative values, and overlap >= chunk_size."""
    # 1. Valid configuration
    cfg = ChunkConfig(chunk_size=500, chunk_overlap=50)
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert cfg.to_dict() == {"chunk_size": 500, "chunk_overlap": 50}

    # 2. Invalid chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=-100, chunk_overlap=0)

    # 3. Invalid chunk_overlap < 0
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        ChunkConfig(chunk_size=500, chunk_overlap=-10)

    # 4. Invalid overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        ChunkConfig(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        ChunkConfig(chunk_size=200, chunk_overlap=250)

    # 5. Boundary checks
    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=1000, min_chunk_overlap=0, max_chunk_overlap=200)
    valid_cfg = ChunkConfig(chunk_size=500, chunk_overlap=50)
    is_valid, _ = valid_cfg.is_valid_for_bounds(bounds)
    assert is_valid is True

    too_small = ChunkConfig(chunk_size=50, chunk_overlap=10)
    is_valid, reason = too_small.is_valid_for_bounds(bounds)
    assert is_valid is False
    assert "min_chunk_size" in reason

    too_large = ChunkConfig(chunk_size=1500, chunk_overlap=50)
    is_valid, reason = too_large.is_valid_for_bounds(bounds)
    assert is_valid is False
    assert "max_chunk_size" in reason


def test_search_bounds_validation():
    """Verify SearchBounds validates limits and prevents inverted or negative ranges."""
    # 1. Negative or zero min_chunk_size
    with pytest.raises(ValueError, match="min_chunk_size must be positive"):
        SearchBounds(min_chunk_size=0)

    # 2. Inverted chunk size limits
    with pytest.raises(ValueError, match="cannot be less than min_chunk_size"):
        SearchBounds(min_chunk_size=500, max_chunk_size=200)

    # 3. Negative min_chunk_overlap
    with pytest.raises(ValueError, match="min_chunk_overlap cannot be negative"):
        SearchBounds(min_chunk_overlap=-1)

    # 4. Inverted chunk overlap limits
    with pytest.raises(ValueError, match="cannot be less than min_chunk_overlap"):
        SearchBounds(min_chunk_overlap=200, max_chunk_overlap=100)

    # 5. Invalid step sizes
    with pytest.raises(ValueError, match="min_step_size must be positive"):
        SearchBounds(min_step_size=0)

    with pytest.raises(ValueError, match="min_overlap_step must be positive"):
        SearchBounds(min_overlap_step=0)


def test_optimization_objective_scoring():
    """Verify composite objective weighting, single-metric selection, and penalty handling."""
    objective = OptimizationObjective(
        weights={
            "faithfulness_ratio": 0.40,
            "context_coverage": 0.30,
            "semantic_similarity": 0.30,
        },
        penalty_weight=0.20,
    )

    metrics = {
        "faithfulness_ratio": 1.0,
        "context_coverage": 0.8,
        "semantic_similarity": 0.9,
        "hallucination_rate": 0.0,
    }
    # Expected: (0.40*1.0 + 0.30*0.8 + 0.30*0.9) / 1.0 = 0.4 + 0.24 + 0.27 = 0.91
    score = objective.calculate_score(metrics)
    assert score == pytest.approx(0.91, abs=0.01)

    # With hallucination penalty: 0.91 - (0.20 * 0.5) = 0.91 - 0.10 = 0.81
    penalized_metrics = dict(metrics)
    penalized_metrics["hallucination_rate"] = 0.5
    pen_score = objective.calculate_score(penalized_metrics)
    assert pen_score == pytest.approx(0.81, abs=0.01)

    # Test single-metric target
    single_obj = OptimizationObjective(target_metric="semantic_similarity")
    assert single_obj.calculate_score(metrics) == 0.9

    # Empty metrics returns 0.0
    assert objective.calculate_score({}) == 0.0


def test_optimizer_candidate_generation_and_rejection():
    """Verify coordinate exploration generates adjacent candidates and rejects boundary/invariant violations."""
    bounds = SearchBounds(min_chunk_size=200, max_chunk_size=800, min_chunk_overlap=0, max_chunk_overlap=100)
    cfg = AdaptiveSearchConfig(bounds=bounds)
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=cfg,
        evaluate_fn=lambda c: 0.5,
    )

    # 4 coordinate directions with step 100, overlap step 25:
    # 1. 600, 50 (valid)
    # 2. 400, 50 (valid)
    # 3. 500, 75 (valid)
    # 4. 500, 25 (valid)
    candidates = optimizer.generate_candidates(ChunkConfig(500, 50), step_size=100, overlap_step=25)
    valid_configs = [c[0] for c in candidates if c[0] is not None]
    assert len(valid_configs) == 4
    assert ChunkConfig(600, 50) in valid_configs
    assert ChunkConfig(400, 50) in valid_configs
    assert ChunkConfig(500, 75) in valid_configs
    assert ChunkConfig(500, 25) in valid_configs

    # Test rejection when candidate violates overlap < chunk_size
    edge_candidates = optimizer.generate_candidates(ChunkConfig(200, 50), step_size=100, overlap_step=50)
    # Direction -size gives chunk_size=100, overlap=50 (< min_chunk_size 200) -> invalid
    # Direction +overlap gives chunk_size=200, overlap=100 (valid)
    invalid_reasons = [c[2] for c in edge_candidates if c[0] is None]
    assert len(invalid_reasons) > 0


def test_optimizer_improvement_acceptance():
    """Verify optimizer accepts candidate that improves objective beyond tolerance."""
    # Mock evaluate function with predictable landscape:
    # Baseline (500, 50) -> 0.70
    # Candidate (600, 50) -> 0.85 (gain: 0.15 > tol 0.01) -> should be accepted!
    # Others -> 0.65
    scores = {
        (500, 50): 0.70,
        (600, 50): 0.85,
        (400, 50): 0.65,
        (500, 75): 0.68,
        (500, 25): 0.60,
    }

    def mock_eval(config: ChunkConfig) -> float:
        return scores.get((config.chunk_size, config.chunk_overlap), 0.50)

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=500,
        initial_chunk_overlap=50,
        initial_step_size=100,
        initial_overlap_step=25,
        improvement_tolerance=0.01,
        max_iterations=1,
    )
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=search_cfg,
        evaluate_fn=mock_eval,
    )

    result = optimizer.optimize()

    assert result.baseline_score == pytest.approx(0.70)
    assert result.selected_config == ChunkConfig(600, 50)
    assert result.selected_score == pytest.approx(0.85)
    assert result.improvement == pytest.approx(0.15)
    assert result.accepted_trials >= 1

    # Check candidate history records
    accepted_trials = [t for t in result.candidate_history if t.status == "accepted"]
    assert len(accepted_trials) == 1
    assert accepted_trials[0].config == ChunkConfig(600, 50)
    assert "Improved objective" in accepted_trials[0].decision_reason


def test_optimizer_rejection_and_step_size_reduction():
    """Verify candidates scoring lower than baseline are rejected and step size decays."""
    # Every candidate scores worse than baseline
    def mock_eval(config: ChunkConfig) -> float:
        if config.chunk_size == 500 and config.chunk_overlap == 50:
            return 0.80
        return 0.60

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=500,
        initial_chunk_overlap=50,
        initial_step_size=100,
        initial_overlap_step=25,
        step_decay=0.5,
        improvement_tolerance=0.01,
        max_iterations=2,
    )
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=search_cfg,
        evaluate_fn=mock_eval,
    )

    result = optimizer.optimize()

    # Selected should remain baseline because nothing improved
    assert result.selected_config == ChunkConfig(500, 50)
    assert result.selected_score == pytest.approx(0.80)
    assert result.improvement == 0.0

    # Iteration 1 candidates should all be marked rejected
    iter1_trials = [t for t in result.candidate_history if t.iteration == 1 and t.status == "rejected"]
    assert len(iter1_trials) > 0
    assert "did not exceed best" in iter1_trials[0].decision_reason

    # Iteration 2 trials should have decayed step size: 100 * 0.5 = 50
    iter2_trials = [t for t in result.candidate_history if t.iteration == 2]
    if iter2_trials:
        assert iter2_trials[0].step_size == 50


def test_optimizer_termination_min_step_size():
    """Verify optimizer terminates when step size falls below min_step_size."""
    bounds = SearchBounds(min_step_size=50)
    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=500,
        initial_chunk_overlap=50,
        initial_step_size=60,
        step_decay=0.5,  # Next step: 30 < min_step_size 50
        max_iterations=10,
        bounds=bounds,
    )

    # Flat landscape - no candidate improves
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=search_cfg,
        evaluate_fn=lambda c: 0.50,
    )

    result = optimizer.optimize()

    assert "below minimum" in result.stopping_reason.lower()
    assert result.total_iterations == 1


def test_optimizer_failed_evaluation_handling():
    """Verify optimizer handles evaluation exceptions without crashing or treating failure as improvement."""
    def faulty_eval(config: ChunkConfig) -> float:
        if config.chunk_size == 600:
            raise RuntimeError("Ollama connection timeout during candidate test")
        if config.chunk_size == 400:
            return 0.85
        return 0.70

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=500,
        initial_chunk_overlap=50,
        initial_step_size=100,
        max_iterations=1,
    )
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=search_cfg,
        evaluate_fn=faulty_eval,
    )

    result = optimizer.optimize()

    # Candidate 600 should be marked failed
    failed_trials = [t for t in result.candidate_history if t.status == "failed"]
    assert len(failed_trials) == 1
    assert "Ollama connection timeout" in failed_trials[0].decision_reason
    assert failed_trials[0].objective_score is None

    # Candidate 400 should be accepted
    assert result.selected_config == ChunkConfig(400, 50)
    assert result.selected_score == pytest.approx(0.85)


def test_optimizer_reproducibility():
    """Verify identical random seeds and evaluations produce identical trial histories and selections."""
    def noisy_eval(config: ChunkConfig) -> float:
        return 0.50 + ((config.chunk_size * 7 + config.chunk_overlap * 13) % 100) / 200.0

    cfg1 = AdaptiveSearchConfig(initial_chunk_size=500, initial_chunk_overlap=50, random_seed=999, max_iterations=2)
    cfg2 = AdaptiveSearchConfig(initial_chunk_size=500, initial_chunk_overlap=50, random_seed=999, max_iterations=2)

    res1 = AdaptiveChunkOptimizer(baseline_config=ChunkConfig(500, 50), search_config=cfg1, evaluate_fn=noisy_eval).optimize()
    res2 = AdaptiveChunkOptimizer(baseline_config=ChunkConfig(500, 50), search_config=cfg2, evaluate_fn=noisy_eval).optimize()

    assert res1.selected_config == res2.selected_config
    assert res1.selected_score == pytest.approx(res2.selected_score)
    assert len(res1.candidate_history) == len(res2.candidate_history)
    for t1, t2 in zip(res1.candidate_history, res2.candidate_history):
        assert t1.config == t2.config
        assert t1.status == t2.status
        assert t1.objective_score == t2.objective_score


def test_optimization_result_save_and_load_json_and_csv(tmp_path: Path):
    """Verify OptimizationResult JSON and CSV serialization and deserialization."""
    base_cfg = ChunkConfig(500, 50)
    sel_cfg = ChunkConfig(600, 50)
    trial = CandidateTrial(
        trial_id=1,
        iteration=0,
        config=base_cfg,
        step_size=100,
        metrics={"semantic_similarity": 0.8},
        objective_score=0.8,
        status="baseline",
        decision_reason="Baseline established",
        execution_time_seconds=1.23,
    )
    result = OptimizationResult(
        baseline_config=base_cfg,
        baseline_metrics={"semantic_similarity": 0.8},
        baseline_score=0.8,
        selected_config=sel_cfg,
        selected_metrics={"semantic_similarity": 0.9},
        selected_score=0.9,
        improvement=0.1,
        candidate_history=[trial],
        stopping_reason="Max iterations reached",
        total_iterations=1,
        total_trials=1,
        accepted_trials=0,
        total_execution_time_seconds=2.5,
        optimization_dataset_id="test_dataset",
    )

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    # Save
    result.save_to_json(json_path)
    result.save_to_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()

    # Load from JSON
    loaded = OptimizationResult.load_from_json(json_path)
    assert loaded.baseline_config == base_cfg
    assert loaded.selected_config == sel_cfg
    assert loaded.improvement == 0.1
    assert len(loaded.candidate_history) == 1
    assert loaded.candidate_history[0].config == base_cfg

    # Check CSV contents
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "trial_id,iteration,chunk_size,chunk_overlap" in csv_content
    assert "Baseline established" in csv_content


def test_optimizer_with_real_pipeline_and_evaluator():
    """Verify end-to-end integration: candidate evaluation actually rebuilds index from raw LoadedDocument."""
    # 1. Documents
    text_content = "Quantum computing leverages superposition and entanglement to solve discrete optimization problems."
    p1 = PageContent(page_number=1, text=text_content, char_count=len(text_content))
    docs = [
        LoadedDocument(
            content=text_content,
            metadata={"document_id": "doc_test", "title": "Quantum Computing"},
            source_path=Path("data/test.txt"),
            pages=[p1],
        )
    ]

    # 2. VectorStore & Retriever
    vector_store = VectorStore(persist_dir=None, collection_name="test_opt_integration")
    retriever = Retriever(vector_store=vector_store, top_k=2, similarity_threshold=0.0)

    # 3. Mock LLM
    mock_llm = MockLLMClient(
        default_response="Quantum computing leverages superposition to solve optimization problems."
    )

    # 4. Pipeline & Evaluator
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)
    evaluator = Evaluator(pipeline=pipeline)

    # 5. Training samples
    samples = [
        EvaluationSample(
            sample_id="q1",
            question="What does quantum computing leverage?",
            reference_answer="Quantum computing leverages superposition and entanglement.",
            document_id="doc_test",
        )
    ]

    # 6. Optimize with 1 iteration
    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=300,
        initial_chunk_overlap=30,
        initial_step_size=100,
        max_iterations=1,
    )
    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(300, 30),
        search_config=search_cfg,
        documents=docs,
        pipeline=pipeline,
        evaluator=evaluator,
        optimization_samples=samples,
        collection_name="test_opt_integration",
    )

    result = optimizer.optimize()

    assert result.baseline_score is not None
    assert result.selected_score is not None
    assert len(result.candidate_history) >= 2


def test_experiment_runner_held_out_isolation(tmp_path: Path, mock_embedder):
    """Verify ExperimentRunner keeps held-out test data isolated during search and evaluates it post-selection."""
    text_content = "Photosynthesis converts solar light into chemical sugars in plant chloroplasts."
    p1 = PageContent(page_number=1, text=text_content, char_count=len(text_content))
    docs = [
        LoadedDocument(
            content=text_content,
            metadata={"document_id": "doc_exp", "title": "Photosynthesis"},
            source_path=Path("data/exp.txt"),
            pages=[p1],
        )
    ]
    train_samples = [
        EvaluationSample(
            sample_id="train_1",
            question="What does photosynthesis convert?",
            reference_answer="Photosynthesis converts solar light into chemical sugars.",
            document_id="doc_exp",
        )
    ]
    test_samples = [
        EvaluationSample(
            sample_id="test_1",
            question="Where does photosynthesis occur?",
            reference_answer="Photosynthesis occurs in plant chloroplasts.",
            document_id="doc_exp",
        )
    ]

    vector_store = VectorStore(persist_dir=None, collection_name="test_runner_isolation", embedding_manager=mock_embedder)
    retriever = Retriever(vector_store=vector_store, embedding_manager=mock_embedder, top_k=2, similarity_threshold=0.0)
    mock_llm = MockLLMClient(default_response="Photosynthesis occurs in chloroplasts.")
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=mock_embedder)

    runner = ExperimentRunner(
        documents=docs,
        train_samples=train_samples,
        test_samples=test_samples,
        pipeline=pipeline,
        evaluator=evaluator,
        collection_name="test_runner_isolation",
        results_dir=tmp_path,
    )

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=300,
        initial_chunk_overlap=30,
        initial_step_size=50,
        max_iterations=1,
    )

    result = runner.run_experiment(
        baseline_config=ChunkConfig(300, 30),
        search_config=search_cfg,
        evaluate_held_out=True,
        save_results=True,
    )

    # 1. Assert training samples were used for optimization
    assert result.optimization_dataset_id == "train_split_n1"

    # 2. Assert held-out test results were recorded
    assert result.held_out_evaluation is not None
    assert result.held_out_evaluation["test_samples_count"] == 1
    assert "baseline_score" in result.held_out_evaluation
    assert "selected_score" in result.held_out_evaluation
    assert "held_out_gain" in result.held_out_evaluation

    # 3. Assert artifacts were persisted
    json_files = list(tmp_path.glob("*.json"))
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(json_files) == 1
    assert len(csv_files) == 1


def test_optimizer_resume_from_checkpoint(tmp_path: Path):
    """Verify an experiment can be checkpointed to JSON and resumed seamlessly."""
    scores = {
        (500, 50): 0.60,
        (600, 50): 0.75,
        (700, 50): 0.88,
    }

    def mock_eval(config: ChunkConfig) -> float:
        return scores.get((config.chunk_size, config.chunk_overlap), 0.50)

    # Phase 1: Run 1 iteration
    cfg1 = AdaptiveSearchConfig(
        initial_chunk_size=500,
        initial_chunk_overlap=50,
        initial_step_size=100,
        max_iterations=1,
    )
    opt1 = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=cfg1,
        evaluate_fn=mock_eval,
    )
    res1 = opt1.optimize()
    checkpoint_file = tmp_path / "checkpoint_phase1.json"
    res1.save_to_json(checkpoint_file)

    assert res1.selected_config == ChunkConfig(600, 50)
    assert res1.total_iterations == 1
    initial_trials_count = len(res1.candidate_history)

    # Phase 2: Resume from checkpoint for 1 additional iteration
    opt2 = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(500, 50),
        search_config=cfg1,
        evaluate_fn=mock_eval,
    )
    res2 = opt2.resume_from_result(checkpoint_file, additional_iterations=1)

    assert res2.total_iterations == 2
    assert len(res2.candidate_history) > initial_trials_count
    assert res2.selected_config == ChunkConfig(700, 50)
    assert res2.selected_score == pytest.approx(0.88)
    assert res2.improvement == pytest.approx(0.28)  # 0.88 - 0.60 baseline


def test_experiment_runner_resume_flow(tmp_path: Path, mock_embedder):
    """Verify ExperimentRunner can resume from an existing JSON checkpoint file."""
    text_content = "Machine learning algorithms learn representations from empirical data."
    p1 = PageContent(page_number=1, text=text_content, char_count=len(text_content))
    docs = [
        LoadedDocument(
            content=text_content,
            metadata={"document_id": "ml_doc"},
            source_path=Path("data/ml.txt"),
            pages=[p1],
        )
    ]
    train_samples = [
        EvaluationSample(
            sample_id="ml_1",
            question="What do machine learning algorithms learn?",
            reference_answer="Machine learning algorithms learn representations from data.",
            document_id="ml_doc",
        )
    ]

    vector_store = VectorStore(persist_dir=None, collection_name="test_runner_resume", embedding_manager=mock_embedder)
    retriever = Retriever(vector_store=vector_store, embedding_manager=mock_embedder, top_k=2, similarity_threshold=0.0)
    mock_llm = MockLLMClient(default_response="Algorithms learn representations.")
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=mock_embedder)

    runner = ExperimentRunner(
        documents=docs,
        train_samples=train_samples,
        test_samples=[],
        pipeline=pipeline,
        evaluator=evaluator,
        collection_name="test_runner_resume",
        results_dir=tmp_path,
    )

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=300,
        initial_chunk_overlap=30,
        initial_step_size=50,
        max_iterations=1,
    )

    # Initial experiment run
    initial_result = runner.run_experiment(
        search_config=search_cfg,
        evaluate_held_out=False,
        save_results=True,
        experiment_name="phase1_test",
    )
    checkpoint_json = tmp_path / "phase1_test.json"
    assert checkpoint_json.exists()
    assert initial_result.total_iterations == 1

    # Resume experiment run
    resumed_result = runner.resume_experiment(
        checkpoint_path=checkpoint_json,
        additional_iterations=1,
        evaluate_held_out=False,
        save_results=True,
        experiment_name="phase2_test",
    )

    assert resumed_result.total_iterations == 2
    assert (tmp_path / "phase2_test.json").exists()
    assert (tmp_path / "phase2_test.csv").exists()
