"""Automated unit and integration tests for RandomSearchOptimizer and Factory."""

import pytest

from src.optimizer import (
    CandidateTrial,
    ChunkConfig,
    ChunkOptimizerFactory,
    OptimizationObjective,
    OptimizationResult,
    RandomSearchConfig,
    RandomSearchOptimizer,
    SearchBounds,
    get_optimizer,
)


def dummy_eval_func(config: ChunkConfig):
    # Simulated objective: peak around (500, 100)
    score = 1.0 - (abs(config.chunk_size - 500) + abs(config.chunk_overlap - 100)) / 2000.0
    return {"faithfulness_ratio": max(0.0, score), "context_coverage": max(0.0, score), "semantic_similarity": max(0.0, score)}, max(0.0, score)


def test_random_search_config_validation():
    """Verify RandomSearchConfig validates search bounds and budgets."""
    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=1000, min_chunk_overlap=0, max_chunk_overlap=300)
    obj = OptimizationObjective()

    cfg = RandomSearchConfig(max_trials=10, bounds=bounds, objective=obj, random_seed=42)
    assert cfg.max_trials == 10
    assert cfg.random_seed == 42

    with pytest.raises(ValueError, match="at least 1"):
        RandomSearchConfig(max_trials=0, bounds=bounds, objective=obj)


def test_random_search_optimizer_execution():
    """Verify RandomSearchOptimizer explores within boundaries and selects the best candidate."""
    bounds = SearchBounds(min_chunk_size=200, max_chunk_size=800, min_chunk_overlap=25, max_chunk_overlap=200, min_step_size=25)
    obj = OptimizationObjective()
    cfg = RandomSearchConfig(max_trials=8, bounds=bounds, objective=obj, random_seed=42)

    optimizer = RandomSearchOptimizer(config=cfg, evaluate_fn=dummy_eval_func)
    baseline = ChunkConfig(300, 50)

    result = optimizer.optimize(baseline_config=baseline)

    assert isinstance(result, OptimizationResult)
    assert result.total_trials == 8
    assert len(result.candidate_history) == 8
    assert result.selected_score >= result.baseline_score or result.selected_config == baseline

    # Check that all candidates satisfy bounds and overlap < size
    for trial in result.candidate_history:
        c = trial.config
        assert bounds.min_chunk_size <= c.chunk_size <= bounds.max_chunk_size
        assert bounds.min_chunk_overlap <= c.chunk_overlap <= bounds.max_chunk_overlap
        assert c.chunk_overlap < c.chunk_size


def test_random_search_evaluation_cache():
    """Verify that RandomSearchOptimizer reuses evaluation cache if duplicate configs are sampled."""
    bounds = SearchBounds(min_chunk_size=200, max_chunk_size=250, min_chunk_overlap=25, max_chunk_overlap=50, min_step_size=25)
    obj = OptimizationObjective()
    cfg = RandomSearchConfig(max_trials=10, bounds=bounds, objective=obj, random_seed=42)

    cache = {}
    eval_call_count = 0

    def counting_eval(config: ChunkConfig):
        nonlocal eval_call_count
        eval_call_count += 1
        return dummy_eval_func(config)

    optimizer = RandomSearchOptimizer(config=cfg, evaluate_fn=counting_eval, evaluation_cache=cache)
    baseline = ChunkConfig(200, 25)
    optimizer.optimize(baseline_config=baseline)

    # In a tiny search space with 10 trials, there must be duplicate configurations cached
    assert len(cache) < 10
    assert eval_call_count == len(cache)


def test_chunk_optimizer_factory():
    """Verify that ChunkOptimizerFactory correctly instantiates all supported optimizer strategies."""
    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=1000, min_chunk_overlap=0, max_chunk_overlap=200)
    obj = OptimizationObjective()

    # 1. Greedy
    opt_greedy = ChunkOptimizerFactory.create(
        strategy="greedy",
        evaluate_fn=dummy_eval_func,
        bounds=bounds,
        objective=obj,
    )
    assert opt_greedy.__class__.__name__ == "AdaptiveChunkOptimizer"

    # 2. Random Search
    opt_random = ChunkOptimizerFactory.create(
        strategy="random_search",
        evaluate_fn=dummy_eval_func,
        bounds=bounds,
        objective=obj,
        max_trials=5,
    )
    assert opt_random.__class__.__name__ == "RandomSearchOptimizer"

    # 3. Simulated Annealing
    opt_sa = ChunkOptimizerFactory.create(
        strategy="simulated_annealing",
        evaluate_fn=dummy_eval_func,
        bounds=bounds,
        objective=obj,
        max_iterations=5,
    )
    assert opt_sa.__class__.__name__ == "SimulatedAnnealingOptimizer"

    # 4. Unknown strategy raises ValueError
    with pytest.raises(ValueError, match="Unknown optimization strategy"):
        ChunkOptimizerFactory.create(
            strategy="quantum_annealing",
            evaluate_fn=dummy_eval_func,
            bounds=bounds,
            objective=obj,
        )
