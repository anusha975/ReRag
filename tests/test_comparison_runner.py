"""Automated tests for ComparisonRunner and multi-method empirical comparisons."""

import json
from pathlib import Path
import pytest

from src.comparison_runner import ComparisonMethodResult, ComparisonResult, ComparisonRunner
from src.config import Settings
from src.document_loader import LoadedDocument
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator
from src.llm import MockLLMClient
from src.optimizer import ChunkConfig, OptimizationObjective, SearchBounds
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore
from tests.conftest import DeterministicMockEmbeddingManager


@pytest.fixture
def test_setup(tmp_path):
    """Fixture providing isolated documents, mock LLM, vector store, and samples."""
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(
        "ReRAG is an adaptive chunking framework designed to reduce hallucinations in RAG pipelines. "
        "It balances faithfulness, context coverage, and semantic similarity. "
        "Traditional RAG pipelines use arbitrary fixed chunk sizes and suffer from hallucination.",
        encoding="utf-8",
    )
    content = doc_path.read_text(encoding="utf-8")
    from src.document_loader import PageContent
    page = PageContent(page_number=1, text=content, char_count=len(content))
    doc = LoadedDocument(
        content=content,
        metadata={"title": "Test Paper", "file_type": "txt"},
        source_path=doc_path,
        pages=[page],
    )

    dataset_path = tmp_path / "test_qa.json"
    qa_data = [
        {
            "sample_id": f"q_{i}",
            "question": f"Question {i} about ReRAG adaptive chunking",
            "ground_truth_answer": "ReRAG adapts chunk size to reduce hallucinations.",
            "expected_passages": ["ReRAG is an adaptive chunking framework"],
            "expected_citations": [1],
            "difficulty": "medium",
            "is_negative": False,
        }
        for i in range(12)
    ]
    dataset_path.write_text(json.dumps(qa_data), encoding="utf-8")

    embed_mgr = DeterministicMockEmbeddingManager()
    vs = VectorStore(persist_dir=None, collection_name=f"test_comp_{tmp_path.name}", embedding_manager=embed_mgr)
    retriever = Retriever(vector_store=vs, embedding_manager=embed_mgr, top_k=2, similarity_threshold=0.0)
    llm = MockLLMClient(default_response="ReRAG adapts chunk size to reduce hallucinations.")
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=embed_mgr)

    settings = Settings()

    return {
        "settings": settings,
        "doc": doc,
        "dataset_path": dataset_path,
        "pipeline": pipeline,
        "evaluator": evaluator,
        "results_dir": tmp_path / "results",
    }


def test_comparison_runner_execution(test_setup):
    """Verify ComparisonRunner runs all 4 methods with data isolation and saves outputs."""
    runner = ComparisonRunner(
        settings=test_setup["settings"],
        documents=[test_setup["doc"]],
        dataset_path=test_setup["dataset_path"],
        train_ratio=0.50,
        val_ratio=0.20,
        test_ratio=0.30,
        random_seed=42,
        pipeline=test_setup["pipeline"],
        evaluator=test_setup["evaluator"],
        results_dir=test_setup["results_dir"],
    )

    bounds = SearchBounds(min_chunk_size=100, max_chunk_size=400, min_chunk_overlap=0, max_chunk_overlap=100, min_step_size=50)
    obj = OptimizationObjective()

    comp_result = runner.run(
        methods=("fixed", "greedy", "random", "simulated_annealing"),
        baseline_config=ChunkConfig(200, 50),
        bounds=bounds,
        objective=obj,
        greedy_kwargs={"max_iterations": 2, "initial_step_size": 50},
        random_search_kwargs={"max_trials": 3, "random_seed": 42},
        simulated_annealing_kwargs={"initial_temp": 1.0, "cooling_rate": 0.8, "max_iterations": 2},
        save_results=True,
    )

    assert isinstance(comp_result, ComparisonResult)
    assert len(comp_result.methods) == 4
    assert "fixed" in comp_result.methods
    assert "greedy" in comp_result.methods
    assert "random" in comp_result.methods
    assert "simulated_annealing" in comp_result.methods

    # Check that summary records contain all methods
    assert len(comp_result.summary_records) == 4

    # Check best method identification
    assert comp_result.best_train_method in comp_result.methods
    assert comp_result.best_test_method in comp_result.methods

    # Verify JSON persistence and load_from_json
    json_files = list(test_setup["results_dir"].glob("*.json"))
    assert len(json_files) >= 1
    loaded = ComparisonResult.load_from_json(json_files[0])
    assert loaded.experiment_id == comp_result.experiment_id
    assert len(loaded.methods) == 4
    assert loaded.best_test_method == comp_result.best_test_method

    # Verify CSV files exist
    summary_csvs = list(test_setup["results_dir"].glob("*_summary.csv"))
    trials_csvs = list(test_setup["results_dir"].glob("*_trials.csv"))
    assert len(summary_csvs) >= 1
    assert len(trials_csvs) >= 1
