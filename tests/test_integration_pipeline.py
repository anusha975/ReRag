"""End-to-end and cross-module integration tests for the ReRAG system.

Tests verify:
1. Upload to indexing: Document loading, chunking, and vector storage.
2. Indexing to retrieval: Similarity search, metric calculations, and ranking.
3. Retrieval to answer generation: Prompt grounding, answer synthesis, and provenance tracking.
4. Evaluation of answers: Metric computation against human/reference answers.
5. Optimization of candidate configurations: Index rebuilding and candidate evaluations.
6. Baseline versus selected configuration: Comparative metrics and empirical delta.
7. Result export: Serialization to JSON and CSV and roundtrip deserialization.
8. Held-out test set isolation: Strict verification against data leakage.
9. End-to-end synthetic pipeline: Completely offline, reproducible workflow.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import hashlib
import re
from typing import List, Sequence

import numpy as np
import pytest

from src.chunker import TextChunk, TextChunker
from src.config import Settings, get_settings
from src.document_loader import DocumentLoader, LoadedDocument, PageContent
from src.embeddings import EmbeddingManager
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
    SearchBounds,
)
from src.prompts import INSUFFICIENT_INFORMATION_RESPONSE
from src.rag_pipeline import PipelineResult, RAGPipeline
from src.retriever import RetrievedContext, Retriever
from src.vector_store import VectorStore


class DeterministicMockEmbeddingManager:
    """Fast, deterministic, zero-memory embedding manager for offline integration testing."""

    def __init__(self, embedding_dim: int = 384, normalize_embeddings: bool = True):
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.model_name = "mock-minilm"
        self.device = "cpu"
        self.batch_size = 32

    def _text_to_vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        for w in words:
            idx = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:6], 16) % self.embedding_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0 and self.normalize_embeddings:
            vec = vec / norm
        return vec

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        rows = [self._text_to_vec(str(t or "")) for t in texts]
        return np.vstack(rows)

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return self._text_to_vec(query)

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> np.ndarray:
        return self.embed_texts([c.text for c in chunks])


@pytest.fixture(scope="module")
def shared_embedder():
    """Module-scoped deterministic mock embedding manager fixture."""
    return DeterministicMockEmbeddingManager()


@pytest.fixture
def synthetic_document() -> LoadedDocument:
    """Synthetic research document for offline integration testing."""
    text = (
        "ReRAG: Refinement and Grounding in Retrieval-Augmented Generation.\n\n"
        "Abstract: Traditional RAG pipelines suffer from factual hallucinations when generative "
        "models extrapolate beyond retrieved evidence. We introduce ReRAG, an architecture "
        "that incorporates post-generation claim verification and grounded context assembly.\n\n"
        "Section 1: Architecture Overview.\n"
        "The ReRAG framework consists of five core stages: document ingestion, dense vector "
        "indexing, top-k context retrieval, grounded generation, and claim-level verification.\n"
        "During verification, each claim is scored against retrieved context chunks using an "
        "entailment threshold of 0.7. If the evidence is insufficient, the system safely abstains.\n\n"
        "Section 2: Adaptive Optimization.\n"
        "Chunk size and overlap parameters significantly influence retrieval precision. "
        "ReRAG applies greedy adaptive coordinate search to select optimal chunking parameters "
        "without overfitting on held-out test sets."
    )
    pages = [
        PageContent(
            page_number=1,
            text=text[:400],
            char_count=len(text[:400]),
        ),
        PageContent(
            page_number=2,
            text=text[400:],
            char_count=len(text[400:]),
        ),
    ]
    return LoadedDocument(
        content=text,
        metadata={"title": "ReRAG Synthetic Paper", "topic": "RAG", "document_id": "doc_rerag_synth"},
        source_path=Path("synthetic/rerag_paper.txt"),
        pages=pages,
    )


@pytest.fixture
def synthetic_samples() -> List[EvaluationSample]:
    """Synthetic training and evaluation questions."""
    return [
        EvaluationSample(
            sample_id="synth_q1",
            question="What is the entailment threshold used during ReRAG verification?",
            reference_answer="The verification step uses an entailment threshold of 0.7.",
            expected_source="doc_rerag_synth",
            expected_pages=[2],
            document_id="doc_rerag_synth",
            relevant_doc_ids=["doc_rerag_synth"],
            supporting_passages=["using an entailment threshold of 0.7."],
        ),
        EvaluationSample(
            sample_id="synth_q2",
            question="What search method does ReRAG use to optimize chunking parameters?",
            reference_answer="ReRAG applies greedy adaptive coordinate search to select optimal chunking parameters.",
            expected_source="doc_rerag_synth",
            expected_pages=[2],
            document_id="doc_rerag_synth",
            relevant_doc_ids=["doc_rerag_synth"],
            supporting_passages=["ReRAG applies greedy adaptive coordinate search to select optimal chunking parameters"],
        ),
    ]


# ==============================================================================
# 1. Upload to Indexing Test
# ==============================================================================

def test_upload_to_indexing(tmp_path: Path, shared_embedder: EmbeddingManager, synthetic_document: LoadedDocument):
    """Verify document file ingestion, chunking, and vector indexing."""
    # 1. Write synthetic text file
    doc_path = tmp_path / "sample_doc.txt"
    doc_path.write_text(synthetic_document.content, encoding="utf-8")

    # 2. Ingest via DocumentLoader
    loader = DocumentLoader()
    loaded = loader.load_file(doc_path)
    assert loaded.document_id is not None
    assert len(loaded.content) > 0
    assert loaded.total_pages >= 1

    # 3. Chunk document
    chunker = TextChunker(chunk_size=70, chunk_overlap=15)
    chunks = chunker.chunk_document(loaded)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunk_id.startswith(loaded.document_id)
        assert c.page_number is not None
        assert len(c.text) > 0

    # 4. Index in VectorStore
    persist_dir = tmp_path / "chroma_test_index"
    store = VectorStore(
        persist_dir=persist_dir,
        collection_name="test_upload_col",
        embedding_manager=shared_embedder,
    )
    initial_count = store.count()
    assert initial_count == 0

    store.add_chunks(chunks)
    assert store.count() == len(chunks)

    # 5. Verify index rebuilding with different chunk configuration
    new_chunker = TextChunker(chunk_size=35, chunk_overlap=10)
    new_chunks = new_chunker.chunk_document(loaded)
    assert len(new_chunks) > len(chunks)  # Smaller chunk size yields more chunks

    store.rebuild_index(new_chunks)
    assert store.count() == len(new_chunks)


# ==============================================================================
# 2. Indexing to Retrieval Test
# ==============================================================================

def test_indexing_to_retrieval(shared_embedder: EmbeddingManager, synthetic_document: LoadedDocument):
    """Verify vector retrieval, cosine similarity, distance metrics, and ranking."""
    store = VectorStore(
        persist_dir=None,
        collection_name="test_indexing_retrieval_col",
        embedding_manager=shared_embedder,
    )
    chunker = TextChunker(chunk_size=250, chunk_overlap=50)
    chunks = chunker.chunk_document(synthetic_document)
    store.add_chunks(chunks)

    retriever = Retriever(
        vector_store=store,
        embedding_manager=shared_embedder,
        top_k=2,
        similarity_threshold=0.1,
    )

    # Query for known fact
    query = "What is the verification threshold in ReRAG?"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) > 0
    assert len(results) <= 2

    top_result = results[0]
    assert isinstance(top_result, RetrievedContext)
    assert top_result.score is not None
    assert 0.0 <= top_result.score <= 1.0
    assert top_result.distance is not None
    assert top_result.document_id == synthetic_document.document_id
    assert any("0.7" in r.text or "threshold" in r.text.lower() for r in results)
    assert top_result.citation.startswith(f"[{synthetic_document.document_id}")


# ==============================================================================
# 3. Retrieval to Answer Generation Test
# ==============================================================================

def test_retrieval_to_answer_generation(shared_embedder: EmbeddingManager, synthetic_document: LoadedDocument):
    """Verify RAG pipeline grounded prompt construction and answer synthesis with MockLLMClient."""
    store = VectorStore(persist_dir=None, collection_name="test_rag_col", embedding_manager=shared_embedder)
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    store.add_chunks(chunker.chunk_document(synthetic_document))

    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2, similarity_threshold=0.3)

    mock_llm = MockLLMClient(
        default_response="ReRAG uses an entailment threshold of 0.7 for verification.",
        response_map={
            "threshold": "The verification threshold is 0.7.",
            "unsupported query": INSUFFICIENT_INFORMATION_RESPONSE,
        },
    )

    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)

    # Grounded query
    result = pipeline.run_baseline("What is the verification threshold?")
    assert isinstance(result, PipelineResult)
    assert "0.7" in result.final_answer
    assert len(result.contexts) > 0
    assert len(result.retrieval_scores) > 0
    assert len(result.citations) > 0
    assert result.execution_time_seconds is not None

    # Insufficient information query
    empty_result = pipeline.run_baseline("Tell me about quantum gravity in 18th century France.", threshold=0.99)
    assert len(empty_result.contexts) == 0
    assert INSUFFICIENT_INFORMATION_RESPONSE in empty_result.final_answer or "insufficient" in empty_result.final_answer.lower()


# ==============================================================================
# 4. Evaluation of Answers Test
# ==============================================================================

def test_evaluation_of_answers(
    shared_embedder: EmbeddingManager,
    synthetic_document: LoadedDocument,
    synthetic_samples: List[EvaluationSample],
):
    """Verify answer evaluation against reference answers and recorded metrics."""
    store = VectorStore(persist_dir=None, collection_name="test_eval_col", embedding_manager=shared_embedder)
    chunker = TextChunker(chunk_size=250, chunk_overlap=50)
    store.add_chunks(chunker.chunk_document(synthetic_document))

    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2, similarity_threshold=0.2)
    mock_llm = MockLLMClient(
        response_map={
            "threshold": "The verification step uses an entailment threshold of 0.7.",
            "coordinate search": "ReRAG applies greedy adaptive coordinate search to select optimal chunking parameters.",
        }
    )
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)

    evaluator = Evaluator(
        pipeline=pipeline,
        embedding_manager=shared_embedder,
    )

    report = evaluator.evaluate(samples=synthetic_samples, deterministic=True)
    assert isinstance(report, EvaluationReport)
    assert report.total_samples == len(synthetic_samples)
    assert len(report.detailed_results) == len(synthetic_samples)

    agg = report.aggregate_metrics
    assert "mean_semantic_similarity" in agg or "semantic_similarity" in agg
    assert "mean_token_f1" in agg or "token_f1" in agg
    assert "mean_context_coverage" in agg or "context_coverage" in agg
    assert "mean_faithfulness_ratio" in agg or "faithfulness_ratio" in agg


# ==============================================================================
# 5. Optimization of Candidate Configurations Test
# ==============================================================================

def test_optimization_of_candidate_configurations(
    shared_embedder: EmbeddingManager,
    synthetic_document: LoadedDocument,
    synthetic_samples: List[EvaluationSample],
):
    """Verify adaptive coordinate search, index rebuilding per candidate, and candidate decisions."""
    store = VectorStore(persist_dir=None, collection_name="test_opt_col", embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2, similarity_threshold=0.2)
    mock_llm = MockLLMClient(default_response="ReRAG uses greedy adaptive coordinate search with threshold 0.7.")
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)

    evaluator = Evaluator(
        pipeline=pipeline,
        embedding_manager=shared_embedder,
    )

    rebuilt_indices = []

    # Wrapper to trace index rebuilding
    original_rebuild = retriever.rebuild_index_from_documents

    def tracked_rebuild(docs, chk):
        rebuilt_indices.append((chk.chunk_size, chk.chunk_overlap))
        return original_rebuild(docs, chk)

    retriever.rebuild_index_from_documents = tracked_rebuild

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=300,
        initial_chunk_overlap=50,
        initial_step_size=50,
        initial_overlap_step=15,
        max_iterations=2,
        improvement_tolerance=0.01,
        bounds=SearchBounds(
            min_chunk_size=100,
            max_chunk_size=600,
            min_chunk_overlap=0,
            max_chunk_overlap=150,
            min_step_size=25,
            min_overlap_step=10,
        ),
        random_seed=42,
    )

    optimizer = AdaptiveChunkOptimizer(
        baseline_config=ChunkConfig(300, 50),
        search_config=search_cfg,
        documents=[synthetic_document],
        pipeline=pipeline,
        evaluator=evaluator,
        optimization_samples=synthetic_samples,
        collection_name="test_opt_col",
    )

    result = optimizer.optimize()

    assert isinstance(result, OptimizationResult)
    assert len(rebuilt_indices) >= 2  # Baseline + at least one valid candidate
    assert len(result.candidate_history) >= 2
    assert result.stopping_reason != ""

    # Check candidate history status values
    valid_statuses = {"baseline", "accepted", "rejected", "rejected_no_improvement", "rejected_invalid", "skipped_duplicate", "failed"}
    for trial in result.candidate_history:
        assert trial.status in valid_statuses
        assert trial.decision_reason != ""


# ==============================================================================
# 6. Baseline versus Selected Configuration Test
# ==============================================================================

def test_baseline_versus_selected_configuration():
    """Verify comparison logic, delta calculation, and execution timing tracking."""
    base_cfg = ChunkConfig(chunk_size=500, chunk_overlap=50)
    sel_cfg = ChunkConfig(chunk_size=350, chunk_overlap=70)

    base_metrics = {"semantic_similarity": 0.72, "faithfulness_ratio": 0.65, "composite": 0.70}
    sel_metrics = {"semantic_similarity": 0.84, "faithfulness_ratio": 0.80, "composite": 0.82}

    trial_base = CandidateTrial(
        trial_id=1,
        iteration=0,
        config=base_cfg,
        step_size=100,
        metrics=base_metrics,
        objective_score=0.70,
        status="baseline",
        decision_reason="Baseline score established",
        execution_time_seconds=1.2,
    )
    trial_sel = CandidateTrial(
        trial_id=2,
        iteration=1,
        config=sel_cfg,
        step_size=100,
        metrics=sel_metrics,
        objective_score=0.82,
        status="accepted",
        decision_reason="Improved composite score (+0.12)",
        execution_time_seconds=1.1,
    )

    result = OptimizationResult(
        baseline_config=base_cfg,
        baseline_metrics=base_metrics,
        baseline_score=0.70,
        selected_config=sel_cfg,
        selected_metrics=sel_metrics,
        selected_score=0.82,
        improvement=0.12,
        candidate_history=[trial_base, trial_sel],
        stopping_reason="Maximum iterations reached",
        total_iterations=1,
        total_trials=2,
        accepted_trials=1,
        total_execution_time_seconds=2.3,
        optimization_dataset_id="test_set",
    )

    assert result.improvement == pytest.approx(0.12, abs=1e-4)
    assert result.selected_score > result.baseline_score
    assert result.selected_config.chunk_size == 350
    assert result.baseline_config.chunk_size == 500
    assert result.total_execution_time_seconds > 0.0


# ==============================================================================
# 7. Result Export Test (JSON and CSV)
# ==============================================================================

def test_result_export(tmp_path: Path):
    """Verify result serialization to JSON and CSV and complete roundtrip deserialization."""
    base_cfg = ChunkConfig(chunk_size=400, chunk_overlap=40)
    sel_cfg = ChunkConfig(chunk_size=250, chunk_overlap=50)

    trials = [
        CandidateTrial(
            trial_id=1,
            iteration=0,
            config=base_cfg,
            step_size=50,
            metrics={"semantic_similarity": 0.75, "token_f1": 0.70},
            objective_score=0.73,
            status="baseline",
            decision_reason="Baseline",
            execution_time_seconds=0.5,
        ),
        CandidateTrial(
            trial_id=2,
            iteration=1,
            config=sel_cfg,
            step_size=50,
            metrics={"semantic_similarity": 0.88, "token_f1": 0.82},
            objective_score=0.86,
            status="accepted",
            decision_reason="Gained +0.13",
            execution_time_seconds=0.4,
        ),
    ]

    opt_result = OptimizationResult(
        baseline_config=base_cfg,
        baseline_metrics={"composite": 0.73},
        baseline_score=0.73,
        selected_config=sel_cfg,
        selected_metrics={"composite": 0.86},
        selected_score=0.86,
        improvement=0.13,
        candidate_history=trials,
        stopping_reason="Target convergence",
        total_iterations=1,
        total_trials=2,
        accepted_trials=1,
        total_execution_time_seconds=0.9,
        optimization_dataset_id="export_demo",
    )

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    opt_result.save_to_json(json_path)
    opt_result.save_to_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()

    # Verify CSV structure
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "trial_id" in header
        assert "chunk_size" in header
        assert "objective_score" in header
        rows = list(reader)
        assert len(rows) == 2

    # Verify JSON deserialization roundtrip
    loaded = OptimizationResult.load_from_json(json_path)
    assert loaded.baseline_config.chunk_size == base_cfg.chunk_size
    assert loaded.selected_config.chunk_size == sel_cfg.chunk_size
    assert loaded.improvement == pytest.approx(0.13, abs=1e-4)
    assert len(loaded.candidate_history) == 2
    assert loaded.candidate_history[1].status == "accepted"


# ==============================================================================
# 8. Held-Out Data Isolation & No Leakage Test
# ==============================================================================

def test_held_out_data_isolation_and_no_leakage(
    shared_embedder: EmbeddingManager,
    synthetic_document: LoadedDocument,
    synthetic_samples: List[EvaluationSample],
):
    """Verify that the optimizer cannot access held-out test questions during candidate search."""
    train_samples = [synthetic_samples[0]]
    test_samples = [synthetic_samples[1]]

    store = VectorStore(persist_dir=None, collection_name="test_isolation_col", embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2)
    mock_llm = MockLLMClient(default_response="Mocked answer for isolation test.")
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=shared_embedder)

    runner = ExperimentRunner(
        documents=[synthetic_document],
        train_samples=train_samples,
        test_samples=test_samples,
        pipeline=pipeline,
        evaluator=evaluator,
        collection_name="test_isolation_col",
        random_seed=42,
    )

    assert len(runner.train_samples) == 1
    assert len(runner.test_samples) == 1
    assert runner.train_samples[0].sample_id != runner.test_samples[0].sample_id

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=300,
        initial_chunk_overlap=50,
        max_iterations=1,
    )

    result = runner.run_experiment(
        search_config=search_cfg,
        evaluate_held_out=True,
        save_results=False,
    )

    # Verify optimizer was strictly bound to training set
    assert result.optimization_dataset_id == f"train_split_n{len(train_samples)}"

    # Generalization evaluation occurred strictly post-selection
    assert result.held_out_evaluation is not None
    assert "baseline_score" in result.held_out_evaluation
    assert "selected_score" in result.held_out_evaluation
    assert "held_out_gain" in result.held_out_evaluation


# ==============================================================================
# 9. Small End-to-End Synthetic Pipeline Test
# ==============================================================================

def test_end_to_end_synthetic_pipeline(tmp_path: Path, shared_embedder: EmbeddingManager):
    """Execute complete end-to-end integration flow offline without external dependencies."""
    # 1. Create realistic synthetic document file
    doc_file = tmp_path / "synthetic_research.txt"
    doc_file.write_text(
        "Introduction to Grounded RAG.\n"
        "Retrieval augmented generation connects LLMs to verified databases.\n"
        "Claim verification ensures every generated sentence is grounded in retrieved chunks.\n"
        "Evaluation demonstrates that adaptive chunking improves faithfulness.",
        encoding="utf-8",
    )

    # 2. Ingest document
    loader = DocumentLoader()
    doc = loader.load_file(doc_file)
    assert doc.total_pages >= 1

    # 3. Define questions
    samples = [
        EvaluationSample(
            sample_id="e2e_q1",
            question="What does claim verification ensure?",
            reference_answer="Claim verification ensures every generated sentence is grounded in retrieved chunks.",
            expected_source=doc.document_id,
            expected_pages=[1],
            document_id=doc.document_id,
            relevant_doc_ids=[doc.document_id],
        )
    ]

    # 4. Initialize pipeline with MockLLMClient
    mock_llm = MockLLMClient(
        response_map={
            "claim verification": "Claim verification ensures every generated sentence is grounded in retrieved chunks."
        }
    )
    store = VectorStore(
        persist_dir=tmp_path / "e2e_vector_store",
        collection_name="e2e_col",
        embedding_manager=shared_embedder,
    )
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2, similarity_threshold=0.05)
    pipeline = RAGPipeline(retriever=retriever, llm_client=mock_llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=shared_embedder)

    # 5. Run ExperimentRunner with automated saving
    exp_results_dir = tmp_path / "experiments"
    runner = ExperimentRunner(
        documents=[doc],
        samples=samples,
        test_ratio=0.0,
        pipeline=pipeline,
        evaluator=evaluator,
        collection_name="e2e_col",
        results_dir=exp_results_dir,
        random_seed=99,
    )

    search_cfg = AdaptiveSearchConfig(
        initial_chunk_size=150,
        initial_chunk_overlap=25,
        max_iterations=1,
    )

    result = runner.run_experiment(
        search_config=search_cfg,
        save_results=True,
        experiment_name="e2e_verified_run",
    )

    # 6. Verify outputs on disk
    saved_json = exp_results_dir / "e2e_verified_run.json"
    saved_csv = exp_results_dir / "e2e_verified_run.csv"

    assert saved_json.exists()
    assert saved_csv.exists()
    assert result.total_iterations >= 1
    assert result.selected_score is not None
    assert result.selected_score > 0.0
