"""Unit tests for semantic retriever and citation preservation."""

from pathlib import Path

import pytest

from src.chunker import TextChunk, TextChunker
from src.document_loader import LoadedDocument, PageContent
from src.embeddings import EmbeddingManager
from src.retriever import RetrievedContext, Retriever
from src.vector_store import VectorStore


@pytest.fixture
def shared_embedder(mock_embedder):
    """Shared EmbeddingManager fixture."""
    return mock_embedder


@pytest.fixture
def populated_retriever(shared_embedder):
    """Retriever fixture loaded with synthetic research chunks across multiple documents."""
    store = VectorStore(persist_dir=None, collection_name="test_retrieval", embedding_manager=shared_embedder)

    chunks = [
        TextChunk(
            chunk_id="paper_rerag::p01::c0000",
            text="ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation.",
            metadata={"document_id": "paper_rerag", "page_number": 1, "topic": "RAG"},
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="paper_rerag::p02::c0001",
            text="The verification step checks candidate claims against retrieved context passages using a consistency threshold.",
            metadata={"document_id": "paper_rerag", "page_number": 2, "topic": "Verification"},
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="paper_opt::p01::c0000",
            text="Hyperparameter optimization identifies the optimal chunk size and overlap for dense passage retrieval.",
            metadata={"document_id": "paper_opt", "page_number": 1, "topic": "Optimization"},
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="paper_astronomy::p05::c0000",
            text="James Webb Space Telescope observes high-redshift galaxies and stellar nurseries in infrared light.",
            metadata={"document_id": "paper_astronomy", "page_number": 5, "topic": "Astronomy"},
            chunk_index=0,
        ),
    ]
    store.add_chunks(chunks)
    thresh = 0.2 if getattr(shared_embedder, "model_name", "") == "mock-minilm" else 0.3
    return Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=2, similarity_threshold=thresh)


def test_retriever_initialization(shared_embedder):
    """Verify retriever configuration initialization."""
    store = VectorStore(persist_dir=None, embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder, top_k=5, similarity_threshold=0.4)
    assert retriever.top_k == 5
    assert retriever.similarity_threshold == 0.4


def test_retrieve_relevant_query(populated_retriever):
    """Verify semantic retrieval ranks the most relevant chunk first with correct citation."""
    results = populated_retriever.retrieve("How does ReRAG reduce hallucination?", top_k=2)

    assert len(results) > 0
    top_result = results[0]

    assert isinstance(top_result, RetrievedContext)
    assert "reducing factual hallucinations" in top_result.text
    assert top_result.document_id == "paper_rerag"
    assert top_result.page_number == 1
    assert top_result.citation == "[paper_rerag, p. 1]"

    # Mathematical relationship: score = 1.0 - distance
    assert abs(top_result.score - (1.0 - top_result.distance)) < 1e-4
    min_score = 0.35 if getattr(populated_retriever.embedding_manager, "model_name", "") == "mock-minilm" else 0.4
    assert top_result.score > min_score


def test_retrieve_page_number_citation(populated_retriever):
    """Verify that citations accurately preserve the page number from metadata."""
    results = populated_retriever.retrieve("consistency threshold and verification step", top_k=1)
    assert len(results) == 1
    match = results[0]
    assert match.page_number == 2
    assert match.citation == "[paper_rerag, p. 2]"


def test_empty_query_and_empty_collection(shared_embedder):
    """Verify retriever handles empty queries and empty collections gracefully."""
    empty_store = VectorStore(persist_dir=None, collection_name="empty_col", embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=empty_store, embedding_manager=shared_embedder)

    # Empty collection query
    assert retriever.retrieve("any query") == []

    # Empty query on populated collection
    populated_store = VectorStore(persist_dir=None, collection_name="pop_col", embedding_manager=shared_embedder)
    populated_store.add_chunks([
        TextChunk(chunk_id="c1", text="Sample valid text", metadata={"document_id": "d1"}, chunk_index=0)
    ])
    retriever_pop = Retriever(vector_store=populated_store, embedding_manager=shared_embedder)

    assert retriever_pop.retrieve("") == []
    assert retriever_pop.retrieve("   ") == []


def test_similarity_threshold_filtering(populated_retriever):
    """Verify that results below the similarity threshold are excluded."""
    # Strict threshold 0.95 should filter out general queries
    strict_results = populated_retriever.retrieve(
        "general machine learning concepts",
        threshold=0.95,
    )
    assert len(strict_results) == 0


def test_rebuild_index_from_documents(shared_embedder):
    """Verify rebuilding the index with a new chunking configuration."""
    store = VectorStore(persist_dir=None, collection_name="test_rebuild", embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder)

    # Synthetic document with 2 pages
    p1 = PageContent(page_number=1, text="First page covering dataset construction.", char_count=42)
    p2 = PageContent(page_number=2, text="Second page analyzing verification accuracy.", char_count=45)
    doc = LoadedDocument(
        content=f"{p1.text}\n\n{p2.text}",
        metadata={"document_id": "doc_rebuild", "title": "Rebuild Study"},
        source_path=Path("doc_rebuild.pdf"),
        pages=[p1, p2],
    )

    # 1. Index with chunk_size=200
    chunker_large = TextChunker(chunk_size=200, chunk_overlap=20, unit="characters")
    count_1 = retriever.rebuild_index_from_documents([doc], chunker_large)
    assert count_1 == 2
    assert store.count() == 2

    # 2. Rebuild with smaller chunk_size=40 producing more chunks
    chunker_small = TextChunker(chunk_size=40, chunk_overlap=10, unit="characters")
    count_2 = retriever.rebuild_index_from_documents([doc], chunker_small)
    assert count_2 > 2
    assert store.count() == count_2

    # Verify query works on rebuilt index
    res = retriever.retrieve("dataset construction", top_k=1)
    assert len(res) == 1
    assert "dataset construction" in res[0].text


def test_citation_multi_page_and_no_page():
    """Verify citation formatting for multi-page, single-page, and unpaged contexts."""
    # Single page
    c1 = RetrievedContext(
        chunk_id="c1",
        text="text",
        score=0.9,
        distance=0.1,
        metadata={},
        page_number=3,
        document_id="paper_a",
    )
    assert c1.citation == "[paper_a, p. 3]"

    # Multi page via metadata
    c2 = RetrievedContext(
        chunk_id="c2",
        text="text",
        score=0.9,
        distance=0.1,
        metadata={"page_numbers": [3, 4, 5]},
        page_number=None,
        document_id="paper_b",
    )
    assert c2.citation == "[paper_b, pp. 3-5]"

    # No page number
    c3 = RetrievedContext(
        chunk_id="c3",
        text="text",
        score=0.8,
        distance=0.2,
        metadata={},
        page_number=None,
        document_id="paper_c",
    )
    assert c3.citation == "[paper_c]"


def test_retriever_with_config(shared_embedder):
    """Verify Retriever initializes properly from RetrieverConfig."""
    from src.config import RetrieverConfig

    store = VectorStore(persist_dir=None, embedding_manager=shared_embedder)
    cfg = RetrieverConfig(top_k=7, similarity_threshold=0.45)
    retriever = Retriever(vector_store=store, config=cfg)
    assert retriever.top_k == 7
    assert retriever.similarity_threshold == 0.45


def test_rebuild_index_from_chunks(shared_embedder):
    """Verify rebuilding index directly from a sequence of TextChunk instances."""
    store = VectorStore(persist_dir=None, collection_name="test_rebuild_chunks", embedding_manager=shared_embedder)
    retriever = Retriever(vector_store=store, embedding_manager=shared_embedder)

    chunks = [
        TextChunk(chunk_id="chk1", text="Neural attention mechanisms in transformers", metadata={"document_id": "d1", "page_number": 1}, chunk_index=0),
        TextChunk(chunk_id="chk2", text="Verification algorithms for hallucination", metadata={"document_id": "d2", "page_number": 1}, chunk_index=0),
    ]

    total = retriever.rebuild_index_from_chunks(chunks)
    assert total == 2
    assert store.count() == 2

    res = retriever.retrieve("transformers attention", top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "chk1"


def test_retriever_with_where_filter(populated_retriever):
    """Verify retriever forwards where_filter to restrict candidate search space."""
    # Filter only for astronomy topic
    res_astro = populated_retriever.retrieve(
        "telescope observations",
        top_k=2,
        where_filter={"topic": "Astronomy"},
    )
    assert len(res_astro) == 1
    assert res_astro[0].document_id == "paper_astronomy"

    # Query without matching topic filter returns empty
    res_empty = populated_retriever.retrieve(
        "telescope observations",
        top_k=2,
        where_filter={"topic": "PhysicsNotPresent"},
    )
    assert len(res_empty) == 0

