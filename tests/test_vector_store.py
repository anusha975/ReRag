"""Unit tests for ChromaDB vector store integration."""

from pathlib import Path

import pytest

from src.chunker import TextChunk
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


@pytest.fixture
def shared_embedder(mock_embedder):
    """Shared EmbeddingManager fixture to reuse cached weights."""
    return mock_embedder


def test_vector_store_initialization_ephemeral(shared_embedder):
    """Verify ephemeral in-memory store creation."""
    store = VectorStore(
        persist_dir=None,
        collection_name="test_ephemeral",
        embedding_manager=shared_embedder,
    )
    assert store.count() == 0
    assert store.collection_name == "test_ephemeral"


def test_add_chunks_and_query(shared_embedder):
    """Verify adding synthetic chunks and semantic querying."""
    store = VectorStore(
        persist_dir=None,
        collection_name="test_querying",
        embedding_manager=shared_embedder,
    )

    chunks = [
        TextChunk(
            chunk_id="doc1::c0",
            text="Quantum computing utilizes qubits exhibiting superposition and entanglement.",
            metadata={"document_id": "doc1", "page_number": 1, "topic": "physics"},
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="doc2::c0",
            text="French cuisine includes croissants, baguettes, and culinary techniques like sous-vide.",
            metadata={"document_id": "doc2", "page_number": 1, "topic": "food"},
            chunk_index=0,
        ),
    ]

    store.add_chunks(chunks)
    assert store.count() == 2

    # Query for quantum physics
    results = store.query(query_text="superposition in quantum algorithms", top_k=1)
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]
    similarities = results["similarities"][0]

    assert len(retrieved_ids) == 1
    assert retrieved_ids[0] == "doc1::c0"
    assert "qubits" in retrieved_docs[0]
    # Cosine similarity should be significantly positive for related text
    min_sim = 0.35 if getattr(shared_embedder, "model_name", "") == "mock-minilm" else 0.4
    assert similarities[0] > min_sim


def test_prevent_stale_chunks_on_reindex(shared_embedder):
    """Verify that re-indexing a document purges old stale chunks."""
    store = VectorStore(
        persist_dir=None,
        collection_name="test_stale",
        embedding_manager=shared_embedder,
    )

    # First indexing run: doc1 has 3 chunks
    initial_chunks = [
        TextChunk(chunk_id=f"doc1::c{i}", text=f"Initial chunk {i}", metadata={"document_id": "doc1"}, chunk_index=i)
        for i in range(3)
    ]
    store.add_chunks(initial_chunks)
    assert store.count() == 3

    # Re-indexing run: doc1 now only has 1 chunk with new content
    updated_chunks = [
        TextChunk(chunk_id="doc1::c0_new", text="Updated single chunk", metadata={"document_id": "doc1"}, chunk_index=0)
    ]
    store.add_chunks(updated_chunks, prevent_stale=True)

    # Stale chunks should be purged: exactly 1 chunk should remain
    assert store.count() == 1
    query_res = store.query(query_text="Updated single chunk", top_k=5)
    remaining_ids = query_res["ids"][0]
    assert remaining_ids == ["doc1::c0_new"]


def test_delete_and_clear(shared_embedder):
    """Verify chunk deletion and complete collection clearing."""
    store = VectorStore(
        persist_dir=None,
        collection_name="test_delete",
        embedding_manager=shared_embedder,
    )

    chunks = [
        TextChunk(chunk_id="c1", text="First chunk", metadata={"document_id": "d1"}, chunk_index=0),
        TextChunk(chunk_id="c2", text="Second chunk", metadata={"document_id": "d2"}, chunk_index=1),
        TextChunk(chunk_id="c3", text="Third chunk", metadata={"document_id": "d2"}, chunk_index=2),
    ]
    store.add_chunks(chunks)
    assert store.count() == 3

    # Delete specific chunk
    store.delete_chunks(["c1"])
    assert store.count() == 2

    # Delete by document ID
    store.delete_document("d2")
    assert store.count() == 0

    # Clear empty store safely
    store.clear()
    assert store.count() == 0


def test_persistent_storage(tmp_path: Path, shared_embedder):
    """Verify persistent disk storage saves and reloads collection."""
    persist_dir = tmp_path / "chroma_test_db"
    store1 = VectorStore(
        persist_dir=persist_dir,
        collection_name="test_persist",
        embedding_manager=shared_embedder,
    )
    chunk = TextChunk(chunk_id="p1", text="Persisted text content", metadata={"document_id": "p"}, chunk_index=0)
    store1.add_chunks([chunk])
    assert store1.count() == 1

    # Re-open with new store instance at same path
    store2 = VectorStore(
        persist_dir=persist_dir,
        collection_name="test_persist",
        embedding_manager=shared_embedder,
    )
    assert store2.count() == 1
    res = store2.query(query_text="Persisted text", top_k=1)
    assert res["ids"][0] == ["p1"]


def test_multi_collection_management(shared_embedder):
    """Verify creating, listing, switching, and deleting multiple collections."""
    store = VectorStore(persist_dir=None, collection_name="col_alpha", embedding_manager=shared_embedder)

    chunk_a = TextChunk(chunk_id="a1", text="Alpha collection chunk", metadata={"document_id": "doc_a"}, chunk_index=0)
    store.add_chunks([chunk_a])
    assert store.count() == 1

    # Create and switch to beta collection
    store.switch_collection("col_beta")
    assert store.collection_name == "col_beta"
    assert store.count() == 0

    chunk_b = TextChunk(chunk_id="b1", text="Beta collection chunk", metadata={"document_id": "doc_b"}, chunk_index=0)
    store.add_chunks([chunk_b])
    assert store.count() == 1

    # List collections contains both
    collections = store.list_collections()
    assert "col_alpha" in collections
    assert "col_beta" in collections

    # Delete col_alpha
    store.delete_collection("col_alpha")
    assert "col_alpha" not in store.list_collections()


def test_metadata_filtering_and_provenance(shared_embedder):
    """Verify metadata filtering and provenance fields are preserved."""
    store = VectorStore(persist_dir=None, collection_name="test_filter", embedding_manager=shared_embedder)

    chunks = [
        TextChunk(
            chunk_id="chunk_cs_1",
            text="Transformer neural networks use self-attention mechanisms.",
            metadata={"document_id": "paper_cs", "topic": "ai", "page_number": 4},
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="chunk_bio_1",
            text="CRISPR Cas9 enables targeted gene editing and genomic modification.",
            metadata={"document_id": "paper_bio", "topic": "genetics", "page_number": 12},
            chunk_index=0,
        ),
    ]
    store.add_chunks(chunks)

    # Filter by topic: ai
    ai_results = store.query(query_text="neural mechanisms", top_k=5, where_filter={"topic": "ai"})
    assert len(ai_results["ids"][0]) == 1
    assert ai_results["ids"][0][0] == "chunk_cs_1"

    # Filter by document_id
    bio_results = store.query(query_text="genomic editing", top_k=5, where_filter={"document_id": "paper_bio"})
    assert len(bio_results["ids"][0]) == 1
    assert bio_results["ids"][0][0] == "chunk_bio_1"
    assert bio_results["metadatas"][0][0]["page_number"] == 12


def test_query_with_precomputed_embedding(shared_embedder):
    """Verify querying with explicit precomputed query embedding vector."""
    store = VectorStore(persist_dir=None, collection_name="test_precomputed", embedding_manager=shared_embedder)
    chunk = TextChunk(chunk_id="c1", text="Natural language understanding", metadata={"document_id": "d1"}, chunk_index=0)
    store.add_chunks([chunk])

    # Precompute vector
    q_vec = shared_embedder.embed_query("understanding language")
    res = store.query(query_embedding=q_vec, top_k=1)
    assert res["ids"][0] == ["c1"]
    assert res["similarities"][0][0] > 0.4


def test_rebuild_index(shared_embedder):
    """Verify rebuilding index clears previous chunks and indexes new chunks."""
    store = VectorStore(persist_dir=None, collection_name="test_rebuild_store", embedding_manager=shared_embedder)

    old_chunks = [TextChunk(chunk_id="old1", text="Old text", metadata={"document_id": "old"}, chunk_index=0)]
    store.add_chunks(old_chunks)
    assert store.count() == 1

    new_chunks = [
        TextChunk(chunk_id="new1", text="First new chunk", metadata={"document_id": "new"}, chunk_index=0),
        TextChunk(chunk_id="new2", text="Second new chunk", metadata={"document_id": "new"}, chunk_index=1),
    ]
    store.rebuild_index(new_chunks)
    assert store.count() == 2
    res = store.query(query_text="new chunk", top_k=5)
    assert set(res["ids"][0]) == {"new1", "new2"}

