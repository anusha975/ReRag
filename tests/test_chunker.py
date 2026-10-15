"""Unit tests for text chunker module."""

from pathlib import Path

import pytest

from src.chunker import TextChunk, TextChunker
from src.document_loader import LoadedDocument, PageContent


def test_chunker_initialization_valid():
    """Verify initialization with valid parameters."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=40, unit="characters")
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 40
    assert chunker.unit == "characters"


def test_chunker_invalid_settings():
    """Verify validation errors for illegal chunk size and overlap configurations."""
    # chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=-50, chunk_overlap=10)

    # chunk_overlap < 0
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-10)

    # chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=150)

    # Unsupported unit
    with pytest.raises(ValueError, match="Unsupported unit"):
        TextChunker(chunk_size=100, chunk_overlap=10, unit="sentences")


def test_chunk_empty_and_whitespace_text():
    """Verify that empty or whitespace-only inputs produce zero chunks."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\t\n   ") == []


def test_chunk_short_document():
    """Verify that a document shorter than chunk_size produces exactly one chunk."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, unit="characters")
    text = "Short sentence with high factual density for ReRAG."
    chunks = chunker.chunk_text(text, document_id="doc_short")

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["document_id"] == "doc_short"
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == len(text)
    assert chunks[0].chunk_index == 0


def test_chunk_long_document_and_overlap_correctness():
    """Verify that a long document is split into multiple chunks with correct overlap."""
    chunker = TextChunker(chunk_size=150, chunk_overlap=30, unit="characters")

    paragraphs = [
        "Paragraph 1: Retrieval-augmented generation combines search with autoregressive language generation.",
        "Paragraph 2: When the context contains incomplete or contradictory data, LLMs are prone to hallucinating facts.",
        "Paragraph 3: ReRAG systematically extracts verifiable claims and checks them against retrieved evidence.",
        "Paragraph 4: If factual consistency falls below the threshold, query refinement issues another retrieval round.",
    ]
    full_text = "\n\n".join(paragraphs)

    chunks = chunker.chunk_text(full_text, document_id="paper_1")

    assert len(chunks) > 1

    # Verify each chunk length does not exceed chunk_size
    for c in chunks:
        assert len(c.text) <= 150
        assert c.text.strip() != ""
        # Check start and end char offsets match the slice in full_text
        extracted = full_text[c.metadata["start_char"] : c.metadata["end_char"]]
        assert extracted == c.text

    # Verify overlap exists between consecutive chunks
    overlap_found = False
    for i in range(len(chunks) - 1):
        # Look for overlapping words between adjacent chunks
        tokens_current = set(chunks[i].text.split())
        tokens_next = set(chunks[i + 1].text.split())
        if tokens_current & tokens_next:
            overlap_found = True
            break
    assert overlap_found, "Consecutive chunks should exhibit vocabulary overlap according to chunk_overlap."


def test_chunk_document_page_provenance():
    """Verify that page numbers, character offsets, and document metadata are preserved."""
    p1 = PageContent(page_number=1, text="Page 1 introduction text for ReRAG.", char_count=35)
    p2 = PageContent(page_number=2, text="Page 2 evaluation methods and empirical benchmarks.", char_count=52)

    doc = LoadedDocument(
        content=f"{p1.text}\n\n{p2.text}",
        metadata={"title": "ReRAG Benchmark", "document_id": "rerag_paper"},
        source_path=Path("data/documents/rerag_paper.pdf"),
        pages=[p1, p2],
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=20, unit="characters")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["document_id"] == "rerag_paper"
    assert chunks[0].metadata["title"] == "ReRAG Benchmark"
    assert chunks[1].metadata["page_number"] == 2
    assert chunks[1].metadata["document_id"] == "rerag_paper"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunker_determinism():
    """Verify that chunking the identical text with identical settings produces identical output."""
    chunker_a = TextChunker(chunk_size=120, chunk_overlap=25, unit="characters")
    chunker_b = TextChunker(chunk_size=120, chunk_overlap=25, unit="characters")

    sample_text = (
        "ReRAG addresses parametric memory hallucinations by grounding answers "
        "in dynamic external corpora. The system maintains an audit trail of every "
        "verification decision."
    )

    chunks_a = chunker_a.chunk_text(sample_text, document_id="doc_det")
    chunks_b = chunker_b.chunk_text(sample_text, document_id="doc_det")

    assert len(chunks_a) == len(chunks_b)
    for ca, cb in zip(chunks_a, chunks_b):
        assert ca.chunk_id == cb.chunk_id
        assert ca.text == cb.text
        assert ca.metadata == cb.metadata
        assert ca.chunk_index == cb.chunk_index


def test_chunker_token_unit():
    """Verify chunking using token units via tiktoken."""
    chunker = TextChunker(chunk_size=20, chunk_overlap=5, unit="tokens")
    text = (
        "Retrieval-Augmented Generation helps ground language models in real facts. "
        "However, single-hop systems may still hallucinate when context is ambiguous. "
        "ReRAG implements iterative query refinement and verification."
    )

    chunks = chunker.chunk_text(text, document_id="token_test")
    assert len(chunks) > 1

    for c in chunks:
        assert c.metadata["chunk_unit"] == "tokens"
        assert c.metadata["token_count"] is not None
        assert c.metadata["token_count"] <= 20
        assert c.token_count <= 20


def test_chunk_documents_batch():
    """Verify chunk_documents processes a collection of documents."""
    doc1 = LoadedDocument(
        content="Doc 1 content.",
        metadata={"document_id": "d1"},
        source_path=Path("doc1.txt"),
        pages=[PageContent(page_number=1, text="Doc 1 content.", char_count=14)],
    )
    doc2 = LoadedDocument(
        content="Doc 2 content.",
        metadata={"document_id": "d2"},
        source_path=Path("doc2.txt"),
        pages=[PageContent(page_number=1, text="Doc 2 content.", char_count=14)],
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    all_chunks = chunker.chunk_documents([doc1, doc2])

    assert len(all_chunks) == 2
    assert all_chunks[0].document_id == "d1"
    assert all_chunks[1].document_id == "d2"
