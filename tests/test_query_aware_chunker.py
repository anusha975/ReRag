"""Automated tests for QueryAwareChunker optional experimental module."""

from pathlib import Path
import pytest

from src.document_loader import LoadedDocument, PageContent
from src.query_aware_chunker import QueryAwareChunker, QueryFocusRegion


def test_query_aware_chunker_focus_regions():
    """Verify that sentences containing query terms are detected as focus regions."""
    text = (
        "Introduction to the architecture. "
        "The hallucination mitigation mechanism uses an adaptive objective function. "
        "Further details on experimental methodology."
    )
    chunker = QueryAwareChunker(base_chunk_size=200, focus_chunk_size=100)
    regions = chunker.find_query_focus_regions(text, query="How does hallucination mitigation work?")

    assert len(regions) >= 1
    assert any("hallucination mitigation mechanism" in r.text for r in regions)


def test_query_aware_chunking_with_query():
    """Verify query-aware chunking generates chunks containing query terms and marks metadata."""
    content = (
        "First section discussing general NLP models and architectures. " * 3
        + "Specific critical finding: ReRAG chunk optimizer reduces hallucination rate by 34%. "
        + "Third section discussing concluding remarks and deployment notes. " * 3
    )
    page = PageContent(page_number=1, text=content, char_count=len(content))
    doc = LoadedDocument(
        content=content,
        metadata={"title": "Test Paper", "file_type": "txt"},
        source_path=Path("test.txt"),
        pages=[page],
    )

    chunker = QueryAwareChunker(base_chunk_size=300, focus_chunk_size=120)
    chunks = chunker.chunk_document(doc, query="ReRAG chunk optimizer hallucination rate")

    assert len(chunks) >= 1
    # Check that query focus was detected
    has_focus_chunk = any(c.metadata.get("query_guided", False) is True for c in chunks)
    assert has_focus_chunk is True


def test_query_aware_chunking_fallback():
    """Verify fallback to standard chunking when query is None or empty."""
    content = "Standard text that should be chunked using normal chunking rules. " * 10
    page = PageContent(page_number=1, text=content, char_count=len(content))
    doc = LoadedDocument(
        content=content,
        metadata={"title": "Test Doc", "file_type": "txt"},
        source_path=Path("test.txt"),
        pages=[page],
    )

    chunker = QueryAwareChunker(base_chunk_size=200, base_chunk_overlap=50)
    chunks_no_query = chunker.chunk_document(doc, query="")

    assert len(chunks_no_query) > 0
    assert all(c.metadata.get("query_guided", False) is False for c in chunks_no_query)
