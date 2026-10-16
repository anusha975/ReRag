"""Query-Aware Dynamic Document Chunking Module for ReRAG.

Implements an optional experimental query-aware dynamic chunking strategy.
Unlike standard static sliding-window chunking (which partitions documents uniformly
regardless of query semantics), query-aware chunking identifies candidate topical
evidence clusters based on query terms and extracts cohesive passages around
the query focus, avoiding artificial boundary splits in the middle of key factual statements.

Scientific Notice:
This module is an optional advanced research extension. It does not replace or modify
the standard TextChunker baseline, allowing direct empirical comparison between
static chunking and query-aware dynamic chunking.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.chunker import TextChunk, TextChunker
from src.document_loader import LoadedDocument, PageContent

logger = logging.getLogger(__name__)

# Standard stopwords
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself",
}


@dataclass
class QueryFocusRegion:
    """Represents a discovered high-density evidence region for a query."""

    start_char: int
    end_char: int
    matched_terms: Set[str]
    score: float
    page_number: Optional[int] = None
    text: str = ""


class QueryAwareChunker:
    """Dynamic query-guided document chunker.

    Analyzes document text to detect clusters of query concepts and constructs
    dynamically bounded chunks centered around these regions.
    """

    def __init__(
        self,
        base_chunk_size: int = 500,
        base_chunk_overlap: int = 50,
        window_expansion: int = 150,
        min_chunk_size: int = 150,
        max_chunk_size: int = 1200,
        focus_chunk_size: Optional[int] = None,
    ) -> None:
        """Initialize query-aware chunker.

        Args:
            base_chunk_size: Default target chunk size.
            base_chunk_overlap: Overlap between adjacent chunks.
            window_expansion: Characters to expand outwards from query term matches.
            min_chunk_size: Lower size boundary for query-aware chunks.
            max_chunk_size: Upper size boundary for query-aware chunks.
            focus_chunk_size: Optional alias for min/focus chunk window size.
        """
        self.base_chunker = TextChunker(chunk_size=base_chunk_size, chunk_overlap=base_chunk_overlap)
        self.base_chunk_size = base_chunk_size
        self.base_chunk_overlap = base_chunk_overlap
        self.window_expansion = window_expansion
        self.min_chunk_size = focus_chunk_size if focus_chunk_size is not None else min_chunk_size
        self.max_chunk_size = max_chunk_size

    def find_query_focus_regions(
        self,
        text: str,
        query: str,
        page_number: Optional[int] = None,
    ) -> List[QueryFocusRegion]:
        """Convenience alias for find_focus_regions."""
        return self.find_focus_regions(text=text, query=query, page_number=page_number)

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract non-stopword query keywords."""
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        return keywords if keywords else words

    def find_focus_regions(
        self,
        text: str,
        query: str,
        page_number: Optional[int] = None,
    ) -> List[QueryFocusRegion]:
        """Locate high-density regions of query terms in the text."""
        keywords = self._extract_query_keywords(query)
        if not keywords:
            return []

        # Find occurrences of each keyword
        match_spans: List[Tuple[int, int, str]] = []
        for kw in keywords:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            for m in pattern.finditer(text):
                match_spans.append((m.start(), m.end(), kw))

        if not match_spans:
            return []

        match_spans.sort(key=lambda s: s[0])

        # Cluster matches within window_expansion distance
        clusters: List[List[Tuple[int, int, str]]] = []
        current_cluster: List[Tuple[int, int, str]] = [match_spans[0]]

        for span in match_spans[1:]:
            prev_end = current_cluster[-1][1]
            if span[0] - prev_end <= self.window_expansion * 2:
                current_cluster.append(span)
            else:
                clusters.append(current_cluster)
                current_cluster = [span]

        if current_cluster:
            clusters.append(current_cluster)

        # Form FocusRegions
        regions: List[QueryFocusRegion] = []
        for cl in clusters:
            start_pos = max(0, cl[0][0] - self.window_expansion)
            end_pos = min(len(text), cl[-1][1] + self.window_expansion)

            # Snap to sentence or whitespace boundaries
            # Search backward for sentence starter or newline
            sub_before = text[max(0, start_pos - 50):start_pos]
            boundary_before = re.search(r"(\. |\n|\n\n)[^\.\n]*$", sub_before)
            if boundary_before:
                start_pos = max(0, start_pos - 50 + boundary_before.end())

            # Search forward for sentence ender
            sub_after = text[end_pos:min(len(text), end_pos + 80)]
            boundary_after = re.search(r"(\.|\n\n|\n)", sub_after)
            if boundary_after:
                end_pos = min(len(text), end_pos + boundary_after.end())

            matched_terms = {span[2] for span in cl}
            # Density score: number of unique terms matched * term occurrences
            score = len(matched_terms) * 1.0 + len(cl) * 0.5

            regions.append(
                QueryFocusRegion(
                    start_char=start_pos,
                    end_char=end_pos,
                    matched_terms=matched_terms,
                    score=score,
                    page_number=page_number,
                    text=text[start_pos:end_pos],
                )
            )

        return regions

    def chunk_document(
        self,
        document: LoadedDocument,
        query: Optional[str] = None,
    ) -> List[TextChunk]:
        """Convenience method chunking a document with optional query guidance."""
        if not query or not query.strip():
            return self.base_chunker.chunk_document(document)
        return self.chunk_query_focused(document, query)

    def chunk_query_focused(
        self,
        document: LoadedDocument,
        query: str,
    ) -> List[TextChunk]:
        """Generate query-focused chunks for a document given a specific inquiry.

        Falls back gracefully to standard base chunking if no query focus is found.

        Args:
            document: Loaded document to chunk.
            query: Target inquiry string.

        Returns:
            List[TextChunk]: Dynamic query-aware chunks with provenance metadata.
        """
        if not query or not query.strip():
            return self.base_chunker.chunk_document(document)

        dynamic_chunks: List[TextChunk] = []
        chunk_idx = 0

        # Process per page if page structure is present, else on full text
        sources = document.pages if document.pages else [
            PageContent(page_number=1, text=document.content, char_count=len(document.content))
        ]

        for page in sources:
            regions = self.find_focus_regions(page.text, query, page.page_number)

            if not regions:
                # Fallback to standard chunking on this page
                sub_doc = LoadedDocument(
                    content=page.text,
                    metadata=document.metadata,
                    source_path=document.source_path,
                    pages=[page],
                )
                base_page_chunks = self.base_chunker.chunk_document(sub_doc)
                for c in base_page_chunks:
                    dynamic_chunks.append(
                        TextChunk(
                            chunk_id=f"{document.document_id}_p{page.page_number}_c{chunk_idx}",
                            text=c.text,
                            metadata={
                                **c.metadata,
                                "chunk_strategy": "static_fallback",
                                "query_guided": False,
                                "query_focused": False,
                            },
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                continue

            # Merge or create chunks for identified regions
            covered_ranges: List[Tuple[int, int]] = []
            for reg in regions:
                chunk_text = page.text[reg.start_char : reg.end_char].strip()
                if not chunk_text or len(chunk_text) < self.min_chunk_size:
                    continue

                # Cap max size
                if len(chunk_text) > self.max_chunk_size:
                    chunk_text = chunk_text[:self.max_chunk_size]

                cid = f"{document.document_id}_qfocused_p{page.page_number}_{reg.start_char}"
                chunk_id_hash = hashlib.md5(cid.encode("utf-8")).hexdigest()[:10]

                dynamic_chunks.append(
                    TextChunk(
                        chunk_id=f"{document.document_id}_dyn_{chunk_id_hash}",
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "document_id": document.document_id,
                            "page_number": page.page_number,
                            "start_char": reg.start_char,
                            "end_char": reg.end_char,
                            "query_guided": True,
                            "query_focused": True,
                            "query_focus_score": reg.score,
                            "matched_terms": list(reg.matched_terms),
                            "chunk_strategy": "query_aware",
                        },
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                covered_ranges.append((reg.start_char, reg.end_char))

        # If no dynamic chunks were produced, fall back to standard chunking
        if not dynamic_chunks:
            return self.base_chunker.chunk_document(document)

        return dynamic_chunks

