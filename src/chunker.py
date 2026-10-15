"""Document chunking and text splitting module for ReRAG.

Splits loaded documents and text into configurable, overlapping chunks. Supports
both token-based (using tiktoken when available) and character-based strategies,
enforces strict overlap constraints, preserves page provenance and character offsets,
prevents empty or duplicate chunks, and ensures deterministic output.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.document_loader import LoadedDocument, PageContent

logger = logging.getLogger(__name__)

# Check for tiktoken availability
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False


@dataclass(frozen=True)
class TextChunk:
    """Represents an individual text chunk with associated provenance metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    chunk_index: int

    @property
    def token_count(self) -> Optional[int]:
        """Return token count if recorded in metadata."""
        return self.metadata.get("token_count")

    @property
    def char_count(self) -> int:
        """Return character length of chunk text."""
        return len(self.text)

    @property
    def page_number(self) -> Optional[int]:
        """Return primary page number if available."""
        return self.metadata.get("page_number")

    @property
    def document_id(self) -> Optional[str]:
        """Return parent document ID."""
        return self.metadata.get("document_id")


class TextChunker:
    """Configurable text chunker supporting token and character splitting with overlap."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        unit: str = "tokens" if _TIKTOKEN_AVAILABLE else "characters",
        separators: Optional[Sequence[str]] = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize text chunker.

        Args:
            chunk_size: Maximum size of each chunk in the chosen unit.
            chunk_overlap: Overlapping size between adjacent chunks.
            unit: Splitting measurement unit ('tokens' or 'characters').
            separators: Hierarchical sequence of boundary separators.
            encoding_name: Tiktoken encoding name (e.g. 'cl100k_base').

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, chunk_overlap >= chunk_size,
                or unit is unsupported.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        unit_clean = unit.lower().strip()
        if unit_clean not in {"tokens", "characters"}:
            raise ValueError(
                f"Unsupported unit '{unit}'. Supported units are 'tokens' and 'characters'."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit_clean
        self.separators = list(separators or self.DEFAULT_SEPARATORS)
        self.encoding_name = encoding_name

        # Setup tokenizer if token-based
        self._tokenizer = None
        if self.unit == "tokens":
            if _TIKTOKEN_AVAILABLE:
                try:
                    self._tokenizer = tiktoken.get_encoding(encoding_name)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Failed to load tiktoken encoding '%s': %s. Falling back to cl100k_base.",
                        encoding_name,
                        exc,
                    )
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            else:  # pragma: no cover
                logger.warning(
                    "tiktoken is not installed; falling back to whitespace tokenization."
                )

    def _length_fn(self, text: str) -> int:
        """Measure length of text in the configured unit (tokens or characters)."""
        if self.unit == "characters":
            return len(text)

        # Token unit
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Fallback whitespace tokens if tiktoken is not present
        return len(text.split())

    def _split_text_by_boundary(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text using the hierarchy of boundary separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        # Find the highest-priority separator present in text
        for idx, sep in enumerate(separators):
            if sep == "":
                separator = ""
                new_separators = []
                break
            if sep in text:
                separator = sep
                new_separators = separators[idx + 1 :]
                break

        # Split text by the selected separator
        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator and good_splits:
                # Retain separator for formatting where appropriate
                pass
            if self._length_fn(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # Lowest level reached: slice forcefully by unit
                    final_chunks.extend(self._slice_forcefully(s))
                else:
                    other_chunks = self._split_text_by_boundary(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _slice_forcefully(self, text: str) -> List[str]:
        """Forcefully slice a large unsplittable string to fit chunk_size."""
        step = max(1, self.chunk_size - self.chunk_overlap)
        if self.unit == "characters" or self._tokenizer is None:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), step)
                if text[i : i + self.chunk_size].strip()
            ]

        # Token-based forceful slicing
        tokens = self._tokenizer.encode(text)
        chunks: List[str] = []
        for i in range(0, len(tokens), step):
            token_slice = tokens[i : i + self.chunk_size]
            decoded = self._tokenizer.decode(token_slice).strip()
            if decoded:
                chunks.append(decoded)
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge smaller splits into chunks respecting chunk_size and chunk_overlap."""
        merged: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for piece in splits:
            piece_len = self._length_fn(piece)
            sep_len = self._length_fn(separator) if current_doc and separator else 0

            if total_len + piece_len + sep_len > self.chunk_size and current_doc:
                chunk_str = separator.join(current_doc).strip()
                if chunk_str:
                    merged.append(chunk_str)

                # Overlap accumulation: keep tail pieces that fit in chunk_overlap
                while current_doc and (
                    total_len > self.chunk_overlap
                    or (total_len + piece_len + sep_len > self.chunk_size and total_len > 0)
                ):
                    removed = current_doc.pop(0)
                    total_len -= self._length_fn(removed)
                    if separator and current_doc:
                        total_len -= self._length_fn(separator)

            current_doc.append(piece)
            total_len += piece_len + (sep_len if len(current_doc) > 1 else 0)

        if current_doc:
            chunk_str = separator.join(current_doc).strip()
            if chunk_str:
                merged.append(chunk_str)

        return merged

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc",
        base_metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None,
        base_char_offset: int = 0,
    ) -> List[TextChunk]:
        """Split a raw text string into chunks with character offsets.

        Args:
            text: Raw input text.
            document_id: Identifier for parent document.
            base_metadata: Optional metadata to attach to each chunk.
            page_number: Optional page number.
            base_char_offset: Starting character offset in parent document.

        Returns:
            List[TextChunk]: List of generated, non-empty, deduplicated chunks.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        # If already fits within a single chunk
        if self._length_fn(cleaned) <= self.chunk_size:
            raw_chunks = [cleaned]
        else:
            raw_chunks = self._split_text_by_boundary(cleaned, self.separators)

        result_chunks: List[TextChunk] = []
        seen_texts: Set[str] = set()
        chunk_idx = 0
        search_start = 0

        for raw_c in raw_chunks:
            c_text = raw_c.strip()
            if not c_text:
                continue

            # Prevent duplicate identical chunks
            if c_text in seen_texts:
                continue
            seen_texts.add(c_text)

            # Find character offset within text
            char_pos = text.find(c_text, search_start)
            if char_pos == -1:
                char_pos = search_start
            search_end = char_pos + len(c_text)
            # Advance search_start for subsequent chunks
            search_start = max(0, search_end - (self.chunk_overlap * 2))

            start_char = base_char_offset + char_pos
            end_char = base_char_offset + search_end

            # Deterministic chunk ID
            chunk_id = f"{document_id}::c{chunk_idx:04d}"
            if page_number is not None:
                chunk_id = f"{document_id}::p{page_number:02d}::c{chunk_idx:04d}"

            meta: Dict[str, Any] = dict(base_metadata or {})
            meta.update(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "start_char": start_char,
                    "end_char": end_char,
                    "char_count": len(c_text),
                    "token_count": self._length_fn(c_text)
                    if self.unit == "tokens"
                    else None,
                    "chunk_unit": self.unit,
                }
            )
            if page_number is not None:
                meta["page_number"] = page_number
                meta["page_numbers"] = [page_number]

            result_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=c_text,
                    metadata=meta,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

        return result_chunks

    def chunk_document(self, document: LoadedDocument) -> List[TextChunk]:
        """Split an ingested LoadedDocument into chunks, preserving page provenance.

        Args:
            document: Ingested document.

        Returns:
            List[TextChunk]: Generated text chunks.
        """
        if not document.content.strip() and not document.pages:
            return []

        doc_id = document.document_id
        all_chunks: List[TextChunk] = []
        chunk_counter = 0

        # If document has distinct pages, chunk page-by-page to preserve provenance
        if document.pages:
            doc_char_offset = 0
            for page in document.pages:
                if not page.text.strip():
                    doc_char_offset += len(page.text) + 2
                    continue

                page_chunks = self.chunk_text(
                    text=page.text,
                    document_id=doc_id,
                    base_metadata=document.metadata,
                    page_number=page.page_number,
                    base_char_offset=doc_char_offset,
                )

                for chunk in page_chunks:
                    # Ensure global chunk index remains strictly monotonically increasing
                    adjusted_meta = dict(chunk.metadata)
                    adjusted_meta["chunk_index"] = chunk_counter
                    adjusted_meta["page_start_char"] = chunk.metadata["start_char"] - doc_char_offset
                    adjusted_meta["page_end_char"] = chunk.metadata["end_char"] - doc_char_offset

                    all_chunks.append(
                        TextChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            metadata=adjusted_meta,
                            chunk_index=chunk_counter,
                        )
                    )
                    chunk_counter += 1

                doc_char_offset += len(page.text) + 2  # Separator allowance

            return all_chunks

        # Fallback for documents loaded without page breakdowns
        return self.chunk_text(
            text=document.content,
            document_id=doc_id,
            base_metadata=document.metadata,
        )

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[TextChunk]:
        """Split multiple documents into chunks.

        Args:
            documents: List of ingested documents.

        Returns:
            List[TextChunk]: Combined list of text chunks.
        """
        combined: List[TextChunk] = []
        for doc in documents:
            combined.extend(self.chunk_document(doc))
        return combined
