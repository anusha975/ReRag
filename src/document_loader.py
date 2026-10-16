"""Document loader module for ReRAG.

Loads and extracts text and metadata page-by-page from PDF files using PyMuPDF,
as well as supporting plain text and markdown documents. Handles empty files,
scanned documents lacking extractable text, invalid structures, and reports OCR
requirements explicitly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import pymupdf
except (ImportError, MemoryError):
    pymupdf = None

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Base exception for document loading and parsing errors."""


class DocumentNotFoundError(DocumentLoaderError, FileNotFoundError):
    """Raised when a requested document does not exist."""


class InvalidDocumentError(DocumentLoaderError, ValueError):
    """Raised when a document is corrupted, malformed, or has an unsupported format."""


class DocumentExtractionError(DocumentLoaderError, RuntimeError):
    """Raised when an unrecoverable error occurs during text extraction."""


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without destroying paragraphs, punctuation, or tables.

    Rules applied:
    - Normalizes line endings (CRLF and CR converted to LF).
    - Removes zero-width characters and replaces non-breaking spaces with standard spaces.
    - Strips trailing whitespace per line to eliminate ragged space noise,
      while preserving leading whitespace and internal column alignment for text tables.
    - Preserves single newlines between table rows or connected text lines.
    - Preserves double newlines between paragraphs.
    - Compresses 3 or more consecutive newlines down to 2 (paragraph boundary).

    Args:
        text: Raw extracted text string.

    Returns:
        str: Cleaned, structure-preserving normalized text.
    """
    if not text:
        return ""

    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces, remove zero-width spaces and BOM
    normalized = (
        normalized.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    # Process line by line: rstrip removes trailing whitespace noise
    # Leading whitespace is intentionally preserved to maintain table indents.
    lines = [line.rstrip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)

    # Compress 3 or more consecutive newlines to double newlines (paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


@dataclass(frozen=True)
class PageContent:
    """Represents text and metadata extracted from an individual page."""

    page_number: int
    text: str
    char_count: int
    has_images: bool = False
    needs_ocr: bool = False


@dataclass
class LoadedDocument:
    """Represents an ingested document with full content, page details, and metadata."""

    content: str
    metadata: Dict[str, Any]
    source_path: Path
    pages: List[PageContent] = field(default_factory=list)
    needs_ocr: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        """Return total page count."""
        return len(self.pages)

    @property
    def document_id(self) -> str:
        """Return unique document identifier derived from metadata or filename."""
        return str(self.metadata.get("document_id") or self.source_path.stem)


class DocumentLoader:
    """Ingests and parses documents into structured text representations."""

    DEFAULT_ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md"]

    def __init__(
        self,
        allowed_extensions: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize document loader.

        Args:
            allowed_extensions: Sequence of supported file extensions (case-insensitive).
        """
        extensions = allowed_extensions or self.DEFAULT_ALLOWED_EXTENSIONS
        self.allowed_extensions = {ext.lower() for ext in extensions}

    def load_file(self, file_path: Path | str) -> LoadedDocument:
        """Load and extract text and metadata from a single document.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            LoadedDocument: Extracted document representation.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            InvalidDocumentError: If the file format is unsupported or corrupted.
            DocumentExtractionError: If extraction fails unrecoverably.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise DocumentNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise InvalidDocumentError(f"Target path is not a file: {path}")

        ext = path.suffix.lower()
        if ext not in self.allowed_extensions:
            raise InvalidDocumentError(
                f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(self.allowed_extensions)}"
            )

        # Handle zero-byte empty files gracefully
        if path.stat().st_size == 0:
            warning = f"Document '{path.name}' is an empty file (0 bytes)."
            logger.warning(warning)
            return LoadedDocument(
                content="",
                metadata={
                    "source_path": str(path),
                    "filename": path.name,
                    "document_id": path.stem,
                    "total_pages": 0,
                    "format": ext.lstrip(".").upper(),
                    "needs_ocr": False,
                    "warnings": [warning],
                },
                source_path=path,
                pages=[],
                needs_ocr=False,
                warnings=[warning],
            )

        if ext == ".pdf":
            return self._load_pdf(path)
        return self._load_text(path)

    def _load_pdf(self, path: Path) -> LoadedDocument:
        """Extract text page-by-page from a PDF using PyMuPDF, falling back to pypdf if memory constrained."""
        global pymupdf
        if pymupdf is None:
            try:
                import pymupdf
            except (ImportError, MemoryError):
                pymupdf = None

        if pymupdf is None:
            return self._load_pdf_fallback_pypdf(path)

        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to open PDF document '{path.name}'. File may be corrupt or invalid: {exc}"
            ) from exc

        try:
            raw_meta = doc.metadata or {}
            total_pages = len(doc)

            metadata: Dict[str, Any] = {
                "source_path": str(path),
                "filename": path.name,
                "document_id": path.stem,
                "format": raw_meta.get("format", "PDF"),
                "title": raw_meta.get("title", ""),
                "author": raw_meta.get("author", ""),
                "subject": raw_meta.get("subject", ""),
                "keywords": raw_meta.get("keywords", ""),
                "creator": raw_meta.get("creator", ""),
                "producer": raw_meta.get("producer", ""),
                "creation_date": raw_meta.get("creationDate", ""),
                "mod_date": raw_meta.get("modDate", ""),
                "total_pages": total_pages,
            }

            # Handle 0-page PDFs
            if total_pages == 0:
                warning = f"PDF '{path.name}' contains 0 pages."
                logger.warning(warning)
                metadata["needs_ocr"] = False
                metadata["warnings"] = [warning]
                return LoadedDocument(
                    content="",
                    metadata=metadata,
                    source_path=path,
                    pages=[],
                    needs_ocr=False,
                    warnings=[warning],
                )

            pages: List[PageContent] = []
            pages_needing_ocr: List[int] = []

            for page_idx, page in enumerate(doc):
                page_number = page_idx + 1
                try:
                    raw_text = page.get_text("text") or ""
                except Exception as exc:
                    logger.error(
                        "Error extracting text from page %d of '%s': %s",
                        page_number,
                        path.name,
                        exc,
                    )
                    raw_text = ""

                clean_text = normalize_whitespace(raw_text)

                # Inspect page visual elements to detect scanned / image-only content
                images = page.get_images()
                drawings = page.get_drawings()
                has_visuals = bool(images or drawings)

                # Detect if page requires OCR
                page_needs_ocr = False
                if not clean_text and has_visuals:
                    page_needs_ocr = True
                    pages_needing_ocr.append(page_number)

                pages.append(
                    PageContent(
                        page_number=page_number,
                        text=clean_text,
                        char_count=len(clean_text),
                        has_images=bool(images),
                        needs_ocr=page_needs_ocr,
                    )
                )

            total_extracted_chars = sum(p.char_count for p in pages)
            doc_needs_ocr = False
            warnings: List[str] = []

            if total_pages > 0 and total_extracted_chars == 0:
                doc_needs_ocr = True
                warning = (
                    f"Document '{path.name}' contains no extractable text across all {total_pages} page(s). "
                    "The document appears to be scanned or image-based and requires OCR to extract content."
                )
                warnings.append(warning)
                logger.warning(warning)
            elif pages_needing_ocr:
                doc_needs_ocr = True
                warning = (
                    f"Page(s) {pages_needing_ocr} of document '{path.name}' contain no extractable text "
                    "but contain visual images/drawings; OCR is required for full extraction."
                )
                warnings.append(warning)
                logger.warning(warning)

            combined_content = "\n\n".join(p.text for p in pages if p.text)

            metadata["needs_ocr"] = doc_needs_ocr
            metadata["warnings"] = warnings

            return LoadedDocument(
                content=combined_content,
                metadata=metadata,
                source_path=path,
                pages=pages,
                needs_ocr=doc_needs_ocr,
                warnings=warnings,
            )

        except Exception as exc:
            if isinstance(exc, DocumentLoaderError):
                raise
            raise DocumentExtractionError(
                f"Unexpected error while extracting content from '{path.name}': {exc}"
            ) from exc
        finally:
            doc.close()

    def _load_pdf_fallback_pypdf(self, path: Path) -> LoadedDocument:
        """Fallback PDF loader using lightweight pypdf when system memory limits C-library loading."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to open PDF document '{path.name}'. File may be corrupt or invalid: {exc}"
            ) from exc

        total_pages = len(reader.pages)
        pages: List[PageContent] = []
        for idx, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            clean_text = normalize_whitespace(raw_text)
            has_images = bool(getattr(page, "images", None))
            pages.append(
                PageContent(
                    page_number=idx + 1,
                    text=clean_text,
                    char_count=len(clean_text),
                    has_images=has_images,
                    needs_ocr=(len(clean_text) == 0 and has_images),
                )
            )

        total_chars = sum(p.char_count for p in pages)
        needs_ocr = total_pages > 0 and total_chars == 0
        warnings: List[str] = []
        if needs_ocr:
            warnings.append(
                f"Document '{path.name}' contains no extractable text across all {total_pages} page(s)."
            )

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "filename": path.name,
            "document_id": path.stem,
            "format": "PDF",
            "total_pages": total_pages,
            "needs_ocr": needs_ocr,
            "warnings": warnings,
        }
        return LoadedDocument(
            content="\n\n".join(p.text for p in pages if p.text),
            metadata=metadata,
            source_path=path,
            pages=pages,
            needs_ocr=needs_ocr,
            warnings=warnings,
        )

    def _load_text(self, path: Path) -> LoadedDocument:
        """Load and normalize plain text or markdown files."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to read text file '{path.name}': {exc}"
            ) from exc

        clean_text = normalize_whitespace(raw_text)
        page = PageContent(
            page_number=1,
            text=clean_text,
            char_count=len(clean_text),
            has_images=False,
            needs_ocr=False,
        )

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "filename": path.name,
            "document_id": path.stem,
            "format": path.suffix.lstrip(".").upper(),
            "total_pages": 1,
            "needs_ocr": False,
            "warnings": [],
        }

        return LoadedDocument(
            content=clean_text,
            metadata=metadata,
            source_path=path,
            pages=[page],
            needs_ocr=False,
            warnings=[],
        )

    def load_directory(
        self,
        directory_path: Path | str,
        recursive: bool = False,
    ) -> List[LoadedDocument]:
        """Load all matching documents from a target directory.

        Args:
            directory_path: Target directory path.
            recursive: If True, search subdirectories recursively.

        Returns:
            List[LoadedDocument]: Collection of parsed documents.

        Raises:
            DocumentNotFoundError: If the directory does not exist.
            InvalidDocumentError: If directory_path is not a directory.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.exists():
            raise DocumentNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise InvalidDocumentError(f"Target path is not a directory: {dir_path}")

        pattern = "**/*" if recursive else "*"
        documents: List[LoadedDocument] = []

        for candidate in sorted(dir_path.glob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in self.allowed_extensions:
                try:
                    doc = self.load_file(candidate)
                    documents.append(doc)
                except Exception as exc:
                    logger.error("Failed to load '%s': %s", candidate.name, exc)

        return documents
