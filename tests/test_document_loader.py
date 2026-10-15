"""Unit tests for document loader module using synthetic test documents."""

from pathlib import Path

import pymupdf
import pytest

from src.document_loader import (
    DocumentExtractionError,
    DocumentLoader,
    DocumentNotFoundError,
    InvalidDocumentError,
    LoadedDocument,
    normalize_whitespace,
)


def _create_synthetic_pdf(
    file_path: Path,
    pages_text: list[str],
    title: str = "Synthetic Test Paper",
) -> Path:
    """Helper to create a synthetic multi-page PDF using PyMuPDF."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), text, fontsize=11)
    doc.set_metadata({"title": title, "author": "ReRAG Research Team"})
    doc.save(file_path)
    doc.close()
    return file_path


def _create_synthetic_scanned_pdf(file_path: Path) -> Path:
    """Helper to create a synthetic scanned/image PDF with drawings but zero text."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Draw vector rectangle without any text
    page.draw_rect(pymupdf.Rect(50, 50, 400, 400), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    doc.save(file_path)
    doc.close()
    return file_path


def test_document_loader_initialization():
    """Verify loader initializes with default and custom extensions."""
    loader = DocumentLoader()
    assert ".pdf" in loader.allowed_extensions
    assert ".txt" in loader.allowed_extensions
    assert ".md" in loader.allowed_extensions

    custom_loader = DocumentLoader(allowed_extensions=[".PDF", ".CUSTOM"])
    assert ".pdf" in custom_loader.allowed_extensions
    assert ".custom" in custom_loader.allowed_extensions


def test_load_normal_single_page_pdf(tmp_path: Path):
    """Verify loading and text extraction from a standard single-page PDF."""
    pdf_path = tmp_path / "single_page.pdf"
    content = "ReRAG introduces a verification step to reduce factual hallucinations in LLMs."
    _create_synthetic_pdf(pdf_path, [content], title="Single Page Study")

    loader = DocumentLoader()
    doc = loader.load_file(pdf_path)

    assert isinstance(doc, LoadedDocument)
    assert doc.total_pages == 1
    assert "ReRAG introduces a verification step" in doc.content
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].char_count > 0
    assert doc.needs_ocr is False
    assert doc.metadata["title"] == "Single Page Study"
    assert doc.metadata["document_id"] == "single_page"


def test_load_multi_page_pdf(tmp_path: Path):
    """Verify page numbers, individual page contents, and combined content across multiple pages."""
    pdf_path = tmp_path / "multi_page.pdf"
    pages_text = [
        "Page 1: Introduction to Retrieval-Augmented Generation.",
        "Page 2: Hallucination rate analysis under baseline single-hop retrieval.",
        "Page 3: Iterative evidence refinement and consistency checking results.",
    ]
    _create_synthetic_pdf(pdf_path, pages_text, title="Multi-Page Paper")

    loader = DocumentLoader()
    doc = loader.load_file(pdf_path)

    assert doc.total_pages == 3
    for idx, expected_text in enumerate(pages_text, start=1):
        page = doc.pages[idx - 1]
        assert page.page_number == idx
        assert expected_text in page.text

    assert "Page 1: Introduction" in doc.content
    assert "Page 2: Hallucination" in doc.content
    assert "Page 3: Iterative" in doc.content
    assert doc.needs_ocr is False


def test_load_scanned_pdf_needs_ocr(tmp_path: Path):
    """Verify that image-only or scanned PDFs are flagged as needing OCR and not silently faked."""
    pdf_path = tmp_path / "scanned_doc.pdf"
    _create_synthetic_scanned_pdf(pdf_path)

    loader = DocumentLoader()
    doc = loader.load_file(pdf_path)

    assert doc.total_pages == 1
    assert doc.content == ""
    assert doc.needs_ocr is True
    assert len(doc.warnings) > 0
    assert any("OCR" in w for w in doc.warnings)
    assert doc.metadata["needs_ocr"] is True


def test_load_empty_pdf_0_pages(tmp_path: Path, monkeypatch):
    """Verify graceful handling when a PDF catalog contains 0 pages."""
    pdf_path = tmp_path / "zero_pages.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy valid bytes")

    class MockDoc:
        metadata = {"format": "PDF"}

        def __len__(self):
            return 0

        def close(self):
            pass

    monkeypatch.setattr(pymupdf, "open", lambda p: MockDoc())

    loader = DocumentLoader()
    loaded = loader.load_file(pdf_path)

    assert loaded.total_pages == 0
    assert loaded.content == ""
    assert loaded.needs_ocr is False
    assert any("0 pages" in w for w in loaded.warnings)


def test_load_blank_page_pdf(tmp_path: Path):
    """Verify handling of PDF with a completely blank page containing no text."""
    pdf_path = tmp_path / "blank_page.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    loader = DocumentLoader()
    loaded = loader.load_file(pdf_path)

    assert loaded.total_pages == 1
    assert loaded.content == ""
    assert loaded.needs_ocr is True
    assert any("OCR" in w for w in loaded.warnings)


def test_load_empty_file_0_bytes(tmp_path: Path):
    """Verify graceful handling of empty zero-byte files."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    loader = DocumentLoader()
    loaded = loader.load_file(empty_pdf)

    assert loaded.total_pages == 0
    assert loaded.content == ""
    assert any("0 bytes" in w for w in loaded.warnings)


def test_load_corrupted_invalid_pdf(tmp_path: Path):
    """Verify corrupted non-PDF files trigger InvalidDocumentError."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nCorrupt header and random bytes \x00\xff\xfe")

    loader = DocumentLoader()
    with pytest.raises(InvalidDocumentError, match="Failed to open PDF document"):
        loader.load_file(corrupt_pdf)


def test_load_nonexistent_file(tmp_path: Path):
    """Verify missing file raises DocumentNotFoundError."""
    missing = tmp_path / "missing.pdf"
    loader = DocumentLoader()
    with pytest.raises(DocumentNotFoundError):
        loader.load_file(missing)


def test_load_unsupported_extension(tmp_path: Path):
    """Verify unsupported file extension raises InvalidDocumentError."""
    unsupported = tmp_path / "document.docx"
    unsupported.write_text("sample content")

    loader = DocumentLoader()
    with pytest.raises(InvalidDocumentError, match="Unsupported file extension"):
        loader.load_file(unsupported)


def test_load_text_file(tmp_path: Path):
    """Verify plain text and markdown file loading."""
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Line 1 of notes.\n\nLine 2 after paragraph break.", encoding="utf-8")

    loader = DocumentLoader()
    doc = loader.load_file(txt_path)

    assert doc.total_pages == 1
    assert "Line 1 of notes." in doc.content
    assert doc.needs_ocr is False
    assert doc.metadata["format"] == "TXT"


def test_load_directory(tmp_path: Path):
    """Verify batch directory loading."""
    _create_synthetic_pdf(tmp_path / "doc1.pdf", ["Document 1 content."])
    _create_synthetic_pdf(tmp_path / "doc2.pdf", ["Document 2 content."])
    (tmp_path / "doc3.txt").write_text("Document 3 text.", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path)

    assert len(docs) == 3
    doc_ids = {d.document_id for d in docs}
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids


def test_whitespace_normalization_preserving_tables_and_paragraphs():
    """Verify whitespace normalization preserves tabular layout, paragraphs, and sentence boundaries."""
    raw_input = (
        "Model               Precision    Recall     F1\r\n"
        "Baseline RAG        0.65         0.70       0.674\r\n"
        "ReRAG (Iterative)   0.82         0.86       0.839\r\n"
        "\r\n"
        "\r\n"
        "\r\n"
        "This is paragraph one.\r\n"
        "It contains two sentences. Sentence two follows.\r\n"
        "\r\n"
        "\r\n"
        "This is paragraph two after multiple blank lines.   \t"
    )

    normalized = normalize_whitespace(raw_input)

    # Paragraph double newlines preserved
    assert "paragraph one.\nIt contains" in normalized
    assert "paragraph one.\nIt contains two sentences. Sentence two follows.\n\nThis is paragraph two" in normalized

    # Excessive 3+ newlines compressed to 2
    assert "\n\n\n" not in normalized

    # Tabular column spaces preserved
    assert "Model               Precision    Recall     F1" in normalized
    assert "ReRAG (Iterative)   0.82         0.86       0.839" in normalized

    # Sentence boundary and punctuation preserved
    assert "It contains two sentences. Sentence two follows." in normalized

    # Trailing spaces stripped
    assert not normalized.endswith(" ")
    assert not normalized.endswith("\t")
