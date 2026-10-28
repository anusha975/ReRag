"""Main Streamlit dashboard page and section controllers for the ReRAG research platform.

Coordinates five primary operational sections:
1. Document Workspace: Document uploading, extraction, text & chunk previews, and indexing.
2. RAG Chat: Interactive grounded question answering, passage citation, and ReRAG refinement traces.
3. Optimization Lab: Greedy adaptive coordinate search for chunking parameters with live progress.
4. Results & Comparison: Empirical comparative analysis, Altair charts, and held-out generalization.
5. Experiment History: Persistent JSON/CSV report loading, auditing, and file downloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.chunker import TextChunk, TextChunker
from src.comparison_runner import ComparisonMethodResult, ComparisonResult, ComparisonRunner
from src.config import Settings, get_settings
from src.dataset_workflow import (
    DatasetSplits,
    DatasetStatistics,
    DatasetValidator,
    DatasetValidationResult,
    audit_split_leakage,
    compute_dataset_statistics,
    export_dataset_to_csv,
    export_dataset_to_json,
    load_dataset_from_csv,
    load_dataset_from_json,
    split_train_val_test,
    split_train_val_test_leakage_resistant,
)
from src.document_loader import (
    DocumentExtractionError,
    DocumentLoader,
    DocumentLoaderError,
    DocumentNotFoundError,
    InvalidDocumentError,
    LoadedDocument,
)
from src.embeddings import EmbeddingManager
from src.evaluator import EvaluationReport, EvaluationSample, Evaluator, load_dataset, split_train_test
from src.experiment_runner import ExperimentRunner
from src.llm import BaseLLMClient, MockLLMClient, OllamaClient
from src.metrics import METRIC_METADATA_REGISTRY, format_metric_value
from src.optimizer import (
    AdaptiveChunkOptimizer,
    AdaptiveSearchConfig,
    CandidateTrial,
    ChunkConfig,
    ChunkOptimizerFactory,
    FixedConfigOptimizer,
    GeneralizationAwareConfig,
    GeneralizationAwareOptimizer,
    OptimizationObjective,
    OptimizationResult,
    RandomSearchConfig,
    RandomSearchOptimizer,
    SearchBounds,
    SimulatedAnnealingConfig,
    SimulatedAnnealingOptimizer,
)
from src.prompts import INSUFFICIENT_INFORMATION_RESPONSE
from src.rag_pipeline import PipelineResult, RAGPipeline
from src.retriever import RetrievedContext, Retriever
from src.vector_store import VectorStore
from ui.components import (
    inject_custom_styles,
    render_candidate_chart,
    render_candidate_table,
    render_comparison_view,
    render_dataset_split_summary,
    render_empty_state,
    render_header,
    render_multi_method_charts,
    render_multi_method_comparison_table,
    render_retrieved_passage,
    render_sidebar_config,
    render_status_badge,
    render_trace_viewer,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_shared_embedding_manager() -> EmbeddingManager:
    """Load and cache the shared SentenceTransformer embedding model."""
    mgr = EmbeddingManager()
    mgr.warmup()
    return mgr


def init_session_state(settings: Settings) -> None:
    """Initialize all persistent session state variables safely."""
    if "settings" not in st.session_state:
        st.session_state.settings = settings
    if "loaded_document" not in st.session_state:
        st.session_state.loaded_document = None
    if "loaded_file_hash" not in st.session_state:
        st.session_state.loaded_file_hash = None
    if "indexed_document_id" not in st.session_state:
        st.session_state.indexed_document_id = None
    if "indexed_chunks_count" not in st.session_state:
        st.session_state.indexed_chunks_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "latest_optimization_result" not in st.session_state:
        st.session_state.latest_optimization_result = None
    if "latest_comparison_result" not in st.session_state:
        st.session_state.latest_comparison_result = None
    if "is_optimizing" not in st.session_state:
        st.session_state.is_optimizing = False
    if "offline_mode" not in st.session_state:
        st.session_state.offline_mode = False
    if "active_collection" not in st.session_state:
        st.session_state.active_collection = settings.vector_store.collection_name
    if "rag_query_input" not in st.session_state:
        st.session_state.rag_query_input = ""
    if "last_index_notice" not in st.session_state:
        st.session_state.last_index_notice = None
    if "active_dataset_samples" not in st.session_state:
        st.session_state.active_dataset_samples = None
    if "active_dataset_name" not in st.session_state:
        st.session_state.active_dataset_name = None
    if "active_dataset_splits" not in st.session_state:
        st.session_state.active_dataset_splits = None
    if "split_leakage_audit" not in st.session_state:
        st.session_state.split_leakage_audit = None

    # Vector store & retriever in session (lazy initialization on first use)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Start non-blocking background model warm-up so embedding model is ready in RAM
    if "warmup_started" not in st.session_state:
        st.session_state.warmup_started = True
        import threading
        threading.Thread(target=get_shared_embedding_manager, daemon=True).start()


def get_vector_store(settings: Settings) -> VectorStore:
    """Lazily load VectorStore and EmbeddingManager on demand."""
    emb_mgr = get_shared_embedding_manager()
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore(
            persist_dir=settings.vector_store.persist_dir,
            collection_name=st.session_state.active_collection,
            embedding_manager=emb_mgr,
        )
    return st.session_state.vector_store


def get_retriever(settings: Settings) -> Retriever:
    """Lazily load Retriever on demand."""
    vs = get_vector_store(settings)
    if st.session_state.retriever is None:
        st.session_state.retriever = Retriever(
            vector_store=vs,
            embedding_manager=vs.embedding_manager,
            top_k=settings.retriever.top_k,
            similarity_threshold=settings.retriever.similarity_threshold,
        )
    return st.session_state.retriever


def get_llm_client(settings: Settings, offline_mode: bool) -> BaseLLMClient:
    """Instantiate appropriate LLM client based on online health and offline mode toggle."""
    if offline_mode:
        response_map = {
            "primary architectural purpose": (
                "ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation."
            ),
            "purpose of rerag": (
                "ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation."
            ),
            "rerag purpose": (
                "ReRAG is an architecture designed for reducing factual hallucinations in retrieval-augmented generation."
            ),
            "verification threshold": (
                "The verification step checks candidate claims against retrieved context passages using a consistency threshold of 0.7."
            ),
            "verification step": (
                "The verification step checks candidate claims against retrieved context passages using a consistency threshold of 0.7."
            ),
            "threshold": (
                "The verification step checks candidate claims against retrieved context passages using a consistency threshold of 0.7."
            ),
            "james webb": (
                "The James Webb Space Telescope observes infrared wavelengths to detect early cosmic structures."
            ),
            "wavelengths": (
                "The James Webb Space Telescope observes infrared wavelengths to detect early cosmic structures."
            ),
            "photosynthesis": (
                "Photosynthesis converts solar light into chemical sugars inside plant chloroplasts."
            ),
            "chloroplasts": (
                "Photosynthesis converts solar light into chemical sugars inside plant chloroplasts."
            ),
            "quantum computing": (
                "Quantum computing leverages superposition and entanglement to solve discrete optimization problems."
            ),
            "hyperparameter": (
                "Chunk size and chunk overlap are the primary hyperparameters adjusted during grid search optimization."
            ),
            "riverbend": (
                "The Riverbend Community Energy Pilot was launched in March 2024 to test whether shared rooftop solar installations can reduce electricity costs."
            ),
            "combined installed capacity": (
                "The combined installed capacity across the three sites is 72 kW (North Library: 24 kW, Eastside Health Centre: 18 kW, Riverbend Skills Hub: 30 kW)."
            ),
            "mira sen": (
                "Mira Sen coordinates schedules, community updates, and quarterly reporting for the Riverbend Community Energy Pilot."
            ),
            "daniel okafor": (
                "Daniel Okafor manages technical troubleshooting and installation records for the Riverbend project."
            ),
            "solar generation": (
                "The three sites reported a combined 8,640 kWh of solar generation for July through September 2024."
            ),
            "communications fault": (
                "On 12 August 2024, the North Library monitoring gateway stopped sending data due to a damaged network cable, which was replaced on 13 August."
            ),
        }
        return MockLLMClient(
            default_response="Based on the retrieved context, the evidence directly supports the stated claims.",
            model="deterministic-mock-llm",
            response_map=response_map,
            is_healthy=True,
        )

    # Online Ollama client
    client = OllamaClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=settings.llm.timeout,
        temperature=settings.llm.temperature,
        seed=settings.llm.seed,
    )
    return client


# ==============================================================================
# SECTION 1: DOCUMENT WORKSPACE
# ==============================================================================


def render_document_workspace(settings: Settings) -> None:
    """Render the Document Workspace section."""
    st.markdown("## 📁 Document Workspace")
    st.markdown(
        "Upload research documents (PDF, TXT, MD), inspect page-by-page text extractions, "
        "configure chunking boundaries, and build search indices with provenance tracking."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📤 Ingestion & Source Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF or text document",
            type=["pdf", "txt", "md"],
            help="Upload a research publication or domain document for chunking and vector indexing.",
        )

        sample_pdf_path = settings.paths.documents_dir / "sample_rerag_paper.pdf"
        load_sample = False
        if sample_pdf_path.exists():
            if st.button("📄 Load Bundled Sample Research Paper (sample_rerag_paper.pdf)"):
                load_sample = True

        target_file_path: Optional[Path] = None

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            target_file_path = settings.paths.documents_dir / uploaded_file.name

            # Avoid re-saving unnecessarily
            if not target_file_path.exists() or st.session_state.loaded_file_hash != file_hash:
                with open(target_file_path, "wb") as f:
                    f.write(file_bytes)
                st.session_state.loaded_file_hash = file_hash
        elif load_sample:
            target_file_path = sample_pdf_path
            with open(target_file_path, "rb") as f:
                st.session_state.loaded_file_hash = hashlib.sha256(f.read()).hexdigest()

        # Document loading & extraction
        if target_file_path and target_file_path.exists():
            try:
                loader = DocumentLoader()
                loaded_doc = loader.load_file(target_file_path)
                st.session_state.loaded_document = loaded_doc
                st.session_state.last_index_notice = None
                st.success(f"Successfully loaded '{target_file_path.name}' ({loaded_doc.total_pages} pages).")
            except (DocumentNotFoundError, InvalidDocumentError, DocumentExtractionError, DocumentLoaderError) as exc:
                st.error(f"Extraction Error: {exc}")
                logger.error("Error loading document %s: %s", target_file_path, exc)
            except Exception as exc:
                st.error(f"Unexpected error loading document: {exc}")
                logger.exception("Unexpected loader exception")

    doc: Optional[LoadedDocument] = st.session_state.loaded_document

    with col2:
        st.markdown("### 📋 Document Status & Metadata")
        if doc is None:
            st.info("No document currently loaded. Upload a document or load the bundled sample.")
        else:
            is_indexed = (st.session_state.indexed_document_id == doc.document_id and st.session_state.indexed_chunks_count > 0)
            st.markdown(
                f"""
                <div class="rerag-card">
                    <p style="margin: 0 0 0.4rem 0;"><b>Document:</b> <code>{doc.source_path.name}</code></p>
                    <p style="margin: 0 0 0.4rem 0;"><b>Document ID:</b> <code>{doc.document_id}</code></p>
                    <p style="margin: 0 0 0.4rem 0;"><b>Total Pages:</b> {doc.total_pages}</p>
                    <p style="margin: 0 0 0.4rem 0;"><b>Total Characters:</b> {len(doc.content):,}</p>
                    <p style="margin: 0 0 0.4rem 0;"><b>OCR Required:</b> {'⚠️ Yes' if doc.needs_ocr else '✅ No'}</p>
                    <p style="margin: 0 0 0.4rem 0;"><b>Indexing Status:</b> {'<span class="rerag-badge rerag-badge-accepted">INDEXED</span>' if is_indexed else '<span class="rerag-badge rerag-badge-rejected">NOT INDEXED</span>'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if doc.warnings:
                for w in doc.warnings:
                    st.warning(f"⚠️ {w}")

    if doc is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Chunking Configuration & Vector Indexing")

        c_col1, c_col2, c_col3 = st.columns([2, 2, 2])
        with c_col1:
            chunk_size = st.number_input(
                "Chunk Size (Characters)",
                min_value=50,
                max_value=3000,
                value=int(settings.document.chunk_size),
                step=50,
                help="Maximum character window length for each passage chunk.",
            )
        with c_col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (Characters)",
                min_value=0,
                max_value=1000,
                value=int(settings.document.chunk_overlap),
                step=25,
                help="Overlapping characters between adjacent chunks to maintain semantic continuity.",
            )
        with c_col3:
            chunk_unit = st.selectbox(
                "Splitting Unit",
                options=["characters", "tokens"],
                index=0,
                help="Measurement unit for sliding chunk window.",
            )

        # Validation check
        if chunk_overlap >= chunk_size:
            st.error("Validation Error: Chunk overlap must be strictly less than chunk size.")
            return

        # Preview & Indexing Buttons
        is_already_indexed = (
            st.session_state.indexed_document_id == doc.document_id
            and st.session_state.indexed_chunks_count > 0
        )
        button_label = "🔄 Re-Index into Vector Store" if is_already_indexed else "🚀 Index into Vector Store"

        b_col1, b_col2 = st.columns([2, 4])
        with b_col1:
            index_btn = st.button(button_label, type="primary", use_container_width=True)

        # Chunker generation
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, unit=chunk_unit)
        preview_chunks = chunker.chunk_document(doc)

        with b_col2:
            st.caption(f"Configured strategy produces **{len(preview_chunks)} chunks** from the active document.")

        if st.session_state.get("last_index_notice"):
            st.success(st.session_state["last_index_notice"])

        if index_btn:
            with st.spinner("Computing neural embeddings & indexing chunks into ChromaDB..."):
                t_idx0 = time.perf_counter()
                vs: VectorStore = get_vector_store(settings)
                vs.add_chunks(preview_chunks, prevent_stale=True)
                elapsed_idx = time.perf_counter() - t_idx0

                st.session_state.indexed_document_id = doc.document_id
                st.session_state.indexed_chunks_count = len(preview_chunks)
                st.session_state.last_index_notice = (
                    f"✅ Successfully indexed {len(preview_chunks)} chunks into collection "
                    f"'{vs.collection_name}' in {elapsed_idx:.2f}s."
                )
                st.rerun()

        # Document Text and Chunk Previews
        st.markdown("---")
        preview_tab1, preview_tab2 = st.tabs(["📄 Extracted Text Preview", "🧩 Representative Chunks Preview"])

        with preview_tab1:
            if doc.pages:
                selected_page_num = st.selectbox(
                    "Select Page to Preview",
                    options=[p.page_number for p in doc.pages],
                    format_func=lambda x: f"Page {x} ({next((p.char_count for p in doc.pages if p.page_number == x), 0)} chars)",
                )
                page_obj = next((p for p in doc.pages if p.page_number == selected_page_num), None)
                if page_obj:
                    st.text_area(f"Page {selected_page_num} Text Content", value=page_obj.text, height=280)
            else:
                st.text_area("Full Document Text Content", value=doc.content, height=280)

        with preview_tab2:
            st.caption(f"Displaying sample chunks (Showing first {min(5, len(preview_chunks))} of {len(preview_chunks)}):")
            for chk in preview_chunks[:5]:
                with st.expander(f"Chunk #{chk.chunk_index} — ID: {chk.chunk_id} ({chk.char_count} chars)", expanded=(chk.chunk_index == 0)):
                    p_num = chk.page_number
                    st.markdown(f"**Source Page:** `{p_num if p_num is not None else 'N/A'}`")
                    st.markdown(f"**Offsets:** `[{chk.metadata.get('start_char', 0)}:{chk.metadata.get('end_char', 0)}]`")
                    st.markdown(f"> {chk.text}")


# ==============================================================================
# SECTION 1B: EXPANDED EXPERT-ANNOTATED QA DATASET LAB
# ==============================================================================


def render_qa_dataset_lab(settings: Settings) -> None:
    """Render the Expanded Expert-Annotated QA Dataset Management & Splitting Lab.

    Provides:
    1. Dataset upload & resolution (CSV and JSON formats).
    2. Data validation: flags missing answers, duplicate questions, and invalid document IDs.
    3. Annotation status inspection: verified expert answers vs. unverified synthetic vs. reviewed.
    4. Dataset summary statistics (questions, documents, topics, verified answers).
    5. Interactive QA pair viewer, editor, and creator.
    6. Leakage-resistant dataset splitting (near-duplicate grouping, document-level, stratified).
    7. Split leakage audit verifying that held-out test answers never leak into training.
    8. Dataset and split exports in CSV and JSON formats.
    """
    st.markdown("## 📚 QA Dataset Lab")
    st.markdown(
        "Manage expert-annotated QA datasets, validate data integrity, inspect annotation status, "
        "and configure leakage-resistant partitions (Optimization / Validation / Held-Out Test). "
        "Ensures strict scientific rigor with zero leakage into the optimization loop."
    )

    eval_dir = settings.paths.evaluation_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset_files = sorted(list(eval_dir.glob("*.json")) + list(eval_dir.glob("*.csv")))

    # 1. Dataset Selection / Upload Section
    tab_select, tab_upload = st.tabs(["📁 Active Dataset Selection", "📤 Upload New Dataset"])

    with tab_select:
        c_sel1, c_sel2 = st.columns([3, 1])
        with c_sel1:
            selected_file = st.selectbox(
                "Select QA Benchmark Dataset:",
                options=dataset_files,
                format_func=lambda p: p.name,
                key="qa_lab_selected_file",
            )
        with c_sel2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Reload File", use_container_width=True):
                st.session_state.active_dataset_samples = None
                st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload QA Dataset (.json or .csv):",
            type=["json", "csv"],
            help="Upload a dataset of question-answer pairs with document references and topics.",
        )
        if uploaded_file is not None:
            save_name = uploaded_file.name
            target_path = eval_dir / save_name
            if st.button(f"💾 Save and Load `{save_name}`", type="primary"):
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded and saved `{save_name}` to {eval_dir}!")
                st.session_state.active_dataset_samples = None
                st.session_state.active_dataset_name = save_name
                st.rerun()

    # Load active dataset samples
    if selected_file and (st.session_state.active_dataset_samples is None or st.session_state.active_dataset_name != selected_file.name):
        try:
            if selected_file.suffix.lower() == ".csv":
                samples = load_dataset_from_csv(selected_file)
            else:
                samples = load_dataset_from_json(selected_file)
            st.session_state.active_dataset_samples = samples
            st.session_state.active_dataset_name = selected_file.name
        except Exception as exc:
            st.error(f"Failed to load dataset `{selected_file.name}`: {exc}")
            return

    samples = st.session_state.active_dataset_samples
    if not samples:
        st.info("No QA records loaded. Please select or upload a dataset.")
        return

    # 2. Dataset Validation & Statistics Card
    stats = compute_dataset_statistics(samples)
    val_res = DatasetValidator.validate_samples(samples)

    st.markdown("### 📊 Dataset Integrity & Verification Statistics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Questions", stats.total_questions)
    with m2:
        st.metric("Verified Expert", stats.verified_expert_count, help="Manually verified gold reference answers.")
    with m3:
        st.metric("Unreviewed Synthetic", stats.unverified_synthetic_count, help="Automatically generated questions awaiting review.")
    with m4:
        st.metric("Reviewed", stats.reviewed_count, help="Human-inspected QA pairs.")
    with m5:
        st.metric("Documents", stats.unique_documents)
    with m6:
        dup_flag = stats.near_duplicate_count + stats.duplicate_questions_count
        st.metric("Duplicate Warnings", dup_flag, help="Exact duplicates and near-duplicate pairs (Jaccard similarity >= 0.85)")

    if not val_res.is_valid:
        st.error(f"⚠️ **Dataset Schema Validation Errors Found:** {', '.join(val_res.errors)}")
    elif val_res.duplicate_questions:
        with st.expander(f"⚠️ {len(val_res.duplicate_questions)} Duplicate / Near-Duplicate Warnings Detected (Click to Inspect)"):
            for dup_msg in val_res.duplicate_questions:
                st.markdown(f"- {dup_msg}")
            st.caption("Tip: Use the 'near_duplicate_grouped' split strategy below to keep these pairs within the same partition.")

    # 3. Interactive QA Explorer & Reviewer
    st.markdown("---")
    st.markdown("### 🔍 QA Pair Explorer & Review Editor")

    f1, f2, f3, f4 = st.columns([2, 1, 1, 2])
    with f1:
        filter_topic = st.selectbox("Filter Topic:", ["All"] + sorted(list(stats.topics)))
    with f2:
        filter_diff = st.selectbox("Filter Difficulty:", ["All"] + sorted(list(stats.difficulties)))
    with f3:
        filter_status = st.selectbox("Filter Status:", ["All", "verified_expert", "unverified_synthetic", "reviewed"])
    with f4:
        search_query = st.text_input("Search Question:", placeholder="Filter by question keyword...")

    filtered_samples = samples
    if filter_topic != "All":
        filtered_samples = [s for s in filtered_samples if s.topic == filter_topic]
    if filter_diff != "All":
        filtered_samples = [s for s in filtered_samples if s.difficulty == filter_diff]
    if filter_status != "All":
        filtered_samples = [s for s in filtered_samples if s.annotation_status == filter_status]
    if search_query.strip():
        q_lower = search_query.strip().lower()
        filtered_samples = [s for s in filtered_samples if q_lower in s.question.lower()]

    st.caption(f"Showing {len(filtered_samples)} of {len(samples)} QA pairs:")

    # Table preview
    preview_rows = []
    for s in filtered_samples:
        preview_rows.append({
            "Sample ID": s.sample_id,
            "Question": s.question,
            "Reference Answer": s.reference_answer[:80] + ("..." if len(s.reference_answer) > 80 else ""),
            "Document ID": s.document_id or "—",
            "Topic": s.topic or "—",
            "Difficulty": s.difficulty or "—",
            "Annotation Status": s.annotation_status,
        })
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    # QA Editor / Review Modal in Expander
    with st.expander("✏️ Edit / Review Selected QA Pair"):
        if filtered_samples:
            edit_sample_id = st.selectbox("Select Sample ID to Review/Edit:", options=[s.sample_id for s in filtered_samples])
            target_sample = next(s for s in filtered_samples if s.sample_id == edit_sample_id)

            e_col1, e_col2 = st.columns(2)
            with e_col1:
                new_q = st.text_area("Question Text:", value=target_sample.question, height=70)
                new_doc = st.text_input("Document ID:", value=target_sample.document_id or "")
                new_topic = st.text_input("Topic:", value=target_sample.topic or "general")
            with e_col2:
                new_ans = st.text_area("Reference Answer:", value=target_sample.reference_answer, height=70)
                new_diff = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=["easy", "medium", "hard"].index(target_sample.difficulty or "medium"))
                status_opts = ["verified_expert", "reviewed", "unverified_synthetic"]
                curr_status_idx = status_opts.index(target_sample.annotation_status) if target_sample.annotation_status in status_opts else 0
                new_status = st.selectbox("Annotation Verification Status:", status_opts, index=curr_status_idx)

            if st.button("💾 Save QA Pair Changes", type="primary"):
                target_sample.question = new_q
                target_sample.reference_answer = new_ans
                target_sample.document_id = new_doc
                target_sample.topic = new_topic
                target_sample.difficulty = new_diff
                target_sample.annotation_status = new_status
                if selected_file:
                    if selected_file.suffix.lower() == ".csv":
                        export_dataset_to_csv(samples, selected_file)
                    else:
                        export_dataset_to_json(samples, selected_file)
                st.success(f"Updated QA pair `{edit_sample_id}` and saved to `{selected_file.name if selected_file else 'disk'}`!")
                st.rerun()

    # Add New QA Pair Expander
    with st.expander("➕ Add New Expert-Annotated QA Pair"):
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            add_q = st.text_area("Question:", placeholder="Enter scientific or technical question...", height=70)
            add_doc = st.text_input("Source Document ID:", placeholder="e.g. doc_rerag_paper")
            add_topic = st.text_input("Topic:", placeholder="e.g. architecture, verification")
        with a_col2:
            add_ans = st.text_area("Ideal Reference Answer:", placeholder="Enter ground truth reference answer...", height=70)
            add_diff = st.selectbox("Difficulty Level:", ["easy", "medium", "hard"], index=1)
            add_status = st.selectbox("Annotation Status:", ["verified_expert", "reviewed", "unverified_synthetic"], index=0)

        if st.button("➕ Add to Dataset"):
            if not add_q.strip() or not add_ans.strip():
                st.error("Both Question and Reference Answer are required.")
            else:
                new_id = f"qa_{len(samples) + 1:03d}"
                new_sample = EvaluationSample(
                    sample_id=new_id,
                    question=add_q.strip(),
                    reference_answer=add_ans.strip(),
                    document_id=add_doc.strip() or None,
                    topic=add_topic.strip() or "general",
                    difficulty=add_diff,
                    annotation_status=add_status,
                )
                samples.append(new_sample)
                if selected_file:
                    if selected_file.suffix.lower() == ".csv":
                        export_dataset_to_csv(samples, selected_file)
                    else:
                        export_dataset_to_json(samples, selected_file)
                st.success(f"Added new sample `{new_id}` to `{selected_file.name if selected_file else 'dataset'}`!")
                st.rerun()

    # Export active dataset buttons
    st.markdown("##### 📥 Export Active Dataset")
    ex_c1, ex_c2, _ = st.columns([1, 1, 2])
    with ex_c1:
        json_data = json.dumps([s.to_dict() for s in samples], indent=2)
        st.download_button(
            "⬇️ Download JSON",
            data=json_data,
            file_name=f"{st.session_state.active_dataset_name or 'dataset'}.json",
            mime="application/json",
            use_container_width=True,
        )
    with ex_c2:
        import io
        import csv
        csv_buf = io.StringIO()
        fieldnames = ["sample_id", "question", "reference_answer", "document_id", "topic", "difficulty", "annotation_status"]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "sample_id": s.sample_id,
                "question": s.question,
                "reference_answer": s.reference_answer,
                "document_id": s.document_id or "",
                "topic": s.topic or "",
                "difficulty": s.difficulty or "",
                "annotation_status": s.annotation_status or "verified_expert",
            })
        st.download_button(
            "⬇️ Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"{st.session_state.active_dataset_name or 'dataset'}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 4. Leakage-Resistant Dataset Splitting & Audit
    st.markdown("---")
    st.markdown("### 🛡️ Leakage-Resistant Dataset Partitioning & Leakage Audit")
    st.caption(
        "Prevent data contamination by isolating Optimization (Train), Validation (Tune), and Held-Out Test sets. "
        "Guarantees that test answers are strictly quarantined from the optimizer."
    )

    sp_c1, sp_c2 = st.columns(2)
    with sp_c1:
        strategy = st.selectbox(
            "Partitioning Strategy:",
            options=[
                "near_duplicate_grouped",
                "document_level",
                "stratified_topic",
                "random_row",
            ],
            format_func=lambda s: {
                "near_duplicate_grouped": "Near-Duplicate Grouped (Clusters overlapping queries together)",
                "document_level": "Document-Level (Quarantines whole documents to single splits)",
                "stratified_topic": "Topic-Stratified (Preserves topic distribution)",
                "random_row": "Random Row Split (Baseline — potential query leakage)",
            }.get(s, s),
            help="Select the strategy to enforce isolation and prevent test data leakage.",
        )
        random_seed = st.number_input("Reproducibility Seed:", min_value=1, max_value=99999, value=42, step=1, key="split_seed")

    with sp_c2:
        train_pct = st.slider("Optimization (Train) %", min_value=10, max_value=80, value=50, step=5)
        val_pct = st.slider("Validation %", min_value=10, max_value=50, value=25, step=5)
        test_pct = max(0, 100 - (train_pct + val_pct))
        st.info(f"Resulting Split: **Train {train_pct}%** | **Val {val_pct}%** | **Held-Out Test {test_pct}%**")

    if st.button("🔬 Compute & Audit Split", type="primary"):
        splits = split_train_val_test_leakage_resistant(
            samples=samples,
            strategy=strategy,
            train_ratio=train_pct / 100.0,
            val_ratio=val_pct / 100.0,
            test_ratio=test_pct / 100.0,
            random_seed=int(random_seed),
        )
        audit = audit_split_leakage(splits)
        st.session_state.active_dataset_splits = splits
        st.session_state.split_leakage_audit = audit

    splits = st.session_state.get("active_dataset_splits")
    audit = st.session_state.get("split_leakage_audit")

    if splits and audit:
        st.markdown("#### 📋 Partition Distribution & Leakage Audit Report")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Optimization (Train)", len(splits.train_samples), help="Used exclusively during parameter search.")
        with s2:
            st.metric("Validation", len(splits.val_samples), help="Used for parameter selection and topic stability tuning.")
        with s3:
            st.metric("Held-Out Test", len(splits.test_samples), help="Strictly quarantined for uncorrupted final testing.")
        with s4:
            leak_status = "✅ CLEAN" if audit["is_leakage_free"] else "⚠️ LEAKAGE"
            st.metric("Leakage Status", leak_status)

        if audit["is_leakage_free"]:
            st.success("✅ **Zero-Leakage Guarantee Verified:** No exact duplicates or near-duplicate query overlaps cross between splits.")
        else:
            st.warning("⚠️ **Leakage Detected Across Splits:**")
            for w in audit.get("warnings", []):
                st.markdown(f"- {w}")

        if audit.get("near_duplicate_cross_split_pairs"):
            with st.expander(f"⚠️ {len(audit['near_duplicate_cross_split_pairs'])} Cross-Split Near-Duplicate Pairs"):
                for p in audit["near_duplicate_cross_split_pairs"]:
                    st.markdown(f"- **{p['sample_a_id']}** ({p['split_a']}) vs **{p['sample_b_id']}** ({p['split_b']}): similarity = {p['similarity']:.2f}")

        # Download splits
        st.markdown("##### 📥 Export Partitioned Splits")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "⬇️ Download Train Split JSON",
                data=json.dumps([s.to_dict() for s in splits.train_samples], indent=2),
                file_name="split_train_optimization.json",
                mime="application/json",
            )
        with d2:
            st.download_button(
                "⬇️ Download Val Split JSON",
                data=json.dumps([s.to_dict() for s in splits.val_samples], indent=2),
                file_name="split_validation.json",
                mime="application/json",
            )
        with d3:
            st.download_button(
                "⬇️ Download Test Split JSON",
                data=json.dumps([s.to_dict() for s in splits.test_samples], indent=2),
                file_name="split_held_out_test.json",
                mime="application/json",
            )


# ==============================================================================
# SECTION 2: RAG CHAT
# ==============================================================================


def render_rag_chat(settings: Settings) -> None:
    """Render the Grounded RAG and ReRAG Chat section."""
    st.markdown("## 💬 RAG Chat & Verification Lab")
    st.markdown(
        "Ask factual questions against indexed document passages. Inspect retrieved contexts, "
        "citation provenance, cosine similarity scores, and claim-level verification traces."
    )

    # 1. Verification of document indexing
    doc: Optional[LoadedDocument] = st.session_state.loaded_document
    vs: VectorStore = get_vector_store(settings)
    total_chunks = st.session_state.get("indexed_chunks_count", 0)
    if total_chunks == 0:
        try:
            total_chunks = vs.count()
            st.session_state.indexed_chunks_count = total_chunks
        except Exception:
            pass

    if total_chunks == 0:
        st.warning(
            "⚠️ No document chunks are currently indexed in the active vector collection. "
            "Please go to the **Document Workspace** to upload and index a document first, "
            "or click below to auto-index the bundled sample research paper."
        )
        sample_pdf = settings.paths.documents_dir / "sample_rerag_paper.pdf"
        if sample_pdf.exists():
            if st.button("📄 Auto-Index Bundled Sample Paper (sample_rerag_paper.pdf)"):
                loader = DocumentLoader()
                loaded_doc = loader.load_file(sample_pdf)
                st.session_state.loaded_document = loaded_doc
                chunker = TextChunker()
                chunks = chunker.chunk_document(loaded_doc)
                vs.add_chunks(chunks, prevent_stale=True)
                st.session_state.indexed_document_id = loaded_doc.document_id
                st.session_state.indexed_chunks_count = len(chunks)
                st.success(f"Indexed {len(chunks)} chunks into ChromaDB!")
                st.rerun()
        return

    # 2. Check LLM Provider status
    llm = get_llm_client(settings, offline_mode=st.session_state.offline_mode)
    is_ollama_healthy = False
    if not st.session_state.offline_mode:
        is_ollama_healthy = llm.check_health()
        if not is_ollama_healthy:
            w_col1, w_col2 = st.columns([3, 1])
            with w_col1:
                st.warning(
                    "⚠️ **Local Ollama Service Offline:** Could not reach Ollama at "
                    f"`{settings.llm.base_url}`. Start Ollama (`ollama serve`) or switch to "
                    "'Offline / Mock LLM Mode' for deterministic demonstration testing."
                )
            with w_col2:
                if st.button("🔌 Enable Offline Mode", use_container_width=True):
                    st.session_state.offline_mode = True
                    st.rerun()

    # 3. Chat Controls
    with st.expander("🛠️ Retrieval & Pipeline Parameters", expanded=False):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=int(settings.retriever.top_k))
        with p_col2:
            sim_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.retriever.similarity_threshold),
                step=0.05,
                help="Minimum cosine similarity required for a passage to be retained.",
            )
        with p_col3:
            pipeline_mode = st.radio(
                "Pipeline Architecture",
                options=["ReRAG (Iterative Refinement)", "Baseline RAG (Single-hop)"],
                index=0,
                help="Choose between single-hop RAG and multi-step ReRAG verification loop.",
            )

    # 4. Sample Question Prompts
    st.markdown("##### Quick-Fill Demonstration Questions:")
    st.caption("Click any sample question below to immediately run it and populate the input field:")

    # Adapt sample questions if active document is Riverbend test document
    doc_name = ""
    if doc is not None and doc.source_path:
        doc_name = doc.source_path.name.lower()

    if "riverbend" in doc_name or "knowledge" in doc_name:
        sample_queries = [
            ("Q1: Installed Capacity", "What is the combined installed capacity across the three sites?"),
            ("Q2: Roles & Leads", "What are the roles of Mira Sen and Daniel Okafor in the project?"),
            ("Q3: Solar Generation", "What was the total reported solar generation for July through September 2024?"),
            ("Q4: Incident Log", "What caused the communications fault at the North Library on 12 August 2024?"),
        ]
    else:
        sample_queries = [
            ("Q1: ReRAG Purpose", "What is the primary architectural purpose of ReRAG?"),
            ("Q2: Verification Threshold", "What threshold does the ReRAG verification step use to check candidate claims?"),
            ("Q3: JWST Wavelengths", "What wavelengths does the James Webb Space Telescope observe to detect early cosmic structures?"),
            ("Q4: Chloroplasts", "What is the role of photosynthesis in plant chloroplasts?"),
        ]

    q_cols = st.columns(len(sample_queries))
    quick_clicked_query = None
    for idx, (label, sq) in enumerate(sample_queries):
        with q_cols[idx]:
            if st.button(label, help=sq, use_container_width=True):
                st.session_state.rag_query_input = sq
                quick_clicked_query = sq

    # 5. Query Input Form
    with st.form("rag_query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter Question:",
            key="rag_query_input",
            placeholder="Ask a factual question about the indexed document...",
        )
        submitted = st.form_submit_button("Ask Question", type="primary")

    active_query = (quick_clicked_query or (user_query if submitted else "")).strip()

    if active_query:
        # Instantiate pipeline
        retriever = Retriever(
            vector_store=vs,
            embedding_manager=vs.embedding_manager,
            top_k=top_k,
            similarity_threshold=sim_threshold,
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm,
            max_refinement_steps=settings.rerag.max_refinement_steps,
            verification_threshold=settings.rerag.verification_threshold,
        )

        with st.spinner("Retrieving evidence and synthesizing grounded response..."):
            t0 = time.perf_counter()
            try:
                if "ReRAG" in pipeline_mode:
                    res: PipelineResult = pipeline.run(
                        active_query,
                        top_k=top_k,
                        threshold=sim_threshold,
                        similarity_threshold=sim_threshold,
                    )
                else:
                    res = pipeline.run_baseline(
                        active_query,
                        top_k=top_k,
                        threshold=sim_threshold,
                        similarity_threshold=sim_threshold,
                    )
                exec_time = time.perf_counter() - t0
            except Exception as exc:
                logger.warning("Primary LLM execution failed (%s), falling back to Offline Mock LLM.", exc)
                fallback_llm = get_llm_client(settings, offline_mode=True)
                fallback_pipeline = RAGPipeline(
                    retriever=retriever,
                    llm_client=fallback_llm,
                    max_refinement_steps=settings.rerag.max_refinement_steps,
                    verification_threshold=settings.rerag.verification_threshold,
                )
                res = fallback_pipeline.run_baseline(
                    active_query,
                    top_k=top_k,
                    threshold=sim_threshold,
                    similarity_threshold=sim_threshold,
                )
                exec_time = time.perf_counter() - t0
                st.info(
                    "ℹ️ Local Ollama service was unreachable, so an offline demonstration answer was synthesized. "
                    "Start Ollama (`ollama serve`) or keep 'Offline Mode' enabled."
                )

        # Save to chat history
        st.session_state.chat_history.append({"query": active_query, "result": res, "time": exec_time, "mode": pipeline_mode})

    # 6. Display Latest Answer & Evidence
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        res: PipelineResult = latest["result"]
        q_text = latest["query"]
        elapsed = latest["time"]
        mode_str = latest["mode"]

        st.markdown("---")
        st.markdown(f"### 💡 Question: *\"{q_text}\"*")
        st.caption(f"Executed via **{mode_str}** | Model: `{res.llm_model}` | Runtime: `{elapsed:.2f}s`")

        # Answer Presentation
        is_insufficient = (
            INSUFFICIENT_INFORMATION_RESPONSE in res.final_answer
            or not res.contexts
            or "does not contain sufficient information" in res.final_answer.lower()
        )

        if is_insufficient:
            st.warning(
                "⚠️ **Insufficient Grounded Information:**\n\n"
                f"{res.final_answer}\n\n"
                "*The system conservative policy prevented hallucination because retrieved passages did not contain verifiable evidence.*"
            )
        else:
            st.markdown(
                f"""
                <div class="rerag-card" style="border-left: 4px solid #10b981; background-color: #f0fdf4;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #15803d;">Generated Answer:</h4>
                    <p style="font-size: 1rem; color: #0f172a; line-height: 1.6; margin: 0;">{res.final_answer}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Retrieved Passages Section
        st.markdown(f"#### 📚 Retrieved Evidence Passages ({len(res.contexts)} Chunks)")
        if not res.contexts:
            st.info("No passages met the cosine similarity threshold.")
        else:
            for idx, ctx in enumerate(res.contexts, 1):
                render_retrieved_passage(
                    chunk_id=ctx.chunk_id,
                    text=ctx.text,
                    score=ctx.score,
                    distance=ctx.distance,
                    page_number=ctx.page_number,
                    document_id=ctx.document_id,
                    citation=ctx.citation,
                    index=idx,
                    metadata=ctx.metadata,
                )

        # ReRAG Iterative Refinement Traces
        if res.steps:
            render_trace_viewer(res.steps)


# ==============================================================================
# ==============================================================================
# SECTION 3: OPTIMIZATION LAB
# ==============================================================================


def render_optimization_lab(settings: Settings) -> None:
    """Render the Adaptive Chunk Optimization Lab section."""
    st.markdown("## 🧪 Optimization Lab")
    st.markdown(
        "Execute chunk parameter optimization across chunk-size and chunk-overlap hyperparameters. "
        "Choose between the original IEEE ReRAG greedy coordinate search baseline, random search, "
        "simulated annealing, or run an empirical controlled multi-method comparison across all strategies."
    )

    doc: Optional[LoadedDocument] = st.session_state.loaded_document

    # 1. Dataset Resolution
    eval_dir = settings.paths.evaluation_dir
    dataset_files = list(eval_dir.glob("*.json")) + list(eval_dir.glob("*.jsonl"))
    if not dataset_files:
        st.warning(f"No benchmark evaluation datasets found in `{eval_dir}`.")
        return

    # Strategy Selection
    opt_mode = st.radio(
        "Optimization Strategy / Mode:",
        options=[
            "Proposed Generalization-Aware ReRAG (Train on Train, Select on Val, Freeze for Test)",
            "Greedy Coordinate Search (Original ReRAG Baseline)",
            "Random Search (Stochastic Uniform Exploration)",
            "Simulated Annealing (Probabilistic Acceptance)",
            "Controlled Multi-Method Comparison (Fixed vs Original ReRAG vs Proposed)",
        ],
        index=0,
        horizontal=True,
    )

    # Configuration layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🎯 Baseline & Search Boundaries")
        base_size = st.number_input("Baseline Chunk Size", min_value=100, max_value=2000, value=int(settings.document.chunk_size), step=50)
        base_overlap = st.number_input("Baseline Chunk Overlap", min_value=0, max_value=500, value=int(settings.document.chunk_overlap), step=25)

        if base_overlap >= base_size:
            st.error("Baseline overlap must be strictly less than baseline chunk size.")
            return

        st.markdown("##### Search Space Limits:")
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            min_chunk_sz = st.number_input("Min Chunk Size", min_value=50, max_value=500, value=100, step=25)
            min_overlap = st.number_input("Min Overlap", min_value=0, max_value=100, value=0, step=10)
            min_step = st.number_input("Min Step Size (Stopping)", min_value=10, max_value=100, value=25, step=5)
        with b_c2:
            max_chunk_sz = st.number_input("Max Chunk Size", min_value=500, max_value=3000, value=2000, step=100)
            max_overlap = st.number_input("Max Overlap", min_value=50, max_value=1000, value=500, step=50)
            initial_step = st.number_input("Initial Step Size", min_value=25, max_value=500, value=100, step=25)

    with col_right:
        st.markdown("### ⚖️ Multi-Metric Objective & Parameters")
        st.caption("Configurable objective weights. Prohibits optimizing semantic similarity alone.")

        w_faith = st.slider("Weight: Faithfulness Ratio", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
        w_cov = st.slider("Weight: Context Coverage", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        w_sem = st.slider("Weight: Semantic Similarity", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

        if (w_faith + w_cov + w_sem) == 0.0:
            st.error("Total metric weights cannot sum to zero.")
            return

        st.markdown("##### Strategy-Specific Hyperparameters:")
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            if "Generalization-Aware" in opt_mode:
                gen_trials = st.number_input("Candidate Evaluation Budget", min_value=3, max_value=30, value=10, step=1)
                stab_weight = st.slider("Stability Penalty (lambda)", min_value=0.0, max_value=0.5, value=0.10, step=0.02, help="Penalizes variance across topics.")
                overfit_weight = st.slider("Overfit Penalty (beta)", min_value=0.0, max_value=0.3, value=0.05, step=0.01, help="Penalizes train-val score gap.")
                max_iterations = 5
                step_decay = 0.5
                rs_trials = 8
                sa_temp = 1.0
                sa_cooling = 0.85
                sa_iters = 8
            elif "Random" in opt_mode:
                rs_trials = st.number_input("Random Search Budget (Trials)", min_value=2, max_value=50, value=8, step=1)
                max_iterations = 5
                step_decay = 0.5
                sa_temp = 1.0
                sa_cooling = 0.85
                sa_iters = 8
            elif "Simulated Annealing" in opt_mode:
                sa_temp = st.number_input("Initial Temperature (T0)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                sa_cooling = st.number_input("Cooling Rate (alpha)", min_value=0.5, max_value=0.99, value=0.85, step=0.05)
                sa_iters = st.number_input("Max SA Iterations", min_value=2, max_value=30, value=8, step=1)
                max_iterations = 5
                step_decay = 0.5
                rs_trials = 8
            elif "Multi-Method" in opt_mode:
                max_iterations = st.number_input("Greedy Max Iterations", min_value=1, max_value=10, value=4, step=1)
                rs_trials = st.number_input("Random Search Budget", min_value=2, max_value=20, value=6, step=1)
                sa_iters = st.number_input("Simulated Annealing Iters", min_value=2, max_value=20, value=6, step=1)
                step_decay = 0.5
                sa_temp = 1.0
                sa_cooling = 0.85
            else:  # Greedy baseline
                max_iterations = st.number_input("Max Search Iterations", min_value=1, max_value=15, value=5, step=1)
                step_decay = st.number_input("Step Decay Factor", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
                rs_trials = 8
                sa_temp = 1.0
                sa_cooling = 0.85
                sa_iters = 8

        with s_c2:
            improvement_tol = st.number_input("Improvement Tolerance", min_value=0.001, max_value=0.100, value=0.005, step=0.001, format="%.3f")
            random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("### 📊 Benchmark Dataset & Generalization Partition")

    d_c1, d_c2 = st.columns([3, 2])
    with d_c1:
        selected_dataset_file = st.selectbox(
            "Evaluation Dataset File",
            options=dataset_files,
            format_func=lambda p: p.name,
        )
    with d_c2:
        eval_held_out = st.checkbox("Strict Zero-Leakage Held-Out Test Evaluation", value=True)

    # Validate dataset and preview partition counts
    if selected_dataset_file:
        try:
            val_result = DatasetValidator.validate_file(selected_dataset_file)
            if val_result.is_valid:
                st.caption(f"✅ Dataset Verified: {val_result.total_samples} samples | Adversarial cases: {val_result.negative_samples_count}")
                # Load sample splits preview
                raw_samples = load_dataset(selected_dataset_file)
                preview_splits = split_train_val_test(raw_samples, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, random_seed=int(random_seed))
                render_dataset_split_summary(preview_splits)
            else:
                st.warning(f"⚠️ Dataset validation warnings: {', '.join(val_result.errors)}")
        except Exception as exc:
            st.caption(f"Dataset check note: {exc}")

    # Concurrency Lock Guard
    is_running = st.session_state.is_optimizing

    st.markdown("<br>", unsafe_allow_html=True)
    button_label = "▶️ Launch Controlled Multi-Method Comparison" if "Multi-Method" in opt_mode else f"▶️ Start {opt_mode.split('(')[0].strip()} Run"
    start_opt_btn = st.button(
        button_label,
        type="primary",
        disabled=is_running,
        help="Launches parameter optimization under the configured search strategy.",
    )

    if start_opt_btn:
        st.session_state.is_optimizing = True

        # Document resolution
        if doc is None:
            sample_pdf = settings.paths.documents_dir / "sample_rerag_paper.pdf"
            if not sample_pdf.exists():
                sample_pdf = settings.paths.documents_dir / "ReRAG_Test_Knowledge_Document.pdf"
            if sample_pdf.exists():
                loader = DocumentLoader()
                doc = loader.load_file(sample_pdf)
                st.session_state.loaded_document = doc
            else:
                st.error("No document available to chunk and evaluate.")
                st.session_state.is_optimizing = False
                return

        docs_list = [doc]

        # LLM Client resolution
        llm_instance = get_llm_client(settings, offline_mode=st.session_state.offline_mode)

        # Objective configuration
        objective = OptimizationObjective(
            weights={
                "faithfulness_ratio": w_faith,
                "context_coverage": w_cov,
                "semantic_similarity": w_sem,
            }
        )

        bounds = SearchBounds(
            min_chunk_size=min_chunk_sz,
            max_chunk_size=max_chunk_sz,
            min_chunk_overlap=min_overlap,
            max_chunk_overlap=max_overlap,
            min_step_size=min_step,
        )

        # Isolated vector store collection
        opt_col_name = f"opt_ui_{int(time.time())}"
        opt_vs = VectorStore(
            persist_dir=None,  # Ephemeral for clean isolation
            collection_name=opt_col_name,
            embedding_manager=get_shared_embedding_manager(),
        )
        opt_retriever = Retriever(vector_store=opt_vs, top_k=settings.retriever.top_k, similarity_threshold=0.0)
        opt_pipeline = RAGPipeline(retriever=opt_retriever, llm_client=llm_instance)
        opt_evaluator = Evaluator(pipeline=opt_pipeline, embedding_manager=opt_vs.embedding_manager)

        try:
            if "Multi-Method" in opt_mode:
                # Execute full controlled comparison
                with st.spinner("Executing controlled comparison across Fixed Baseline, Greedy, Random Search, and Simulated Annealing..."):
                    comp_runner = ComparisonRunner(
                        settings=settings,
                        documents=docs_list,
                        dataset_path=selected_dataset_file,
                        train_ratio=0.50,
                        val_ratio=0.20,
                        test_ratio=0.30 if eval_held_out else 0.0,
                        random_seed=int(random_seed),
                        pipeline=opt_pipeline,
                        evaluator=opt_evaluator,
                        collection_name=opt_col_name,
                        results_dir=settings.paths.results_dir,
                    )
                    comp_result = comp_runner.run(
                        baseline_config=ChunkConfig(base_size, base_overlap),
                        bounds=bounds,
                        objective=objective,
                        greedy_kwargs={"max_iterations": int(max_iterations), "initial_step_size": int(initial_step), "step_decay": float(step_decay)},
                        random_search_kwargs={"max_trials": int(rs_trials), "random_seed": int(random_seed)},
                        simulated_annealing_kwargs={"initial_temp": float(sa_temp), "cooling_rate": float(sa_cooling), "max_iterations": int(sa_iters)},
                        save_results=True,
                    )

                st.session_state.latest_comparison_result = comp_result
                st.session_state.is_optimizing = False
                st.success(
                    f"🏆 Multi-Method Comparison Complete! Best Held-Out Generalization: "
                    f"**{comp_result.best_test_method}** | Best Optimization: **{comp_result.best_train_method}**"
                )

            else:
                # Single Strategy Run
                prog_bar = st.progress(0.0)
                status_box = st.empty()

                def on_trial_progress(iter_num: int, total_iters: int, trial: CandidateTrial) -> None:
                    prog_val = min(1.0, max(0.0, iter_num / max(1, total_iters)))
                    prog_bar.progress(prog_val)
                    c_cfg = getattr(trial, "config", None)
                    c_sz = getattr(c_cfg, "chunk_size", "?")
                    c_ov = getattr(c_cfg, "chunk_overlap", "?")
                    status_box.markdown(
                        f"**Evaluating:** Trial #{trial.trial_id} | Iteration {iter_num}/{total_iters} | "
                        f"Config: `(Size: {c_sz}, Overlap: {c_ov})` | Status: `{trial.status.upper()}`"
                    )

                if "Random" in opt_mode:
                    search_cfg = RandomSearchConfig(
                        max_trials=int(rs_trials),
                        bounds=bounds,
                        objective=objective,
                        random_seed=int(random_seed),
                    )
                elif "Simulated Annealing" in opt_mode:
                    search_cfg = SimulatedAnnealingConfig(
                        initial_temperature=float(sa_temp),
                        cooling_rate=float(sa_cooling),
                        max_iterations=int(sa_iters),
                        step_size=int(initial_step),
                        bounds=bounds,
                        objective=objective,
                        random_seed=int(random_seed),
                    )
                else:
                    search_cfg = AdaptiveSearchConfig(
                        initial_chunk_size=base_size,
                        initial_chunk_overlap=base_overlap,
                        initial_step_size=initial_step,
                        step_decay=step_decay,
                        improvement_tolerance=improvement_tol,
                        max_iterations=max_iterations,
                        bounds=bounds,
                        objective=objective,
                        random_seed=random_seed,
                    )

                runner = ExperimentRunner(
                    settings=settings,
                    documents=docs_list,
                    dataset_path=selected_dataset_file,
                    test_ratio=0.33 if eval_held_out else 0.0,
                    random_seed=random_seed,
                    pipeline=opt_pipeline,
                    evaluator=opt_evaluator,
                    collection_name=opt_col_name,
                    results_dir=settings.paths.results_dir,
                )

                with st.spinner(f"Executing {opt_mode.split('(')[0].strip()}..."):
                    result: OptimizationResult = runner.run_experiment(
                        baseline_config=ChunkConfig(base_size, base_overlap),
                        search_config=search_cfg,
                        evaluate_held_out=eval_held_out,
                        save_results=True,
                        progress_callback=on_trial_progress,
                    )

                prog_bar.progress(1.0)
                status_box.success(
                    f"🎉 Optimization Complete! Selected: Chunk Size {result.selected_config.chunk_size}, "
                    f"Overlap {result.selected_config.chunk_overlap} (Gain: {result.improvement:+.4f}) in {result.total_execution_time_seconds:.1f}s."
                )

                st.session_state.latest_optimization_result = result
                st.session_state.is_optimizing = False

        except Exception as exc:
            st.session_state.is_optimizing = False
            st.error(f"Optimization run failed: {exc}")
            logger.exception("Error in optimization run")

    # Display active results if available
    comp_obj: Optional[ComparisonResult] = st.session_state.latest_comparison_result
    if comp_obj is not None:
        st.markdown("---")
        st.markdown("### 📊 Active Multi-Method Comparison Study")
        render_multi_method_comparison_table(comp_obj.summary_records)
        render_multi_method_charts(comp_obj)

    res_obj: Optional[OptimizationResult] = st.session_state.latest_optimization_result
    if res_obj is not None and comp_obj is None:
        st.markdown("---")
        st.markdown(f"#### 📜 Candidate Trials History ({len(res_obj.candidate_history)} Trials Recorded)")
        render_candidate_table(res_obj.candidate_history)


# ==============================================================================
# SECTION 4: RESULTS AND COMPARISON
# ==============================================================================


def render_results_and_comparison(settings: Settings) -> None:
    """Render the Empirical Results and Comparative Analysis section."""
    st.markdown("## 📊 Results and Comparison")
    st.markdown(
        "Examine empirical metrics from completed optimization experiments. "
        "Inspect multi-method comparative performance, parameter progression, and held-out test generalization."
    )

    comp_res: Optional[ComparisonResult] = st.session_state.latest_comparison_result
    single_res: Optional[OptimizationResult] = st.session_state.latest_optimization_result

    if comp_res is None and single_res is None:
        render_empty_state(
            title="No Experiment Results Available",
            message="No optimization run has been executed in this session or loaded from history. "
            "Visit the **Optimization Lab** to run an experiment, or **Experiment History** to load past results.",
            icon="📊",
        )
        return

    # If both exist, allow selecting between multi-method study and single-run inspection
    if comp_res is not None and single_res is not None:
        view_choice = st.radio(
            "Select View Mode:",
            options=["Multi-Method Comparison Study", "Single Run In-Depth Inspection"],
            horizontal=True,
        )
    elif comp_res is not None:
        view_choice = "Multi-Method Comparison Study"
    else:
        view_choice = "Single Run In-Depth Inspection"

    # 1. Multi-Method Comparative View
    if view_choice == "Multi-Method Comparison Study" and comp_res is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"🏆 **Best Generalization (Held-Out Test):** `{comp_res.best_test_method}`")
        with c2:
            st.info(f"🎯 **Best Optimization (Train):** `{comp_res.best_train_method}`")

        # Metric Classification & Transparency Legend
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.82rem;">
                <span style="font-weight: 700; color: #1e293b;">🔬 Metric Classification & Scientific Transparency:</span><br>
                <span style="background-color: #dcfce7; color: #15803d; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Deterministic</span> Exact lexical/structural metrics (Token F1, Faithfulness, Coverage, Exact Match).<br>
                <span style="background-color: #fef3c7; color: #b45309; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Neural Proxy</span> Dense cosine similarity (SentenceTransformers). Evaluates topical alignment, <em>not</em> factual truth.<br>
                <span style="background-color: #f3e8ff; color: #7e22ce; padding: 2px 6px; border-radius: 4px; font-weight: 600;">LLM Judge</span> Heuristic model-based consistency checking.<br>
                <span style="background-color: #f1f5f9; color: #475569; padding: 2px 6px; border-radius: 4px; font-weight: 600;">System</span> End-to-end latency, number of trials, and computational indexing cost proxy.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📋 Controlled Multi-Method Comparison Table")
        st.caption("Controlled evaluation across identical corpus and held-out test split. Optimizers never accessed test answers during search.")
        render_multi_method_comparison_table(comp_res.summary_records)

        st.markdown("---")
        render_multi_method_charts(comp_res)

        st.markdown("---")
        st.markdown("#### 🔬 Scientific Insights & Generalization")
        st.markdown(
            "- **Zero-Leakage Guarantee:** Parameter optimization took place exclusively on the training split. "
            "Held-out test answers were never inspected during parameter selection.\n"
            "- **Generalization Gap:** The drop between train and held-out test score reflects whether an optimizer "
            "overfitted chunk boundaries to the specific training questions.\n"
            "- **Multi-Metric Integrity:** Objective balances faithfulness, coverage, and similarity to prevent "
            "superficial semantic hallucination masking."
        )

        # Export options
        st.markdown("##### 📥 Export Study Results")
        ex_r1, ex_r2 = st.columns(2)
        with ex_r1:
            json_str = json.dumps(comp_res.to_dict(), indent=2)
            st.download_button(
                "⬇️ Download Experiment JSON",
                data=json_str,
                file_name=f"{comp_res.experiment_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with ex_r2:
            import io
            import csv
            buf = io.StringIO()
            if comp_res.summary_records:
                writer = csv.DictWriter(buf, fieldnames=list(comp_res.summary_records[0].keys()))
                writer.writeheader()
                for r in comp_res.summary_records:
                    writer.writerow(r)
            st.download_button(
                "⬇️ Download Summary CSV",
                data=buf.getvalue(),
                file_name=f"{comp_res.experiment_id}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 2. Single Optimizer Inspection View
    elif single_res is not None:
        render_comparison_view(
            baseline_metrics=single_res.baseline_metrics,
            selected_metrics=single_res.selected_metrics,
            baseline_config=single_res.baseline_config,
            selected_config=single_res.selected_config,
            held_out_evaluation=single_res.held_out_evaluation,
        )

        st.markdown("---")
        st.markdown("#### 📈 Objective Score Progression across Candidate Trials")
        render_candidate_chart(single_res.candidate_history, baseline_score=single_res.baseline_score)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 📌 Experiment Metadata")
            st.write({
                "Stopping Reason": single_res.stopping_reason,
                "Total Search Iterations": single_res.total_iterations,
                "Total Candidate Trials": single_res.total_trials,
                "Accepted Candidates": single_res.accepted_trials,
                "Total Execution Time": f"{single_res.total_execution_time_seconds:.2f} seconds",
                "Optimization Dataset ID": single_res.optimization_dataset_id,
            })
        with col_b:
            st.markdown("##### 🔬 Scientific Integrity Findings")
            gain = single_res.improvement
            h_gain = single_res.held_out_evaluation.get("held_out_gain", 0.0) if single_res.held_out_evaluation else None
            findings = [f"- **Optimization Gain:** {gain:+.4f} on tuning questions."]
            if h_gain is not None:
                findings.append(f"- **Generalization Gain:** {h_gain:+.4f} on unseen held-out test questions.")
                if h_gain < gain:
                    findings.append("- *Observation:* Mild generalization drop observed on held-out data (expected in empirical research).")
                else:
                    findings.append("- *Observation:* Optimization gains successfully generalized to held-out distribution.")
            else:
                findings.append("- *Note:* Held-out test evaluation was not enabled for this experiment run.")
            st.markdown("\n".join(findings))

        st.markdown("---")
        st.markdown("#### 📋 Candidate Breakdown")
        status_filter = st.selectbox("Filter by Status", ["All", "Accepted Only", "Rejected Only", "Invalid / Failed"])
        filtered_trials = single_res.candidate_history
        if status_filter == "Accepted Only":
            filtered_trials = [t for t in single_res.candidate_history if t.status in {"accepted", "baseline"}]
        elif status_filter == "Rejected Only":
            filtered_trials = [t for t in single_res.candidate_history if t.status == "rejected"]
        elif status_filter == "Invalid / Failed":
            filtered_trials = [t for t in single_res.candidate_history if t.status in {"rejected_invalid", "failed"}]

        render_candidate_table(filtered_trials)


# ==============================================================================
# SECTION 5: EXPERIMENT HISTORY
# ==============================================================================


def render_experiment_history(settings: Settings) -> None:
    """Render the Historical Experiments loader, inspector, and exporter."""
    st.markdown("## 📜 Experiment History")
    st.markdown(
        "Browse previously executed optimization results stored in `results/`. "
        "Load past experiments for comparative analysis or download JSON/CSV audit logs."
    )

    results_dir = settings.paths.results_dir
    json_files = sorted(results_dir.glob("*.json"), key=os.path.getmtime, reverse=True)

    if not json_files:
        render_empty_state(
            title="No Saved Experiments Found",
            message=f"No `.json` experiment files were found in `{results_dir}`. "
            "Run an optimization experiment in the Optimization Lab to generate reproducible logs.",
            icon="📁",
        )
        return

    st.markdown(f"Found **{len(json_files)} saved experiment logs**:")

    selected_file = st.selectbox(
        "Select Saved Experiment",
        options=json_files,
        format_func=lambda p: f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})",
    )

    col1, col2 = st.columns([2, 4])
    with col1:
        load_btn = st.button("📂 Load Selected Experiment", type="primary", use_container_width=True)

    if load_btn and selected_file:
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if "methods" in raw_data and "summary_records" in raw_data:
                loaded_comp = ComparisonResult.load_from_json(selected_file)
                st.session_state.latest_comparison_result = loaded_comp
                st.success(f"Loaded multi-method comparison study '{selected_file.name}' into active session state!")
            else:
                loaded_res = OptimizationResult.load_from_json(selected_file)
                st.session_state.latest_optimization_result = loaded_res
                st.success(f"Loaded single optimization result '{selected_file.name}' into active session state!")
        except Exception as exc:
            st.error(f"Error loading experiment file: {exc}")

    # File Download Actions
    if selected_file and selected_file.exists():
        with open(selected_file, "r", encoding="utf-8") as f:
            raw_json_str = f.read()

        summary_csv = selected_file.parent / f"{selected_file.stem}_summary.csv"
        trials_csv = selected_file.parent / f"{selected_file.stem}_trials.csv"
        csv_equiv = selected_file.with_suffix(".csv")

        d_cols = st.columns(3)
        with d_cols[0]:
            st.download_button(
                "📥 Download Experiment JSON",
                data=raw_json_str,
                file_name=selected_file.name,
                mime="application/json",
                use_container_width=True,
            )
        with d_cols[1]:
            csv_to_use = summary_csv if summary_csv.exists() else (csv_equiv if csv_equiv.exists() else None)
            if csv_to_use and csv_to_use.exists():
                with open(csv_to_use, "r", encoding="utf-8") as f:
                    summary_csv_str = f.read()
                st.download_button(
                    "📥 Download Summary CSV",
                    data=summary_csv_str,
                    file_name=csv_to_use.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        with d_cols[2]:
            if trials_csv.exists():
                with open(trials_csv, "r", encoding="utf-8") as f:
                    trials_csv_str = f.read()
                st.download_button(
                    "📥 Download Trials CSV",
                    data=trials_csv_str,
                    file_name=trials_csv.name,
                    mime="text/csv",
                    use_container_width=True,
                )

    # Preview loaded result
    active_comp: Optional[ComparisonResult] = st.session_state.latest_comparison_result
    active_single: Optional[OptimizationResult] = st.session_state.latest_optimization_result

    if active_comp is not None:
        st.markdown("---")
        st.markdown(f"### 🔍 Active Multi-Method Comparison: Best Test = `{active_comp.best_test_method}`")
        render_multi_method_comparison_table(active_comp.summary_records)
        render_multi_method_charts(active_comp)
    elif active_single is not None:
        st.markdown("---")
        st.markdown(f"### 🔍 Active Single Run: `{active_single.selected_config}` (Score: {active_single.selected_score:.4f})")
        render_comparison_view(
            baseline_metrics=active_single.baseline_metrics,
            selected_metrics=active_single.selected_metrics,
            baseline_config=active_single.baseline_config,
            selected_config=active_single.selected_config,
            held_out_evaluation=active_single.held_out_evaluation,
        )
        st.markdown("#### Trial Progression")
        render_candidate_chart(active_single.candidate_history, baseline_score=active_single.baseline_score)



# ==============================================================================
# MASTER DASHBOARD RENDERER
# ==============================================================================


def render_dashboard() -> None:
    """Render the primary research dashboard layout with sidebar navigation."""
    settings = get_settings()
    settings.paths.ensure_directories()

    # Session State
    init_session_state(settings)

    # Header & Styles
    inject_custom_styles()

    vs_chunks = st.session_state.get("indexed_chunks_count", 0)
    if st.session_state.vector_store is not None:
        try:
            vs_chunks = st.session_state.vector_store.count()
        except Exception:
            pass
    doc_name = st.session_state.loaded_document.source_path.name if st.session_state.loaded_document else "None"
    status_indicators = [
        {"label": "Document", "value": doc_name},
        {"label": "Indexed Chunks", "value": str(vs_chunks)},
        {"label": "Mode", "value": "Offline / Mock" if st.session_state.offline_mode else "Online / Ollama"},
    ]

    render_header(status_badges=status_indicators)

    # Sidebar Navigation & Settings
    sidebar_controls = render_sidebar_config(settings, offline_mode_toggle=st.session_state.offline_mode)
    st.session_state.offline_mode = sidebar_controls["offline_mode"]

    with st.sidebar:
        st.markdown("### 🧭 Platform Navigation")
        nav_selection = st.radio(
            "Navigate to Section:",
            options=[
                "📁 Document Workspace",
                "📚 QA Dataset Lab",
                "💬 RAG Chat",
                "🧪 Optimization Lab",
                "📊 Results and Comparison",
                "📜 Experiment History",
            ],
            index=0,
            label_visibility="collapsed",
        )

    # Route navigation
    if nav_selection == "📁 Document Workspace":
        render_document_workspace(settings)
    elif nav_selection == "📚 QA Dataset Lab":
        render_qa_dataset_lab(settings)
    elif nav_selection == "💬 RAG Chat":
        render_rag_chat(settings)
    elif nav_selection == "🧪 Optimization Lab":
        render_optimization_lab(settings)
    elif nav_selection == "📊 Results and Comparison":
        render_results_and_comparison(settings)
    elif nav_selection == "📜 Experiment History":
        render_experiment_history(settings)
