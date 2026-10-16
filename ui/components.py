"""Reusable Streamlit UI presentation components for the ReRAG research platform.

Provides modular, cleanly styled UI components for:
- Custom styling and theme injection
- Status badges and metric KPI cards
- Retrieved passage cards with cosine similarity & distance inspection
- Refinement trace viewer
- Candidate trial tables and Altair charts
- Side-by-side configuration and metric comparison views
- Professional empty states
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def inject_custom_styles() -> None:
    """Inject custom CSS styles for clean, professional typography and cards."""
    css = """
    <style>
    /* Global container adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }

    /* Main header styling */
    .rerag-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .rerag-header h1 {
        color: #f8fafc !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .rerag-header p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0.25rem 0 0 0;
    }

    /* Card styling */
    .rerag-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Status badges */
    .rerag-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        border-radius: 9999px;
        letter-spacing: 0.04em;
    }
    .rerag-badge-accepted {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .rerag-badge-rejected {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .rerag-badge-invalid {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
    .rerag-badge-failed {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .rerag-badge-baseline {
        background-color: #e0e7ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
    }
    .rerag-badge-info {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    /* Passage card */
    .rerag-passage-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    .rerag-passage-meta {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rerag-passage-text {
        font-size: 0.88rem;
        color: #1e293b;
        line-height: 1.5;
        font-family: inherit;
    }

    /* Empty state styling */
    .rerag-empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        background-color: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    .rerag-empty-icon {
        font-size: 2.8rem;
        margin-bottom: 0.6rem;
    }
    .rerag-empty-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.35rem;
    }
    .rerag-empty-desc {
        font-size: 0.88rem;
        color: #64748b;
        max-width: 480px;
        margin: 0 auto;
    }

    /* Score callout */
    .rerag-score-tag {
        font-family: monospace;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.82rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(
    title: str = "ReRAG: Research & Optimization Platform",
    subtitle: str = "Adaptive Hyperparameter Optimization & Hallucination Reduction in Retrieval-Augmented Generation",
    status_badges: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Render top application banner with optional status indicators."""
    badge_html = ""
    if status_badges:
        badge_html = "<div style='margin-top: 0.6rem; display: flex; gap: 0.5rem; flex-wrap: wrap;'>"
        for b in status_badges:
            badge_html += f"<span class='rerag-badge rerag-badge-info'>{b.get('label', '')}: <b>{b.get('value', '')}</b></span>"
        badge_html += "</div>"

    html = f"""
    <div class="rerag-header">
        <h1>🔬 {title}</h1>
        <p>{subtitle}</p>
        {badge_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Return HTML string for a colored status badge."""
    st_clean = str(status).lower().strip()
    if st_clean == "accepted":
        cls = "rerag-badge-accepted"
        lbl = "Accepted"
    elif st_clean == "rejected":
        cls = "rerag-badge-rejected"
        lbl = "Rejected"
    elif "invalid" in st_clean:
        cls = "rerag-badge-invalid"
        lbl = "Invalid"
    elif st_clean == "failed":
        cls = "rerag-badge-failed"
        lbl = "Failed"
    elif st_clean == "baseline":
        cls = "rerag-badge-baseline"
        lbl = "Baseline"
    else:
        cls = "rerag-badge-info"
        lbl = status.capitalize()

    return f"<span class='rerag-badge {cls}'>{lbl}</span>"


def render_empty_state(
    title: str,
    message: str,
    icon: str = "🧪",
    action_label: Optional[str] = None,
) -> None:
    """Render a clean, responsive empty state card."""
    html = f"""
    <div class="rerag-empty-state">
        <div class="rerag-empty-icon">{icon}</div>
        <div class="rerag-empty-title">{title}</div>
        <div class="rerag-empty-desc">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_retrieved_passage(
    chunk_id: str,
    text: str,
    score: float,
    distance: float,
    page_number: Optional[int] = None,
    document_id: Optional[str] = None,
    citation: Optional[str] = None,
    index: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a single retrieved context passage with citation and metrics."""
    doc_label = document_id or "Doc"
    page_label = f"Page {page_number}" if page_number is not None else "Page N/A"
    cite_label = citation or f"[{doc_label}, {page_label}]"

    header_cols = st.columns([3, 1, 1])
    with header_cols[0]:
        st.markdown(
            f"**Passage #{index}** &nbsp; <span class='rerag-badge rerag-badge-info'>{cite_label}</span>",
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown(
            f"Cosine Sim: <span class='rerag-score-tag'>{score:.4f}</span>",
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        st.markdown(
            f"Distance: <span class='rerag-score-tag'>{distance:.4f}</span>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="rerag-passage-card">
            <div class="rerag-passage-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Inspect Passage Metadata (#{index} - {chunk_id})", expanded=False):
        meta_display = dict(metadata or {})
        meta_display.update({
            "chunk_id": chunk_id,
            "document_id": doc_label,
            "page_number": page_number,
            "cosine_similarity": score,
            "cosine_distance": distance,
        })
        st.json(meta_display)


def render_trace_viewer(steps: List[Any]) -> None:
    """Render interactive trace viewer of ReRAG refinement steps.

    Args:
        steps: List of PipelineStepTrace instances or step dictionaries.
    """
    if not steps:
        st.info("No iterative refinement steps were triggered (single-hop generation satisfied threshold).")
        return

    st.markdown(f"#### 🔄 ReRAG Verification & Refinement Traces ({len(steps)} Steps)")

    for i, step in enumerate(steps, 1):
        step_idx = getattr(step, "step_index", i)
        query = getattr(step, "query", "")
        cand_ans = getattr(step, "candidate_answer", "")
        verif = getattr(step, "verification_result", {}) or {}
        ref_q = getattr(step, "refinement_query", None)
        contexts = getattr(step, "retrieved_contexts", [])

        with st.expander(f"Step {step_idx}: Query: '{query[:60]}...'", expanded=(i == len(steps))):
            st.markdown(f"**Step Query:** `{query}`")
            st.markdown(f"**Candidate Generation:**\n> {cand_ans}")

            v_cols = st.columns(3)
            with v_cols[0]:
                faith_score = verif.get("faithfulness_score", verif.get("score"))
                st.metric("Faithfulness Score", f"{faith_score:.2f}" if isinstance(faith_score, (int, float)) else "N/A")
            with v_cols[1]:
                verified = verif.get("is_verified", verif.get("verified"))
                st.metric("Threshold Met", "✅ True" if verified else "❌ False")
            with v_cols[2]:
                st.metric("Passages Retrieved", len(contexts))

            if ref_q:
                st.markdown(f"**Refinement Query Formulated:** `{ref_q}`")


def render_comparison_view(
    baseline_result: Optional[Dict[str, Any]] = None,
    rerag_result: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    selected_metrics: Optional[Dict[str, Any]] = None,
    baseline_config: Optional[Any] = None,
    selected_config: Optional[Any] = None,
    held_out_evaluation: Optional[Dict[str, Any]] = None,
) -> None:
    """Render side-by-side comparison between Baseline and Selected/ReRAG results.

    Handles both RAG answer comparisons and optimization metric comparisons.
    """
    # 1. RAG Answer Comparison Mode
    if baseline_result is not None and rerag_result is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔹 Baseline RAG")
            st.markdown(f"> {baseline_result.get('final_answer', 'No answer')}")
            st.caption(f"Retrieved {len(baseline_result.get('contexts', []))} chunks | Runtime: {baseline_result.get('execution_time_seconds', 0.0):.2f}s")

        with c2:
            st.markdown("### 🔸 ReRAG (Refined)")
            st.markdown(f"> {rerag_result.get('final_answer', 'No answer')}")
            steps = rerag_result.get("steps", [])
            st.caption(f"Refinement Steps: {len(steps)} | Runtime: {rerag_result.get('execution_time_seconds', 0.0):.2f}s")
        return

    # 2. Optimization Metrics Comparison Mode
    if baseline_metrics is not None and selected_metrics is not None:
        b_score = baseline_metrics.get("objective_score", baseline_metrics.get("composite_score", 0.0))
        s_score = selected_metrics.get("objective_score", selected_metrics.get("composite_score", 0.0))
        delta = s_score - b_score

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Baseline Objective",
                f"{b_score:.4f}",
                help="Objective score on training/optimization set using baseline chunk configuration.",
            )
        with col2:
            st.metric(
                "Selected Objective",
                f"{s_score:.4f}",
                delta=f"{delta:+.4f}",
                help="Objective score achieved by selected chunk configuration.",
            )
        with col3:
            b_cfg_str = f"{baseline_config.chunk_size} / {baseline_config.chunk_overlap}" if baseline_config else "N/A"
            st.metric("Baseline Config", b_cfg_str, help="Chunk Size / Overlap")
        with col4:
            s_cfg_str = f"{selected_config.chunk_size} / {selected_config.chunk_overlap}" if selected_config else "N/A"
            st.metric("Selected Config", s_cfg_str, help="Chunk Size / Overlap")

        # Detailed metric breakdown table
        metric_keys = [
            ("faithfulness_ratio", "Faithfulness Ratio"),
            ("context_coverage", "Context Coverage"),
            ("semantic_similarity", "Semantic Similarity"),
            ("token_f1", "Token F1"),
            ("hallucination_rate", "Hallucination Rate"),
        ]

        rows = []
        for key, label in metric_keys:
            b_val = baseline_metrics.get(key)
            s_val = selected_metrics.get(key)
            if b_val is not None or s_val is not None:
                b_float = float(b_val) if b_val is not None else 0.0
                s_float = float(s_val) if s_val is not None else 0.0
                diff = s_float - b_float
                rows.append({
                    "Evaluation Metric": label,
                    "Baseline Configuration": f"{b_float:.4f}",
                    "Selected Configuration": f"{s_float:.4f}",
                    "Absolute Delta": f"{diff:+.4f}",
                })

        if rows:
            st.markdown("#### 📊 Core Metric Comparison")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Held-out generalization evaluation
        if held_out_evaluation:
            st.markdown("#### 🔬 Held-Out Test Generalization")
            st.caption("Evaluated on unseen held-out test questions to assess whether optimization gains generalize.")
            h_b = held_out_evaluation.get("baseline_score", 0.0)
            h_s = held_out_evaluation.get("selected_score", 0.0)
            h_gain = held_out_evaluation.get("held_out_gain", h_s - h_b)

            h_c1, h_c2, h_c3 = st.columns(3)
            with h_c1:
                st.metric("Held-Out Baseline", f"{h_b:.4f}")
            with h_c2:
                st.metric("Held-Out Selected", f"{h_s:.4f}", delta=f"{h_gain:+.4f}")
            with h_c3:
                st.metric("Test Samples Count", held_out_evaluation.get("test_samples_count", 0))


def render_candidate_table(trials: Sequence[Any]) -> None:
    """Render candidate history table with status badges and decision details."""
    if not trials:
        st.info("No candidate trials recorded.")
        return

    table_data = []
    for t in trials:
        cfg = getattr(t, "config", None)
        c_size = getattr(cfg, "chunk_size", None) if cfg else getattr(t, "attempted_params", {}).get("chunk_size", "N/A")
        c_over = getattr(cfg, "chunk_overlap", None) if cfg else getattr(t, "attempted_params", {}).get("chunk_overlap", "N/A")
        score = getattr(t, "objective_score", None)
        score_str = f"{score:.4f}" if score is not None else "—"

        status_str = getattr(t, "status", "unknown")
        reason = getattr(t, "decision_reason", "")
        time_sec = getattr(t, "execution_time_seconds", 0.0)

        table_data.append({
            "Trial": getattr(t, "trial_id", 0),
            "Iter": getattr(t, "iteration", 0),
            "Chunk Size": c_size,
            "Overlap": c_over,
            "Step": getattr(t, "step_size", 0),
            "Score": score_str,
            "Status": status_str.capitalize(),
            "Decision Reason": reason,
            "Time (s)": f"{time_sec:.2f}",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_candidate_chart(trials: Sequence[Any], baseline_score: Optional[float] = None) -> None:
    """Render an interactive Altair chart tracking objective score progression."""
    records = []
    for t in trials:
        score = getattr(t, "objective_score", None)
        if score is not None:
            cfg = getattr(t, "config", None)
            c_size = getattr(cfg, "chunk_size", 0) if cfg else 0
            c_over = getattr(cfg, "chunk_overlap", 0) if cfg else 0
            records.append({
                "trial_id": getattr(t, "trial_id", 0),
                "iteration": getattr(t, "iteration", 0),
                "objective_score": float(score),
                "status": getattr(t, "status", "unknown").capitalize(),
                "config": f"Size {c_size}, Overlap {c_over}",
                "reason": getattr(t, "decision_reason", ""),
            })

    if not records:
        st.info("No evaluated candidates with numerical objective scores to display.")
        return

    df = pd.DataFrame(records)

    # Base chart
    color_scale = alt.Scale(
        domain=["Accepted", "Rejected", "Baseline", "Failed", "Rejected_invalid"],
        range=["#10b981", "#f59e0b", "#3b82f6", "#ef4444", "#94a3b8"],
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("trial_id:O", title="Candidate Trial ID"),
            y=alt.Y("objective_score:Q", title="Objective Score", scale=alt.Scale(zero=False)),
            color=alt.Color("status:N", scale=color_scale, title="Trial Status"),
            tooltip=["trial_id", "iteration", "config", "objective_score", "status", "reason"],
        )
        .properties(height=320)
    )

    if baseline_score is not None:
        rule = (
            alt.Chart(pd.DataFrame([{"baseline": baseline_score}]))
            .mark_rule(color="#94a3b8", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(y="baseline:Q")
        )
        chart = chart + rule

    st.altair_chart(chart, use_container_width=True)



def render_multi_method_comparison_table(summary_records: Sequence[Dict[str, Any]]) -> None:
    """Render a styled comparative table across all evaluated optimization strategies.

    Displays Chunk Size, Chunk Overlap, Optimization (Train) Score, Validation Score,
    Held-Out Test Score, Generalization Gap, Faithfulness, Coverage, Similarity,
    MRR@k, Citation Precision, Latency, and Computational Cost.
    """
    if not summary_records:
        st.info("No comparative summary records available.")
        return

    df = pd.DataFrame(summary_records)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_multi_method_charts(comparison_result: Any) -> None:
    """Render comparative Altair visualizations for multi-method optimization experiments.

    Generates:
    1. Grouped bar chart comparing Train vs. Held-Out Test performance across methods.
    2. Generalization gap bar chart highlighting potential overfitting.
    3. Multi-series trial convergence trajectories across candidate search steps.
    """
    if comparison_result is None:
        st.info("No comparative data available to visualize.")
        return

    # Extract methods dict or dict representation
    methods = getattr(comparison_result, "methods", {})
    if isinstance(comparison_result, dict):
        methods = comparison_result.get("methods", {})

    if not methods:
        st.info("No method details found in comparison result.")
        return

    # 1. Train vs Test Performance Bar Chart
    perf_records = []
    gap_records = []
    trial_records = []

    for m_id, m_data in methods.items():
        disp_name = getattr(m_data, "display_name", None) or (m_data.get("display_name") if isinstance(m_data, dict) else m_id)
        train_score = getattr(m_data, "train_score", None) or (m_data.get("train_score") if isinstance(m_data, dict) else 0.0)
        test_score = getattr(m_data, "test_score", None) or (m_data.get("test_score") if isinstance(m_data, dict) else 0.0)
        gen_gap = getattr(m_data, "generalization_gap", None) or (m_data.get("generalization_gap") if isinstance(m_data, dict) else 0.0)

        perf_records.append({"Method": disp_name, "Split": "Optimization / Train", "Score": float(train_score)})
        perf_records.append({"Method": disp_name, "Split": "Held-Out Test", "Score": float(test_score)})
        gap_records.append({"Method": disp_name, "Generalization Gap": float(gen_gap)})

        # Trial trajectories
        candidates = getattr(m_data, "candidate_history", []) or (m_data.get("candidate_history", []) if isinstance(m_data, dict) else [])
        for cand in candidates:
            score = getattr(cand, "objective_score", None) or (cand.get("objective_score") if isinstance(cand, dict) else None)
            if score is not None:
                t_id = getattr(cand, "trial_id", 0) or (cand.get("trial_id", 0) if isinstance(cand, dict) else 0)
                status = getattr(cand, "status", "unknown") or (cand.get("status", "unknown") if isinstance(cand, dict) else "unknown")
                cfg = getattr(cand, "config", None) or (cand.get("config") if isinstance(cand, dict) else None)
                c_sz = getattr(cfg, "chunk_size", 0) if cfg and hasattr(cfg, "chunk_size") else (cfg.get("chunk_size", 0) if isinstance(cfg, dict) else 0)
                c_ov = getattr(cfg, "chunk_overlap", 0) if cfg and hasattr(cfg, "chunk_overlap") else (cfg.get("chunk_overlap", 0) if isinstance(cfg, dict) else 0)
                trial_records.append({
                    "Method": disp_name,
                    "Trial": int(t_id),
                    "Score": float(score),
                    "Status": str(status).capitalize(),
                    "Config": f"{c_sz}/{c_ov}",
                })

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### 📊 Train vs. Held-Out Test Generalization")
        df_perf = pd.DataFrame(perf_records)
        chart_perf = (
            alt.Chart(df_perf)
            .mark_bar()
            .encode(
                x=alt.X("Method:N", title=None, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Score:Q", title="Objective Score", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Split:N",
                    scale=alt.Scale(domain=["Optimization / Train", "Held-Out Test"], range=["#3b82f6", "#10b981"]),
                    title="Evaluation Split",
                ),
                xOffset="Split:N",
                tooltip=["Method", "Split", alt.Tooltip("Score:Q", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_perf, use_container_width=True)

    with c2:
        st.markdown("##### 📉 Generalization Gap (Train - Held-Out)")
        df_gap = pd.DataFrame(gap_records)
        chart_gap = (
            alt.Chart(df_gap)
            .mark_bar(color="#f59e0b")
            .encode(
                x=alt.X("Method:N", title=None, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Generalization Gap:Q", title="Score Drop (Gap)"),
                tooltip=["Method", alt.Tooltip("Generalization Gap:Q", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_gap, use_container_width=True)

    # 2. Multi-Method Convergence Trajectory Chart
    if trial_records:
        st.markdown("##### 📈 Trial Trajectories & Search Convergence")
        df_trials = pd.DataFrame(trial_records)
        chart_traj = (
            alt.Chart(df_trials)
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2)
            .encode(
                x=alt.X("Trial:Q", title="Candidate Trial Index"),
                y=alt.Y("Score:Q", title="Objective Score", scale=alt.Scale(zero=False)),
                color=alt.Color("Method:N", title="Method", scale=alt.Scale(scheme="category10")),
                tooltip=["Method", "Trial", alt.Tooltip("Score:Q", format=".4f"), "Config", "Status"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_traj, use_container_width=True)


def render_dataset_split_summary(splits: Any) -> None:
    """Render visual summary badges for train, validation, and held-out test splits."""
    if splits is None:
        return

    n_train = len(getattr(splits, "train_samples", []))
    n_val = len(getattr(splits, "val_samples", []))
    n_test = len(getattr(splits, "test_samples", []))
    total = n_train + n_val + n_test

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total QA Questions", total)
    with c2:
        st.metric("Optimization (Train)", n_train, help="Used exclusively during parameter search.")
    with c3:
        st.metric("Validation", n_val, help="Used for checkpointing/tuning.")
    with c4:
        st.metric("Held-Out Test", n_test, help="Strictly isolated for final uncorrupted evaluation.")


def render_sidebar_config(
    settings: Any,
    offline_mode_toggle: bool = False,
) -> Dict[str, Any]:
    """Render sidebar configuration overview and runtime controls."""
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")

        st.markdown(
            f"""
            - **Embedding Model:** `{getattr(settings.embedding, 'model_name', 'all-MiniLM-L6-v2')}`
            - **Compute Device:** `{getattr(settings.embedding, 'device', 'cpu')}`
            - **Storage Engine:** `ChromaDB`
            - **LLM Model:** `{getattr(settings.llm, 'model', 'llama3.2:latest')}`
            """
        )

        st.markdown("---")
        st.markdown("### 🧪 Offline & Test Controls")

        offline_mode = st.toggle(
            "Offline / Mock LLM Mode",
            value=offline_mode_toggle,
            help="Enable deterministic offline responses when local Ollama service is not running.",
        )

        st.caption("Allows complete verification of document ingestion, vector retrieval, and optimization search.")

        if st.button("🧹 Clear Session & Cache", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 0.76rem; color: #64748b; text-align: center;'>"
            "ReRAG Research Platform v1.0.0<br>Scientific Rigor & Hallucination Reduction"
            "</div>",
            unsafe_allow_html=True,
        )

    return {"offline_mode": offline_mode}

