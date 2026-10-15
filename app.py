"""ReRAG Research Platform Streamlit Application Entrypoint.

Provides interactive research dashboard for investigating and optimizing retrieval-augmented
generation hyperparameters, claim-level factual verification, and hallucination reduction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings, setup_logging
from ui.dashboard import render_dashboard

setup_logging()
logger = logging.getLogger("rerag.app")


def main() -> None:
    """Streamlit application main entrypoint."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Please run 'pip install -r requirements.txt'")
        sys.exit(1)

    settings = get_settings()
    settings.paths.ensure_directories()

    # Streamlit master page configuration
    st.set_page_config(
        page_title="ReRAG Research Platform",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        render_dashboard()
    except Exception as exc:
        logger.exception("Fatal error rendering ReRAG dashboard: %s", exc)
        st.error(f"An unexpected application error occurred: {exc}")


if __name__ == "__main__":
    main()
