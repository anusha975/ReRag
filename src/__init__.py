"""ReRAG: Research implementation of Retrieval-Augmented Generation for reducing hallucination."""

import os
from pathlib import Path

# Redirect HF cache to D:\hf_cache if present to avoid C: drive space constraints
if "HF_HOME" not in os.environ and Path(r"D:\hf_cache").exists():
    os.environ["HF_HOME"] = r"D:\hf_cache"

__version__ = "0.1.0"

