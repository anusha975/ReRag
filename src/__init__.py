"""ReRAG: Research implementation of Retrieval-Augmented Generation for reducing hallucination."""

import os
from pathlib import Path

import sys

# Redirect HF cache to D:\hf_cache if present to avoid C: drive space constraints
if "HF_HOME" not in os.environ and Path(r"D:\hf_cache").exists():
    os.environ["HF_HOME"] = r"D:\hf_cache"

# Ensure Python 3.13 site-packages are accessible if chromadb isn't in current environment
_SITE_PACKAGES_313 = Path(r"C:\Users\DELL\AppData\Local\Programs\Python\Python313\Lib\site-packages")
if _SITE_PACKAGES_313.exists() and str(_SITE_PACKAGES_313) not in sys.path:
    sys.path.append(str(_SITE_PACKAGES_313))

__version__ = "0.1.0"


