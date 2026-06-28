# Sphinx configuration for rk-recall / rk-recall Sphinx 配置
# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."

from __future__ import annotations

import os
import sys

# §将项目根目录加入 sys.path, 以便 autodoc 发现 rk_recall
# §Add project root to sys.path so autodoc can find rk_recall
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "rk-recall"
copyright = "2026, rk-recall Contributors <17319939601@163.com>"
author = "rk-recall Contributors"
release = "1.0.0"
version = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Math --------------------------------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
