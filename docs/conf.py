"""Sphinx configuration for the pf-chees-hmc documentation.

Build from the repository root, in an environment where ``pfcheeshmc`` imports (the API page reads its docstrings):

    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html
"""

import importlib.metadata

project = "pf-chees-hmc"
author = "Rizqiansyah"
copyright = "2025-2026, Rizqiansyah"
try:
    release = importlib.metadata.version("pfcheeshmc")
except importlib.metadata.PackageNotFoundError:
    release = "unknown"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# The case study is a draft until 0.3.1; _build is the output.
exclude_patterns = ["_build", "requirements.txt"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo"
html_title = f"pf-chees-hmc {release}"
html_theme_options = {
    "source_repository": "https://github.com/Rizqiansyah/pf-chees-hmc/",
    "source_branch": "main",
    "source_directory": "docs/",
}
copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True

# Types from other projects named in signatures; no intersphinx (the build needs no network).
nitpick_ignore = [("py:class", "pm.Model"), ("py:class", "arviz.InferenceData"),
                  ("py:class", "os.PathLike"), ("py:class", "pathlib.Path")]
