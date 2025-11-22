# docs/conf.py
import os
import sys
from datetime import datetime

# Add project root to path so Sphinx can import pyfundlib
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "PyFundLib"
copyright = f"{datetime.now().year}, Himanshu Dixit"
author = "Himanshu Dixit"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # Beautiful type hints
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy code buttons
    "sphinxext.opengraph",  # Sexy social previews
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Type hints
set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"  # The most beautiful modern theme
html_static_path = ["_static"]
html_title = "PyFundLib ⚡"
html_logo = "_static/logo.png"  # Add your logo!
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2E86AB",
        "color-brand-content": "#F18F01",
    },
    "dark_css_variables": {
        "color-brand-primary": "#F18F01",
        "color-brand-content": "#2E86AB",
    },
    "announcement": "<b>🚀 PyFundLib 1.0 is here!</b> The future of open-source quant trading.",
}

# Furo is amazing — install with:
# pip install furo sphinx-autodoc-typehints sphinx-copybutton sphinxext-opengraph myst-parser

# -- OpenGraph (social sharing) ---------------------------------------------
ogp_site_url = "https://pyfundlib.com"
ogp_image = "_static/og-image.png"  # 1200x630 recommended
ogp_description_length = 200
ogp_type = "website"

# -- MyST Parser (Markdown support) -----------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Autodoc settings -------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autosummary_generate = True

# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/latest/", None),
}
