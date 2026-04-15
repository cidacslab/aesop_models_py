import os
import sys

project = "aesop_models_py"
copyright = "2026, Seu Nome"
author = "Seu Nome"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_rtd_theme"

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
