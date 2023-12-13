# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "semantic-segmentation-tgce"
copyright = "2023, Brandon Lotero"
author = "Brandon Lotero"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_wagtail_theme", 
]

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# include the theme in the list of extensions to be loaded

# select the theme
html_theme = "sphinx_wagtail_theme"

html_static_path = ["_static"]

highlight_language = 'bash'

