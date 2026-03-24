# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'depp_sfa'
copyright = '2026, Anashkina, Lucereau, Pyle, Tastet'
author = 'Anashkina, Lucereau, Pyle, Tastet'
release = '1.0.0'


html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) 

extensions = [
    'sphinx.ext.autodoc',      
    'sphinx.ext.viewcode',     
]

html_theme = 'sphinx_rtd_theme'
