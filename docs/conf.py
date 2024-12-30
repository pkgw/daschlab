project = "daschlab"
author = "Peter K. G. Williams"
copyright = "2024, " + author

release = "1.0.0"  # cranko project-version
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "numpydoc",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False

html_theme = "bootstrap-astropy"
html_theme_options = {
    "logotext1": "daschlab",
    "logotext2": "",
    "logotext3": ":docs",
    "astropy_project_menubar": False,
}
html_static_path = ["_static"]
htmlhelp_basename = "daschlabdoc"

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://data.astropy.org/intersphinx/python3.inv"),
    ),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "bokeh": ("https://docs.bokeh.org/en/latest/", None),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://data.astropy.org/intersphinx/numpy.inv"),
    ),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "pywwt": ("https://pywwt.readthedocs.io/en/stable/", None),
}

numpydoc_show_class_members = False

nitpicky = True

default_role = "obj"

# html_logo = "images/logo.png"

linkcheck_retries = 5
linkcheck_timeout = 10
