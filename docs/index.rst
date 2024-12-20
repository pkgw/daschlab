=========================================
daschlab: The DASCH Data Analysis Package
=========================================

daschlab_ is a Python package that assists with astrophysical analysis of data
from DASCH_, the effort to scan Harvard College Observatory’s collection of
`astronomical glass plates`_. This irreplaceable resource provides a means
for systematic study of the sky on 100-year time scales.

.. _daschlab: https://daschlab.readthedocs.io/
.. _DASCH: https://dasch.cfa.harvard.edu/
.. _astronomical glass plates: https://platestacks.cfa.harvard.edu/

This website contains only the **Python API reference material**. For tutorials
and howtos, see `the DASCH DR7 documentation`_.

.. _the DASCH DR7 documentation: https://dasch.cfa.harvard.edu/dr7/

This package is designed for primarily interactive usage in a JupyterLab
environment. The most important item provided in this module is the
`~daschlab.Session` class, which defines a daschlab analysis session. Obtain a
session by calling the `~daschlab.open_session()` function::

   from daschlab import open_session

   sess = open_session(".")

Virtually all subsequent analysis occurs through actions connected to an
initialized `~daschlab.Session` instance.

The version of *daschlab* described by this documentation has a DOI of
`10.5281/zenodo.14537904`_. You can obtain the DOI of the version of
*daschlab* that you are running via the `~daschlab.get_version_doi` function.
This DOI should be reported in scholarly publications that made use of
*daschlab*.

.. _10.5281/zenodo.14537904: https://doi.org/10.5281/zenodo.14537904


Table of Contents
=================

.. toctree::
   :maxdepth: 1

   api/daschlab
   api/daschlab.cutouts
   api/daschlab.exposures
   api/daschlab.extracts
   api/daschlab.lightcurves
   api/daschlab.photometry
   api/daschlab.query
   api/daschlab.refcat
   api/daschlab.series


Getting help
============

If you run into any issues when using daschlab_, please review `the DASCH DR7
documentation`_, inquire with `the DASCH Astrophysics email group
<https://gaggle.email/join/dasch@gaggle.email>`_ or open an issue `on its GitHub
repository <https://github.com/pkgw/daschlab/issues>`_.
