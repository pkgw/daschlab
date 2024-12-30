# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
The toplevel Python module of the `daschlab`_ package.

.. _daschlab: https://daschlab.readthedocs.io/

daschlab_ is a Python package that assists with astrophysical analysis of data
from DASCH_, the effort to scan Harvard College Observatory’s collection of
`astronomical glass plates`_. This irreplaceable resource provides a means for
systematic study of the sky on 100-year time scales.

.. _daschlab: https://daschlab.readthedocs.io/
.. _DASCH: https://dasch.cfa.harvard.edu/
.. _astronomical glass plates: https://platestacks.cfa.harvard.edu/

This website contains only the **Python API reference material**. For tutorials
and howtos, see `the DASCH DR7 documentation`_.

.. _the DASCH DR7 documentation: https://dasch.cfa.harvard.edu/dr7/

This package is designed for primarily interactive usage in a JupyterLab
environment. The most important item provided in this module is the `Session`
class, which defines a daschlab analysis session. Obtain a session by calling
the `open_session()` function::

   from daschlab import open_session

   sess = open_session(".")

Virtually all subsequent analysis occurs through actions connected to an
initialized `Session` instance.
"""

from contextlib import contextmanager
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
from typing import Dict, FrozenSet, Iterable, Optional
import warnings

from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
import astropy.utils.exceptions
import astropy.version
import numpy as np
from pywwt.jupyter import connect_to_app, WWTJupyterWidget

from .apiclient import ApiClient
from .query import SessionQuery
from .refcat import RefcatSources, RefcatSourceRow, _query_refcat
from .exposures import Exposures, ExposureRow, _query_exposures
from .lightcurves import Lightcurve, _query_lc
from .cutouts import _query_cutout
from .extracts import Extract, _query_extract

__all__ = [
    "__version__",
    "SUPPORTED_REFCATS",
    "InteractiveError",
    "Session",
    "get_concept_doi",
    "get_version_doi",
    "open_session",
    "source_name_to_fs_name",
]

__version__ = "1.0.0"  # cranko project-version


if astropy.version.major < 6:
    warnings.warn(
        f"daschlab requires Astropy version 6.0 or greater, but you have Astropy version {astropy.__version__}. "
        "If you don't upgrade your Astropy, expect strange crashes."
    )


def get_version_doi() -> str:
    """
    Get the DOI associated with the exact version of *daschlab* that you are
    running.

    Returns
    =======
    str

    Notes
    =====
    If you are reporting the results of a scientific analysis based on
    *daschlab*, you are strongly recommended to use this function to ensure that
    you cite the DOI of the exact version of the software that you used. If you
    use *daschlab* to generate data files that will be archived, we suggest
    embedding using this function to determine the DOI of *daschlab* dynamically
    and embedding it in those files as metadata.

    See also `How to Cite DASCH`_.

    .. _How to Cite DASCH: https://dasch.cfa.harvard.edu/citing/

    The returned `DOI`_ will start with the characters "10.". If you are running
    a development build of the software, the returned value will instead start
    with "xx.", and will not be a valid DOI.

    .. _DOI: https://www.doi.org/the-identifier/what-is-a-doi/

    See Also
    ========
    get_concept_doi : Get the DOI that identifies the *daschlab* software
    """
    return "10.5281/zenodo.14574817"


def get_concept_doi() -> str:
    """
    Get the "concept DOI" that identifies the *daschlab* software.

    Returns
    =======
    str

    Notes
    =====
    The "concept DOI" associated with *daschlab* identifies the *daschlab*
    software in general. The value returned by this function should never
    change. In most cases, you probably should use the "version DOI",
    identifying the exact version of the software that is running. You can
    obtain this with the function `get_version_doi`.

    See also `How to Cite DASCH`_.

    .. _How to Cite DASCH: https://dasch.cfa.harvard.edu/citing/

    The returned `DOI`_ will start with the characters "10.". If you are running
    a development build of the software, the returned value will instead start
    with "xx.", and will not be a valid DOI.

    .. _DOI: https://www.doi.org/the-identifier/what-is-a-doi/

    See Also
    ========
    get_version_doi : Get the DOI of the precise version of *daschlab* in use
    """
    return "10.5281/zenodo.14537903"


SUPPORTED_REFCATS: FrozenSet[str] = frozenset(("apass", "atlas"))


CUTOUT_HALFSIZE: Angle = Angle(600 * u.arcsec)

# In order to get ~all catalog sources on a cutout, the radius should be the
# size of such a square's diagonal.
REFCAT_RADIUS: Angle = Angle(850 * u.arcsec)


class InteractiveError(Exception):
    """
    A error class for errors that should be shown to a human user without
    traceback context.

    If you initialize an interactive daschlab session in a Jupyter/IPython
    environment, this package will install a custom exception handler that
    customizes the display of `InteractiveError` instances. Rather than printing
    a full traceback, just the exception message will be printed. This is
    intended to produce easier-to-read output during interactive analysis
    sessions.
    """

    pass


def _ipython_custom_exception_formatter(_shell, etype, evalue, _tb, tb_offset=None):
    print("error:", evalue, f"({etype.__name__})", file=sys.stderr)


def _maybe_install_custom_exception_formatter():
    try:
        ip = get_ipython()
    except NameError:
        return

    ip.set_custom_exc((InteractiveError,), _ipython_custom_exception_formatter)


def _formatsc(sc: SkyCoord) -> str:
    r = sc.ra.to_string(sep=":", precision=1, unit=u.hour)
    d = sc.dec.to_string(sep=":", precision=0, unit=u.deg, alwayssign=True)
    return f"{r} {d}"


class Session:
    """A daschlab analysis session.

    Do not construct instances of this class directly. Instead, use the function
    `daschlab.open_session()`, which may perform additional, helpful
    initializations of the analysis environment.

    Once you have obtained a `Session` instance, you should configure its key
    parameters with a series of function calls resembling the following::

        from daschlab import open_session

        sess = open_session(".")
        sess.select_target("V* RY Cnc")
        sess.select_refcat("apass")

        # after opening the WWT JupyterLab application:
        await sess.connect_to_wwt()

        # You will generally also want to run:
        from bokeh.io import output_notebook
        output_notebook()

    After this initialization is complete, you can obtain various data products
    relevant to your session with the following methods:

    - `~Session.refcat()` to access a table of DASCH catalog sources near
      your target
    - `~Session.exposures()` to access a table of exposures overlapping
      your target
    - `~Session.lightcurve()` to access lightcurves for the catalog sources
    - `~Session.cutout()` to download exposure cutout images centered on
      your target
    - `~Session.mosaic()` to download full-plate "mosaic" data and synthesize
      a "value-added" FITS image
    - `~Session.extract()` to download photometric data from a single plate,
      centered on your target
    """

    _root: pathlib.Path
    _interactive: bool = True
    _apiclient: ApiClient
    _internal_simg: str = ""
    _query: Optional[SessionQuery] = None
    _refcat: Optional[RefcatSources] = None
    _exposures: Optional[Exposures] = None
    _wwt: Optional[WWTJupyterWidget] = None
    _refcat_table_layer: Optional["pywwt.layers.TableLayer"] = None
    _lc_cache: Dict[str, Lightcurve] = None
    _extract_cache: Dict[str, Extract] = None
    _exposure_image_layer_cache: dict = None
    _extract_table_layer_cache: Dict[str, "pywwt.layers.TableLayer"] = None

    def __init__(self, root: str, interactive: bool = True, _internal_simg: str = ""):
        self._root = pathlib.Path(root)
        self._interactive = interactive
        self._apiclient = ApiClient()
        self._internal_simg = _internal_simg
        self._lc_cache = {}
        self._extract_cache = {}
        self._exposure_image_layer_cache = {}
        self._extract_table_layer_cache = {}

        try:
            self._root.mkdir(parents=True)
            self._info(f"Created new DASCH session at disk location `{self._root}`")
        except FileExistsError:
            self._info(f"Opened DASCH session at disk location `{self._root}`")

        # Recover query?

        try:
            f_query = self.path("query.json").open("rt")
        except FileNotFoundError:
            self._query = None
            self._info(
                f"- Query target not yet defined - run something like `{self._my_var_name()}.select_target(name='HD 209458')`"
            )
        else:
            self._query = SessionQuery.schema().load(json.load(f_query))

            if self._query.name:
                self._info(f"- Query target: `{self._query.name}`")
            else:
                self._info(
                    f"- Query target: `{_formatsc(self._query.pos_as_skycoord())}"
                )

        # Recover refcat?

        try:
            self._refcat = RefcatSources.read(
                str(self.path("refcat.ecsv")), format="ascii.ecsv"
            )
            self._refcat.meta["daschlab_sess_key"] = str(self._root)
        except FileNotFoundError:
            self._refcat = None
            self._info(
                f"- Refcat not yet fetched - after setting target, run something like `{self._my_var_name()}.select_refcat('apass')`"
            )
        else:
            if len(self._refcat):
                self._info(
                    f"- Refcat: {len(self._refcat)} sources from `{self._refcat_name()}`"
                )
            else:
                self._info("- Refcat: present but empty")

        # Recover exposures?

        try:
            self._exposures = Exposures.read(
                str(self.path("exposures.ecsv")), format="ascii.ecsv"
            )
            self._exposures.meta["daschlab_sess_key"] = str(self._root)
        except FileNotFoundError:
            self._exposures = None
        else:
            self._info(f"- Exposures: {len(self._exposures)} relevant exposures")

    def _my_var_name(self) -> str:
        for name, value in globals().items():
            if value is self:
                return name

        return "sess"

    def _info(self, msg: str):
        # TODO: use logging or something if not interactive
        print(msg)

    def _warn(self, msg: str):
        # TODO: use logging or something if not interactive
        print("warning:", msg, file=sys.stderr)

    def _require_internal(self):
        if self._internal_simg:
            return

        raise InteractiveError(
            "sorry, this functionality requires a direct login to the DASCH HPC cluster"
        )

    @contextmanager
    def _save_atomic(self, *relpath_pieces: Iterable[str], mode: str = "wt"):
        """
        A context manager for saving a file into the session tree atomically, so
        that the partially-written form of the file is not observable under its
        final name.
        """

        with tempfile.NamedTemporaryFile(
            mode=mode, dir=str(self._root), delete=False
        ) as f:
            yield f

        os.rename(f.name, self.path(*relpath_pieces))

    def path(self, *pieces: Iterable[str]) -> pathlib.Path:
        """Generate a filesystem path within this session.

        Parameters
        ==========
        *pieces : sequence of `str`
          Path components

        Returns
        =======
        A `pathlib.Path` relative to this session's "root" directory.
        """
        return self._root.joinpath(*pieces)

    def select_target(
        self,
        name: Optional[str] = None,
        ra_deg: Optional[float] = None,
        dec_deg: Optional[float] = None,
        coords: Optional[SkyCoord] = None,
    ) -> "Session":
        """
        Specify the center of the session's target area.

        Parameters
        ==========
        name : optional `str`
          If specified, target this session based on the given Simbad-resolvable
          source name.
        ra_deg : optional `float`
          If specified, target this session based on given equatorial right
          ascension and declination, both measured in degrees.
        dec_deg : optional `float`
          Companion to the *ra_deg* argument.
        coords : optional `astropy.coordinates.SkyCoord`
          If specified, target this session based on the given Astropy
          coordinates, which will be converted to equatorial form.

        Returns
        =======
        *self*, for chaining convenience

        Notes
        =====
        There are three ways to identify the session target: a
        Simbad/Sesame-resolvable name; explicit equatorial (RA/dec) coordinates
        in degrees; or an Astropy `~astropy.coordinates.SkyCoord` object. You
        may only provide one form of identification.

        The first time you call this method for a given session, it resolve the
        target location (potentially making a Simbad/Sesame API call) and save
        the resulting information in a file named ``query.json``. Subsequent
        calls (i.e., ones made with the ``query.json`` file already existing)
        will merely check for consistency.

        After calling this method, you may use `~Session.query()` to obtain the
        cached information about your session’s target. This provides the RA and
        dec.
        """
        has_name = int(bool(name))
        has_radec = int(ra_deg is not None and dec_deg is not None)
        has_coords = int(coords is not None)

        if has_name + has_radec + has_coords != 1:
            raise ValueError(
                "one of `name`, `coords`, or both `ra_deg` and `dec_deg` must be specified"
            )

        if self._query is not None:
            if has_name:
                if not self._query.name:
                    self._warn(
                        f"on-disk query target name `{self._query.name}` does not agree with in-code name `{name}`"
                    )
                elif self._query.name != name:
                    raise InteractiveError(
                        f"on-disk query target name `{self._query.name}` does not agree with in-code name `{name}`"
                    )
            else:
                if has_coords:
                    ra_deg = coords.ra.deg
                    dec_deg = coords.dec.deg

                if self._query.ra_deg != ra_deg or self._query.dec_deg != dec_deg:
                    raise InteractiveError(
                        f"on-disk query target location (RA={self._query.ra_deg}, dec={self._query.dec_deg}) "
                        f"does not agree with in-code location (RA={ra_deg}, dec={dec_deg})"
                    )

            return self

        # First-time invocation; set up the query file
        print("- Querying API ...", flush=True)
        nametext = ""

        if has_name:
            self._query = SessionQuery.new_from_name(name)
            nametext = f" with name `{name}`"
        elif has_radec:
            self._query = SessionQuery.new_from_radec(ra_deg, dec_deg)
        else:
            assert has_coords
            self._query = SessionQuery.new_from_coords(coords)

        with self._save_atomic("query.json") as f_new:
            json.dump(self._query.to_dict(), f_new, ensure_ascii=False, indent=2)
            print(file=f_new)  # `json` doesn't do a trailing newline

        self._info(
            f"- Saved `query.json`{nametext} resolved to: {_formatsc(self._query.pos_as_skycoord())}"
        )
        return self

    def query(self) -> SessionQuery:
        """
        Obtain the session "query" specifying the center of the analysis region.

        Returns
        =======
        A `daschlab.query.SessionQuery` instance

        Notes
        =====
        You must call `~Session.select_target()` before calling this method.
        """

        if self._query is None:
            raise InteractiveError(
                f"you must select the target first - run something like `{self._my_var_name()}.select_target(name='Algol')`"
            )
        return self._query

    def select_refcat(self, name: str) -> "Session":
        """
        Specify which DASCH reference catalog to use.

        Parameters
        ==========
        name : `str`
          The name of the reference catalog. Supported values are ``"apass"``
          and ``"atlas"``, with ``"apass"`` being strongly recommended.

        Returns
        =======
        *self*, for chaining convenience

        Notes
        =====
        You must have called `~Session.select_target()` before calling this
        method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch information from the specified catalog,
        saving the resulting information in a file named ``refcat.ecsv``.
        Subsequent calls (i.e., ones made with the ``refcat.ecsv`` file already
        existing) will merely check for consistency.

        After calling this method, you may use `~Session.refcat()` to obtain the
        reference catalog data table. This table is a
        `daschlab.refcat.RefcatSources` object, which is a subclass of
        `astropy.table.Table`.
        """

        if name not in SUPPORTED_REFCATS:
            raise InteractiveError(
                f"invalid refcat name {name!r}; must be one of: {' '.join(SUPPORTED_REFCATS)}"
            )

        if self._refcat is not None:
            if not len(self._refcat):
                self._warn(
                    f"on-disk refcat is empty; assuming that it is for refcat `{name}`"
                )
            elif self._refcat_name() != name:
                raise InteractiveError(
                    f"on-disk refcat name `{self._refcat_name()}` does not agree with in-code name `{name}`"
                )

            return self

        # First-time invocation; query the catalog

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve refcat before setting target - run something like `{self._my_var_name()}.select_target(name='HD 209458')`"
            )

        t0 = time.time()
        print("- Querying API ...", flush=True)
        self._refcat = _query_refcat(
            self._apiclient, name, self._query.pos_as_skycoord(), REFCAT_RADIUS
        )
        self._refcat.meta["daschlab_sess_key"] = str(self._root)

        with self._save_atomic("refcat.ecsv") as f_new:
            self._refcat.write(f_new.name, format="ascii.ecsv", overwrite=True)

        elapsed = time.time() - t0
        self._info(
            f'- Retrieved {len(self._refcat)} sources from reference catalog "{name}" in {elapsed:.0f} seconds and saved as `refcat.ecsv`'
        )
        return self

    def refcat(self) -> RefcatSources:
        """
        Obtain the table of reference catalog sources associated with this
        session.

        Returns
        =======
        A `daschlab.refcat.RefcatSources` instance.

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.select_refcat()`
        before calling this method.
        """

        if self._refcat is None:
            raise InteractiveError(
                f"you must select the refcat first - run something like `{self._my_var_name()}.select_refcat('apass')`"
            )
        return self._refcat

    def _refcat_name(self) -> str:
        """
        Get the name of the currently active refcat.
        """
        return self.refcat()["refcat"][0]

    def exposures(self) -> Exposures:
        """
        Obtain the table of exposures overlapping the session target area.

        Returns
        =======
        A `daschlab.exposures.Exposures` instance.

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.select_refcat()`
        before calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch information from the exposure database, saving
        the resulting information in a file named ``exposures.ecsv``. Subsequent
        calls (i.e., ones made with the ``exposures.ecsv`` file already existing)
        will merely check for consistency and load the saved file.
        """

        if self._exposures is not None:
            return self._exposures

        # First-time invocation; query the database

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve exposures before setting target - run something like `{self._my_var_name()}.select_target(name='HD 209458')`"
            )

        t0 = time.time()
        print("- Querying API ...", flush=True)
        self._exposures = _query_exposures(
            self._apiclient, self._query.pos_as_skycoord()
        )
        self._exposures.meta["daschlab_sess_key"] = str(self._root)

        with self._save_atomic("exposures.ecsv") as f_new:
            self._exposures.write(f_new.name, format="ascii.ecsv", overwrite=True)

        elapsed = time.time() - t0
        self._info(
            f"- Retrieved {len(self._exposures)} relevant exposures in {elapsed:.0f} seconds and saved as `exposures.ecsv`"
        )
        return self._exposures

    def _resolve_src_reference(self, src_ref: "SourceReferenceType") -> RefcatSourceRow:
        if isinstance(src_ref, str) and src_ref == "click":
            info = self.wwt().most_recent_source

            if info is None:
                raise InteractiveError(
                    "trying to pick a source based on the latest WWT click, but nothing has been clicked"
                )

            src_id = info["layerData"].get("local_id")
            if src_id is None:
                raise InteractiveError(
                    'trying to pick a source based on the latest WWT click, but the clicked source has no "local_id" metadatum'
                )

            return self.refcat()[int(src_id)]

        if isinstance(src_ref, int) or isinstance(src_ref, np.integer):
            return self.refcat()[src_ref]

        assert isinstance(src_ref, RefcatSourceRow)
        return src_ref

    def lightcurve(self, src_ref: "SourceReferenceType") -> Lightcurve:
        """
        Obtain a table of lightcurve data for the specified source.

        Parameters
        ==========
        src_ref : `int` or `~daschlab.refcat.RefcatSourceRow` or ``"click"``.
          If this argument is an integer, it is interpreted as the "local ID" of
          a row in the reference catalog. Local IDs are assigned in order of
          distance from the query center, so a value of ``0`` fetches the
          lightcurve for the catalog source closest to the query target. This is
          probably what you want.

          If this argument is an instance of `~daschlab.refcat.RefcatSourceRow`,
          the lightcurve for the specified source is obtained.

          If this argument is the literal string value ``"click"``, the
          lightcurve for the catalog source that was most recently clicked in
          the WWT app is obtained. In particular, the most recently-clicked
          source must have a piece of metadata tagged ``local_id``, which is
          interpreted as a refcat local ID.

        Returns
        =======
        A `daschlab.lightcurves.Lightcurve` instance.

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.select_refcat()`
        before calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch information from the lightcurve database,
        saving the resulting information in a file inside the session's
        ``lightcurves`` subdirectory. Subsequent calls (i.e., ones made with the
        data file already existing) will merely check for consistency and load
        the saved file.
        """
        src = self._resolve_src_reference(src_ref)

        # We're going to need this later; emit the output now.
        exposures = self.exposures()

        name = src["ref_text"]
        lc = self._lc_cache.get(name)
        if lc is not None:
            return lc

        relpath = f"lightcurves/{name}.ecsv"

        try:
            lc = Lightcurve.read(str(self.path(relpath)), format="ascii.ecsv")
            self._lc_cache[name] = lc
            return lc
        except FileNotFoundError:
            pass

        # We need to fetch it

        self.path("lightcurves").mkdir(exist_ok=True)

        t0 = time.time()
        print("- Querying API ...", flush=True)
        lc = _query_lc(
            self._apiclient, src["refcat"], src["gsc_bin_index"], src["ref_number"]
        )

        # Cross-match with the exposures. We can assume that these are ones with
        # imaging, since those are the only ones that can yield photometry.

        exp_lookup = {}

        for exp in exposures:
            if exp["mosnum"] is not np.ma.masked:
                exp_lookup[
                    (exp["series"], exp["platenum"], exp["mosnum"], exp["solnum"])
                ] = exp["local_id"]

        lc["exp_local_id"] = -1

        for p in lc:
            p["exp_local_id"] = exp_lookup.get(
                (p["series"], p["platenum"], p["mosnum"], p["solnum"]), -1
            )

        # All done

        with self._save_atomic(relpath) as f_new:
            lc.write(f_new.name, format="ascii.ecsv", overwrite=True)

        self._lc_cache[name] = lc
        elapsed = time.time() - t0
        self._info(
            f"- Fetched {len(lc)} rows in {elapsed:.0f} seconds and saved as `{self.path(relpath)}`"
        )
        return lc

    def merge_lightcurves(
        self, src_refs: Iterable["SourceReferenceType"]
    ) -> Lightcurve:
        """
        Obtain a lightcurve merging rows from multiple catalog source
        lightcurves, under the assumption that they all contain data for the
        same astronomical source.

        Parameters
        ==========
        lcs : list of `int` or `~daschlab.refcat.RefcatSourceRow` or ``"click"``
            These parameters specify the refcat sources whose lightcurves will
            be merged together. Individual values are interpreted as in
            `~Session.lightcurve()`.

        Returns
        =======
        merged_lc : `~daschlab.lightcurves.Lightcurve`
            The merged lightcurve.

        Examples
        ========
        Obtain a merged lightcurve from the four refcat sources closest to the
        session's query position::

            lc = session.merge_lightcurves(range(4))

        Notes
        =====
        See the documentation for `daschlab.lightcurves.merge()` for details on
        the motivation, merge algorithm, and return value of this method.
        """
        from .lightcurves import merge

        return merge([self.lightcurve(r) for r in src_refs])

    def _resolve_exposure_reference(
        self, exp_ref: "ExposureReferenceType"
    ) -> ExposureRow:
        if isinstance(exp_ref, int) or isinstance(exp_ref, np.integer):
            return self.exposures()[exp_ref]

        assert isinstance(exp_ref, ExposureRow)
        return exp_ref

    def cutout(
        self, exp_ref: "ExposureReferenceType", no_download: bool = False
    ) -> Optional[str]:
        """
        Obtain a FITS cutout for the specified exposure.

        Parameters
        ==========
        exp_ref : `int` or `~daschlab.exposures.ExposureRow`
            If this argument is an integer, it is interpreted as the "local ID"
            of a row in the exposures table.

            If this argument is an instance of `~daschlab.exposures.ExposureRow`, the
            cutout for the specified exposure is obtained.

        no_download : optional `bool`, default False
            If set to true, no attempt will be made to download the requested
            cutout. If it is already cached on disk, the path will be returned;
            otherwise, `None` will be returned.

        Returns
        =======
        A `str` or `None`.
            If the former, the cutout was successfully fetched; the value is a
            path to a local FITS file containing the cutout data, *relative to the
            session root*. If the latter, a cutout could not be obtained. This
            could happen if the exposure has not been scanned, among other reasons.

        Examples
        ========
        Load the cutout of the chronologically-first calibrated observation of
        the target field (the exposure list is in chronological order)::

            from astropy.io import fits

            imaged_exposures = sess.exposures().keep_only.has_imaging()
            exp = imaged_exposures[0]

            relpath = sess.cutout(exp)
            assert relpath, f"could not get cutout of {exp.exp_id()}"

            # str() needed here because Astropy does not accept Path objects:
            hdu_list = fits.open(str(sess.path(relpath))

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.exposures()` before
        calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch the cutout, saving the resulting data in a
        file inside the session's ``cutouts`` subdirectory. Subsequent calls
        (i.e., ones made with the data file already existing) will merely check
        for consistency and load the saved file.

        See Also
        ========
        mosaic : obtain a full-plate "mosaic" image
        daschlab.exposures.Exposures.show : to show a cutout in the WWT view
        """
        from astropy.io import fits

        exp = self._resolve_exposure_reference(exp_ref)
        exp_id = exp.exp_id()
        if not exp.has_imaging():
            self._warn(
                f"cannot get a cutout for exposure {exp_id}: it does not have an imaging solution"
            )
            return None

        local_id = exp["local_id"]
        dest_relpath = f"cutouts/{local_id:05d}_{exp_id}.fits"

        if self.path(dest_relpath).exists():
            return dest_relpath

        if no_download:
            return None

        # Try to fetch it

        self.path("cutouts").mkdir(exist_ok=True)
        t0 = time.time()
        print("- Querying API ...", flush=True)
        center = self.query().pos_as_skycoord()

        try:
            fits_data = _query_cutout(
                self._apiclient, exp["series"], exp["platenum"], exp["solnum"], center
            )
        except Exception as e:
            # Now that we are better about distinguishing between exposures that
            # do and don't have imaging, this should probably be upgraded to an
            # exception again.
            self._warn(f"failed to fetch cutout for {exp_id}: {e}")
            return None

        print(f"- Fetched {len(fits_data)} bytes in {time.time()-t0:.0f} seconds")

        # Add a bunch of headers using the metadata that we have

        fits_data = io.BytesIO(fits_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with fits.open(fits_data) as hdul:
                h = hdul[0].header

                h["D_SCNNUM"] = exp["scannum"]
                h["D_EXPNUM"] = exp["expnum"]
                h["D_SOLNUM"] = exp["solnum"]
                # the 'class' column is handled as a masked array and when we convert it
                # to a row, the masked value seems to become a float64?
                h["D_PCLASS"] = "" if exp["class"] is np.ma.masked else exp["class"]

                if not np.isnan(exp["exptime"]):
                    h["EXPTIME"] = exp["exptime"] * 60

                h["DATE-OBS"] = exp["obs_date"].unmasked.fits
                h["MJD-OBS"] = exp["obs_date"].unmasked.mjd
                h["DATE-SCN"] = exp["scan_date"].unmasked.fits
                h["MJD-SCN"] = exp["scan_date"].unmasked.mjd
                h["DATE-MOS"] = exp["mos_date"].unmasked.fits
                h["MJD-MOS"] = exp["mos_date"].unmasked.mjd

                with self._save_atomic(dest_relpath) as f_new:
                    hdul.writeto(f_new.name, overwrite=True)

        self._info(f"- Saved `{self.path(dest_relpath)}`")
        return dest_relpath

    def extract(self, exp_ref: "ExposureReferenceType") -> Extract:
        """
        Obtain an extract of photometric data for the specified exposure.

        Parameters
        ==========
        exp_ref : `int` or ...

        Returns
        =======
        A `daschlab.extracts.Extract` instance.

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.select_refcat()`
        before calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch information from the lightcurve database,
        saving the resulting information in a file inside the session's
        ``lightcurves`` subdirectory. Subsequent calls (i.e., ones made with the
        data file already existing) will merely check for consistency and load
        the saved file.
        """
        exp = self._resolve_exposure_reference(exp_ref)
        plate_id = exp.plate_id()
        exp_id = exp.exp_id()

        if not exp.has_phot():
            self._warn(
                f"cannot get a photometry extract for exposure {exp_id}: it does not have photometry data (in this refcat)"
            )
            return None

        extract = self._extract_cache.get(exp_id)
        if extract is not None:
            return extract

        relpath = f"extracts/{exp_id}.ecsv"

        try:
            extract = Extract.read(str(self.path(relpath)), format="ascii.ecsv")
            extract.meta["daschlab_sess_key"] = str(self._root)
            extract.meta["daschlab_exposure_id"] = exp_id
            self._extract_cache[exp_id] = extract
            return extract
        except FileNotFoundError:
            pass

        # We need to fetch it

        self.path("extracts").mkdir(exist_ok=True)

        t0 = time.time()
        print("- Querying API ...", flush=True)
        extract = _query_extract(
            self._apiclient,
            self._refcat_name(),
            plate_id,
            exp["solnum"],
            self._query.pos_as_skycoord(),
        )
        extract.meta["daschlab_sess_key"] = str(self._root)
        extract.meta["daschlab_exposure_id"] = exp_id
        self._extract_cache[exp_id] = extract

        # This is a bit silly, but helps us treat lightcurves and extracts more
        # consistently

        extract["exp_local_id"] = exp_id

        # All done

        with self._save_atomic(relpath) as f_new:
            extract.write(f_new.name, format="ascii.ecsv", overwrite=True)

        elapsed = time.time() - t0
        self._info(
            f"- Fetched {len(extract)} rows in {elapsed:.0f} seconds and saved as `{self.path(relpath)}`"
        )
        return extract

    def mosaic(self, plate_id: str, binning: int) -> str:
        """
        Obtain a "value-added" full-plate FITS mosaic.

        Parameters
        ==========
        plate_id : `str`
            The ID of the plate whose mosaic will be fetched. This should have
            the form ``{series}{platenum}``, where the plate number is
            zero-padded to be five digits wide. Given an exposure table record,
            you can obtain its plate ID with
            `~daschlab.exposures.ExposureRow.plate_id()`.

        binning : int
            Allowed values are 1 or 16. If 1, the full-resolution mosaic is
            downloaded. If 16, the 16×16 binned mosaic is downloaded. The
            binned file is smaller by a factor of 256, so the binned mosaics
            should be preferred if at all possible.

        Returns
        =======
        `str`
            The returned value is a path to a local FITS file containing the
            value-added mosaic, *relative to the session root*.

        Examples
        ========
        Construct a binned FITS file for the plate with chronologically-last
        calibrated observation of the target field (the exposure list is in
        chronological order), and add it to the WWT display::

            # Assumes `sess` is a Session object and it has been connected to
            # WWT.

            from toasty import TilingMethod

            imaged_exposures = sess.exposures().keep_only.has_imaging()
            exp = imaged_exposures[-1]

            # Generate the value-add bin16 mosaic
            relpath = sess.mosaic(exp.plate_id(), 16)

            # Important to force TOAST tiling to avoid distortions with the
            # large angular sizes of DASCH plates.
            diskpath = str(sess.path(relpath))
            sess.wwt().layers.add_image_layer(diskpath, tiling_method=TilingMethod.TOAST)

        Notes
        =====
        The first time you call this method for a given plate, it will perform
        several DASCH API queries and download the "base" DASCH mosaic, saving
        the resulting file inside the session's ``base_mosaics`` subdirectory.
        This download may take a while; the largest DASCH base mosaics are more
        than a gigabyte in size. *daschlab* will then create the "value-add"
        mosaic inside the session's ``mosaics`` subdirectory, normalizing the
        data structure and inserting a variety of headers into the final FITS
        file. Subsequent invocations do no extra work if the final output file
        already exists.

        Unbinned full-plate mosaics are large, and they can therefore be slow
        to load into visualization tools. Prefer using the `cutout` method
        unless you are really going to want to examine the entire mosaic at
        the pixel level.

        Once a value-added mosaic is created, you can delete the base mosaics
        in the ``base_mosaics`` directory to save disk space.

        See Also
        ========
        cutout : get a full-resolution cutout around the session query point
        """
        from .mosaics import _get_mosaic

        return _get_mosaic(self, plate_id, binning)

    async def connect_to_wwt(self):
        """
        Connect this session to the WorldWide Telescope JupyterLab app.

        Notes
        =====
        This is an asynchonous function and should generally be called as::

           await sess.connect_to_wwt()

        After calling this function, the session will be able to automatically
        display various catalogs and images in the WWT viewer.
        """
        if self._wwt is not None:
            return

        self._wwt = await connect_to_app().becomes_ready()

        if self._wwt.foreground_opacity != 0:
            self._wwt.background = "Digitized Sky Survey (Color)"
            self._wwt.foreground_opacity = 0

        if self._query is not None:
            self._wwt.center_on_coordinates(
                self._query.pos_as_skycoord(), fov=REFCAT_RADIUS * 1.2
            )

    def wwt(self) -> WWTJupyterWidget:
        """
        Obtain the WorldWide Telescope "widget" handle associated with this session.

        Returns
        =======
        `pywwt.jupyter.WWTJupyterWidget`
          The WWT widget object.

        Notes
        =====
        You must call `~Session.connect_to_wwt()` before you can use this method.
        """
        if self._wwt is None:
            raise InteractiveError(
                f"you must connect to WWT asynchronously first - run `await {self._my_var_name()}.connect_to_wwt()`"
            )

        return self._wwt

    def delete_data(self):
        """
        Delete all files stored on-disk for this session.

        Notes
        =====
        This command will also clear any data tables stored in-memory, forcing
        them to be re-fetched from the data API as needed. It will recreate the
        session directory afterwards.
        """

        self._info(
            f"Deleting DASCH session data at disk location `{self._root}` and clearing tables"
        )

        try:
            shutil.rmtree(self._root)
        except FileNotFoundError:
            pass

        try:
            # Everything else assumes that this directory exists, since we
            # create it in the constructor; so we have to recreate it.
            self._root.mkdir(parents=True)
        except FileExistsError:
            pass

        self._query = None
        self._refcat = None
        self._exposures = None
        self._lc_cache = {}


_NAME_CONVERSION_FILTER_RE = re.compile("[^-+_.a-z0-9]")
_NAME_CONVERSION_CONDENSE_RE = re.compile("__*")


def source_name_to_fs_name(name: str) -> str:
    """
    Convert the name of an astronomical source to something convenient to use in
    file names.

    Parameters
    ==========
    name : `str`
        The astronomical source name.

    Returns
    =======
    The converted filesystem name.

    Notes
    =====
    This function is a helper that is used to automatically generate names for
    `Session` data directories. The transformation that it performs is
    intentionally not precisely specified, but it does things like lowercasing
    the input, removing spaces and special characters, and so on.
    """
    name = name.lower()
    name = _NAME_CONVERSION_FILTER_RE.sub("_", name)
    name = _NAME_CONVERSION_CONDENSE_RE.sub("_", name)
    return name


# We're not really that interested in maintaining a session cache ... but we
# want some of our tables to "know about" their associated session. The sensible
# way to do this is via their metadata, but Table metadata should be
# plain-old-data: they are always deepcopied when copying tables, and we do
# *not* want to be duplicating Sessions in these circumstances. By maintaining
# this dict, the tables can locate their sessions based on a key rather than a
# reference to the whole object.
_session_cache: Dict[str, Session] = {}


def open_session(
    root: str = ".",
    source: str = "",
    interactive: bool = True,
    _internal_simg: str = "",
) -> Session:
    """
    Open or create a new daschlab analysis session.

    Parameters
    ==========
    root : optional `str`, default ``"."``
        A path to a directory that will contain all of the data files associated
        with this analysis session. Overridden if *source_name* is specified.

    source : optional `str`, default ``""``
        If specified, derive *root* by processing this value with the function
        `source_name_to_fs_name`. See the examples below. The resulting name
        will be relative to the current directory and begin with ``daschlab_``.

    interactive : optional `bool`, default True
        Whether this is an interactive analysis session. If True, various
        warnings and errors will be printed under the assumption that a human
        will read them, as opposed to raising exceptions with proper tracebacks.

    Returns
    =======
    An initialized `Session` instance.

    Examples
    ========
    Open a session whose data are stored in a specific directory::

        from daschlab import open_session

        sess = open_session(root="snia/1987a")

    Open a session with a name automatically derived from a source name::

        from daschlab import open_session

        source = "SN 1987a"
        sess = open_session(source=source)
        sess.select_target(name=source)

    Notes
    =====
    It is possible to construct a `Session` directly, but this function may
    perform additional, helpful initialization of the analysis environment. We
    can’t stop you from calling the `Session` constructor directly, but there
    should be no reason to do so.
    """
    if interactive and not len(_session_cache):
        _maybe_install_custom_exception_formatter()

        # ERFA thinks that our years are "dubious". Feh!
        warnings.simplefilter("ignore", astropy.utils.exceptions.ErfaWarning)

    if source:
        root = "daschlab_" + source_name_to_fs_name(source)

    sess = _session_cache.get(root)
    if sess is None:
        sess = Session(root, interactive=interactive, _internal_simg=_internal_simg)
        _session_cache[root] = sess

    return sess


def _lookup_session(root: str) -> Session:
    return _session_cache[root]


# Typing stuff

from .refcat import SourceReferenceType
from .exposures import ExposureReferenceType
