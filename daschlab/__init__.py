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
and howtos, see `the DASCH DRnext documentation`_.

.. _the DASCH DRnext documentation: https://dasch.cfa.harvard.edu/drnext/

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
import sys
import tempfile
import time
from typing import Dict, FrozenSet, Iterable, Optional
import warnings

from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
import astropy.utils.exceptions
import numpy as np
from pywwt.jupyter import connect_to_app, WWTJupyterWidget

from .query import SessionQuery
from .refcat import RefcatSources, RefcatSourceRow, _query_refcat
from .plates import Plates, PlateRow, _query_plates
from .lightcurves import Lightcurve, _query_lc
from .cutouts import _query_cutout


__all__ = [
    "SUPPORTED_REFCATS",
    "InteractiveError",
    "Session",
    "open_session",
    "source_name_to_fs_name",
]


SUPPORTED_REFCATS: FrozenSet[str] = frozenset(("apass", "atlas"))


CUTOUT_HALFSIZE: Angle = Angle(600 * u.arcsec)

# In order to get ~all catalog sources on a cutout, the radius should be the
# size of such a square's diagonal.
REFCAT_RADIUS: Angle = Angle(850 * u.arcsec)

PLATES_RADIUS: Angle = Angle(10 * u.arcsec)


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
    - `~Session.plates()` to access a table of DASCH plates overlapping
      your target
    - `~Session.lightcurve()` to access lightcurves for the catalog sources
    - `~Session.cutout()` to download plate cutout images centered on
      your target
    """

    _root: pathlib.Path
    _interactive: bool = True
    _internal_simg: str = ""
    _query: Optional[SessionQuery] = None
    _refcat: Optional[RefcatSources] = None
    _plates: Optional[Plates] = None
    _wwt: Optional[WWTJupyterWidget] = None
    _refcat_table_layer: Optional["pywwt.layers.TableLayer"] = None
    _lc_cache: Dict[str, Lightcurve] = None
    _plate_image_layer_cache: dict = None

    def __init__(self, root: str, interactive: bool = True, _internal_simg: str = ""):
        self._root = pathlib.Path(root)
        self._interactive = interactive
        self._internal_simg = _internal_simg
        self._lc_cache = {}
        self._plate_image_layer_cache = {}

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
                    f"- Refcat: {len(self._refcat)} sources from `{self._refcat['refcat'][0]}`"
                )
            else:
                self._info("- Refcat: present but empty")

        # Recover plates?

        try:
            self._plates = Plates.read(
                str(self.path("plates.ecsv")), format="ascii.ecsv"
            )
            self._plates.meta["daschlab_sess_key"] = str(self._root)
        except FileNotFoundError:
            self._plates = None
        else:
            self._info(f"- Plates: {len(self._plates)} relevant plates")

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

    def select_target(self, name: str = None) -> "Session":
        """
        Specify the center of the session's target area.

        Parameters
        ==========
        name : `str`
          The Simbad-resolvable name of this session's target area.

        Returns
        =======
        *self*, for chaining convenience

        Notes
        =====
        The first time you call this method for a given session, it will perform
        a Simbad/Sesame API query to resolve the target name and save the
        resulting information in a file named ``query.json``. Subsequent calls
        (i.e., ones made with the ``query.json`` file already existing) will
        merely check for consistency.

        After calling this method, you may use `~Session.query()` to obtain the
        cached information about your session’s target. This merely provides the
        RA and dec.

        This method could easily support queries based on RA/Dec rather than
        name resolution. If that’s functionality you would like, please consider
        filing a pull request.
        """
        if not name:
            raise ValueError("`name` must be specified")

        if self._query is not None:
            if not self._query.name:
                self._warn(
                    f"on-disk query target name `{self._query.name}` does not agree with in-code name `{name}`"
                )
            elif self._query.name != name:
                raise InteractiveError(
                    f"on-disk query target name `{self._query.name}` does not agree with in-code name `{name}`"
                )

            return self

        # First-time invocation; set up the query file
        print("- Querying API ...", flush=True)
        self._query = SessionQuery.new_from_name(name)

        with self._save_atomic("query.json") as f_new:
            json.dump(self._query.to_dict(), f_new, ensure_ascii=False, indent=2)
            print(file=f_new)  # `json` doesn't do a trailing newline

        self._info(
            f"- Saved `query.json` with name `{name}` resolved to: {_formatsc(self._query.pos_as_skycoord())}"
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
            elif self._refcat["refcat"][0] != name:
                raise InteractiveError(
                    f"on-disk refcat name `{self._refcat['refcat'][0]}` does not agree with in-code name `{name}`"
                )

            return self

        # First-time invocation; query the catalog

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve refcat before setting target - run something like `{self._my_var_name()}.select_target(name='HD 209458')`"
            )

        t0 = time.time()
        print("- Querying API ...", flush=True)
        self._refcat = _query_refcat(name, self._query.pos_as_skycoord(), REFCAT_RADIUS)
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

    def plates(self) -> Plates:
        """
        Obtain the table of plates overlapping the session target area.

        Returns
        =======
        A `daschlab.plates.Plates` instance.

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.select_refcat()`
        before calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch information from the plate database, saving
        the resulting information in a file named ``plates.ecsv``. Subsequent
        calls (i.e., ones made with the ``plates.ecsv`` file already existing)
        will merely check for consistency and load the saved file.
        """

        if self._plates is not None:
            return self._plates

        # First-time invocation; query the database

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve plates before setting target - run something like `{self._my_var_name()}.select_target(name='HD 209458')`"
            )

        t0 = time.time()
        print("- Querying API ...", flush=True)
        self._plates = _query_plates(self._query.pos_as_skycoord(), PLATES_RADIUS)
        self._plates.meta["daschlab_sess_key"] = str(self._root)

        with self._save_atomic("plates.ecsv") as f_new:
            self._plates.write(f_new.name, format="ascii.ecsv", overwrite=True)

        elapsed = time.time() - t0
        self._info(
            f"- Retrieved {len(self._plates)} relevant plates in {elapsed:.0f} seconds and saved as `plates.ecsv`"
        )
        return self._plates

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

        if isinstance(src_ref, int):
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
        plates = self.plates()

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
        lc = _query_lc(src["refcat"], name, src["gsc_bin_index"])

        # Cross-match with the plates

        plate_lookup = {}

        for p in plates:
            if p["mosnum"] is not np.ma.masked:
                plate_lookup[(p["series"], p["platenum"], p["mosnum"])] = p["local_id"]

        lc["plate_local_id"] = -1

        for p in lc:
            p["plate_local_id"] = plate_lookup.get(
                (p["series"], p["platenum"], p["mosnum"]), -1
            )

        # All done

        with self._save_atomic(relpath) as f_new:
            lc.write(f_new.name, format="ascii.ecsv", overwrite=True)

        elapsed = time.time() - t0
        self._info(
            f"- Fetched {len(lc)} rows in {elapsed:.0f} seconds and saved as `{self.path(relpath)}`"
        )
        return lc

    def _resolve_plate_reference(self, plate_ref: "PlateReferenceType") -> PlateRow:
        if isinstance(plate_ref, int):
            return self.plates()[plate_ref]

        assert isinstance(plate_ref, PlateRow)
        return plate_ref

    def cutout(
        self, plate_ref: "PlateReferenceType", no_download: bool = False
    ) -> Optional[str]:
        """
        Obtain a FITS cutout for the specified plate.

        Parameters
        ==========
        plate_ref : `int` or `~daschlab.plates.PlateRow`
            If this argument is an integer, it is interpreted as the "local ID"
            of a row in the plates table.

            If this argument is an instance of `~daschlab.plates.PlateRow`, the
            cutout for the specified plate is obtained.

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
            could happen if the plate has not been scanned, among other reasons.

        Examples
        ========
        Load the cutout of the chronologically-first calibrated observation of
        the target field (the plate list is in chronological order)::

            from astropy.io import fits

            solved_plates = sess.plates().keep_only.wcs_solved()
            plate = solved_plates[0]

            relpath = sess.cutout(plate)
            assert relpath, f"could not get cutout of {plate.plate_id()}"

            # str() needed here because Astropy does not accept Path objects:
            hdu_list = fits.open(str(sess.path(relpath))

        Notes
        =====
        You must call `~Session.select_target()` and `~Session.plates()` before
        calling this method.

        The first time you call this method for a given session, it will perform
        a DASCH API query to fetch the cutout, saving the resulting data in a
        file inside the session's ``cutouts`` subdirectory. Subsequent calls
        (i.e., ones made with the data file already existing) will merely check
        for consistency and load the saved file.

        See Also
        ========
        daschlab.plates.Plates.show : to show a cutout in the WWT view
        """
        from astropy.io import fits

        plate = self._resolve_plate_reference(plate_ref)

        local_id = plate["local_id"]
        plate_id = f"{plate['series']}{plate['platenum']:05d}_{plate['mosnum']:02d}"
        dest_relpath = f"cutouts/{local_id:05d}_{plate_id}.fits"

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
                plate["series"], plate["platenum"], plate["mosnum"], center
            )
        except Exception as e:
            # Right now, this could happen either because the API failed in a
            # transient way, or because the plate was not scanned and
            # WCS-solved. It would be good to be able to distinguish these
            # cases.
            self._warn(f"failed to fetch cutout for {plate_id}: {e}")
            return None

        print(f"- Fetched {len(fits_data)} bytes in {time.time()-t0:.0f} seconds")

        # Add a bunch of headers using the metadata that we have

        fits_data = io.BytesIO(fits_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with fits.open(fits_data) as hdul:
                h = hdul[0].header

                h["D_SCNNUM"] = plate["scannum"]
                h["D_EXPNUM"] = plate["expnum"]
                h["D_SOLNUM"] = plate["solnum"]
                # the 'class' column is handled as a masked array and when we convert it
                # to a row, the masked value seems to become a float64?
                h["D_PCLASS"] = "" if plate["class"] is np.ma.masked else plate["class"]
                h["D_ROTFLG"] = plate["rotation_deg"]
                h["EXPTIME"] = plate["exptime"] * 60
                h["DATE-OBS"] = plate["obs_date"].fits
                h["MJD-OBS"] = plate["obs_date"].mjd
                h["DATE-SCN"] = plate["scan_date"].unmasked.fits
                h["MJD-SCN"] = plate["scan_date"].unmasked.mjd
                h["DATE-MOS"] = plate["mos_date"].unmasked.fits
                h["MJD-MOS"] = plate["mos_date"].unmasked.mjd

                with self._save_atomic(dest_relpath) as f_new:
                    hdul.writeto(f_new.name, overwrite=True)

        self._info(f"- Saved `{self.path(dest_relpath)}`")
        return dest_relpath

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
from .plates import PlateReferenceType
