# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
The main ``daschlab`` package.
"""

from contextlib import contextmanager
import json
import os
import pathlib
import sys
import tempfile
from typing import Dict, FrozenSet, Iterable, Optional

from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from pywwt.jupyter import connect_to_app, WWTJupyterWidget

from .query import SessionQuery
from .refcat import RefcatSources, RefcatSourceRow, _query_refcat
from .plates import Plates, _query_plates
from .lightcurves import Lightcurve, _query_lc

__all__ = [
    "SUPPORTED_REFCATS",
    "InteractiveError",
    "Session",
    "open_session",
]


SUPPORTED_REFCATS: FrozenSet[str] = frozenset(("apass", "atlas"))


# The default cutout is a square with half-size 600 arcsec. In order to get ~all
# catalog sources on the cutout, th radius should be the size of such a square's
# diagonal.
REFCAT_RADIUS: Angle = Angle(850 * u.arcsec)

PLATES_RADIUS: Angle = Angle(10 * u.arcsec)


class InteractiveError(Exception):
    pass


def _ipython_custom_exception_formatter(_shell, etype, evalue, _tb, tb_offset=None):
    print("error:", evalue, f"({etype.__name__})", file=sys.stderr)


def _maybe_install_custom_exception_formatter():
    try:
        ip = get_ipython()
    except NameError:
        pass

    ip.set_custom_exc((InteractiveError,), _ipython_custom_exception_formatter)


def _formatsc(sc: SkyCoord) -> str:
    r = sc.ra.to_string(sep=":", precision=1)
    d = sc.dec.to_string(sep=":", precision=0, alwayssign=True)
    return f"{r} {d}"


class Session:
    """A daschlab analysis session."""

    _root: pathlib.Path
    _interactive: bool = True
    _internal_simg: str = ""
    _query: Optional[SessionQuery] = None
    _refcat: Optional[RefcatSources] = None
    _plates: Optional[Plates] = None
    _wwt: Optional[WWTJupyterWidget] = None
    _lc_cache: Dict[str, Lightcurve] = None

    def __init__(self, root: str, interactive: bool = True, _internal_simg: str = ""):
        self._root = pathlib.Path(root)
        self._interactive = interactive
        self._internal_simg = _internal_simg
        self._lc_cache = {}

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
                f"- Query target not yet defined - run something like `{self._my_var_name()}.target_by_name('HD 209458')`"
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
            self._refcat = RefcatSources.read(str(self.path("refcat.ecsv")), format="ascii.ecsv")
            self._refcat._sess = self
        except FileNotFoundError:
            self._refcat = None

            if self._query is not None:
                self._info(
                   f"- Refcat not yet fetched - run something like `{self._my_var_name()}.refcat('apass')`"
                )
        else:
            if len(self._refcat):
                self._info(f"- Refcat: {len(self._refcat)} sources from `{self._refcat['refcat'][0]}`")
            else:
                self._info("- Refcat: present but empty")

        # Recover plates?

        try:
            self._plates = Plates.read(str(self.path("plates.ecsv")), format="ascii.ecsv")
        except FileNotFoundError:
            self._plates = None

            if self._query is not None:
                self._info(
                    f"- Plates not yet fetched - run something like `{self._my_var_name()}.plates()`"
                )
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
        """Generate a path within this session."""
        return self._root.joinpath(*pieces)

    def select_target(self, name: str = None) -> "Session":
        """
        Specify the center of the session's target area, by resolving a source
        name with Simbad/Sesame.
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
        q = SessionQuery.new_from_name(name)

        with self._save_atomic("query.json") as f_new:
            json.dump(q.to_dict(), f_new, ensure_ascii=False, indent=2)
            print(file=f_new)  # `json` doesn't do a trailing newline

        self._info(
            f"- Saved `query.json` with name `{name}` resolved to: {_formatsc(q.pos_as_skycoord())}"
        )
        return self

    def select_refcat(self, name: str) -> "Session":
        """
        Specify which DASCH reference catalog to use.
        """

        if name not in SUPPORTED_REFCATS:
            raise InteractiveError(f"invalid refcat name {name!r}; must be one of: {' '.join(SUPPORTED_REFCATS)}")

        if self._refcat is not None:
            if not len(self._refcat):
                self._warn(f"on-disk refcat is empty; assuming that it is for refcat `{name}`")
            elif self._refcat["refcat"][0] != name:
                raise InteractiveError(
                    f"on-disk refcat name `{self._refcat["refcat"][0]}` does not agree with in-code name `{name}`")

            return self

        # First-time invocation; query the catalog

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve refcat before setting target - run something like `{self._my_var_name()}.target_by_name('HD 209458')`"
            )

        print("- Querying API ...", flush=True)
        self._refcat = _query_refcat(name, self._query.pos_as_skycoord(), REFCAT_RADIUS)
        self._refcat._sess = self

        with self._save_atomic("refcat.ecsv") as f_new:
            self._refcat.write(f_new.name, format="ascii.ecsv", overwrite=True)

        self._info(
            f"- Saved `refcat.ecsv` for reference catalog \"{name}\" ({len(self._refcat)} sources)"
        )
        return self

    def refcat(self) -> RefcatSources:
        if self._refcat is None:
            raise InteractiveError(f"you must select the refcat first - run something like `{self._my_var_name()}.select_refcat('apass')`")
        return self._refcat

    def plates(self) -> Plates:
        """
        Ensure that we have a list of plates relevant to this session.
        """

        if self._plates is not None:
            return self._plates

        # First-time invocation; query the database

        if self._query is None:
            raise InteractiveError(
                f"cannot retrieve plates before setting target - run something like `{self._my_var_name()}.target_by_name('HD 209458')`"
            )

        print("- Querying API ...", flush=True)
        self._plates = _query_plates(self._query.pos_as_skycoord(), PLATES_RADIUS)

        with self._save_atomic("plates.ecsv") as f_new:
            self._plates.write(f_new.name, format="ascii.ecsv", overwrite=True)

        self._info(
            f"- Saved `plates.ecsv` ({len(self._plates)} relevant plates)"
        )
        return self._plates

    def lightcurve(self, src: RefcatSourceRow):
        name = src["ref_text"]
        lc = self._lc_cache.get(name)
        if lc is not None:
            return lc

        p = self.path("lightcurves") / f"{name}.ecsv"

        try:
            lc = Lightcurve.read(str(p), format="ascii.ecsv")
            self._lc_cache[name] = lc
            return lc
        except FileNotFoundError:
            pass

        # We need to fetch it

        self.path("lightcurves").mkdir(exist_ok=True)

        print("- Querying API ...", flush=True)
        lc = _query_lc(src["refcat"], name, src["gsc_bin_index"])

        with self._save_atomic(p) as f_new:
            lc.write(f_new.name, format="ascii.ecsv", overwrite=True)

        self._info(
            f"- Saved `{p}` ({len(lc)} relevant rows)"
        )
        return lc

    async def connect_to_wwt(self):
        if self._wwt is not None:
            return

        self._wwt = await connect_to_app().becomes_ready()
        self._wwt.foreground_opacity = 0
        
        if self._query is not None:
            self._wwt.center_on_coordinates(self._query.pos_as_skycoord(), fov=REFCAT_RADIUS * 1.2)

    def wwt(self) -> WWTJupyterWidget:
        if self._wwt is None:
            raise InteractiveError(
                f"you must connect to WWT asynchronously first - run `await {self._my_var_name()}.connect_to_wwt()`"
            )

        return self._wwt


def open_session(
    root: str = ".", interactive: bool = True, _internal_simg: str = ""
) -> Session:
    """
    Open or create a new daschlab analysis session.
    """
    _maybe_install_custom_exception_formatter()
    return Session(root, interactive=interactive, _internal_simg=_internal_simg)
