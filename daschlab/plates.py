# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lists of plates.
"""

import re
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlencode

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table, Row
from astropy.time import Time
from astropy import units as u
from bokeh.plotting import figure, show
import numpy as np
from pywwt.layers import ImageLayer
import requests

from .series import SERIES, SeriesKind

__all__ = ["Plates", "PlateRow", "PlateSelector"]


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/queryplates.php"


def _daschtime_to_isot(t: str) -> str:
    outer_bits = t.split("T", 1)
    inner_bits = outer_bits[-1].split("-")
    outer_bits[-1] = ":".join(inner_bits)
    return "T".join(outer_bits)


_PLATE_NAME_REGEX = re.compile(r"^([a-zA-Z]+)([0-9]{1,5})$")


def _parse_plate_name(name: str) -> Tuple[str, int]:
    try:
        m = _PLATE_NAME_REGEX.match(name)
        series = m[1].lower()
        platenum = int(m[2])
    except Exception:
        raise Exception(f"invalid plate name `{name}`")

    return series, platenum


_COLTYPES = {
    "series": str,
    "platenum": int,
    "scannum": int,
    "mosnum": int,
    "expnum": int,
    "solnum": int,
    "class": str,
    "ra": float,
    "dec": float,
    "exptime": float,
    "jd": float,
    # "epoch": float,
    "wcssource": str,
    "scandate": _daschtime_to_isot,
    "mosdate": _daschtime_to_isot,
    "rotation": int,
    "binflags": int,
    "centerdist": float,
    "edgedist": float,
}


class PlateRow(Row):
    def show(self) -> ImageLayer:
        return self._table.show(self)

    def plate_id(self) -> str:
        return f"{self['series']}{self['platenum']:05d}_{self['mosnum']:02d}"


class PlateSelector:
    """
    A magic object to help enable plate-list filtering.
    """

    _plates: "Plates"
    _apply = None

    def __init__(self, plates: "Plates", apply):
        self._plates = plates
        self._apply = apply

    def where(self, row_mask, **kwargs) -> "Plates":
        return self._apply(self, row_mask, **kwargs)

    def rejected(self, **kwargs) -> "Plates":
        m = self._plates["reject"] != 0
        return self._apply(self, m, **kwargs)

    def rejected_with(self, tag: str, strict: bool = False, **kwargs) -> "Plates":
        bitnum0 = None

        if self._plates._rejection_tags is not None:
            bitnum0 = self._plates._rejection_tags.get(tag)

        if bitnum0 is not None:
            m = (self._plates["reject"] & (1 << bitnum0)) != 0
        elif strict:
            raise Exception(f"unknown rejection tag `{tag}`")
        else:
            m = np.zeros(len(self._plates), dtype=bool)

        return self._apply(self, m, **kwargs)

    def local_id(self, local_id: int, **kwargs) -> "Plates":
        m = self._plates["local_id"] == local_id
        return self._apply(self, m, **kwargs)

    def series(self, series: str, **kwargs) -> "Plates":
        m = self._plates["series"] == series
        return self._apply(self, m, **kwargs)

    def narrow(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.NARROW for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def patrol(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.PATROL for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def meteor(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.METEOR for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def plate_names(self, names: Iterable[str], **kwargs) -> "Plates":
        # This feels so inefficient, but it's not obvious to me how to do any better
        m = np.zeros(len(self._plates), dtype=bool)

        for name in names:
            series, platenum = _parse_plate_name(name)
            this_one = (self._plates["series"] == series) & (
                self._plates["platenum"] == platenum
            )
            m |= this_one

        return self._apply(self, m, **kwargs)


class Plates(Table):
    _sess: "daschlab.Session" = None
    _rejection_tags: Optional[Dict[str, int]] = None
    _layers: Dict[int, ImageLayer] = None
    Row = PlateRow

    # Filtering infrastructure

    def _copy_subset(self, keep, verbose: bool) -> "Plates":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        new._rejection_tags = self._rejection_tags
        return new

    @property
    def match(self) -> PlateSelector:
        return PlateSelector(self, lambda _sel, m: m)

    @property
    def count(self) -> PlateSelector:
        return PlateSelector(self, lambda _sel, m: m.sum())

    def _apply_keep_only(self, _selector, flags, verbose=True) -> "Plates":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> PlateSelector:
        return PlateSelector(self, self._apply_keep_only)

    def _apply_drop(self, _selector, flags, verbose=True) -> "Plates":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> PlateSelector:
        return PlateSelector(self, self._apply_drop)

    def _make_reject_selector(
        self, tag: str, verbose: bool, apply_func
    ) -> PlateSelector:
        if self._rejection_tags is None:
            self._rejection_tags = {}

        bitnum0 = self._rejection_tags.get(tag)

        if bitnum0 is None:
            bitnum0 = len(self._rejection_tags)
            if bitnum0 > 63:
                raise Exception("you cannot have more than 64 distinct rejection tags")

            if verbose:
                print(f"Assigned rejection tag `{tag}` to bit number {bitnum0 + 1}")

            self._rejection_tags[tag] = bitnum0

        selector = PlateSelector(self, apply_func)
        selector._bitnum0 = bitnum0
        return selector

    def _apply_reject(self, selector, flags, verbose: bool = True):
        n_before = (self["reject"] != 0).sum()
        self["reject"][flags] |= 1 << selector._bitnum0
        n_after = (self["reject"] != 0).sum()

        if verbose:
            print(
                f"Marked {n_after - n_before} new rows as rejected; {n_after} total are rejected"
            )

    def reject(self, tag: str, verbose: bool = True) -> PlateSelector:
        return self._make_reject_selector(tag, verbose, self._apply_reject)

    def _apply_reject_unless(self, selector, flags, verbose: bool = True):
        return self._apply_reject(selector, ~flags, verbose)

    def reject_unless(self, tag: str, verbose: bool = True) -> PlateSelector:
        return self._make_reject_selector(tag, verbose, self._apply_reject_unless)

    # Non-filtering actions on this list

    def series_info(self) -> Table:
        g = self.drop.rejected(verbose=False).group_by("series")

        t = Table()
        t["series"] = [t[0] for t in g.groups.keys]
        t["count"] = np.diff(g.groups.indices)
        t["kind"] = [SERIES[t[0]].kind for t in g.groups.keys]
        t["plate_scale"] = (
            np.array([SERIES[t[0]].plate_scale for t in g.groups.keys])
            * u.arcsec
            / u.mm
        )
        t["aperture"] = np.array([SERIES[t[0]].aperture for t in g.groups.keys]) * u.m
        t["description"] = [SERIES[t[0]].description for t in g.groups.keys]

        t.sort(["count"], reverse=True)

        return t

    def time_coverage(self) -> figure:
        from scipy.stats import gaussian_kde

        plot_years = np.linspace(1875, 1995, 200)
        plate_years = self.drop.rejected(verbose=False)["obs_date"].jyear

        kde = gaussian_kde(plate_years, bw_method="scott")
        plate_years_smoothed = kde(plot_years)

        # try to get the normalization right: integral of the
        # curve is equal to the number of plates, so that the
        # Y axis is plates per year.

        integral = plate_years_smoothed.sum() * (plot_years[1] - plot_years[0])
        plate_years_smoothed *= len(self) / integral

        p = figure(
            x_axis_label="Year",
            y_axis_label="Plates per year (smoothed)",
        )
        p.line(plot_years, plate_years_smoothed)
        show(p)
        return p

    def show(self, plate_ref: "PlateReferenceType") -> ImageLayer:
        plate = self._sess._resolve_plate_reference(plate_ref)

        if self._layers is None:
            self._layers = {}

        local_id = plate["local_id"]
        il = self._layers.get(local_id)

        if il is not None:
            return il  # TODO: bring it to the top, etc?

        fits_relpath = self._sess.cutout(plate)
        if fits_relpath is None:
            from . import InteractiveError

            raise InteractiveError(
                f"cannot create a cutout for plate #{local_id:05d} ({plate.plate_id()})"
            )

        il = self._sess.wwt().layers.add_image_layer(str(self._sess.path(fits_relpath)))
        self._layers[local_id] = il
        return il


def _query_plates(
    center: SkyCoord,
    radius: u.Quantity,
) -> Plates:
    radius = Angle(radius)

    # API-based query

    url = (
        _API_URL
        + "?"
        + urlencode(
            {
                "cra_deg": center.ra.deg,
                "cdec_deg": center.dec.deg,
                "radius_arcsec": radius.arcsec,
            }
        )
    )

    colnames = None
    coltypes = None
    coldata = None
    saw_sep = False

    with requests.get(url, stream=True) as resp:
        for line in resp.iter_lines():
            line = line.decode("utf-8")
            pieces = line.rstrip().split("\t")

            if colnames is None:
                colnames = pieces
                coltypes = [_COLTYPES.get(c) for c in colnames]
                coldata = [[] if t is not None else None for t in coltypes]
            elif not saw_sep:
                saw_sep = True
            else:
                for row, ctype, cdata in zip(pieces, coltypes, coldata):
                    if ctype is not None:
                        cdata.append(ctype(row))

    # Postprocess

    input_cols = dict(t for t in zip(colnames, coldata) if t[1] is not None)

    table = Plates()
    table["series"] = input_cols["series"]
    table["platenum"] = np.array(input_cols["platenum"], dtype=np.uint32)
    table["scannum"] = np.array(input_cols["scannum"], dtype=np.uint8)
    table["mosnum"] = np.array(input_cols["mosnum"], dtype=np.uint8)
    table["expnum"] = np.array(input_cols["expnum"], dtype=np.uint8)
    table["solnum"] = np.array(input_cols["solnum"], dtype=np.uint8)
    table["class"] = input_cols["class"]
    table["pos"] = SkyCoord(
        ra=input_cols["ra"] * u.deg,
        dec=input_cols["dec"] * u.deg,
        frame="icrs",
    )
    table["exptime"] = np.array(input_cols["exptime"], dtype=np.float32) * u.minute
    table["obs_date"] = Time(input_cols["jd"], format="jd")
    table["wcssource"] = input_cols["wcssource"]
    table["scan_date"] = Time(input_cols["scandate"], format="isot")
    table["mos_date"] = Time(input_cols["mosdate"], format="isot")
    table["rotation_deg"] = np.array(input_cols["rotation"], dtype=np.uint16)
    table["binflags"] = np.array(input_cols["binflags"], dtype=np.uint8)
    table["center_distance"] = (
        np.array(input_cols["centerdist"], dtype=np.float32) * u.cm
    )
    table["edge_distance"] = np.array(input_cols["edgedist"], dtype=np.float32) * u.cm

    table.sort(["obs_date"])

    table["local_id"] = np.arange(len(table))
    table["reject"] = np.zeros(len(table), dtype=np.uint64)

    return table
