# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lists of plates.
"""

from typing import Dict
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

__all__ = ["Plates", "PlateRow"]


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/queryplates.php"


def _daschtime_to_isot(t: str) -> str:
    outer_bits = t.split("T", 1)
    inner_bits = outer_bits[-1].split("-")
    outer_bits[-1] = ":".join(inner_bits)
    return "T".join(outer_bits)


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


class Plates(Table):
    _sess: "daschlab.Session" = None
    _layers: Dict[int, ImageLayer] = None
    Row = PlateRow

    def only_narrow(self) -> "Plates":
        mask = [SERIES[k].kind == SeriesKind.NARROW for k in self["series"]]
        return self[mask]

    def series_info(self) -> Table:
        g = self.group_by("series")

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
        plate_years = self["obs_date"].jyear

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

    def find(self, series: str, platenum: int, mosnum: int) -> "Plates":
        keep = (
            (self["series"] == series)
            & (self["platenum"] == platenum)
            & (self["mosnum"] == mosnum)
        )
        return self[keep]

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

    return table
