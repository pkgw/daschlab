# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lists of plates.
"""

from urllib.parse import urlencode

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
import numpy as np
import requests

__all__ = ["Plates"]


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


class Plates(Table):
    def series_info(self):
        g = self.group_by("series")

        t = Table()
        t["series"] = [t[0] for t in g.groups.keys]
        t["count"] = np.diff(g.groups.indices)

        return t


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

    return table
