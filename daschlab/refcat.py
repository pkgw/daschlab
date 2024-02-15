# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Data extracted from one of the DASCH reference catalogs in the query region.
"""

from typing import Optional
from urllib.parse import urlencode

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Row, Table
from astropy.time import Time
from astropy import units as u
import numpy as np
import requests
from pywwt.layers import TableLayer

__all__ = ["RefcatSources"]


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/querycat.php"

_COLTYPES = {
    "ref_text": str,
    "ref_number": int,
    "gscBinIndex": int,
    "raDeg": float,
    "decDeg": float,
    "draAsec": float,
    "ddecAsec": float,
    "posEpoch": float,
    "pmRaMasyr": float,
    "pmDecMasyr": float,
    "uPMRaMasyr": float,
    "uPMDecMasyr": float,
    "stdmag": float,
    "color": float,
    "vFlag": int,
    "magFlag": int,
    "class": int,
}


class RefcatSourceRow(Row):
    def lightcurve(self) -> "daschlab.Lightcurve":
        return self._table._sess.lightcurve(self)


class RefcatSources(Table):
    _sess: "daschlab.Session" = None
    _layer: Optional[TableLayer] = None
    Row = RefcatSourceRow

    def show(self) -> TableLayer:
        if self._layer is not None:
            return self._layer

        wwt = self._sess.wwt()
        tl = wwt.layers.add_table_layer(self)
        self._layer = tl

        tl.size_scale = 20
        tl.marker_type = "circle"
        return tl


def _query_refcat(
    name: str,
    center: SkyCoord,
    radius: u.Quantity,
) -> RefcatSources:
    radius = Angle(radius)

    # API-based query

    url = (
        _API_URL
        + "?"
        + urlencode(
            {
                "name": name,
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

    table = RefcatSources()
    table["ref_text"] = input_cols["ref_text"]
    table["ref_number"] = np.array(input_cols["ref_number"], dtype=np.uint64)
    table["gsc_bin_index"] = np.array(input_cols["gscBinIndex"], dtype=np.uint32)
    table["pos"] = SkyCoord(
        ra=input_cols["raDeg"] * u.deg,
        dec=input_cols["decDeg"] * u.deg,
        pm_ra_cosdec=input_cols["pmRaMasyr"] * u.mas / u.yr,
        pm_dec=input_cols["pmDecMasyr"] * u.mas / u.yr,
        obstime=Time(input_cols["posEpoch"], format="jyear"),
        frame="icrs",
    )
    table["dra"] = np.array(input_cols["draAsec"], dtype=np.float32) * u.arcsec
    table["ddec"] = np.array(input_cols["ddecAsec"], dtype=np.float32) * u.arcsec
    table["u_pm_ra_cosdec"] = (
        np.array(input_cols["uPMRaMasyr"], dtype=np.float32) * u.mas / u.yr
    )
    table["u_pm_dec"] = (
        np.array(input_cols["uPMDecMasyr"], dtype=np.float32) * u.mas / u.yr
    )
    table["stdmag"] = np.array(input_cols["stdmag"], dtype=np.float32) * u.mag
    table["color"] = np.array(input_cols["color"], dtype=np.float32) * u.mag
    table["vFlag"] = np.array(input_cols["vFlag"], dtype=np.uint16)
    table["magFlag"] = np.array(input_cols["magFlag"], dtype=np.uint16)
    table["class"] = np.array(input_cols["class"], dtype=np.uint16)
    table["refcat"] = [name] * len(input_cols["ref_text"])

    # Sort by distance from the query point. I believe that we need to use a
    # temporary column for this.
    table["_distsq"] = table["dra"] ** 2 + table["ddec"] ** 2
    table.sort(["_distsq"])
    del table["_distsq"]

    table["local_id"] = np.arange(len(table))

    return table
