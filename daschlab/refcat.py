# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Data extracted from one of the DASCH reference catalogs in the query region.

The main class provided by this module is `RefcatSources`, instances of which
can be obtained with the `daschlab.Session.refcat()` method.
"""

from copy import copy
from typing import Literal, Optional, Union
from urllib.parse import urlencode

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Row, Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
import numpy as np
import requests
from pywwt.layers import TableLayer

__all__ = ["RefcatSources", "RefcatSourceRow", "SourceReferenceType"]


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
    """
    A single row from a `RefcatSources` table.

    You do not need to construct these objects manually. Indexing a `RefcatSources`
    table with a single integer will yield an instance of this class, which is a
    subclass of `astropy.table.Row`.
    """

    def lightcurve(self) -> "daschlab.lightcurves.Lightcurve":
        """
        Obtain a table of lightcurve data for this specified source.

        Returns
        =======
        A `daschlab.lightcurves.Lightcurve` instance.

        Notes
        =====
        For details, see `daschlab.Session.lightcurve()`, which implements this
        functionality.
        """
        return self._table._sess.lightcurve(self)


SourceReferenceType = Union[RefcatSourceRow, int, Literal["click"]]


class RefcatSources(Table):
    """
    A table of sources from a DASCH reference catalog.

    A `RefcatSources` is a subclass of `astropy.table.Table` containing DASCH
    catalog data and associated catalog-specific methods. You can use all of the
    usual methods and properties made available by the `astropy.table.Table`
    class. Items provided by the `~astropy.table.Table` class are not documented
    here.

    You should not construct `RefcatSources` instances directly. Instead, obtain
    the full table using the `daschlab.Session.refcat()` method.

    **Columns are not documented here!** They are (**FIXME: will be**)
    documented more thoroughly in the DASCH data description pages.
    """

    _sess: "daschlab.Session" = None
    _layer: Optional[TableLayer] = None
    Row = RefcatSourceRow

    def show(self) -> TableLayer:
        """
        Display the catalog contents in the WWT view.

        Returns
        =======
        `pywwt.layers.TableLayer`
            This is the WWT table layer object corresponding to the displayed
            catalog. You can use it to programmatically control aspects of how
            the data are displayed, such as the which column sets the point
            size.

        Notes
        =====
        In order to use this method, you must first have called
        `daschlab.Session.connect_to_wwt()`.
        """
        if self._layer is not None:
            return self._layer

        # TODO: pywwt can't handle Astropy tables that use a SkyCoord
        # to hold positional information. That should be fixed, but it
        # will take some time. In the meantime, hack around it.

        compat_table = copy(self)
        compat_table["ra"] = self["pos"].ra.deg
        compat_table["dec"] = self["pos"].dec.deg
        del compat_table["pos"]

        wwt = self._sess.wwt()
        tl = wwt.layers.add_table_layer(compat_table)
        self._layer = tl

        tl.marker_type = "circle"
        tl.size_att = "stdmag"
        tl.size_vmin = compat_table["stdmag"].max()
        tl.size_vmax = compat_table["stdmag"].min()
        tl.size_scale = 10.0
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

    stdmag = np.array(input_cols["stdmag"], dtype=np.float32)
    table["stdmag"] = Masked(u.Quantity(stdmag, u.mag), stdmag >= 99)

    color = np.array(input_cols["color"], dtype=np.float32)
    table["color"] = Masked(u.Quantity(color, u.mag), color >= 99)

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
