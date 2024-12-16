# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Data extracted from one of the DASCH reference catalogs in the query region.

The main class provided by this module is `RefcatSources`, instances of which
can be obtained with the `daschlab.Session.refcat()` method.
"""

from copy import copy
from typing import Literal, Optional, Union

from astropy.coordinates import Angle, SkyCoord
from astropy.table import Row, Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
import numpy as np
from pywwt.layers import TableLayer

from .apiclient import ApiClient


__all__ = ["RefcatSources", "RefcatSourceRow", "SourceReferenceType"]


def maybe_int(s: str, default: int = 0) -> int:
    if s:
        return int(s)
    return default


_COLTYPES = {
    "ref_text": str,
    "ref_number": int,
    "gsc_bin_index": int,
    "ra_deg": float,
    "dec_deg": float,
    "dra_asec": float,
    "ddec_asec": float,
    "pos_epoch": float,
    "pm_ra_masyr": float,
    "pm_dec_masyr": float,
    "u_pm_ra_masyr": float,
    "u_pm_dec_masyr": float,
    "stdmag": float,
    "color": float,
    "class": int,
    "v_flag": int,
    "mag_flag": int,
    "num_matches": maybe_int,
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
        return self._table._sess().lightcurve(self)


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

    Row = RefcatSourceRow

    def _sess(self) -> "daschlab.Session":
        from . import _lookup_session

        return _lookup_session(self.meta["daschlab_sess_key"])

    def show(
        self, mag_limit: Optional[float] = None, size_vmin_bias: float = 1.0
    ) -> TableLayer:
        """
        Display the catalog contents in the WWT view.

        Parameters
        ==========
        mag_limit : optional `float` or `None`
            For display purposes, source magnitudes fainter (larger) than this
            value, or missing magnitudes, will be filled in with this value.
            If unspecified (`None`), the maximum unmasked value will be used.
        size_vmin_bias : optional `float`, default 1.0
            The WWT layer's ``size_vmin`` setting is set to the ``mag_limit``
            plus this number. Larger values cause relatively faint sources
            to be rendered with relatively larger indicators. This makes them
            easier to see, at a cost of somewhat reducing the dynamic range of
            the indicator sizing.

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
        sess = self._sess()
        if sess._refcat_table_layer is not None:
            return sess._refcat_table_layer

        # TODO: pywwt can't handle Astropy tables that use a SkyCoord
        # to hold positional information. That should be fixed, but it
        # will take some time. In the meantime, hack around it.

        compat_table = copy(self)
        compat_table["ra"] = self["pos"].ra.deg
        compat_table["dec"] = self["pos"].dec.deg
        del compat_table["pos"]

        # Fill in unlisted magnitudes with the limit value.

        if mag_limit is None:
            mag_limit = compat_table["stdmag"].max()

        compat_table["viz_mag"] = np.minimum(
            compat_table["stdmag"].filled(mag_limit), mag_limit
        )

        wwt = sess.wwt()
        tl = wwt.layers.add_table_layer(compat_table)
        sess._refcat_table_layer = tl

        tl.marker_type = "circle"
        tl.size_att = "viz_mag"
        tl.size_vmin = mag_limit + size_vmin_bias
        tl.size_vmax = compat_table["viz_mag"].min()
        tl.size_scale = 10.0
        return tl


def _query_refcat(
    client: ApiClient,
    name: str,
    center: SkyCoord,
    radius: u.Quantity,
) -> RefcatSources:
    radius = Angle(radius)
    payload = {
        "refcat": name,
        "ra_deg": center.ra.deg,
        "dec_deg": center.dec.deg,
        "radius_arcsec": radius.arcsec,
    }

    colnames = None
    coltypes = None
    coldata = None

    data = client.invoke("/dasch/dr7/querycat", payload)
    if not isinstance(data, list):
        from . import InteractiveError

        raise InteractiveError(f"querycat API request failed: {data!r}")

    for line in data:
        pieces = line.split(",")

        if colnames is None:
            colnames = pieces
            coltypes = [_COLTYPES.get(c) for c in colnames]
            coldata = [[] if t is not None else None for t in coltypes]
        else:
            for row, ctype, cdata in zip(pieces, coltypes, coldata):
                if ctype is not None:
                    cdata.append(ctype(row))

    # Postprocess

    input_cols = dict(t for t in zip(colnames, coldata) if t[1] is not None)

    table = RefcatSources(masked=True)
    table["ref_text"] = input_cols["ref_text"]
    table["ref_number"] = np.array(input_cols["ref_number"], dtype=np.uint64)
    table["gsc_bin_index"] = np.array(input_cols["gsc_bin_index"], dtype=np.uint32)
    table["pos"] = SkyCoord(
        ra=input_cols["ra_deg"] * u.deg,
        dec=input_cols["dec_deg"] * u.deg,
        pm_ra_cosdec=input_cols["pm_ra_masyr"] * u.mas / u.yr,
        pm_dec=input_cols["pm_dec_masyr"] * u.mas / u.yr,
        obstime=Time(input_cols["pos_epoch"], format="jyear"),
        frame="icrs",
    )
    table["dra"] = np.array(input_cols["dra_asec"], dtype=np.float32) * u.arcsec
    table["ddec"] = np.array(input_cols["ddec_asec"], dtype=np.float32) * u.arcsec
    table["u_pm_ra_cosdec"] = (
        np.array(input_cols["u_pm_ra_masyr"], dtype=np.float32) * u.mas / u.yr
    )
    table["u_pm_dec"] = (
        np.array(input_cols["u_pm_dec_masyr"], dtype=np.float32) * u.mas / u.yr
    )

    stdmag = np.array(input_cols["stdmag"], dtype=np.float32)
    table["stdmag"] = Masked(u.Quantity(stdmag, u.mag), stdmag >= 99)

    color = np.array(input_cols["color"], dtype=np.float32)
    table["color"] = Masked(u.Quantity(color, u.mag), color >= 99)

    table["v_flag"] = np.array(input_cols["v_flag"], dtype=np.uint16)
    table["mag_flag"] = np.array(input_cols["mag_flag"], dtype=np.uint16)
    table["class"] = np.array(input_cols["class"], dtype=np.uint16)
    table["num_matches"] = np.array(input_cols["num_matches"], dtype=np.uint32)
    table["refcat"] = [name] * len(input_cols["ref_text"])

    # Sort by distance from the query point. I believe that we need to use a
    # temporary column for this.
    table["_distsq"] = table["dra"] ** 2 + table["ddec"] ** 2
    table.sort(["_distsq"])
    del table["_distsq"]

    table["local_id"] = np.arange(len(table))

    return table
