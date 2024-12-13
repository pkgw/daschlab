# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
DASCH lightcurve data.

The main class provided by this module is `Lightcurve`, instances of which can
be obtained with the `daschlab.Session.lightcurve()` method.

.. _lc-rejection:

Rejecting Data Points
=====================

`Lightcurve` tables come with a uint64 column named ``"reject"`` that is always
initialized with zeros. If you are analyzing a lightcurve and wish to discard
various bad data points from it, you should set their ``"reject"`` values to
something non-zero. Most lightcurve analysis functions will automatically drop
points with non-zero ``"reject"`` values.


.. _lc-filtering:

Filtering and Subsetting
========================

`Lightcurve` objects are instances of the `astropy.timeseries.TimeSeries` class,
which in turn are instances of the `astropy.table.Table` class. In your data
analysis you can use all of the usual methods and functions that these
superclasses provide. In addition, the `Lightcurve` class provides a convenient
set of tools for subsetting and filtering data.

These tools take the form of paired “actions” and “selections”. The syntax for
using them is as follows::

    lc = sess.lightcurve(some_local_id)

    # Get a subset of the data containing only the detections
    # brighter than 13th mag. The action is `keep_only` and
    # the selection is `brighter`.
    bright = lc.keep_only.brighter(13)

    # From that subset, remove points from the "ai" series.
    # The action is `drop` and the selection is `series`.
    no_ai = bright.drop.series("ai")

    # Count remaining points not from narrow-field telescopes.
    # The action is `count` and the selection is `narrow`, and
    # `not_` is a modifier with the intuitive behavior.
    n = no_ai.count.not_.narrow()

These operations can be conveniently chained together. More complex boolean
logic can be implemented with the
`~daschlab.photometry.PhotometrySelector.where` selection and the
`~daschlab.photometry.Photometry.match` action.

In terms of implementation, an item like ``lc.keep_only`` is a property that
returns a helper `~daschlab.photometry.PhotometrySelector` object, which is what
actually implements methods such as
`~daschlab.photometry.PhotometrySelector.brighter()`. The selector object helps
pair the desired action with the desired selection while maintaining a
convenient code syntax.

DASCH lightcurves often contain many nondetections (upper limits), which means
that you may need to be careful about selection logic. For instance::

    lc = sess.lightcurve(some_local_id)
    n1 = lc.count.brighter(13)
    n2 = lc.count.not_.detected_and_fainter(13)

    # This will NOT generally hold, because n2 will include
    # nondetections while n1 will not.
    assert n1 == n2
"""

from typing import List, Optional

from astropy.coordinates import SkyCoord
from astropy.table import Column, MaskedColumn
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy import units as u
from astropy.utils.masked import Masked
from bokeh.plotting import figure, show
import numpy as np

from .apiclient import ApiClient
from .photometry import (
    _tabulate_phot_data,
    _postproc_phot_table,
    Photometry,
    PhotometrySelector,
)

__all__ = [
    "Lightcurve",
    "LightcurveSelector",
    "merge",
]


class LightcurveSelector(PhotometrySelector):
    """
    A helper object that supports `Lightcurve` filtering functionality.

    Lightcurve selector objects are returned by lightcurve selection "action
    verbs" such as `~daschlab.photometry.Photometry.keep_only`. Calling one of
    the methods on a selector instance will apply the associated action to the
    specified portion of the lightcurve data.
    """

    def sep_below(
        self,
        sep_limit: u.Quantity = 20 * u.arcsec,
        pos: Optional[SkyCoord] = None,
        **kwargs,
    ) -> "Lightcurve":
        """
        Act on rows whose positional separations from the source location
        are below the limit.

        Parameters
        ==========
        sep_limit : optional `astropy.units.Quantity`, default 20 arcsec
            The separation limit. This should be an angular quantity.
        pos : optional `astropy.coordinates.SkyCoord` or `None`
            The position relative to which the separation is computed. If
            unspecified, the lightcurve `~daschlab.lightcurves.Lightcurve.mean_pos()`
            is used.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `~daschlab.photometry.Photometry.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only detections within 10 arcsec
        of the mean source position::

            from astropy import units as u

            lc = sess.lightcurve(some_local_id)
            near = lc.keep_only.sep_below(10 * u.arcsec)

        Notes
        =====
        Nondetection rows do not have an associated position, and will never
        match this filter.
        """
        if pos is None:
            pos = self._table.mean_pos()

        seps = pos.separation(self._table["pos"])
        m = seps < sep_limit
        return self._apply(m, **kwargs)

    def sep_above(
        self,
        sep_limit: u.Quantity = 20 * u.arcsec,
        pos: Optional[SkyCoord] = None,
        **kwargs,
    ) -> "Lightcurve":
        """
        Act on rows whose positional separations from the source location
        are above the limit.

        Parameters
        ==========
        sep_limit : optional `astropy.units.Quantity`, default 20 arcsec
            The separation limit. This should be an angular quantity.
        pos : optional `astropy.coordinates.SkyCoord` or `None`
            The position relative to which the separation is computed. If
            unspecified, the lightcurve `~Lightcurve.mean_pos()`
            is used.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `~daschlab.photometry.Photometry.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only detections beyond 10 arcsec
        from the mean source position::

            from astropy import units as u

            lc = sess.lightcurve(some_local_id)
            near = lc.keep_only.sep_above(10 * u.arcsec)

        Notes
        =====
        Nondetection rows do not have an associated position, and will never
        match this filter.
        """
        if pos is None:
            pos = self._table.mean_pos()

        seps = pos.separation(self._table["pos"])
        m = seps > sep_limit
        return self._apply(m, **kwargs)


class Lightcurve(Photometry, TimeSeries):
    """
    A DASCH lightcurve data table.

    A `Lightcurve` is a specialized form of `astropy.table.Table`. It is a
    subclass of both `astropy.timeseries.TimeSeries` and
    `daschlab.photometry.Photometry`, equipping it with the methods associated
    with both of those specializations. You can use all of the usual methods and
    properties made available by these parent classes; items they provide are
    not documented here.

    The actual data contained in these tables — the columns — are documented
    elsewhere, `on the main DASCH website`_.

    .. _on the main DASCH website: https://dasch.cfa.harvard.edu/dr7/lightcurve-columns/

    You should not construct `Lightcurve` instances directly. Instead, obtain
    lightcurves using the `daschlab.Session.lightcurve()` method.

    See :ref:`the module-level documentation <lc-filtering>` for a summary of
    the filtering and subsetting functionality provided by this class.

    Cheat sheet:

    - ``time`` is HJD midpoint
    - ``magcal_magdep`` is preferred calibrated phot measurement
    - legacy plotter error bar is ``magcal_local_rms` * `error_bar_factor``, the
      latter being set to match the error bars to the empirical RMS, if this
      would shrink the error bars
    """

    Selector = LightcurveSelector

    def mean_pos(self) -> SkyCoord:
        """
        Obtain the mean source position from the lightcurve points.

        Returns
        =======
        `astropy.coordinates.SkyCoord`
            The mean position of the non-rejected detections

        Notes
        =====
        Average is done in degrees naively, so if your source has RA values of
        both 0 and 360, you might get seriously bogus results.
        """
        detns = self.keep_only.nonrej_detected(verbose=False)
        mra = detns["pos"].ra.deg.mean()
        mdec = detns["pos"].dec.deg.mean()
        return SkyCoord(mra, mdec, unit=u.deg, frame="icrs")

    def plot(
        self, x_axis: str = "year", callout: Optional[np.ndarray] = None
    ) -> figure:
        """
        Plot the lightcurve using default settings.

        Parameters
        ==========
        x_axis : optional `str`, default ``"year"``
            The name of the column to use for the X axis. ``"year"`` is a
            synthetic column calculated on-the-fly from the ``"time"`` column,
            corresponding to the ``jyear`` property of the `astropy.time.Time`
            object.

        callout : optional `numpy.ndarray`, default `None`
            If provided, this should be a boolean array of the same size as the
            lightcurve table. Points (both detections and nondetections) for
            which the array is true will be visually "called out" in the plot,
            highlighted in red.

        Returns
        =======
        `bokeh.plotting.figure`
            A plot.

        Examples
        ========
        The `~daschlab.photometry.Photometry.match` selector combines
        conveniently with the *callout* functionality in constructs like this::

            # Call out points in the lightcurve with the "suspected defect" flag
            lc.plot(callout=lc.match.any_aflags(AFlags.SUSPECTED_DEFECT))

        Notes
        =====
        The function `bokeh.io.show` (imported as ``bokeh.plotting.show``) is
        called on the figure before it is returned, so you don't need to do that
        yourself.
        """
        # We have to split by `callout` before dropping any rows, becauase it is
        # a boolean filter array sized just for us. After that's done, the next
        # order of business is to drop rejected rows. Then we can add helper
        # columns that we don't want to preserve in `self`.

        if callout is None:
            main = self.drop.rejected(verbose=False)
            called_out = None
            callout_detect = None
            callout_limit = None
        else:
            called_out, main = self.split_by.where(callout)
            called_out = called_out.drop.rejected(verbose=False)
            called_out["year"] = called_out["time"].jyear

            main = main.drop.rejected(verbose=False)

        main["year"] = main["time"].jyear
        main_detect, main_limit = main.split_by.detected()

        if callout is not None:
            callout_detect, callout_limit = called_out.split_by.detected()

        p = figure(
            tools="pan,wheel_zoom,box_zoom,reset,hover",
            tooltips=[
                ("LocalID", "@local_id"),
                ("Mag.", "@magcal_magdep"),
                ("Lim. Mag.", "@limiting_mag_local"),
                ("Epoch", "@year{0000.0000}"),
                (
                    "Plate",
                    "@series@platenum / expLocID @exp_local_id",
                ),
            ],
        )

        if len(main_limit):
            p.scatter(
                x_axis,
                "limiting_mag_local",
                marker="inverted_triangle",
                fill_color="lightgray",
                line_color=None,
                source=main_limit.to_pandas(),
            )

        if callout_limit and len(callout_limit):
            p.scatter(
                x_axis,
                "limiting_mag_local",
                marker="inverted_triangle",
                fill_color="lightcoral",
                line_color=None,
                source=callout_limit.to_pandas(),
            )

        if len(main_detect):
            p.scatter(x_axis, "magcal_magdep", source=main_detect.to_pandas())

        if callout_detect and len(callout_detect):
            p.scatter(
                x_axis,
                "magcal_magdep",
                fill_color="crimson",
                line_color=None,
                source=callout_detect.to_pandas(),
            )

        p.y_range.flipped = True
        show(p)
        return p


def _query_lc(
    client: ApiClient,
    refcat: str,
    gsc_bin_index: int,
    ref_number: int,
) -> Lightcurve:
    payload = {
        "refcat": refcat,
        "gsc_bin_index": int(gsc_bin_index),  # `json` will reject `np.uint64`
        "ref_number": int(ref_number),
    }

    data = client.invoke("/dasch/dr7/lightcurve", payload)
    if not isinstance(data, list):
        from . import InteractiveError

        raise InteractiveError(f"lightcurve API request failed: {data!r}")

    return _postproc_phot_table(Lightcurve, _tabulate_phot_data(data))


# Merging lightcurves


def merge(lcs: List[Lightcurve]) -> Lightcurve:
    """
    Merge multiple lightcurves under the assumption that they all contain data
    for the same source.

    Parameters
    ==========
    lcs : list of `Lightcurve`
        The lightcurves to be merged. The provided value must be iterable and
        indexable.

    Returns
    =======
    merged_lc : `Lightcurve`
        The merged lightcurve.

    Notes
    =====
    This function is intended to address the `source splitting`_ issue that can
    affect DASCH lightcurves. In some cases, detections of the same astronomical
    source end up split across multiple DASCH lightcurves. When this happens,
    for each exposure containing a detection of the source, the detection will
    be assigned to one of several lightcurves quasi-randomly, depending on
    factors like image defects and the local errors of the WCS solution for the
    exposure in question.

    .. _source splitting: https://dasch.cfa.harvard.edu/dr7/ki/source-splitting/

    This function accomplishes the merger by unifying all of the input
    lightcurves as keyed by their exposure identifier. For each exposure where
    there is only one non-rejected detection among all the inputs, that
    detection is chosen as the "best" one for that particular exposure. These
    unique detections are used to calculate a mean magnitude for the source. For
    cases where more than one input lightcurve has a non-rejected detection for
    a given exposure, the detection with the magnitude closest to this mean
    value is selected. Finally, for any exposures where *no* lightcurves have a
    non-rejected detection, the data from the zero'th input lightcurve are used.

    The returned lightcurve has columns matching those of the zero'th input
    lightcurve, plus the following:

    - ``source_lc_index`` specifies which input lightcurve a particular row's
      data came from. This index number is relative to the *lcs* argument, which
      does not necessarily have to align with reference catalog source numbers.
    - ``source_lc_row`` specifies which row of the input lightcurve was the
      source of the output row.
    - ``merge_n_hits`` specifies how many lightcurves contained a non-rejected
      detection for this exposure. When things are working well, most rows
      should have only one hit. You may wish to reject rows where this quantity
      is larger than one, since they may indicate that the source flux is split
      between multiple SExtractor detections.
    """
    lc0 = lcs[0]
    colnames = lc0.colnames

    # Analyze the first curve to check that we know what to do with all of its columns

    T_SKYCOORD, T_TIME, T_COL, T_MASKCOL, T_MASKQUANT, T_QUANT = range(6)
    coltypes = np.zeros(len(lc0.colnames), dtype=int)

    for idx, cname in enumerate(colnames):
        col = lc0[cname]

        if isinstance(col, SkyCoord):
            coltypes[idx] = T_SKYCOORD
        elif isinstance(col, Time):
            coltypes[idx] = T_TIME
        elif isinstance(col, MaskedColumn):
            coltypes[idx] = T_MASKCOL
        elif isinstance(col, Column):
            coltypes[idx] = T_COL
        elif isinstance(col, Masked):
            coltypes[idx] = T_MASKQUANT
        elif isinstance(col, u.Quantity):
            coltypes[idx] = T_QUANT
        else:
            raise ValueError(
                f"unrecognized type of input column `{cname}`: {col.__class__.__name__}"
            )

    # Group detections in each candidate lightcurve by exposure. Some rows have
    # exp_local_id = -1 if the exposure is missing from the session's exposure
    # list; we want/need to distinguish these, so we assign them temporary ID's
    # that are negative.

    dets_by_exp = {}
    untabulated_exps = {}

    for i_lc, lc in enumerate(lcs):
        for i_row, row in enumerate(lc):
            expid = row["exp_local_id"]

            if expid == -1:
                key = (row["series"], row["platenum"], row["mosnum"], row["solnum"])
                expid = untabulated_exps.get(key)
                if expid is None:
                    expid = -(len(untabulated_exps) + 2)
                    untabulated_exps[key] = expid

            if not row["magcal_magdep"].mask and row["reject"] == 0:
                dets_by_exp.setdefault(expid, []).append((i_lc, i_row))

    # Go back to lc0 and figure out which rows have no detections in any
    # lightcurve. We could in principle do this for *all* input lightcurves, but
    # the marginal gain ought to be tiny at best, unless something very weird is
    # happening.

    no_dets = []

    for i_row, row in enumerate(lc0):
        expid = row["exp_local_id"]

        if expid == -1:
            key = (row["series"], row["platenum"], row["mosnum"], row["solnum"])
            expid = untabulated_exps[key]

        if dets_by_exp.get(expid) is None:
            no_dets.append((expid, i_row))

    # We now know how many output rows we're going to have

    n_det = len(dets_by_exp)
    n_tot = n_det + len(no_dets)

    # Allocate buffers that will hold the merged table data

    buffers = []

    for cname, ctype in zip(colnames, coltypes):
        if ctype == T_SKYCOORD:
            b = (np.zeros(n_tot), np.zeros(n_tot))
        elif ctype == T_TIME:
            b = np.zeros(n_tot)
        elif ctype == T_COL:
            b = np.zeros(n_tot, dtype=lc0[cname].dtype)
        elif ctype == T_MASKCOL:
            b = (np.zeros(n_tot, dtype=lc0[cname].dtype), np.zeros(n_tot, dtype=bool))
        elif ctype == T_MASKQUANT:
            b = (np.zeros(n_tot), np.zeros(n_tot, dtype=bool))
        elif ctype == T_QUANT:
            b = np.zeros(n_tot)

        buffers.append(b)

    # Calculate a mean magnitude for all unambiguous detections

    mags = np.zeros(n_det)
    umags = np.zeros(n_det)
    idx = 0

    for expid, hits in dets_by_exp.items():
        if len(hits) == 1:
            i_lc, i_row = hits[0]
            row = lcs[i_lc][i_row]
            mag = row["magcal_magdep"].filled(np.nan).value
            umag = row["magcal_local_rms"].filled(np.nan).value

            if np.isfinite(umag) and umag > 0:
                mags[idx] = mag
                umags[idx] = umag
                idx += 1

    mags = mags[:idx]
    umags = umags[:idx]
    weights = umags**-2
    mean_mag = (mags * weights).sum() / weights.sum()
    del mags, umags, weights

    # Use the mean mag to decide which point we'll prefer if there are ambiguous
    # detections. This is a super naive approach and will probably benefit from
    # refinement.

    det_expids = np.zeros(n_det, dtype=int)
    det_i_lcs = np.zeros(n_det, dtype=np.uint32)
    det_i_rows = np.zeros(n_det, dtype=np.uint32)
    det_n_hits = np.zeros(n_det, dtype=np.uint32)

    for idx, (expid, hits) in enumerate(dets_by_exp.items()):
        det_expids[idx] = expid
        n_hits = len(hits)

        i_lc = i_row = best_absdelta = None

        for j_lc, j_row in hits:
            row = lcs[j_lc][j_row]
            mag = row["magcal_magdep"]
            absdelta = np.abs(mag.filled(np.nan).value - mean_mag)

            if best_absdelta is None or absdelta < best_absdelta:
                best_absdelta = absdelta
                i_lc = j_lc
                i_row = j_row

        det_i_lcs[idx] = i_lc
        det_i_rows[idx] = i_row
        det_n_hits[idx] = n_hits
        # TODO: Preserve info about other candidates?

    # Figure out the timestamps of every output row so that we can sort them. If
    # a row in `origins` is nonnegative, it represents a detection, and indexes
    # into the `det_*` arrays; if it is negative, it is a non-detection, and it
    # indexes into no_dets (with an offset).

    timestamps = np.zeros(n_tot)
    origins = np.zeros(n_tot, dtype=int)
    alloced_expids = set()
    idx = 0

    for i in range(n_det):
        expid = det_expids[i]
        assert expid not in alloced_expids
        alloced_expids.add(expid)

        origins[idx] = i
        timestamps[idx] = lcs[det_i_lcs[i]][det_i_rows[i]]["time"].jd
        idx += 1

    for i, (expid, i_row) in enumerate(no_dets):
        assert expid not in alloced_expids
        alloced_expids.add(expid)

        origins[idx] = -(i + 1)
        timestamps[idx] = lc0[i_row]["time"].jd
        idx += 1

    assert idx == n_tot
    sort_args = np.argsort(timestamps)

    # Now we can populate the column buffers, with appropriate sorting.

    source_id = np.zeros(n_tot, dtype=np.uint64)
    source_row = np.zeros(n_tot, dtype=np.uint64)
    n_hits = np.zeros(n_tot, dtype=np.uint8)

    for sort_idx, presort_idx in enumerate(sort_args):
        origin = origins[presort_idx]

        if origin >= 0:
            i_lc = det_i_lcs[origin]
            i_row = det_i_rows[origin]
            i_n_hits = det_n_hits[origin]
        else:
            i_lc = 0
            i_row = (-origin) - 1
            i_n_hits = 0

        row = lcs[i_lc][i_row]
        source_id[sort_idx] = i_lc
        source_row[sort_idx] = i_row
        n_hits[sort_idx] = i_n_hits

        for cname, ctype, cbuf in zip(colnames, coltypes, buffers):
            val = row[cname]

            if ctype == T_SKYCOORD:
                cbuf[0][sort_idx] = val.ra.deg
                cbuf[1][sort_idx] = val.dec.deg
            elif ctype == T_TIME:
                cbuf[sort_idx] = val.jd
            elif ctype == T_COL:
                cbuf[sort_idx] = val
            elif ctype == T_MASKCOL:
                if val is np.ma.masked:
                    cbuf[1][sort_idx] = True
                else:
                    cbuf[0][sort_idx] = val
                    cbuf[1][sort_idx] = False
            elif ctype == T_MASKQUANT:
                cbuf[0][sort_idx] = val.filled(0).value
                cbuf[1][sort_idx] = val.mask
            elif ctype == T_QUANT:
                cbuf[sort_idx] = val.value

    # Build the actual table, and we're done.

    result = Lightcurve(masked=True, meta=lc0.meta)

    for cname, ctype, cbuf in zip(colnames, coltypes, buffers):
        if ctype == T_SKYCOORD:
            result[cname] = SkyCoord(
                ra=cbuf[0] * u.deg, dec=cbuf[1] * u.deg, frame="icrs"
            )
        elif ctype == T_TIME:
            result[cname] = Time(cbuf, format="jd")
        elif ctype == T_COL:
            result[cname] = cbuf
        elif ctype == T_MASKCOL:
            result[cname] = np.ma.array(cbuf[0], mask=cbuf[1])
        elif ctype == T_MASKQUANT:
            result[cname] = Masked(u.Quantity(cbuf[0], lc0[cname].unit), cbuf[1])
        elif ctype == T_QUANT:
            result[cname] = u.Quantity(cbuf, lc0[cname].unit)

    result["source_lc_index"] = source_id
    result["source_lc_row"] = source_row
    result["merge_n_hits"] = n_hits
    result["local_id"] = np.arange(n_tot)
    return result
