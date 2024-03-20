# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Tables of information about photographic plates.

The main class provided by this module is `Plates`, instances of which can be
obtained with the `daschlab.Session.plates()` method.

The nomenclature in this module somewhat glosses over the distinction between
physical plates and “mosaics”, which is the DASCH terminology for a single large
image made from a scan of a plate. Multiple mosaics of a single plate can, in
principle, exist. To a good approximation, however, there is a 1:1 relationship
between the two.
"""

from datetime import datetime
import io
import os
import re
import time
from typing import Dict, Iterable, Optional, Tuple, Union
from urllib.parse import urlencode
import warnings

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table, Row
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
from astropy.wcs import WCS
from bokeh.plotting import figure, show
import cairo
import numpy as np
from PIL import Image
from pytz import timezone
from pywwt.layers import ImageLayer
import requests

from .series import SERIES, SeriesKind

__all__ = ["Plates", "PlateReferenceType", "PlateRow", "PlateSelector"]


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
    """
    A single row from a `Plates` table.

    You do not need to construct these objects manually. Indexing a `Plates`
    table with a single integer will yield an instance of this class, which is a
    subclass of `astropy.table.Row`.
    """

    def show(self) -> ImageLayer:
        """
        Display the cutout of this plate in the WWT view.

        Returns
        =======
        `pywwt.layers.ImageLayer`
            This is the WWT image layer object corresponding to the displayed
            FITS file. You can use it to programmatically control aspects of how
            the file is displayed, such as the colormap.

        Notes
        =====
        In order to use this method, you must first have called
        `daschlab.Session.connect_to_wwt()`. If needed, this method will execute
        an API call and download the cutout to be displayed, which may be slow.
        """
        return self._table.show(self)

    def plate_id(self) -> str:
        """
        Get a textual identifier for this plate.

        Returns
        =======
        `str`
            The returned string has the form ``{series}{platenum}_{mosnum}``,
            where the plate number is zero-padded to be five digits wide, and
            the mosaic number is zero-padded to be two digits wide.
        """
        return f"{self['series']}{self['platenum']:05d}_{self['mosnum']:02d}"


PlateReferenceType = Union[PlateRow, int]


class PlateSelector:
    """
    A helper object that supports `Plates` filtering functionality.

    Plate selector objects are returned by plate selection "action verbs" such
    as `Plates.keep_only`. Calling one of the methods on a selector instance
    will apply the associated action to the specified portion of the lightcurve
    data.

    See the introduction to the `daschlab.lightcurves` module for an overview of
    the filtering framework used here.
    """

    _plates: "Plates"
    _apply = None

    def __init__(self, plates: "Plates", apply):
        self._plates = plates
        self._apply = apply

    def _apply_not(self, flags, **kwargs):
        "This isn't a lambda because it needs to accept and relay **kwargs"
        return self._apply(~flags, **kwargs)

    @property
    def not_(self) -> "PlateSelector":
        """
        Get a selector that will act on an inverted row selection.

        Examples
        ========
        Create a plate-list subset only those plates without good WCS
        solutions::

            from astropy import units as u

            pl = sess.plates()
            unsolved = plates.keep_only.not_.wcs_solved()

        In general, the function of this modifier is such that::

            pl.ACTION.not_.CONDITION() # should be equivalent to:
            pl.ACTION.where(~pl.match.CONDITION())
        """
        return PlateSelector(self._plates, self._apply_not)

    def where(self, row_mask, **kwargs) -> "Plates":
        """
        Act on exactly the specified list of rows.

        Parameters
        ==========
        row_mask : boolean `numpy.ndarray`
            A boolean array of exactly the size of the input plate list, with true
            values indicating rows that should be acted upon.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only points that are from A-series
        plates without good WCS solutions::

            pl = sess.plates()
            subset = pl.keep_only.where(
                pl.match.series("a") & pl.match.not_.wcs_solved()
            )
        """
        return self._apply(row_mask, **kwargs)

    def rejected(self, **kwargs) -> "Plates":
        """
        Act on rows with a non-zero ``"reject"`` value.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only rejected rows::

            pl = sess.plates()
            rejects = pl.keep_only.rejected()
        """
        m = self._plates["reject"] != 0
        return self._apply(m, **kwargs)

    def rejected_with(self, tag: str, strict: bool = False, **kwargs) -> "Plates":
        """
        Act on rows that have been rejected with the specified tag.

        Parameters
        ==========
        tag : `str`
            The tag used to identify the rejection reason
        strict : optional `bool`, default `False`
            If true, and the specified tag has not been defined, raise an
            exception. Otherwise, the action will be invoked with no rows
            selected.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only rows rejected with the
        "astrom" tag:

            pl = sess.plates()
            astrom_rejects = pl.keep_only.rejected_with("astrom")
        """
        bitnum0 = self._plates._rejection_tags().get(tag)

        if bitnum0 is not None:
            m = (self._plates["reject"] & (1 << bitnum0)) != 0
        elif strict:
            raise Exception(f"unknown rejection tag `{tag}`")
        else:
            m = np.zeros(len(self._plates), dtype=bool)

        return self._apply(m, **kwargs)

    def local_id(self, local_id: int, **kwargs) -> "Plates":
        """
        Act on the row with the specified local ID.

        Parameters
        ==========
        local_id : `int`
            The local ID to select.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only the chronologically first
        row::

            pl = sess.plates()
            first = pl.keep_only.local_id(0)

        Notes
        =====
        Plate local IDs are unique, and so this filter should only
        ever match at most one row.
        """
        m = self._plates["local_id"] == local_id
        return self._apply(m, **kwargs)

    def scanned(self, **kwargs) -> "Plates":
        """
        Act on rows corresponding to plates that have been scanned.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only *unscanned* plates::

            pl = sess.plates()
            unscanned = pl.drop.scanned()
            # or equivalently:
            unscanned = pl.keep_only.not_.scanned()

        Notes
        =====
        Some plates have been scanned, but do not have astrometric (WCS) solutions.
        These are not processed by the DASCH photometric pipeline since catalog
        cross-matching is not possible. Use `wcs_solved()` to act on such plates.
        """
        m = ~self._plates["scannum"].mask
        return self._apply(m, **kwargs)

    def wcs_solved(self, **kwargs) -> "Plates":
        """
        Act on rows corresponding to plates that have astrometric (WCS) solutions.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only WCS-solved plates::

            pl = sess.plates()
            solved = pl.keep_only.wcs_solved()

        Notes
        =====
        All WCS-solved plates are by definition scanned. Unfortunately, some
        of the WCS solutions are erroneous.
        """
        m = self._plates["wcssource"] == "imwcs"
        return self._apply(m, **kwargs)

    def series(self, series: str, **kwargs) -> "Plates":
        """
        Act on rows associated with the specified plate series.

        Parameters
        ==========
        series : `str`
            The plate series to select.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only points from the MC series::

            pl = sess.plates()
            mcs = pl.keep_only.series("mc")
        """
        m = self._plates["series"] == series
        return self._apply(m, **kwargs)

    def narrow(self, **kwargs) -> "Plates":
        """
        Act on rows associated with narrow-field telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only points from narrow-field
        telescopes::

            pl = sess.plates()
            narrow = pl.keep_only.narrow()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.NARROW for k in self._plates["series"]]
        )
        return self._apply(m, **kwargs)

    def patrol(self, **kwargs) -> "Plates":
        """
        Act on rows associated with low-resolution "patrol" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only points from patrol
        telescopes::

            pl = sess.plates()
            patrol = pl.keep_only.patrol()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.PATROL for k in self._plates["series"]]
        )
        return self._apply(m, **kwargs)

    def meteor(self, **kwargs) -> "Plates":
        """
        Act on rows associated with ultra-low-resolution "meteor" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing only points from meteor
        telescopes::

            pl = sess.plates()
            meteor = pl.keep_only.meteor()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.METEOR for k in self._plates["series"]]
        )
        return self._apply(m, **kwargs)

    def plate_names(self, names: Iterable[str], **kwargs) -> "Plates":
        """
        Act on rows associated with the specified plate names.

        Parameters
        ==========
        names : iterable of `str`
            Each name should be of the form ``{series}{platenum}``. Capitalization
            and zero-padding of the plate number are not important. This is different
            than a plate-ID, which also includes the mosaic number.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing the specified plates:

            pl = sess.plates()
            subset = pl.keep_only.plate_names(["A10000", "mc1235"])
        """
        # This feels so inefficient, but it's not obvious to me how to do any better
        m = np.zeros(len(self._plates), dtype=bool)

        for name in names:
            series, platenum = _parse_plate_name(name)
            this_one = (self._plates["series"] == series) & (
                self._plates["platenum"] == platenum
            )
            m |= this_one

        return self._apply(m, **kwargs)

    def jyear_range(self, jyear_min: float, jyear_max: float, **kwargs) -> "Plates":
        """
        Act on plates observed within the specified Julian-year time range.

        Parameters
        ==========
        jyear_min : `float`
            The lower limit of the time range (inclusive).
        jyear_max : `float`
            The upper limit of the time range (inclusive).
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Plates`
            However, different actions may return different types. For instance,
            the `Plates.count` action will return an integer.

        Examples
        ========
        Create a plate-list subset containing plates observed in the 1920's::

            pl = sess.plates()
            subset = pl.keep_only.jyear_range(1920, 1930)

        Notes
        =====
        The comparison is performed against the ``jyear`` attribute of the
        contents of the ``"obs_date"`` column.
        """
        m = (self._plates["obs_date"].jyear >= jyear_min) & (
            self._plates["obs_date"].jyear <= jyear_max
        )
        return self._apply(m, **kwargs)


class Plates(Table):
    """
    A table of DASCH plate information.

    A `Plates` is a subclass of `astropy.table.Table` containing DASCH plate
    data and associated plate-specific methods. You can use all of the usual
    methods and properties made available by the `astropy.table.Table` class.
    Items provided by the `~astropy.table.Table` class are not documented here.

    You should not construct `Plates` instances directly. Instead, obtain the
    full table using the `daschlab.Session.plates()` method.

    **Columns are not documented here!** They are (**FIXME: will be**)
    documented more thoroughly in the DASCH data description pages.

    See :ref:`the module-level documentation <lc-filtering>` of the
    `daschlab.lightcurves` for a summary of the filtering and subsetting
    functionality provided by this class.
    """

    Row = PlateRow

    def _sess(self) -> "daschlab.Session":
        from . import _lookup_session

        return _lookup_session(self.meta["daschlab_sess_key"])

    def _rejection_tags(self) -> Dict[str, int]:
        return self.meta.setdefault("daschlab_rejection_tags", {})

    def _layers(self) -> Dict[int, ImageLayer]:
        return self._sess()._plate_image_layer_cache

    # Filtering infrastructure

    def _copy_subset(self, keep, verbose: bool) -> "Plates":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        return new

    @property
    def match(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` returning a boolean array identifying selected rows.

        Unlike many actions, this does not return a new `Plates`. It can be used
        to implement arbitrary boolean logic within the action/selection framework::

            pl = sess.plates()
            subset = pl.keep_only.where(
                pl.match.series("a") & pl.match.wcs_solved()
            )
        """
        return PlateSelector(self, lambda m: m)

    @property
    def count(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` returning the number of selected rows

        Unlike many actions, this returns an `int`, not a new `Plates`.
        """
        return PlateSelector(self, lambda m: m.sum())

    def _apply_keep_only(self, flags, verbose=True) -> "Plates":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` returning a `Plates` copy containing only the selected rows.
        """
        return PlateSelector(self, self._apply_keep_only)

    def _apply_drop(self, flags, verbose=True) -> "Plates":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` returning a `Plates` copy dropping
        the selected rows; all non-selected rows are retained.
        """
        return PlateSelector(self, self._apply_drop)

    def _process_reject_tag(self, tag: str, verbose: bool) -> int:
        if not tag:
            raise Exception(
                'you must specify a rejection tag with a `tag="text"` keyword argument'
            )

        rt = self._rejection_tags()
        bitnum0 = rt.get(tag)

        if bitnum0 is None:
            bitnum0 = len(rt)
            if bitnum0 > 63:
                raise Exception("you cannot have more than 64 distinct rejection tags")

            if verbose:
                print(f"Assigned rejection tag `{tag}` to bit number {bitnum0 + 1}")

            rt[tag] = bitnum0

        return bitnum0

    def _apply_reject(self, flags, tag: str = None, verbose: bool = True):
        bitnum0 = self._process_reject_tag(tag, verbose)
        n_before = (self["reject"] != 0).sum()
        self["reject"][flags] |= 1 << bitnum0
        n_after = (self["reject"] != 0).sum()

        if verbose:
            print(
                f"Marked {n_after - n_before} new rows as rejected; {n_after} total are rejected"
            )

    @property
    def reject(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` modifying the plate-list in-place,
        rejecting the selected rows.

        Usage is as follows::

            pl = sess.plates()

            # Mark all points from meteor telescopes as rejected:
            pl.reject.meteor(tag="meteor")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.
        """
        return PlateSelector(self, self._apply_reject)

    def _apply_reject_unless(self, flags, tag: str = None, verbose: bool = True):
        return self._apply_reject(~flags, tag, verbose)

    @property
    def reject_unless(self) -> PlateSelector:
        """
        An :ref:`action <lc-filtering>` modifying the plate-list in-place,
        rejecting rows not matching the selection.

        Usage is as follows::

            pl = sess.plates()

            # Mark all plates *not* from narrow-field telescopes as rejected:
            pl.reject_unless.narrow(tag="lowrez")

            # This is equivalent to:
            pl.reject.not_.narrow(tag="lowrez")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.
        """
        return PlateSelector(self, self._apply_reject_unless)

    # Non-filtering actions on this list

    def series_info(self) -> Table:
        """
        Obtain a table summarizing information about the different plate series
        in the non-rejected rows of the current table.

        Returns
        =======
        `astropy.table.Table`
            A table of summary information

        Notes
        =====
        Columns in the returned table include:

        - ``"series"``: the plate series in question
        - ``"count"``: the number of plates from the series in *self*
        - ``"kind"``: the `daschlab.series.SeriesKind` of this series
        - ``"plate_scale"``: the typical plate scale of this series (in arcsec/mm)
        - ``"aperture"``: the telescope aperture of this series (in meters)
        - ``"description"``: a textual description of this series

        The table is sorted by decreasing ``"count"``.
        """
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

    def candidate_nice_cutouts(
        self,
        source_of_interest=0,
        limit_cols: bool = True,
        limit_rows: Optional[int] = 8,
    ) -> "daschlab.lightcurves.Lightcurve":
        """
        Return a table with information about plates that might contain "nice"
        imagery of the target, in some vague sense.

        Parameters
        ==========
        source_of_interest : optional source reference, default ``0``
            A source of interest that will be used to evaluate imaging quality.
            The lightcurve for this source will be downloaded, with the
            reference resolved as per the function
            `daschlab.Session.lightcurve()`.

        limit_cols : optional `bool`, default `True`
            If true, drop various columns from the returned table. The preserved
            columns will be only the ones relevant to inferring the plate image
            quality.

        limit_rows : optional `int` or None, default 10
            If a positive integer, limit the returned table to only this many
            rows.

        Returns
        =======
        `daschlab.lightcurves.Lightcurve`
            The returned table is actually a `~daschlab.lightcurves.Lightcurve`
            table, re-sorted and potentially subsetted. The sort order will be
            in decreasing predicted image quality based on the lightcurve data.

        Notes
        =====
        The primary sort key is based on the limiting magnitude, which is a
        quantity available even in rows of the lightcurve that do not detect the
        source of interest. So, this method should be useful even if the source
        is not easily detected. Other image-quality metadata, however, may be
        missing, so this method will restrict itself to lightcurve rows with
        detections if there are a sufficient number of them.
        """
        sess = self._sess()
        lc = sess.lightcurve(source_of_interest)
        lc = lc.drop.rejected(verbose=False)

        # If we have an acceptable number of detections, consider only plates
        # that detect the SOI. The cutoff here is totally arbitrary.
        if lc.count.nonrej_detected() > 10:
            lc = lc.keep_only.nonrej_detected(verbose=False)

        lc.sort(["limiting_mag_local"], reverse=True)

        if limit_cols:
            cols = "limiting_mag_local plate_local_id series platenum fwhm_world ellipticity background".split()
            lc = lc[cols]

        if limit_rows and limit_rows > 0:
            lc = lc[:limit_rows]

        return lc

    def time_coverage(self) -> figure:
        """
        Plot the observing time coverage of the non-rejected plates in this list.

        Returns
        =======
        `bokeh.plotting.figure`
            A plot.

        Notes
        =====
        The plot is generated by using a Gaussian kernel density estimator to
        smooth out the plate observation times.

        The function `bokeh.io.show` (imported as ``bokeh.plotting.show``) is
        called on the figure before it is returned, so you don't need to do that
        yourself.
        """
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

    def show(self, plate_ref: PlateReferenceType) -> ImageLayer:
        """
        Display the cutout of the specified plate in the WWT view.

        Parameters
        ==========
        plate_ref : `PlateRow` or `int`
            If this argument is an integer, it is interpreted as the "local ID"
            of a plate. Local IDs are assigned in chronological order of
            observing time, so a value of ``0`` shows the first plate. This will
            only work if the selected plate is scanned and WCS-solved.

            If this argument is an instance of `PlateRow`, the specified plate
            is shown.

        Returns
        =======
        `pywwt.layers.ImageLayer`
          This is the WWT image layer object corresponding to the displayed FITS
          file. You can use it to programmatically control aspects of how the
          file is displayed, such as the colormap.

        Notes
        =====
        In order to use this method, you must first have called
        `daschlab.Session.connect_to_wwt()`. If needed, this method will execute
        an API call and download the cutout to be displayed, which may be slow.
        """
        plate = self._sess()._resolve_plate_reference(plate_ref)

        local_id = plate["local_id"]
        il = self._layers().get(local_id)

        if il is not None:
            return il  # TODO: bring it to the top, etc?

        fits_relpath = self._sess().cutout(plate)
        if fits_relpath is None:
            from . import InteractiveError

            raise InteractiveError(
                f"cannot create a cutout for plate #{local_id:05d} ({plate.plate_id()})"
            )

        il = (
            self._sess()
            .wwt()
            .layers.add_image_layer(str(self._sess().path(fits_relpath)))
        )
        self._layers()[local_id] = il
        return il

    def export_cutouts_to_pdf(self, pdfpath: str, **kwargs):
        """
        Export cutouts of the plates in this list to a PDF file.

        Parameters
        ==========
        pdf_path : `str`
            The path at which the output file will be written. This path is
            *not* relative to the `daschlab.Session` root.
        refcat_stdmag_limit : optional `float`
            If specified, only sources from the reference catalog brigher than
            the specified ``stdmag`` limit will be overlaid on the cutouts.
            Otherwise, all refcat sources are overlaid, including very faint
            ones and ones without ``stdmag`` records.

        Notes
        =====
        This operation will attempt to download a cutout for every WCS-solved
        plate in the list. This can be extremely slow, as well as creating a lot
        of work for the API server. Only use it for short plate lists.

        The format of the created file needs to be documented.
        """
        _pdf_export(pdfpath, self._sess(), self, **kwargs)


def _query_plates(
    center: SkyCoord,
    radius: u.Quantity,
) -> Plates:
    return _postproc_plates(_get_plate_cols(center, radius))


def _get_plate_cols(
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

    with requests.get(url, stream=True) as resp:
        for line in resp.iter_lines():
            line = line.decode("utf-8")
            pieces = line.rstrip().split("\t")

            if colnames is None:
                colnames = pieces
                coltypes = [_COLTYPES.get(c) for c in colnames]
                coldata = [[] if t is not None else None for t in coltypes]
            else:
                for row, ctype, cdata in zip(pieces, coltypes, coldata):
                    if ctype is not None:
                        cdata.append(ctype(row))

    if colnames is None:
        raise Exception("empty plate-table data response")

    return dict(t for t in zip(colnames, coldata) if t[1] is not None)


def _dasch_date_as_datetime(date: str) -> Optional[datetime]:
    if not date:
        return None

    p = date.split("T")
    p[1] = p[1].replace("-", ":")
    naive = datetime.fromisoformat("T".join(p))
    tz = timezone("US/Eastern")
    return tz.localize(naive)


def _dasch_dates_as_time_array(dates) -> Time:
    """
    Convert an iterable of DASCH dates to an Astropy Time array, with masking of
    missing values.
    """
    # No one had any photographic plates in 1800! But the exact value here
    # doesn't matter; we just need something accepted by the datetime-Time
    # constructor.
    invalid_dt = datetime(1800, 1, 1)

    invalid_indices = []
    dts = []

    for i, dstr in enumerate(dates):
        dt = _dasch_date_as_datetime(dstr)
        if dt is None:
            invalid_indices.append(i)
            dt = invalid_dt

        dts.append(dt)

    times = Time(dts, format="datetime")

    for i in invalid_indices:
        times[i] = np.ma.masked

    # If we don't do this, Astropy is unable to roundtrip masked times out of
    # the ECSV format.
    times.format = "isot"

    return times


def _postproc_plates(input_cols) -> Plates:
    table = Plates(masked=True)

    scannum = np.array(input_cols["scannum"], dtype=np.int8)
    mask = scannum == -1

    # create a unitless (non-Quantity) column, unmasked:
    all_c = lambda c, dt: np.array(input_cols[c], dtype=dt)

    # unitless column, masked:
    mc = lambda c, dt: np.ma.array(input_cols[c], mask=mask, dtype=dt)

    # Quantity column, unmasked:
    all_q = lambda c, dt, unit: u.Quantity(np.array(input_cols[c], dtype=dt), unit)

    # Quantity column, masked:
    mq = lambda c, dt, unit: Masked(
        u.Quantity(np.array(input_cols[c], dtype=dt), unit), mask
    )

    # Quantity column, masked, with extra flag values to mask:
    def extra_mq(c, dt, unit, flagval):
        a = np.array(input_cols[c], dtype=dt)
        extra_mask = mask | (a == flagval)
        return Masked(u.Quantity(a, unit), extra_mask)

    table["series"] = input_cols["series"]
    table["platenum"] = np.array(input_cols["platenum"], dtype=np.uint32)
    table["scannum"] = mc("scannum", np.uint8)
    table["mosnum"] = mc("mosnum", np.uint8)
    table["expnum"] = np.array(input_cols["expnum"], dtype=np.uint8)
    table["solnum"] = mc("solnum", np.uint8)
    table["class"] = input_cols["class"]
    table["pos"] = SkyCoord(
        ra=input_cols["ra"] * u.deg,
        dec=input_cols["dec"] * u.deg,
        frame="icrs",
    )
    table["exptime"] = np.array(input_cols["exptime"], dtype=np.float32) * u.minute
    table["obs_date"] = Time(input_cols["jd"], format="jd")
    table["wcssource"] = input_cols["wcssource"]
    table["scan_date"] = _dasch_dates_as_time_array(input_cols["scandate"])
    table["mos_date"] = _dasch_dates_as_time_array(input_cols["mosdate"])
    table["rotation_deg"] = mc("rotation", np.uint16)
    table["binflags"] = mc("binflags", np.uint8)
    table["center_distance"] = (
        np.array(input_cols["centerdist"], dtype=np.float32) * u.cm
    )
    table["edge_distance"] = np.array(input_cols["edgedist"], dtype=np.float32) * u.cm

    table.sort(["obs_date"])

    table["local_id"] = np.arange(len(table))
    table["reject"] = np.zeros(len(table), dtype=np.uint64)

    return table


# PDF export


PAGE_WIDTH = 612  # US Letter paper size, points
PAGE_HEIGHT = 792
MARGIN = 36  # points; 0.5 inch
CUTOUT_WIDTH = PAGE_WIDTH - 2 * MARGIN
DSF = 4  # downsampling factor
LINE_HEIGHT = 14  # points


def _data_to_image(data: np.array) -> Image:
    """
    Here we implement the grayscale method used by the DASCH website. The input is a
    float array and the output is a PIL/Pillow image.

    1. linearly rescale data clipping bottom 2% and top 1% of pixel values
       (pnmnorm)
    2. scale to 0-255 (pnmdepth)
    3. invert scale (pnminvert)
    4. flip vertically (pnmflip; done prior to this function)
    5. apply a 2.2 gamma scale (pnmgamma)

    We reorder a few steps here to simplify the computations. The results aren't
    exactly identical to what you get across the NetPBM steps but they're
    visually indistinguishable in most cases.
    """

    dmin, dmax = np.percentile(data, [2, 99])

    if dmax == dmin:
        data = np.zeros(data.shape, dtype=np.uint8)
    else:
        data = np.clip(data, dmin, dmax)
        data = (data - dmin) / (dmax - dmin)
        data = 1 - data
        data **= 1 / 2.2
        data *= 255
        data = data.astype(np.uint8)

    return Image.fromarray(data, mode="L")


def _pdf_export(
    pdfpath: str,
    sess: "Session",
    plates: Plates,
    refcat_stdmag_limit: Optional[float] = None,
):
    PLOT_CENTERING_CIRCLE = True
    PLOT_SOURCES = True

    center_pos = sess.query().pos_as_skycoord()
    ra_text = center_pos.ra.to_string(unit=u.hour, sep=":", precision=3)
    dec_text = center_pos.dec.to_string(
        unit=u.degree, sep=":", alwayssign=True, precision=2
    )
    center_text = f"{ra_text} {dec_text} J2000"

    name = sess.query().name
    if name:
        center_text = f"{name} = {center_text}"

    # First thing, ensure that we have all of our cutouts

    n_orig = len(plates)
    plates = plates.keep_only.wcs_solved(verbose=False).drop.rejected(verbose=False)
    n_filtered = len(plates)

    if not n_filtered:
        raise Exception("no plates remain after filtering")

    if n_filtered != n_orig:
        print(
            f"Filtered input list of {n_orig} plates down to {n_filtered} for display"
        )

    print(
        f"Ensuring that we have cutouts for {n_filtered} plates, this may take a while ..."
    )
    t0 = time.time()

    for plate in plates:
        if not sess.cutout(plate):
            raise Exception(
                f"unable to fetch cutout for plate LocalId {plate['local_id']}"
            )

    elapsed = time.time() - t0
    print(f"... completed in {elapsed:.0f} seconds")

    # Set up refcat sources (could add filtering, etc.)

    if PLOT_SOURCES:
        refcat = sess.refcat()

        if refcat_stdmag_limit is not None:
            refcat = refcat[~refcat["stdmag"].mask]
            refcat = refcat[refcat["stdmag"] <= refcat_stdmag_limit]

        cat_pos = refcat["pos"]
        print(f"Overlaying {len(refcat)} refcat sources")

    # We're ready to start rendering pages

    t0 = time.time()
    print("Rendering pages ...")

    n_pages = 0
    label_y0 = None
    wcs = None

    with cairo.PDFSurface(pdfpath, PAGE_WIDTH, PAGE_HEIGHT) as pdf:
        for plate in plates:
            local_id = plate["local_id"]
            series = plate["series"]
            platenum = plate["platenum"]
            mosnum = plate["mosnum"]
            plateclass = plate["class"] or "(none)"
            jd = plate["obs_date"].jd
            epoch = plate["obs_date"].jyear
            # platescale = plate["plate_scale"]
            exptime = plate["exptime"]
            center_distance_cm = plate["center_distance"]
            edge_distance_cm = plate["edge_distance"]

            fits_relpath = sess.cutout(plate)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with fits.open(str(sess.path(fits_relpath))) as hdul:
                    data = hdul[0].data
                    astrom_type = hdul[0].header["D_ASTRTY"]

                    if wcs is None:
                        wcs = WCS(hdul[0].header)

                        # Calculate the size of a 20 arcsec circle
                        x0 = data.shape[1] / 2
                        y0 = data.shape[0] / 2
                        c0 = wcs.pixel_to_world(x0, y0)

                        if c0.dec.deg >= 0:
                            c1 = c0.directional_offset_by(180 * u.deg, 1 * u.arcmin)
                        else:
                            c1 = c0.directional_offset_by(0 * u.deg, 1 * u.arcmin)

                        x1, y1 = wcs.world_to_pixel(c1)
                        ref_radius = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                        ref_radius /= DSF  # account for the downsampling

            # downsample to reduce data volume
            height = data.shape[0] // DSF
            width = data.shape[1] // DSF
            h_trunc = height * DSF
            w_trunc = width * DSF
            data = data[:h_trunc, :w_trunc].reshape((height, DSF, width, DSF))
            data = data.mean(axis=(1, 3))

            # Important! FITS is rendered "bottoms-up" so we must flip vertically for
            # the RGB imagery, while will be top-down!
            data = data[::-1]

            # Our data volumes get out of control unless we JPEG-compress
            pil_image = _data_to_image(data)
            jpeg_encoded = io.BytesIO()
            pil_image.save(jpeg_encoded, format="JPEG")
            jpeg_encoded.seek(0)
            jpeg_encoded = jpeg_encoded.read()

            img = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
            img.set_mime_data("image/jpeg", jpeg_encoded)

            cr = cairo.Context(pdf)
            cr.save()
            cr.translate(MARGIN, MARGIN)
            f = CUTOUT_WIDTH / width
            cr.scale(f, f)
            cr.set_source_surface(img, 0, 0)
            cr.paint()
            cr.restore()

            # reference circle overlay

            if PLOT_CENTERING_CIRCLE:
                cr.save()
                cr.set_source_rgb(0.5, 0.5, 0.9)
                cr.set_line_width(1)
                cr.arc(
                    CUTOUT_WIDTH / 2 + MARGIN,  # x
                    CUTOUT_WIDTH / 2 + MARGIN,  # y -- hardcoding square aspect ratio
                    ref_radius * f,  # radius
                    0.0,  # angle 1
                    2 * np.pi,  # angle 2
                )
                cr.stroke()
                cr.restore()

            # Source overlay

            if PLOT_SOURCES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    this_pos = cat_pos.apply_space_motion(
                        Time(epoch, format="decimalyear")
                    )

                this_px = np.array(wcs.world_to_pixel(this_pos))
                this_px /= DSF  # account for our downsampling
                x = this_px[0]
                y = this_px[1]
                y = height - y  # account for FITS/JPEG y axis parity flip
                ok = (x > -0.5) & (x < width - 0.5) & (y > -0.5) & (y < height - 0.5)

                cr.save()
                cr.set_source_rgb(0, 1, 0)
                cr.set_line_width(1)

                for i in range(len(ok)):
                    if not ok[i]:
                        continue

                    ix = x[i] * f + MARGIN
                    iy = y[i] * f + MARGIN
                    cr.move_to(ix - 4, iy - 4)
                    cr.line_to(ix + 4, iy + 4)
                    cr.stroke()
                    cr.move_to(ix - 4, iy + 4)
                    cr.line_to(ix + 4, iy - 4)
                    cr.stroke()

                cr.restore()

            # Labeling

            if label_y0 is None:
                label_y0 = 2 * MARGIN + height * f

            with warnings.catch_warnings():
                # Lots of ERFA complaints for our old years
                warnings.simplefilter("ignore")
                t_iso = Time(jd, format="jd").isot

            cr.save()
            cr.select_font_face(
                "mono", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            linenum = 0

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.save()
            cr.set_font_size(18)
            cr.show_text(
                f"Plate {series.upper()}{platenum:05d} - localId {local_id:5d} - {epoch:.5f}"
            )
            cr.restore()
            linenum += 1.5

            # XXX hardcoding params
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(f"Image: center {center_text}")
            linenum += 1
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text("  halfsize 600 arcsec, orientation N up, E left")
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(
                f"Mosaic {series:>4}{platenum:05d}_{mosnum:02d}: "
                f"astrom {astrom_type.upper():3} "
                f"class {plateclass:12} "
                f"exp {exptime:5.1f} (min) "
                # f"scale {platescale:7.2f} (arcsec/mm)"
            )
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            url = f"http://dasch.rc.fas.harvard.edu/showplate.php?series={series}&plateNumber={platenum}"
            cr.tag_begin(cairo.TAG_LINK, f"uri='{url}'")
            cr.set_source_rgb(0, 0, 1)
            cr.show_text(url)
            cr.set_source_rgb(0, 0, 0)
            cr.tag_end(cairo.TAG_LINK)
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(f"Midpoint: epoch {epoch:.5f} = HJD {jd:.6f} = {t_iso}")
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(
                f"Cutout location: {center_distance_cm:4.1f} cm from center, {edge_distance_cm:4.1f} cm from edge"
            )
            linenum += 1
            cr.restore()

            pdf.show_page()
            n_pages += 1

            if n_pages % 100 == 0:
                print(f"- page {n_pages} - {fits_relpath}")

    elapsed = time.time() - t0
    info = os.stat(pdfpath)
    print(
        f"... completed in {elapsed:.0f} seconds and saved to `{pdfpath}` ({info.st_size / 1024**2:.1f} MiB)"
    )
