# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Tables of information about exposures on photographic plates.

The main class provided by this module is `Exposures`, instances of which can be
obtained with the `daschlab.Session.exposures()` method.

Every photographic plate worth scanning was exposed at least once, but some were
exposed multiple times. This means that on a given plate image, the same pixel
may correspond to multiple different RA/Dec values, and a single RA/Dec may
occur at several different pixel locations, if the multiple exposures overlapped
each other.

For each plate that was successfully scanned, there is at least one “mosaic”,
which is the DASCH terminology for a single large FITS image made from a plate
scan. There can in principle be multiple mosaics per plate. Each mosaic is
analyzed to search for as many WCS solutions as can be found.

Ideally, each WCS solution corresponds to an exposure of the plate. However, we
also have external information about what plates have exposures, based on
historical observing logbooks. WCS solutions may or may not be identifiable with
known exposures. Sometimes there are fewer WCS solutions than logged exposures;
sometimes there are more. Furthermore, the logbooks may provide information
about exposures on plates that haven't been scanned at all.

An exposure associated with a WCS solution to a mosaic is said to "have imaging"
data available. This module deals with both exposures that have imaging, and
those that do not. Some exposures lack imaging because the associated plate was
not scanned at all; some lack imaging because it was not possible to associate
that exposure with a WCS solution.
"""

import io
import os
import re
import time
from typing import Dict, Iterable, Optional, Tuple, Union
import warnings

from astropy.coordinates import SkyCoord
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
from pywwt.layers import ImageLayer, TableLayer

from .apiclient import ApiClient
from .series import SERIES, SeriesKind
from .timeutil import dasch_time_as_isot, dasch_isots_as_time_array


__all__ = ["Exposures", "ExposureReferenceType", "ExposureRow", "ExposureSelector"]


_PLATE_NAME_REGEX = re.compile(r"^([a-zA-Z]+)([0-9]{1,5})$")


def _parse_plate_name(name: str) -> Tuple[str, int]:
    try:
        m = _PLATE_NAME_REGEX.match(name)
        series = m[1].lower()
        platenum = int(m[2])
    except Exception:
        raise Exception(f"invalid plate name `{name}`")

    return series, platenum


def maybe_float(s: str) -> float:
    if s:
        return float(s)
    return np.nan


def maybe_nonneg_int(s: str) -> int:
    if s:
        return int(s)
    return -1


_COLTYPES = {
    "series": str,
    "platenum": int,
    "scannum": int,
    "mosnum": int,
    "expnum": int,
    "solnum": int,
    "class": str,
    "ra": maybe_float,
    "dec": maybe_float,
    "exptime": maybe_float,
    "expdate": dasch_time_as_isot,
    # "epoch": float,
    "wcssource": lambda s: s.lower(),
    "scandate": dasch_time_as_isot,
    "mosdate": dasch_time_as_isot,
    "centerdist": maybe_float,
    "edgedist": maybe_float,
    "limMagApass": maybe_float,
    "limMagAtlas": maybe_float,
    "medianColortermApass": maybe_float,
    "medianColortermAtlas": maybe_float,
    "nSolutionsApass": maybe_nonneg_int,
    "nSolutionsAtlas": maybe_nonneg_int,
    "nMagdepApass": maybe_nonneg_int,
    "nMagdepAtlas": maybe_nonneg_int,
    "resultIdApass": str,
    "resultIdAtlas": str,
}


class ExposureRow(Row):
    """
    A single row from an `Exposures` table.

    You do not need to construct these objects manually. Indexing an `Exposures`
    table with a single integer will yield an instance of this class, which is a
    subclass of `astropy.table.Row`.
    """

    def show(self) -> ImageLayer:
        """
        Display the cutout of this exposure in the WWT view.

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
        if not self.has_imaging():
            raise Exception("Cannot show an exposure that has no associated imaging")
        return self._table.show(self)

    def has_imaging(self) -> bool:
        """
        Test whether this exposure has associated imaging data or not.

        Returns
        =======
        `bool`
            True if the exposure has imaging.

        Notes
        =====
        Exposures do not have imaging when they are obtained from logbook data
        but the associated plate has not been scanned. Or, the associated plate
        may have been scanned, but the WCS analysis of the imagery was unable to
        determine a solution matching the information associated with this
        exposure.
        """
        return self["wcssource"] == "imwcs"  # same test as in the selector framework

    def has_phot(self) -> bool:
        """
        Test whether this exposure has associated photometric data for the session's
        reference catalog, or not.

        Returns
        =======
        `bool`
            True if the exposure has photometry.

        Notes
        =====
        The test checks whether this exposure has data in the
        ``lim_mag_{refcat}`` field, where ``{refcat}`` is the reference catalog
        being used by this session: either ``apass`` or ``atlas``. If present,
        this implies that exposure has imaging and that the DASCH photometric
        calibration pipeline successfully processed this exposure. Exposures
        that have photometry for one refcat may not have photometry for the
        other.
        """
        sess_refcat = self._table._sess()._refcat_name()
        val = self[
            f"lim_mag_{sess_refcat}"
        ]  # this might be a float or a MaskedConstant

        if hasattr(val, "mask"):
            return not val.mask

        return np.isfinite(val)

    def exp_id(self) -> str:
        """
        Get a textual identifier for this exposure.

        Returns
        =======
        `str`
            The returned string has the form ``{series}{platenum}{tail}``, where
            the plate number is zero-padded to be five digits wide. If the
            exposure corresponds to a WCS solution on a mosaic, the tail has the
            form ``m{mosnum}s{solnum}``. If it corresponds to a logged exposure
            that is not assignable to a mosaic WCS solution, it has the form
            ``e{expnum}``.

        See Also
        ========
        plate_id : Get the plate ID
        """
        m = self["mosnum"]

        if m is np.ma.masked:
            tail = f"e{self['expnum']}"
        else:
            tail = f"m{m}s{self['solnum']}"

        return f"{self['series']}{self['platenum']:05d}{tail}"

    def plate_id(self) -> str:
        """
        Get the name of the plate that this exposure is associated with.

        Returns
        =======
        `str`
            The returned string has the form ``{series}{platenum}``, where
            the plate number is zero-padded to be five digits wide.

        Notes
        =====
        In most cases, you should work with exposure IDs, obtainable with
        the `exp_id` method. This is because some plates were exposured
        multiple times, and each exposure has its own duration, WCS solution,
        and source catalog. Multiple exposures in one exposure list may be
        associated with the same plate.

        See Also
        ========
        exp_id : Get the unique ID of this particular exposure
        """
        return f"{self['series']}{self['platenum']:05d}"

    def photcal_asdf_url(self, refcat: Optional[str] = None) -> str:
        """
        Obtain a temporary URL that can be used to download photometric
        calibration metadata associated with this exposure.

        Parameters
        ==========
        refcat : optional `str`, default `None`
            There are separate calibration data files for the two DASCH DR7
            reference catalogs: ``"apass"`` or ``"atlas"``. This field indicates
            which file you’re interested in. If unspecified, will default to the
            reference catalog being used by the currently active session. This
            is almost surely what you want.

        Returns
        =======
        url : `str`
            A URL that can be used to download the metadata file.

        Examples
        ========

        Assuming that the first row in the exposures table has a valid
        photometric calibration, download its ASDF file and save it to disk::

            import shutil
            import requests

            # Assume `sess` is a standard daschlab session object

            asdf_url = sess.exposures()[0].photcal_asdf_url()

            with requests.get(asdf_url, stream=True) as resp:
                with open("exposure0.asdf", "wb") as f_out:
                    shutil.copyfileobj(resp.raw, f_out)

        Notes
        =====
        The returned URL is a presigned AWS S3 download link with a lifetime of
        around 15 minutes. You can fetch this URL using a browser or any
        standard HTTP library to obtain the data.

        The resulting data file is stored in the `ASDF`_ format. Its contents
        are documented on the DASCH DR7 `Photometric Calibration ASDF File
        Contents`_ page. Typical file sizes are a few megabytes, ranging up to
        tens of megabytes.

        .. _ASDF: https://asdf.readthedocs.io/
        .. _Photometric Calibration ASDF File Contents: https://dasch.cfa.harvard.edu/dr7/photcal-asdf-contents/

        Each ASDF file is uniquely identified by a hexadecimal “result ID”. The
        value to use is taken from one table columns ``result_id_apass`` or
        ``result_id_atlas``. This value may be empty if the exposure in question
        was not able to be calibrated against the specified refcat. In that
        case, this method will raise an exception.
        """
        sess = self._table._sess()

        if refcat is None:
            refcat = sess._refcat_name()
        else:
            from . import SUPPORTED_REFCATS

            if refcat not in SUPPORTED_REFCATS:
                raise ValueError(
                    f"invalid refcat name {refcat!r}; must be one of: {' '.join(SUPPORTED_REFCATS)}"
                )

        hexid = self[f"result_id_{refcat}"]
        if not hexid:
            raise Exception(
                f"no photometric calibration ASDF file available for {self.exp_id()}/{refcat}"
            )

        data = sess._apiclient.invoke(
            f"/dasch/dr7/asset/photcal_asdf/{hexid}", None, method="get"
        )
        if not isinstance(data, dict):
            from . import InteractiveError

            raise InteractiveError(f"photcal ASDF asset API request failed: {data!r}")

        return data["location"]


ExposureReferenceType = Union[ExposureRow, int]


class ExposureSelector:
    """
    A helper object that supports `Exposures` filtering functionality.

    Exposure selector objects are returned by exposure selection "action verbs" such
    as `Exposures.keep_only`. Calling one of the methods on a selector instance
    will apply the associated action to the specified portion of the lightcurve
    data.

    See the introduction to the `daschlab.lightcurves` module for an overview of
    the filtering framework used here.
    """

    _exposures: "Exposures"
    _apply = None

    def __init__(self, exposures: "Exposures", apply):
        self._exposures = exposures
        self._apply = apply

    def _apply_not(self, flags, **kwargs):
        "This isn't a lambda because it needs to accept and relay **kwargs"
        return self._apply(~flags, **kwargs)

    @property
    def not_(self) -> "ExposureSelector":
        """
        Get a selector that will act on an inverted row selection.

        Examples
        ========
        Create an exposure-list subset containing only those exposures that do
        not have associated imaging data::

            from astropy import units as u

            exp = sess.exposures()
            unsolved = exposures.keep_only.not_.has_imaging()

        In general, the function of this modifier is such that::

            exp.ACTION.not_.CONDITION() # should be equivalent to:
            exp.ACTION.where(~exp.match.CONDITION())
        """
        return ExposureSelector(self._exposures, self._apply_not)

    def where(self, row_mask, **kwargs) -> "Exposures":
        """
        Act on exactly the specified list of rows.

        Parameters
        ==========
        row_mask : boolean `numpy.ndarray`
            A boolean array of exactly the size of the input exposure list, with true
            values indicating rows that should be acted upon.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures that are from A-series
        exposures and do not have associated imaging data::

            exp = sess.exposures()
            subset = exp.keep_only.where(
                exp.match.series("a") & exp.match.not_.has_imaging()
            )
        """
        return self._apply(row_mask, **kwargs)

    def rejected(self, **kwargs) -> "Exposures":
        """
        Act on rows with a non-zero ``"reject"`` value.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only rejected rows::

            exp = sess.exposures()
            rejects = exp.keep_only.rejected()
        """
        m = self._exposures["reject"] != 0
        return self._apply(m, **kwargs)

    def rejected_with(self, tag: str, strict: bool = False, **kwargs) -> "Exposures":
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
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only rows rejected with the
        "astrom" tag:

            exp = sess.exposures()
            astrom_rejects = exp.keep_only.rejected_with("astrom")
        """
        bitnum0 = self._exposures._rejection_tags().get(tag)

        if bitnum0 is not None:
            m = (self._exposures["reject"] & (1 << bitnum0)) != 0
        elif strict:
            raise Exception(f"unknown rejection tag `{tag}`")
        else:
            m = np.zeros(len(self._exposures), dtype=bool)

        return self._apply(m, **kwargs)

    def local_id(self, local_id: int, **kwargs) -> "Exposures":
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
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only the chronologically first
        row::

            exp = sess.exposures()
            first = exp.keep_only.local_id(0)

        Notes
        =====
        Exposure local IDs are unique, and so this filter should only
        ever match at most one row.
        """
        m = self._exposures["local_id"] == local_id
        return self._apply(m, **kwargs)

    def scanned(self, **kwargs) -> "Exposures":
        """
        Act on exposures corresponding to plates that have been scanned.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures from *unscanned* plates::

            exp = sess.exposures()
            unscanned = exp.drop.scanned()
            # or equivalently:
            unscanned = exp.keep_only.not_.scanned()

        Notes
        =====
        Some exposures are known to have occurred, and are associated with plates that
        have been scanned, but the pipeline was unable to find an associated WCS solution
        in the plate imagery. Use `has_imaging()` to act only on exposures with WCS
        solutions.
        """
        m = ~self._exposures["scannum"].mask
        return self._apply(m, **kwargs)

    def has_imaging(self, **kwargs) -> "Exposures":
        """
        Act on exposures that can be associated to imaging data.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures with imaging::

            exp = sess.exposures()
            solved = exp.keep_only.has_imaging()

        Notes
        =====
        Unfortunately, some of the DASCH WCS solutions are erroneous.
        """
        m = self._exposures["wcssource"] == "imwcs"
        return self._apply(m, **kwargs)

    def has_phot(self, **kwargs) -> "Exposures":
        """
        Act on exposures that were able to be photometrically calibrated.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Count the number of exposures with photometry::

            exp = sess.exposures()
            n_phot = exp.count.has_phot()

        Notes
        =====
        The selector acts on exposures that have data in the
        ``lim_mag_{refcat}`` field, where ``{refcat}`` is the reference catalog
        being used by this session: either ``apass`` or ``atlas``. This field is
        only present for exposures that have imaging and were successfully
        processed by the DASCH photometric pipeline for that refcat.
        """
        sess_refcat = self._exposures._sess()._refcat_name()
        col = self._exposures[f"lim_mag_{sess_refcat}"]
        m = np.isfinite(col.filled(np.nan))
        return self._apply(m, **kwargs)

    def series(self, series: str, **kwargs) -> "Exposures":
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
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures from the MC series::

            exp = sess.exposures()
            mcs = exp.keep_only.series("mc")
        """
        m = self._exposures["series"] == series
        return self._apply(m, **kwargs)

    def narrow(self, **kwargs) -> "Exposures":
        """
        Act on rows associated with narrow-field telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures from narrow-field
        telescopes::

            exp = sess.exposures()
            narrow = exp.keep_only.narrow()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.NARROW for k in self._exposures["series"]]
        )
        return self._apply(m, **kwargs)

    def patrol(self, **kwargs) -> "Exposures":
        """
        Act on rows associated with low-resolution "patrol" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures from patrol
        telescopes::

            exp = sess.exposures()
            patrol = exp.keep_only.patrol()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.PATROL for k in self._exposures["series"]]
        )
        return self._apply(m, **kwargs)

    def meteor(self, **kwargs) -> "Exposures":
        """
        Act on rows associated with ultra-low-resolution "meteor" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing only exposures from meteor
        telescopes::

            exp = sess.exposures()
            meteor = exp.keep_only.meteor()
        """
        m = np.array(
            [SERIES[k].kind == SeriesKind.METEOR for k in self._exposures["series"]]
        )
        return self._apply(m, **kwargs)

    def plate_names(self, names: Iterable[str], **kwargs) -> "Exposures":
        """
        Act on rows associated with the specified plate names.

        Parameters
        ==========
        names : iterable of `str`
            Each name should be of the form ``{series}{platenum}``. Capitalization
            and zero-padding of the plate number are not important.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing exposures on the specified plates:

            exp = sess.exposures()
            subset = exp.keep_only.plate_names(["A10000", "mc1235"])
        """
        # This feels so inefficient, but it's not obvious to me how to do any better
        m = np.zeros(len(self._exposures), dtype=bool)

        for name in names:
            series, platenum = _parse_plate_name(name)
            this_one = (self._exposures["series"] == series) & (
                self._exposures["platenum"] == platenum
            )
            m |= this_one

        return self._apply(m, **kwargs)

    def jyear_range(self, jyear_min: float, jyear_max: float, **kwargs) -> "Exposures":
        """
        Act on exposures observed within the specified Julian-year time range.

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
        Usually, another `Exposures`
            However, different actions may return different types. For instance,
            the `Exposures.count` action will return an integer.

        Examples
        ========
        Create an exposure-list subset containing exposures observed in the 1920's::

            exp = sess.exposures()
            subset = exp.keep_only.jyear_range(1920, 1930)

        Notes
        =====
        The comparison is performed against the ``jyear`` attribute of the
        contents of the ``"obs_date"`` column.
        """
        m = (self._exposures["obs_date"].jyear >= jyear_min) & (
            self._exposures["obs_date"].jyear <= jyear_max
        )
        return self._apply(m, **kwargs)


class Exposures(Table):
    """
    A table of DASCH exposure information.

    An `Exposures` table is a subclass of `astropy.table.Table` containing DASCH
    exposure data and associated exposure-specific methods. You can use all of
    the usual methods and properties made available by the `astropy.table.Table`
    class. Items provided by the `~astropy.table.Table` class are not documented
    here.

    You should not construct `Exposures` instances directly. Instead, obtain the
    full table using the `daschlab.Session.exposures()` method.

    The actual data contained in these tables — the columns — are documented
    elsewhere, `on the main DASCH website`_.

    .. _on the main DASCH website: https://dasch.cfa.harvard.edu/dr7/exposurelist-columns/

    See :ref:`the module-level documentation <lc-filtering>` of the
    `daschlab.lightcurves` for a summary of the filtering and subsetting
    functionality provided by this class.
    """

    Row = ExposureRow

    def _sess(self) -> "daschlab.Session":
        from . import _lookup_session

        return _lookup_session(self.meta["daschlab_sess_key"])

    def _rejection_tags(self) -> Dict[str, int]:
        return self.meta.setdefault("daschlab_rejection_tags", {})

    def _layers(self) -> Dict[int, ImageLayer]:
        return self._sess()._exposure_image_layer_cache

    # Filtering infrastructure

    def _copy_subset(self, keep, verbose: bool) -> "Exposures":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        return new

    @property
    def match(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` returning a boolean array identifying selected rows.

        Unlike many actions, this does not return a new `Exposures`. It can be used
        to implement arbitrary boolean logic within the action/selection framework::

            exp = sess.exposures()
            subset = exp.keep_only.where(
                exp.match.series("a") & exp.match.has_imaging()
            )
        """
        return ExposureSelector(self, lambda m: m)

    @property
    def count(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` returning the number of selected rows

        Unlike many actions, this returns an `int`, not a new `Exposures`.
        """
        return ExposureSelector(self, lambda m: m.sum())

    def _apply_keep_only(self, flags, verbose=True) -> "Exposures":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` returning an `Exposures` copy containing only the selected rows.
        """
        return ExposureSelector(self, self._apply_keep_only)

    def _apply_drop(self, flags, verbose=True) -> "Exposures":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` returning an `Exposures` copy dropping
        the selected rows; all non-selected rows are retained.
        """
        return ExposureSelector(self, self._apply_drop)

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
    def reject(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` modifying the exposure-list in-place,
        rejecting the selected rows.

        Usage is as follows::

            exp = sess.exposures()

            # Mark all exposures from meteor telescopes as rejected:
            exp.reject.meteor(tag="meteor")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.
        """
        return ExposureSelector(self, self._apply_reject)

    def _apply_reject_unless(self, flags, tag: str = None, verbose: bool = True):
        return self._apply_reject(~flags, tag, verbose)

    @property
    def reject_unless(self) -> ExposureSelector:
        """
        An :ref:`action <lc-filtering>` modifying the exposure-list in-place,
        rejecting rows not matching the selection.

        Usage is as follows::

            exp = sess.exposures()

            # Mark all exposures *not* from narrow-field telescopes as rejected:
            exp.reject_unless.narrow(tag="lowrez")

            # This is equivalent to:
            exp.reject.not_.narrow(tag="lowrez")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.
        """
        return ExposureSelector(self, self._apply_reject_unless)

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
        - ``"count"``: the number of exposures from the series in *self*
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
        Return a table with information about exposures that might contain "nice"
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
            columns will be only the ones relevant to inferring the exposure image
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

        # If we have an acceptable number of detections, consider only exposures
        # that detect the SOI. The cutoff here is totally arbitrary.
        if lc.count.nonrej_detected() > 10:
            lc = lc.keep_only.nonrej_detected(verbose=False)

        lc.sort(["limiting_mag_local"], reverse=True)

        if limit_cols:
            cols = "limiting_mag_local exp_local_id series platenum fwhm_world ellipticity background".split()
            lc = lc[cols]

        if limit_rows and limit_rows > 0:
            lc = lc[:limit_rows]

        return lc

    def time_coverage(self) -> figure:
        """
        Plot the observing time coverage of the non-rejected exposures in this list.

        Returns
        =======
        `bokeh.plotting.figure`
            A plot.

        Notes
        =====
        The plot is generated by using a Gaussian kernel density estimator to
        smooth out the exposure observation times.

        The function `bokeh.io.show` (imported as ``bokeh.plotting.show``) is
        called on the figure before it is returned, so you don't need to do that
        yourself.
        """
        from scipy.stats import gaussian_kde

        plot_years = np.linspace(1875, 1995, 200)
        exp_years = self.drop.rejected(verbose=False)["obs_date"].jyear

        kde = gaussian_kde(exp_years, bw_method="scott")
        exp_years_smoothed = kde(plot_years)

        # try to get the normalization right: integral of the
        # curve is equal to the number of exposures, so that the
        # Y axis is exposures per year.

        integral = exp_years_smoothed.sum() * (plot_years[1] - plot_years[0])
        exp_years_smoothed *= len(self) / integral

        p = figure(
            x_axis_label="Year",
            y_axis_label="Exposures per year (smoothed)",
        )
        p.line(plot_years, exp_years_smoothed)
        show(p)
        return p

    def show(self, exp_ref: ExposureReferenceType) -> ImageLayer:
        """
        Display the cutout of the specified exposure in the WWT view.

        Parameters
        ==========
        exp_ref : `ExposureRow` or `int`
            If this argument is an integer, it is interpreted as the "local ID"
            of an exposure. Local IDs are assigned in chronological order of
            observing time, so a value of ``0`` shows the first exposure. This will
            only work if the selected exposure is scanned and WCS-solved.

            If this argument is an instance of `ExposureRow`, the specified exposure
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
        exposure = self._sess()._resolve_exposure_reference(exp_ref)

        local_id = exposure["local_id"]
        il = self._layers().get(local_id)

        if il is not None:
            return il  # TODO: bring it to the top, etc?

        fits_relpath = self._sess().cutout(exposure)
        if fits_relpath is None:
            from . import InteractiveError

            raise InteractiveError(
                f"cannot create a cutout for exposure #{local_id:05d} ({exposure.exp_id()})"
            )

        il = (
            self._sess()
            .wwt()
            .layers.add_image_layer(str(self._sess().path(fits_relpath)))
        )
        self._layers()[local_id] = il
        return il

    def show_extract(self, exp_ref: ExposureReferenceType) -> TableLayer:
        """
        Display the catalog extract of the specified exposure in the WWT view.

        Parameters
        ==========
        exp_ref : `ExposureRow` or `int`
            If this argument is an integer, it is interpreted as the "local ID"
            of an exposure. Local IDs are assigned in chronological order of
            observing time, so a value of ``0`` shows the first exposure. This will
            only work if the selected exposure is scanned and WCS-solved.

            If this argument is an instance of `ExposureRow`, the specified exposure
            is shown.

        Returns
        =======
        `pywwt.layers.TableLayer`
          This is the WWT table layer object corresponding to the displayed catalog
          extract. You can use it to programmatically control aspects of how the
          data are displayed.

        Notes
        =====
        In order to use this method, you must first have called
        `daschlab.Session.connect_to_wwt()`. If needed, this method will execute
        an API call and download the data to be displayed, which may be slow.

        This method is equivalent to calling ``sess.extract(exp_ref).show()``.
        """
        return self._sess().extract(exp_ref).show()

    def export(self, path: str, format: str = "csv", drop_rejects: bool = True):
        """
        Save a copy of this exposure list to a file.

        Parameters
        ==========
        path : `str`
            The path at which the exported exposure list data will be saved.
        format : `str`, default ``"csv"``
            The file format in which to save the data. Currently, the only
            allowed value is ``"csv"``.
        drop_rejects : `bool`, default `True`
            If True, rejected exposures will not appear in the output file at all.
            Otherwise, they will be included, and there will be a ``"reject"``
            column reproducing their rejection status.

        Notes
        =====
        The default ``"csv"`` output format only exports a subset of the table columns
        known as the “medium” subset. It is defined in `the exposure-list table documentation`_.

        .. _the exposure-list table documentation: https://dasch.cfa.harvard.edu/dr7/exposurelist-columns/#medium-columns
        """

        if format != "csv":
            raise ValueError(f"illegal `format`: {format!r}")

        if drop_rejects:
            elc = self.drop.rejected(verbose=False)
            del elc["reject"]
        else:
            elc = self.copy(True)

        pos_index = list(elc.columns).index("pos")
        ra_deg = self["pos"].ra.deg
        pos_mask = ~np.isfinite(ra_deg)
        elc.add_columns(
            (
                np.ma.array(ra_deg, mask=pos_mask),
                np.ma.array(self["pos"].dec.deg, mask=pos_mask),
            ),
            indexes=[pos_index, pos_index],
            names=["ra_deg", "dec_deg"],
            copy=False,
        )
        del elc["pos"]

        DEL_COLS = [
            "local_id",
            "mos_date",
            "scannum",
            "scan_date",
        ]

        for c in DEL_COLS:
            del elc[c]

        elc.write(path, format="ascii.csv", overwrite=True)

    def export_cutouts_to_pdf(self, pdfpath: str, **kwargs):
        """
        Export cutouts of the exposures in this list to a PDF file.

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
        no_download : optional `bool`, default False
            If set to true, no cutouts will be downloaded. Only cutouts
            that are already cached locally will be used to construct the PDF.

        Notes
        =====
        This operation will by default attempt to download a cutout for every
        exposure with imaging in the list. This can be extremely slow, as well as
        creating a lot of work for the API server. Only use it for short exposure
        lists.

        The format of the created file needs to be documented.
        """
        _pdf_export(pdfpath, self._sess(), self, **kwargs)


def _query_exposures(
    client: ApiClient,
    center: SkyCoord,
) -> Exposures:
    return _postproc_exposures(_get_exposure_cols(client, center))


def _get_exposure_cols(
    client: ApiClient,
    center: SkyCoord,
) -> Exposures:
    payload = {
        "ra_deg": center.ra.deg,
        "dec_deg": center.dec.deg,
    }

    colnames = None
    coltypes = None
    coldata = None

    data = client.invoke("/dasch/dr7/queryexps", payload)
    if not isinstance(data, list):
        from . import InteractiveError

        raise InteractiveError(f"queryexps API request failed: {data!r}")

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

    if colnames is None:
        raise Exception("empty exposure-table data response")

    return dict(t for t in zip(colnames, coldata) if t[1] is not None)


def _postproc_exposures(input_cols) -> Exposures:
    table = Exposures(masked=True)

    scannum = np.array(input_cols["scannum"], dtype=np.int8)
    scanned_mask = scannum == -1

    # unitless column, masked by scanned status:
    smc = lambda c, dt: np.ma.array(input_cols[c], mask=scanned_mask, dtype=dt)

    # unitless column, with infinities/NaNs masked
    def finite_mc(c, dt):
        a = np.array(input_cols[c], dtype=dt)
        return np.ma.array(a, mask=~np.isfinite(a))

    # unitless column, with negative values masked
    def nonneg_mc(c, dt):
        a = np.array(input_cols[c], dtype=dt)
        return np.ma.array(a, mask=(a < 0))

    # quantity (unit) column, with NaNs masked
    def nan_mq(c, dt, unit):
        a = np.array(input_cols[c], dtype=dt)
        return Masked(u.Quantity(a, unit), ~np.isfinite(a))

    table["series"] = input_cols["series"]
    table["platenum"] = np.array(input_cols["platenum"], dtype=np.uint32)
    table["scannum"] = smc("scannum", np.int8)
    table["mosnum"] = smc("mosnum", np.int8)
    table["expnum"] = nonneg_mc("expnum", np.int8)
    table["solnum"] = smc("solnum", np.int8)
    table["class"] = input_cols["class"]
    table["exptime"] = nan_mq("exptime", np.float32, u.minute)
    table["obs_date"] = dasch_isots_as_time_array(input_cols["expdate"])
    table["wcssource"] = input_cols["wcssource"]
    table["scan_date"] = dasch_isots_as_time_array(input_cols["scandate"])
    table["mos_date"] = dasch_isots_as_time_array(input_cols["mosdate"])
    table["center_distance"] = (
        np.array(input_cols["centerdist"], dtype=np.float32) * u.cm
    )
    table["edge_distance"] = np.array(input_cols["edgedist"], dtype=np.float32) * u.cm
    table["lim_mag_apass"] = nan_mq("limMagApass", np.float32, u.mag)
    table["lim_mag_atlas"] = nan_mq("limMagAtlas", np.float32, u.mag)
    table["n_solutions_apass"] = nonneg_mc("nSolutionsApass", np.int8)
    table["n_magdep_apass"] = nonneg_mc("nMagdepApass", np.int8)
    table["median_colorterm_apass"] = finite_mc("medianColortermApass", np.float32)
    table["result_id_apass"] = input_cols["resultIdApass"]
    table["n_solutions_atlas"] = nonneg_mc("nSolutionsAtlas", np.int8)
    table["n_magdep_atlas"] = nonneg_mc("nMagdepAtlas", np.int8)
    table["median_colorterm_atlas"] = finite_mc("medianColortermAtlas", np.float32)
    table["result_id_atlas"] = input_cols["resultIdAtlas"]

    # Some exposures are missing position data, flagged with 999/99's.
    ra = np.array(input_cols["ra"])
    dec = np.array(input_cols["dec"])
    bad_pos = (ra == 999) | (dec == 99)
    ra[bad_pos] = np.nan
    dec[bad_pos] = np.nan
    table["pos"] = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
    )

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
    exposures: Exposures,
    refcat_stdmag_limit: Optional[float] = None,
    no_download: bool = False,
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

    n_orig = len(exposures)
    exposures = exposures.keep_only.has_imaging(verbose=False).drop.rejected(
        verbose=False
    )
    n_filtered = len(exposures)

    if not n_filtered:
        raise Exception("no exposures remain after filtering")

    if n_filtered != n_orig:
        print(
            f"Filtered input list of {n_orig} exposures down to {n_filtered} for display"
        )

    if not no_download:
        print(
            f"Ensuring that we have cutouts for {n_filtered} exposures, this may take a while ..."
        )
        t0 = time.time()

        for exposure in exposures:
            if not sess.cutout(exposure):
                raise Exception(
                    f"unable to fetch cutout for exposure LocalId {exposure['local_id']}"
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
        for exposure in exposures:
            local_id = exposure["local_id"]
            series = exposure["series"]
            platenum = exposure["platenum"]
            plateclass = exposure["class"] or "(none)"
            jd = exposure["obs_date"].jd
            epoch = exposure["obs_date"].jyear
            # platescale = exposure["plate_scale"]
            exptime = exposure["exptime"]
            center_distance_cm = exposure["center_distance"]
            edge_distance_cm = exposure["edge_distance"]

            fits_relpath = sess.cutout(exposure, no_download=no_download)
            if fits_relpath is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with fits.open(str(sess.path(fits_relpath))) as hdul:
                    data = hdul[0].data

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

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(
                f"Exposure {exposure.exp_id():>14}: "
                f"class {plateclass:12} "
                f"exp {exptime:5.1f} (min) "
                # f"scale {platescale:7.2f} (arcsec/mm)"
            )
            linenum += 1

            # XXX hardcoding params
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(f"Image: center {center_text}")
            linenum += 1
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text("  halfsize 600 arcsec, orientation N up, E left")
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            url = f"https://starglass.cfa.harvard.edu/plate/{series}{platenum:05d}"
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
        f"... completed in {elapsed:.0f} seconds and saved {n_pages} pages to `{pdfpath}` ({info.st_size / 1024**2:.1f} MiB)"
    )
