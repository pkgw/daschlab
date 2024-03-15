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
logic can be implemented with the `~LightcurveSelector.where` selection and the
`~Lightcurve.match` action.

In terms of implementation, an item like ``lc.keep_only`` is a property that
returns a helper `LightcurveSelector` object, which is what actually implements
methods such as `~LightcurveSelector.brighter()`. The selector object helps pair
the desired action with the desired selection while maintaining a convenient
code syntax.

DASCH lightcurves often contain many nondetections (upper limits), which means
that you may need to be careful about selection logic. For instance::

    lc = sess.lightcurve(some_local_id)
    n1 = lc.count.brighter(13)
    n2 = lc.count.not_.detected_and_fainter(13)

    # This will NOT generally hold, because n2 will include
    # nondetections while n1 will not.
    assert n1 == n2
"""

from enum import IntFlag
from urllib.parse import urlencode
from typing import Dict, Optional, Tuple
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import Table, Row
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy import units as u
from astropy.utils.masked import Masked
from bokeh.plotting import figure, show
import numpy as np
import requests

from .series import SERIES, SeriesKind

__all__ = [
    "AFlags",
    "BFlags",
    "Lightcurve",
    "LightcurvePoint",
    "LightcurveSelector",
    "LocalBinRejectFlags",
    "PlateQualityFlags",
]


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/querylc.php"


_COLTYPES = {
    # "REFNumber": int,
    "X_IMAGE": float,
    "Y_IMAGE": float,
    "MAG_ISO": float,
    "ra": float,
    "dec": float,
    "magcal_iso": float,
    "magcal_iso_rms": float,
    "magcal_local": float,
    "magcal_local_rms": float,
    "Date": float,
    "FLUX_ISO": float,
    "MAG_APER": float,
    "MAG_AUTO": float,
    "KRON_RADIUS": float,
    "BACKGROUND": float,
    "FLUX_MAX": float,
    "THETA_J2000": float,
    "ELLIPTICITY": float,
    "ISOAREA_WORLD": float,
    "FWHM_IMAGE": float,
    "FWHM_WORLD": float,
    "plate_dist": float,
    "Blendedmag": float,
    "limiting_mag_local": float,
    "magcal_local_error": float,
    "dradRMS2": float,
    "magcor_local": float,
    "extinction": float,
    "gsc_bin_index": int,
    "series": str,
    "plateNumber": int,
    "NUMBER": int,
    "versionId": int,
    "AFLAGS": int,
    "BFLAGS": int,
    "ISO0": int,
    "ISO1": int,
    "ISO2": int,
    "ISO3": int,
    "ISO4": int,
    "ISO5": int,
    "ISO6": int,
    "ISO7": int,
    "npoints_local": int,
    "rejectFlag": int,
    "local_bin_index": int,
    "seriesId": int,
    "exposureNumber": int,
    "solutionNumber": int,
    "spatial_bin": int,
    "mosaicNumber": int,
    "quality": int,
    "plateVersionId": int,
    "magdep_bin": int,
    "magcal_magdep": float,
    "magcal_magdep_rms": float,
    "ra_2": float,
    "dec_2": float,
    "RaPM": float,
    "DecPM": float,
    "A2FLAGS": int,
    "B2FLAGS": int,
    "timeAccuracy": float,
    "maskIndex": int,
    # "catalogNumber": int,
}


class AFlags(IntFlag):
    """
    DASCH photometry data quality warning flags.

    The "AFLAGS" value is an integer where each bit indicates a data quality
    concern. The highest-quality data have no bits set, i.e., an integer value
    of zero.

    **Flag documentation is intentionally superficial.** Flag semantics are
    (**FIXME: will be**) documented more thoroughly in the DASCH data
    description pages.
    """

    HIGH_BACKGROUND = 1 << 6  # NB, this is "bit 7" since we count them 1-based
    "Bit 7: High SExtractor background level at object position"

    BAD_PLATE_QUALITY = 1 << 7
    "Bit 8: Plate fails general quality checks"

    MULT_EXP_UNMATCHED = 1 << 8
    "Bit 9: Object is unmatched and this is a multiple-exposure plate"

    UNCERTAIN_DATE = 1 << 9
    "Bit 10: Observation time is too uncertain to calculate extinction accurately"

    MULT_EXP_BLEND = 1 << 10
    "Bit 11: Object is a blend and this is a multiple-exposure plate"

    LARGE_ISO_RMS = 1 << 11
    "Bit 12: SExtractor isophotonic RMS is suspiciously large"

    LARGE_LOCAL_SMOOTH_RMS = 1 << 12
    "Bit 13: Local-binning RMS is suspiciously large"

    CLOSE_TO_LIMITING = 1 << 13
    "Bit 14: Object brightness is too close to the local limiting magnitude"

    RADIAL_BIN_9 = 1 << 14
    "Bit 15: Object is in radial bin 9 (close to the plate edge)"

    BIN_DRAD_UNKNOWN = 1 << 15
    "Bit 16: Object's spatial bin has unmeasured ``drad``"

    UNCERTAIN_CATALOG_MAG = 1 << 19
    "Bit 20: Magnitude of the catalog source is uncertain/variable"

    CASE_B_BLEND = 1 << 20
    'Bit 21: "Case B" blend - multiple catalog entries for one imaged star'

    CASE_C_BLEND = 1 << 21
    'Bit 22: "Case C" blend - mutiple imaged stars for one catalog entry'

    CASE_BC_BLEND = 1 << 22
    'Bit 23: "Case B/C" blend  - multiple catalog entries and imaged stars all mixed up'

    LARGE_DRAD = 1 << 23
    "Bit 24: Object ``drad`` is large relative to its bin, or its spatial or local bin is bad"

    PICKERING_WEDGE = 1 << 24
    "Bit 25: Object is a Pickering Wedge image"

    SUSPECTED_DEFECT = 1 << 25
    "Bit 26: Object is a suspected plate defect"

    SXT_BLEND = 1 << 26
    "Bit 27: SExtractor flags the object as a blend"

    REJECTED_BLEND = 1 << 27
    "Bit 28: Rejected blended object"

    LARGE_SMOOTHING_CORRECTION = 1 << 28
    "Bit 29: Smoothing correction is suspiciously large"

    TOO_BRIGHT = 1 << 29
    "Bit 30: Object is too bright for accurate calibration"

    LOW_ALTITUDE = 1 << 30
    "Bit 31: Low altitude - object is within 23.5 deg of the horizon"


_AFLAG_DESCRIPTIONS = [
    "(bit 1 unused)",
    "(bit 2 unused)",
    "(bit 3 unused)",
    "(bit 4 unused)",
    "(bit 5 unused)",
    "(bit 6 unused)",
    "High SExtractor background level at object position",
    "Plate fails general quality checks",
    "Object is unmatched and this is a multiple-exposure plate",
    "Observation time is too uncertain to calculate extinction accurately",
    "Object is a blend and this is a multiple-exposure plate",
    "SExtractor isophotonic RMS is suspiciously large",
    "Local-binning RMS is suspiciously large",
    "Object brightness is too close to the local limiting magnitude",
    "Object is in radial bin 9 (close to the plate edge)",
    "Object's spatial bin has unmeasured `drad`",
    "(bit 17 unused)",
    "(bit 18 unused)",
    "(bit 19 unused)",
    "Magnitude of the catalog source is uncertain/variable",
    '"Case B" blend - multiple catalog entries for one imaged star',
    '"Case C" blend - mutiple imaged stars for one catalog entry',
    '"Case B/C" blend  - multiple catalog entries and imaged stars all mixed up',
    "Object `drad` is large relative to its bin, or its spatial or local bin is bad",
    "Object is a Pickering Wedge image",
    "Object is a suspected plate defect",
    "SExtractor flags the object as a blend",
    "Rejected blended object",
    "Smoothing correction is suspiciously large",
    "Object is too bright for accurate calibration",
    "Low altitude - object is within 23.5 deg of the horizon",
]


class BFlags(IntFlag):
    """
    DASCH photometry data processing flags.

    The "BFLAGS" value is an integer where each bit indicates something about
    the data processing of this photometric point. Unlike the "AFLAGS", these
    are not necessarily "good" or "bad".

    **Flag documentation is intentionally superficial.** Flag semantics are
    (**FIXME: will be**) documented more thoroughly in the DASCH data
    description pages.
    """

    NEIGHBORS = 1 << 0
    "Bit 1: Object has nearby neighbors"

    BLEND = 1 << 1
    "Bit 2: Object was blended with another"

    SATURATED = 1 << 2
    "Bit 3: At least one image pixel was saturated"

    NEAR_BOUNDARY = 1 << 3
    "Bit 4: Object is too close to the image boundary"

    APERTURE_INCOMPLETE = 1 << 4
    "Bit 5: Object aperture data incomplete or corrupt"

    ISOPHOT_INCOMPLETE = 1 << 5
    "Bit 6: Object isophotal data incomplete or corrupt"

    DEBLEND_OVERFLOW = 1 << 6
    "Bit 7: Memory overflow during deblending"

    EXTRACTION_OVERFLOW = 1 << 7
    "Bit 8: Memory overflow during extraction"

    CORRECTED_FOR_BLEND = 1 << 8
    "Bit 9: Magnitude corrected for blend"

    LARGE_BIN_DRAD = 1 << 9
    "Bit 10: Object ``drad`` is low, but its bin ``drad`` is large"

    PSF_SATURATED = 1 << 10
    "Bit 11: Object PSF considered saturated"

    MAG_DEP_CAL_APPLIED = 1 << 11
    "Bit 12: Magnitude-dependent calibration has been applied (this is good)"

    GOOD_STAR = 1 << 16
    "Bit 17: Appears to be a good star (this is good)"

    LOWESS_CAL_APPLIED = 1 << 17
    "Bit 18: Lowess calibration has been applied (this is good)"

    LOCAL_CAL_APPLIED = 1 << 18
    "Bit 19: Local calibration has been applied (this is good)"

    EXTINCTION_CAL_APPLIED = 1 << 19
    "Bit 20: Extinction calibration has been applied (this is good)"

    TOO_BRIGHT = 1 << 20
    "Bit 21: Object is too bright to calibrate"

    COLOR_CORRECTION_APPLIED = 1 << 21
    "Bit 22: Color correction has been applied (this is good)"

    COLOR_CORRECTION_USED_METROPOLIS = 1 << 22
    "Bit 23: Color correction used the Metropolis algorithm (this is good)"

    LATE_CATALOG_MATCH = 1 << 24
    "Bit 25: Object was only matched to catalog at the end of the pipeline"

    LARGE_PROMO_UNCERT = 1 << 25
    "Bit 26: Object has high proper motion uncertainty"

    LARGE_SPATIAL_BIN_COLORTERM = 1 << 27
    "Bit 28: Spatial bin color-term calibration fails quality check"

    POSITION_ADJUSTED = 1 << 28
    "Bit 29: RA/Dec have been adjusted by bin medians"

    LARGE_JD_UNCERT = 1 << 29
    "Bit 30: Plate date is uncertain"

    PROMO_APPLIED = 1 << 30
    "Bit 31: Catalog position has been corrected for proper motion (this is good)"


_BFLAG_DESCRIPTIONS = [
    "Object has nearby neighbors",
    "Object was blended with another",
    "At least one image pixel was saturated",
    "Object is too close to the image boundary",
    "Object aperture data incomplete or corrupt",
    "Object isophotal data incomplete or corrupt",
    "Memory overflow during deblending",
    "Memory overflow during extraction",
    "Magnitude corrected for blend",
    "Object `drad` is low, but its bin `drad` is large",
    "Object PSF considered saturated",
    "Magnitude-dependent calibration has been applied",
    "(bit 12 unused)",
    "(bit 13 unused)",
    "(bit 14 unused)",
    "(bit 15 unused)",
    "Appears to be a good star",
    "Lowess calibration has been applied",
    "Local calibration has been applied",
    "Extinction calibration has been applied",
    "Object is too bright to calibrate",
    "Color correction has been applied",
    "Color correction used the Metropolis algorithm",
    "(bit 23 unused)",
    "Object was only matched to catalog at the end of the pipeline",
    "Object has high proper motion uncertainty",
    "(bit 26 unused)",
    "Spatial bin color-term calibration fails quality check",
    "RA/Dec have been adjusted by bin medians",
    "Plate date is uncertain",
    "Catalog position has been corrected for proper motion",
]


class LocalBinRejectFlags(IntFlag):
    """
    DASCH photometry data processing warning flags regarding the "local bin"
    calibration.

    The ``local_bin_reject_flag`` value is an integer where each bit indicates
    something about the local-bin calibration of this photometric point. The
    highest-quality data have no bits set, i.e., an integer value of zero.

    **Flag documentation is intentionally superficial.** Flag semantics are
    (**FIXME: will be**) documented more thoroughly in the DASCH data
    description pages.
    """

    LARGE_CORRECTION = 1 << 0
    "Bit 1: Local correction is out of range"

    TOO_DIM = 1 << 1
    "Bit 2: Median brightness is below the limiting magnitude"

    LARGE_DRAD = 1 << 2
    "Bit 3: Majority of stars have a high ``drad``"


_LOCAL_BIN_REJECT_FLAG_DESCRIPTIONS = [
    "Local correction is out of range",
    "Median brightness is below the limiting magnitude",
    "Majority of stars have a high `drad`",
]


class PlateQualityFlags(IntFlag):
    """
    DASCH photometry data processing warning flags based on plate-level
    diagnostics.

    The ``plate_quality_flag`` value is an integer where each bit indicates
    something about the quality of the plate associated with this photometric
    point. In most but not all cases, the "good" flag setting is zero.

    **Flag documentation is intentionally superficial.** Flag semantics are
    (**FIXME: will be**) documented more thoroughly in the DASCH data
    description pages.
    """

    MULTIPLE_EXPOSURE = 1 << 0
    "Bit 1: Multiple-exposure plate"

    GRATING = 1 << 1
    "Bit 2: Grating plate"

    COLORED_FILTER = 1 << 2
    "Bit 3: Color-filter plate"

    COLORTERM_OUT_OF_BOUNDS = 1 << 3
    "Bit 4: Image fails colorterm limits"

    PICKERING_WEDGE = 1 << 4
    "Bit 5: Pickering Wedge plate"

    SPECTRUM = 1 << 5
    "Bit 6: Spectrum plate"

    SATURATED = 1 << 6
    "Bit 7: Saturated images"

    MAG_DEP_CAL_UNAVAILABLE = 1 << 7
    "Bit 8: Magnitude-dependent calibration unavailable"

    PATROL_TELESCOPE = 1 << 8
    "Bit 9: Patrol-telescope plate"

    NARROW_TELESCOPE = 1 << 9
    "Bit 10: Narrow-field-telescope plate (this is good!)"

    SHOW_LIMITING = 1 << 10
    "Bit 11: (internal only? plotter shows limiting magnitudes)"

    SHOW_UNDETECTED = 1 << 11
    "Bit 12: (internal only? plotter shows nondetections)"

    TRAILED = 1 << 12
    "Bit 13: Image is trailed -- ellipticity > 0.6"


_PLATE_QUALITY_FLAG_DESCRIPTIONS = [
    "Multiple-exposure plate",
    "Grating plate",
    "Color-filter plate",
    "Image fails colorterm limits",
    "Pickering Wedge plate",
    "Spectrum plate",
    "Saturated images",
    "Magnitude-dependent calibration unavailable",
    "Patrol-telescope plate",
    "Narrow-field-telescope plate",
    "(plotter shows limiting magnitudes)",
    "(plotter shows nondetections)",
    "Image is trailed -- ellipticity > 0.6",
]


def _report_flags(desc, observed, enumtype, descriptions):
    print(f"{desc}: 0x{observed:08X}")

    if not observed:
        return

    accum = 0

    for iflag in enumtype:
        if observed & iflag.value:
            bit = int(np.log2(iflag.value)) + 1
            print(f"  bit {bit:2d}: {iflag.name:32} - {descriptions[bit - 1]}")
            accum |= iflag.value

    if accum != observed:
        print(f"  !! unexpected leftover bits: {observed & ~accum:08X}")


class LightcurvePoint(Row):
    """
    A single row from a `Lightcurve` table.

    You do not need to construct these objects manually. Indexing a `Lightcurve`
    table with a single integer will yield an instance of this class, which is a
    subclass of `astropy.table.Row`.
    """

    def flags(self):
        """
        Print out a textual summary of the quality flags associated with this
        row.

        Notes
        =====
        The output from this method summarizes the contents of this row's
        `AFlags`, `BFlags`, `LocalBinRejectFlags`, and `PlateQualityFlags`.
        """

        _report_flags("AFLAGS", self["aflags"], AFlags, _AFLAG_DESCRIPTIONS)
        _report_flags("BFLAGS", self["bflags"], BFlags, _BFLAG_DESCRIPTIONS)
        _report_flags(
            "LocalBinReject",
            self["local_bin_reject_flag"],
            LocalBinRejectFlags,
            _LOCAL_BIN_REJECT_FLAG_DESCRIPTIONS,
        )
        _report_flags(
            "PlateQuality",
            self["plate_quality_flag"],
            PlateQualityFlags,
            _PLATE_QUALITY_FLAG_DESCRIPTIONS,
        )


class LightcurveSelector:
    """
    A helper object that supports `Lightcurve` filtering functionality.

    Lightcurve selector objects are returned by lightcurve selection "action
    verbs" such as `Lightcurve.keep_only`. Calling one of the methods on a
    selector instance will apply the associated action to the specified portion
    of the lightcurve data.
    """

    _lc: "Lightcurve"
    _apply = None

    def __init__(self, lc: "Lightcurve", apply):
        self._lc = lc
        self._apply = apply

    def _apply_not(self, flags, **kwargs):
        "This isn't a lambda because it needs to accept and relay **kwargs"
        return self._apply(~flags, **kwargs)

    @property
    def not_(self) -> "LightcurveSelector":
        """
        Get a selector that will act on an inverted row selection.

        Examples
        ========
        Create a lightcurve subset containing nondetections and detections that
        are *not* within 10 arcsec of the mean source position::

            from astropy import units as u

            lc = sess.lightcurve(some_local_id)
            far = lc.keep_only.not_.sep_below(10 * u.arcsec)

        In general, the function of this modifier is such that::

            lc.ACTION.not_.CONDITION()
            # should be equivalent to:
            lc.ACTION.where(~lc.match.CONDITION())

        The logical negation implied by this method can cause problems if your
        lightcurve contains nondetections. For instance, the above logic will select
        both detections with measured separations above 10 arcsec, and nondetections
        where the separation is undefined. If you are using this selection to
        reject data points, the latter aspect may be undesirable.
        """
        return LightcurveSelector(self._lc, self._apply_not)

    def where(self, row_mask: np.ndarray, **kwargs) -> "Lightcurve":
        """
        Act on exactly the specified list of rows.

        Parameters
        ==========
        row_mask : boolean `numpy.ndarray`
            A boolean array of exactly the size of the input lightcurve, with true
            values indicating rows that should be acted upon.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points that are from A-series
        plates with detections brighter than 13th magnitude::

            lc = sess.lightcurve(some_local_id)
            subset = lc.keep_only.where(
                lc.match.series("a") & lc.match.brighter(13)
            )
        """
        return self._apply(row_mask, **kwargs)

    def detected(self, **kwargs) -> "Lightcurve":
        """
        Act on rows corresponding to detections.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only detections::

            lc = sess.lightcurve(some_local_id)
            detns = lc.keep_only.detected()
        """
        m = ~self._lc["magcal_magdep"].mask
        return self._apply(m, **kwargs)

    def undetected(self, **kwargs) -> "Lightcurve":
        """
        Act on rows corresponding to nondetections.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only nondetections::

            lc = sess.lightcurve(some_local_id)
            nondetns = lc.keep_only.undetected()
        """
        m = self._lc["magcal_magdep"].mask
        return self._apply(m, **kwargs)

    def rejected(self, **kwargs) -> "Lightcurve":
        """
        Act on rows with a non-zero ``"reject"`` value.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only rejected rows::

            lc = sess.lightcurve(some_local_id)
            rejects = lc.keep_only.rejected()
        """
        m = self._lc["reject"] != 0
        return self._apply(m, **kwargs)

    def rejected_with(self, tag: str, strict: bool = False, **kwargs) -> "Lightcurve":
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
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only rows rejected with the
        "astrom" tag:

            lc = sess.lightcurve(some_local_id)
            astrom_rejects = lc.keep_only.rejected_with("astrom")
        """
        bitnum0 = self._lc._rejection_tags().get(tag)

        if bitnum0 is not None:
            m = (self._lc["reject"] & (1 << bitnum0)) != 0
        elif strict:
            raise Exception(f"unknown rejection tag `{tag}`")
        else:
            m = np.zeros(len(self._lc), dtype=bool)

        return self._apply(m, **kwargs)

    def nonrej_detected(self, **kwargs) -> "Lightcurve":
        """
        Act on rows that are non-rejected detections.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only non-rejected detections::

            lc = sess.lightcurve(some_local_id)
            good = lc.keep_only.nonrej_detected()

        Notes
        =====
        This selector is a shorthand combination of
        `LightcurveSelector.detected()` and (a logical inversion of)
        `LightcurveSelector.rejected()`.
        """
        m = ~self._lc["magcal_magdep"].mask & (self._lc["reject"] == 0)
        return self._apply(m, **kwargs)

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
            the `Lightcurve.count` action will return an integer.

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
            pos = self._lc.mean_pos()

        seps = pos.separation(self._lc["pos"])
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
            the `Lightcurve.count` action will return an integer.

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
            pos = self._lc.mean_pos()

        seps = pos.separation(self._lc["pos"])
        m = seps > sep_limit
        return self._apply(m, **kwargs)

    def any_aflags(self, aflags: int, **kwargs) -> "Lightcurve":
        """
        Act on rows that have any of the specific AFLAGS bits set.

        Parameters
        ==========
        aflags : `int` or `AFlags`
            The flag or flags to check for. If this value contains multiple
            non-zero bits, a row will be selected if its AFLAGS contain
            *any* of the specified bits.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only rows with the specified flags::

            from astropy import units as u
            from daschlab.lightcurves import AFlags

            lc = sess.lightcurve(some_local_id)
            filter = AFlags.LARGE_DRAD | AFlags.RADIAL_BIN_9
            bad = lc.keep_only.any_aflags(filter)
        """
        m = (self._lc["aflags"] & aflags) != 0
        m = m.filled(False)  # do *not* affect rows where this column is masked
        return self._apply(m, **kwargs)

    def local_id(self, local_id: int, **kwargs) -> "Lightcurve":
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
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only the chronologically first
        row::

            lc = sess.lightcurve(some_local_id)
            first = lc.keep_only.local_id(0)

        Notes
        =====
        Lightcurve point local IDs are unique, and so this filter should only
        ever match at most one row.
        """
        m = self._lc["local_id"] == local_id
        return self._apply(m, **kwargs)

    def series(self, series: str, **kwargs) -> "Lightcurve":
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
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points from the MC series::

            lc = sess.lightcurve(some_local_id)
            mcs = lc.keep_only.series("mc")
        """
        m = self._lc["series"] == series.lower()
        return self._apply(m, **kwargs)

    def brighter(self, cutoff_mag: float, **kwargs) -> "Lightcurve":
        """
        Act on detections brighter than the specified cutoff magnitude.

        Parameters
        ==========
        cutoff_mag : `float`
            The cutoff magnitude.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points brighter than 13th
        magnitude::

            lc = sess.lightcurve(some_local_id)
            bright = lc.keep_only.brighter(13)

        Notes
        =====
        The cutoff is exclusive, i.e., the comparison is perform with a
        less-than rather than less-than-or-equals comparison.
        """
        m = self._lc["magcal_magdep"] < cutoff_mag
        m = m.filled(False)  # do *not* affect rows where this column is masked
        return self._apply(m, **kwargs)

    def detected_and_fainter(self, cutoff_mag: float, **kwargs) -> "Lightcurve":
        """
        Act on detections fainter than the specified cutoff magnitude.

        Parameters
        ==========
        cutoff_mag : `float`
            The cutoff magnitude.
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points fainter than 13th
        magnitude::

            lc = sess.lightcurve(some_local_id)
            faint = lc.keep_only.detected_and_fainter(13)

        Notes
        =====
        The cutoff is exclusive, i.e., the comparison is perform with a
        greater-than rather than greater-than-or-equals comparison.
        """
        m = self._lc["magcal_magdep"] > cutoff_mag
        m = m.filled(False)  # do *not* affect rows where this column is masked
        return self._apply(m, **kwargs)

    def narrow(self, **kwargs) -> "Lightcurve":
        """
        Act on rows associated with narrow-field telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points from narrow-field
        telescopes::

            lc = sess.lightcurve(some_local_id)
            narrow = lc.keep_only.narrow()
        """
        m = [SERIES[k].kind == SeriesKind.NARROW for k in self._lc["series"]]
        return self._apply(m, **kwargs)

    def patrol(self, **kwargs) -> "Lightcurve":
        """
        Act on rows associated with low-resolution "patrol" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points from patrol
        telescopes::

            lc = sess.lightcurve(some_local_id)
            patrol = lc.keep_only.patrol()
        """
        m = [SERIES[k].kind == SeriesKind.PATROL for k in self._lc["series"]]
        return self._apply(m, **kwargs)

    def meteor(self, **kwargs) -> "Lightcurve":
        """
        Act on rows associated with ultra-low-resolution "meteor" telescopes.

        Parameters
        ==========
        **kwargs
            Parameters forwarded to the action.

        Returns
        =======
        Usually, another `Lightcurve`
            However, different actions may return different types. For instance,
            the `Lightcurve.count` action will return an integer.

        Examples
        ========
        Create a lightcurve subset containing only points from meteor
        telescopes::

            lc = sess.lightcurve(some_local_id)
            meteor = lc.keep_only.meteor()
        """
        m = [SERIES[k].kind == SeriesKind.METEOR for k in self._lc["series"]]
        return self._apply(m, **kwargs)


class Lightcurve(TimeSeries):
    """
    A DASCH lightcurve data table.

    A `Lightcurve` is a subclass of `astropy.timeseries.TimeSeries`, which in
    turn is a subclass of `astropy.table.Table`. It contains DASCH lightcurve
    data and associated lightcurve-specific methods. You can use all of the
    usual methods and properties made available by the
    `astropy.timeseries.TimeSeries` and `astropy.table.Table` classes. Items
    provided by those classes are not documented here.

    You should not construct `Lightcurve` instances directly. Instead, obtain
    lightcurves using the `daschlab.Session.lightcurve()` method.

    **Columns are not documented here!** They are (**FIXME: will be**)
    documented more thoroughly in the DASCH data description pages.

    See :ref:`the module-level documentation <lc-filtering>` for a summary of
    the filtering and subsetting functionality provided by this class.

    Cheat sheet:

    - ``time`` is HJD midpoint
    - ``magcal_magdep`` is preferred calibrated phot measurement
    - legacy plotter error bar is ``magcal_local_rms` * `error_bar_factor``, the
      latter being set to match the error bars to the empirical RMS, if this
      would shrink the error bars
    """

    Row = LightcurvePoint

    def _rejection_tags(self) -> Dict[str, int]:
        return self.meta.setdefault("daschlab_rejection_tags", {})

    # Filtering utilities

    def _copy_subset(self, keep, verbose: bool) -> "Lightcurve":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        return new

    @property
    def match(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` returning a boolean array identifying selected rows.

        Unlike many actions, this does not return a new `Lightcurve`. It can be used
        to implement arbitrary boolean logic within the action/selection framework::

            lc = sess.lightcurve(some_local_id)
            subset = lc.keep_only.where(
                lc.match.series("a") & lc.match.brighter(13)
            )
        """
        return LightcurveSelector(self, lambda m: m)

    @property
    def count(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` returning the number of selected rows

        Unlike many actions, this returns an `int`, not a new `Lightcurve`
        """
        return LightcurveSelector(self, lambda m: m.sum())

    def _apply_keep_only(self, flags, verbose=True) -> "Lightcurve":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` returning a `Lightcurve` copy containing only the selected rows.
        """
        return LightcurveSelector(self, self._apply_keep_only)

    def _apply_drop(self, flags, verbose=True) -> "Lightcurve":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` returning a `Lightcurve` copy dropping
        the selected rows; all non-selected rows are retained.
        """
        return LightcurveSelector(self, self._apply_drop)

    def _apply_split_by(self, flags) -> Tuple["Lightcurve", "Lightcurve"]:
        lc_left = self._copy_subset(flags, False)
        lc_right = self._copy_subset(~flags, False)
        return (lc_left, lc_right)

    @property
    def split_by(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` returning two `Lightcurve` copies,
        partitioning by the selection.

        Unlike many actions, this returns a tuple of two new `Lightcurve`
        instances, not just one. The first returned instance contains only
        the selected points; the second instance returns all others. This
        can be useful for statistical comparisons::

            from astropy import units as u

            lc = sess.lightcurve(some_local_id)
            neardet, fardet = lc.keep_only.nonrej_detected().split_by.sep_below(20 * u.arcsec)
            near_std = neardet["magcal_magdep"].std()
            far_std = neardet["magcal_magdep"].std()

        """
        return LightcurveSelector(self, self._apply_split_by)

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
    def reject(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` modifying the lightcurve in-place,
        rejecting the selected rows.

        Usage is as follows::

            lc = sess.lightcurve(some_local_id)

            # Mark all points from meteor telescopes as rejected:
            lc.reject.meteor(tag="meteor")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.
        """
        return LightcurveSelector(self, self._apply_reject)

    def _apply_reject_unless(self, flags, tag: str = None, verbose: bool = True):
        return self._apply_reject(~flags, tag, verbose)

    @property
    def reject_unless(self) -> LightcurveSelector:
        """
        An :ref:`action <lc-filtering>` modifying the lightcurve in-place,
        rejecting rows not matching the selection.

        Usage is as follows::

            lc = sess.lightcurve(some_local_id)

            # Mark all points *not* from narrow-field telescopes as rejected:
            lc.reject_unless.narrow(tag="lowrez")

            # This is equivalent to:
            lc.reject.not_.narrow(tag="lowrez")

        The ``tag`` keyword argument to the selector is mandatory. It specifies
        a short, arbitrary "tag" documenting the reason for rejection. Each
        unique tag is associated with a binary bit of the ``"reject"`` column,
        and these bits are logically OR'ed together as rejections are established.
        The maximum number of distinct rejection tags is 64, since the ``"reject"``
        column is stored as a 64-bit integer.

        The logical negation used by this function can cause problems if your
        lightcurve contains nondetections. For instance, if you intend to reject
        points *unless* they are brighter than a mag of 15.0, you will also reject
        all of the nondetections, even shallow upper limits consistent with that
        cutoff.
        """
        return LightcurveSelector(self, self._apply_reject_unless)

    def summary(self):
        """
        Print a brief textual summary of the lightcurve contents.
        """
        print(f"Total number of rows: {len(self)}")

        nonrej = self.drop.rejected(verbose=False)
        print(f"Number of rejected rows: {len(self) - len(nonrej)}")

        detns = nonrej.keep_only.detected(verbose=False)
        print(f"Number of unrejected detections: {len(detns)}")

        if len(detns):
            mm = detns["magcal_magdep"].mean()
            rm = ((detns["magcal_magdep"] - mm) ** 2).mean() ** 0.5
            print(f"Mean/RMS mag: {mm:.3f} ± {rm:.3f}")

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

    def plot(self, x_axis="year") -> figure:
        """
        Plot the lightcurve using default settings.

        Parameters
        ==========
        x_axis : optional `str`, default ``"year"``
            The name of the column to use for the X axis. ``"year"`` is a
            synthetic column calculated on-the-fly from the ``"time"`` column,
            corresponding to the ``jyear`` property of the `astropy.time.Time`
            object.

        Returns
        =======
        `bokeh.plotting.figure`
            A plot.

        Notes
        =====
        The function `bokeh.io.show` (imported as ``bokeh.plotting.show``) is
        called on the figure before it is returned, so you don't need to do that
        yourself.
        """
        detect, limit = self.drop.rejected(verbose=False).split_by.detected()

        with warnings.catch_warnings():
            # Shush ERFA warnings about dubious years -- if we don't change the
            # `time` column to not be an Astropy Time, the warnings come out in
            # the `to_pandas()` call(s).
            warnings.simplefilter("ignore")

            time = detect["time"]
            detect["time"] = time.jd
            detect["year"] = time.jyear

            time = limit["time"]
            limit["time"] = time.jd
            limit["year"] = time.jyear

        p = figure(
            tools="pan,wheel_zoom,box_zoom,reset,hover",
            tooltips=[
                ("LocalID", "@local_id"),
                ("Mag.", "@magcal_magdep"),
                ("Epoch", "@year"),
                ("Plate", "@series@platenum mosnum @mosnum pl_loc_id @plate_local_id"),
            ],
        )

        if len(limit):
            p.inverted_triangle(
                x_axis,
                "limiting_mag_local",
                fill_color="lightgray",
                line_color=None,
                source=limit.to_pandas(),
            )

        if len(detect):
            p.scatter(x_axis, "magcal_magdep", source=detect.to_pandas())

        p.y_range.flipped = True
        show(p)
        return p

    def scatter(self, x_axis: str, y_axis: str, rejects: bool = False) -> figure:
        """
        Make a Bokeh scatter plot of lightcurve data.

        Parameters
        ==========
        x_axis : `str`
            The name of the column to use for the X axis.
        y_axis : `str`
            The name of the column to use for the Y axis.
        rejects : optional `bool`, default `False`
            Whether to include rejected points in the resulting plot.

        Returns
        =======
        `bokeh.plotting.figure`
            A plot.

        Notes
        =====
        The function `bokeh.io.show` (imported as ``bokeh.plotting.show``) is
        called on the figure before it is returned, so you don't need to do that
        yourself.
        """
        table = self

        if not rejects:
            table = table.drop.rejected()

        p = figure(
            tools="pan,wheel_zoom,box_zoom,reset,hover",
            tooltips=[
                ("LocalID", "@local_id"),
                ("Mag.", "@magcal_magdep"),
                ("Epoch", "@year"),
                ("Plate", "@series@platenum mosnum @mosnum pl_loc_id @plate_local_id"),
            ],
        )

        p.scatter(x_axis, y_axis, source=table.to_pandas())

        if y_axis == "magcal_magdep":
            p.y_range.flipped = True

        show(p)
        return p


def _query_lc(
    refcat: str,
    name: str,
    gsc_bin_index: int,
) -> Lightcurve:
    return _postproc_lc(_get_lc_cols(refcat, name, gsc_bin_index))


def _get_lc_cols(
    refcat: str,
    name: str,
    gsc_bin_index: int,
) -> dict:
    url = (
        _API_URL
        + "?"
        + urlencode(
            {
                "refcat": refcat,
                "name": name,
                "gsc_bin_index": gsc_bin_index,
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

    if colnames is None or not saw_sep:
        raise Exception("incomplete lightcurve data response")

    return dict(t for t in zip(colnames, coldata) if t[1] is not None)


def _postproc_lc(input_cols) -> Lightcurve:
    table = Lightcurve(masked=True)

    gsc_bin_index = np.array(input_cols["gsc_bin_index"], dtype=np.uint32)
    sxt_number = np.array(input_cols["NUMBER"], dtype=np.uint32)
    mask = (gsc_bin_index == 0) & (sxt_number == 0)

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

    # Columns are displayed in the order that they're added to the table,
    # so we try to register the most important ones first.

    # this must be the first column:
    table["time"] = Time(input_cols["Date"], format="jd")
    # this will be filled in for real at the end:
    table["local_id"] = np.zeros(len(gsc_bin_index))
    table["magcal_magdep"] = mq("magcal_magdep", np.float32, u.mag)
    table["magcal_magdep_rms"] = extra_mq("magcal_magdep_rms", np.float32, u.mag, 99.0)
    table["limiting_mag_local"] = all_q("limiting_mag_local", np.float32, u.mag)

    # This appears to be the least-bad way to mask SkyCoords right now.
    # Cf. https://github.com/astropy/astropy/issues/13041
    mask_nans = np.zeros(len(gsc_bin_index))
    mask_nans[mask] = np.nan

    table["pos"] = SkyCoord(
        ra=(input_cols["ra"] + mask_nans) * u.deg,
        dec=(input_cols["dec"] + mask_nans) * u.deg,
        frame="icrs",
    )

    table["fwhm_world"] = mq("FWHM_WORLD", np.float32, u.deg)
    table["ellipticity"] = mc("ELLIPTICITY", np.float32)
    table["theta_j2000"] = mq("THETA_J2000", np.float32, u.deg)
    table["iso_area_world"] = mq("ISOAREA_WORLD", np.float32, u.deg**2)

    table["aflags"] = mc("AFLAGS", np.uint32)
    table["bflags"] = mc("BFLAGS", np.uint32)
    table["plate_quality_flag"] = all_c("quality", np.uint32)
    table["local_bin_reject_flag"] = mc("rejectFlag", np.uint32)

    table["series"] = input_cols["series"]
    table["platenum"] = all_c("plateNumber", np.uint32)
    table["mosnum"] = all_c("mosaicNumber", np.uint8)
    table["solnum"] = all_c("solutionNumber", np.uint8)
    table["expnum"] = all_c("exposureNumber", np.uint8)

    # Astrometry: image-level

    table["image_x"] = mq("X_IMAGE", np.float32, u.pixel)
    table["image_y"] = mq("Y_IMAGE", np.float32, u.pixel)

    # Photometry: calibrated

    table["magcal_iso"] = mq("magcal_iso", np.float32, u.mag)
    table["magcal_iso_rms"] = mq("magcal_iso_rms", np.float32, u.mag)
    table["magcal_local"] = mq("magcal_local", np.float32, u.mag)
    table["magcal_local_rms"] = extra_mq("magcal_local_rms", np.float32, u.mag, 99.0)
    table["magcal_local_error"] = extra_mq(
        "magcal_local_error", np.float32, u.mag, 99.0
    )

    # Photometry: calibration/quality information

    table["drad_rms2"] = extra_mq("dradRMS2", np.float32, u.arcsec, 99.0)
    table["extinction"] = mq("extinction", np.float32, u.mag)
    table["magcor_local"] = extra_mq("magcor_local", np.float32, u.mag, 0.0)
    table["blended_mag"] = extra_mq("Blendedmag", np.float32, u.mag, 0.0)
    table["a2flags"] = mc("A2FLAGS", np.uint32)
    table["b2flags"] = mc("B2FLAGS", np.uint32)
    table["npoints_local"] = mc("npoints_local", np.uint32)
    table["local_bin_index"] = mc("local_bin_index", np.uint16)
    table["spatial_bin"] = mc("spatial_bin", np.uint16)
    table["magdep_bin"] = mc("magdep_bin", np.uint16)

    # Photometry: image-level

    table["mag_iso"] = mq("MAG_ISO", np.float32, u.mag)
    table["mag_aper"] = mq("MAG_APER", np.float32, u.mag)
    table["mag_auto"] = mq("MAG_AUTO", np.float32, u.mag)
    table["flux_iso"] = mc("FLUX_ISO", np.float32)
    table["background"] = mc("BACKGROUND", np.float32)
    table["flux_max"] = mc("FLUX_MAX", np.float32)

    # Morphology: image-level

    table["fwhm_image"] = mq("FWHM_IMAGE", np.float32, u.pixel)
    table["kron_radius"] = mc("KRON_RADIUS", np.float32)
    # these have units of px^2, but making them a quantity forces them to be floats:
    table["sxt_iso0"] = mc("ISO0", np.uint32)
    table["sxt_iso1"] = mc("ISO1", np.uint32)
    table["sxt_iso2"] = mc("ISO2", np.uint32)
    table["sxt_iso3"] = mc("ISO3", np.uint32)
    table["sxt_iso4"] = mc("ISO4", np.uint32)
    table["sxt_iso5"] = mc("ISO5", np.uint32)
    table["sxt_iso6"] = mc("ISO6", np.uint32)
    table["sxt_iso7"] = mc("ISO7", np.uint32)

    # Timing

    table["time_accuracy"] = mq("timeAccuracy", np.float32, u.day)

    # Information about the associated catalog source

    table["catalog_ra"] = mq("ra_2", np.float32, u.deg)
    table["catalog_dec"] = mq("dec_2", np.float32, u.deg)
    table["pm_ra_cosdec"] = mq("RaPM", np.float32, u.mas / u.yr)
    table["pm_dec"] = mq("DecPM", np.float32, u.mas / u.yr)

    # Information about the associated plate

    table["series_id"] = all_c("seriesId", np.uint8)
    table["plate_dist"] = mq("plate_dist", np.float32, u.deg)

    # SExtractor supporting info

    table["sxt_number"] = mc("NUMBER", np.uint32)

    # This depends on the dec, and could vary over time with the right proper motion

    table["gsc_bin_index"] = mc("gsc_bin_index", np.uint32)

    # DASCH supporting info

    table["dasch_photdb_version_id"] = all_c("versionId", np.uint16)
    table["dasch_plate_version_id"] = all_c("plateVersionId", np.uint8)
    table["dasch_mask_index"] = all_c("maskIndex", np.uint8)

    table.sort(["time"])

    table["local_id"] = np.arange(len(table))
    table["reject"] = np.zeros(len(table), dtype=np.uint64)

    return table
