# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lightcurves
"""

from enum import IntFlag
from urllib.parse import urlencode
from typing import Tuple

from astropy.coordinates import SkyCoord
from astropy.table import Table, Row
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
from bokeh.plotting import figure, show
import numpy as np
import requests

__all__ = [
    "AFlags",
    "BFlags",
    "Lightcurve",
    "LightcurvePoint",
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
    HIGH_BACKGROUND = 1 << 6  # NB, this is "bit 7" since we count them 1-based
    BAD_PLATE_QUALITY = 1 << 7
    MULT_EXP_UNMATCHED = 1 << 8
    UNCERTAIN_DATE = 1 << 9
    MULT_EXP_BLEND = 1 << 10
    LARGE_ISO_RMS = 1 << 11
    LARGE_LOCAL_SMOOTH_RMS = 1 << 12
    CLOSE_TO_LIMITING = 1 << 13
    RADIAL_BIN_9 = 1 << 14
    BIN_DRAD_UNKNOWN = 1 << 15
    UNCERTAIN_CATALOG_MAG = 1 << 19
    CASE_B_BLEND = 1 << 20
    CASE_C_BLEND = 1 << 21
    CASE_BC_BLEND = 1 << 22
    LARGE_DRAD = 1 << 23
    PICKERING_WEDGE = 1 << 24
    SUSPECTED_DEFECT = 1 << 25
    SXT_BLEND = 1 << 26
    REJECTED_BLEND = 1 << 27
    LARGE_SMOOTHING_CORRECTION = 1 << 28
    TOO_BRIGHT = 1 << 29
    LOW_ALTITUDE = 1 << 30


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
    NEIGHBORS = 1 << 0
    BLEND = 1 << 1
    SATURATED = 1 << 2
    NEAR_BOUNDARY = 1 << 3
    APERTURE_INCOMPLETE = 1 << 4
    ISOPHOT_INCOMPLETE = 1 << 5
    DEBLEND_OVERFLOW = 1 << 6
    EXTRACTION_OVERFLOW = 1 << 7
    CORRECTED_FOR_BLEND = 1 << 8
    LARGE_BIN_DRAD = 1 << 9
    PSF_SATURATED = 1 << 10
    MAG_DEP_CAL_APPLIED = 1 << 11
    GOOD_STAR = 1 << 16
    LOWESS_CAL_APPLIED = 1 << 17
    LOCAL_CAL_APPLIED = 1 << 18
    EXTINCTION_CAL_APPLIED = 1 << 19
    TOO_BRIGHT = 1 << 20
    COLOR_CORRECTION_APPLIED = 1 << 21
    COLOR_CORRECTION_USED_METROPOLIS = 1 << 22
    LATE_CATALOG_MATCH = 1 << 24
    LARGE_PROMO_UNCERT = 1 << 25
    LARGE_SPATIAL_BIN_COLORTERM = 1 << 27
    POSITION_ADJUSTED = 1 << 28
    LARGE_JD_UNCERT = 1 << 29
    PROMO_APPLIED = 1 << 30


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
    LARGE_CORRECTION = 1 << 0
    TOO_DIM = 1 << 1
    LARGE_DRAD = 1 << 2


_LOCAL_BIN_REJECT_FLAG_DESCRIPTIONS = [
    "Local correction is out of range",
    "Median brightness is below the limiting magnitude",
    "Majority of stars have a high `drad`",
]


class PlateQualityFlags(IntFlag):
    MULTIPLE_EXPOSURE = 1 << 0
    GRATING = 1 << 1
    COLORED_FILTER = 1 << 2
    COLORTERM_OUT_OF_BOUNDS = 1 << 3
    PICKERING_WEDGE = 1 << 4
    SPECTRUM = 1 << 5
    SATURATED = 1 << 6
    MAG_DEP_CAL_UNAVAILABLE = 1 << 7
    PATROL_TELESCOPE = 1 << 8
    NARROW_TELESCOPE = 1 << 9
    SHOW_LIMITING = 1 << 10
    SHOW_UNDETECTED = 1 << 11
    TRAILED = 1 << 12


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
    def flags(self):
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


class Selector:
    """
    A magic object to help enable lightcurve filtering.
    """

    _lc: "Lightcurve"
    _apply = None

    def __init__(self, lc: "Lightcurve", apply):
        self._lc = lc
        self._apply = apply

    def where(self, row_mask, **kwargs) -> "Lightcurve":
        return self._apply(row_mask, **kwargs)

    def detected(self, **kwargs) -> "Lightcurve":
        m = ~self._lc["magcal_magdep"].mask
        return self._apply(m, **kwargs)

    def undetected(self, **kwargs) -> "Lightcurve":
        m = self._lc["magcal_magdep"].mask
        return self._apply(m, **kwargs)

    def sep_below(
        self, sep_limit: u.Quantity = 20 * u.arcsec, **kwargs
    ) -> "Lightcurve":
        mp = self._lc.mean_pos()
        seps = mp.separation(self._lc["pos"])
        m = seps < sep_limit
        return self._apply(m, **kwargs)

    def any_aflags(self, aflags: int, **kwargs) -> "Lightcurve":
        m = (self._lc["aflags"] & aflags) != 0
        return self._apply(m, **kwargs)


class Lightcurve(Table):
    """
    Cheat sheet:

    - `date` is HJD midpoint
    - `magcal_magdep` is preferred calibrated phot measurement
    - legacy plotter error bar is `magcal_local_rms` * `error_bar_factor`, the
      latter being set to match the error bars to the empirical RMS, if this
      would shrink the error bars
    """

    Row = LightcurvePoint

    # Filtering utilities

    def _copy_subset(self, keep, verbose: bool) -> "Lightcurve":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        return new

    @property
    def match(self) -> Selector:
        return Selector(self, lambda m: m)

    def _apply_keep_only(self, flags, verbose=True) -> "Lightcurve":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> Selector:
        return Selector(self, self._apply_keep_only)

    def _apply_drop(self, flags, verbose=True) -> "Lightcurve":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> Selector:
        return Selector(self, self._apply_drop)

    def _apply_split_by(self, flags) -> Tuple["Lightcurve", "Lightcurve"]:
        lc_left = self._copy_subset(flags, False)
        lc_right = self._copy_subset(~flags, False)
        return (lc_left, lc_right)

    @property
    def split_by(self) -> Selector:
        return Selector(self, self._apply_split_by)

    def summary(self):
        print(f"Number of rows: {len(self)}")

        detns = self.keep_only.detected(verbose=False)
        print(f"Number of detections: {len(detns)}")

        if len(detns):
            mm = detns["magcal_magdep"].mean()
            rm = ((detns["magcal_magdep"] - mm) ** 2).mean() ** 0.5
            print(f"Mean/RMS mag: {mm:.3f} ± {rm:.3f}")

    def mean_pos(self) -> SkyCoord:
        detns = self.keep_only.detected(verbose=False)
        mra = detns["pos"].ra.deg.mean()
        mdec = detns["pos"].dec.deg.mean()
        return SkyCoord(mra, mdec, unit=u.deg, frame="icrs")

    def plot(self, x_axis="year") -> figure:
        detect, limit = self.split_by.detected()

        detect["year"] = detect["date"].jyear
        limit["year"] = limit["date"].jyear

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

    def scatter(self, x_axis: str, y_axis: str) -> figure:
        p = figure(
            tools="pan,wheel_zoom,box_zoom,reset,hover",
            tooltips=[
                ("LocalID", "@local_id"),
                ("Mag.", "@magcal_magdep"),
                ("Epoch", "@year"),
                ("Plate", "@series@platenum mosnum @mosnum pl_loc_id @plate_local_id"),
            ],
        )

        p.scatter(x_axis, y_axis, source=self.to_pandas())

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

    # this will be filled in for real at the end:
    table["local_id"] = np.zeros(len(gsc_bin_index))
    table["date"] = Time(input_cols["Date"], format="jd")
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

    table.sort(["date"])

    table["local_id"] = np.arange(len(table))

    return table
