# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lightcurves
"""

from enum import Enum
from urllib.parse import urlencode

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
import numpy as np
import requests

__all__ = ["AFlags", "BFlags", "Lightcurve", "LocalBinRejectFlags", "PlateQualityFlags"]


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


class AFlags(Enum):
    HIGH_BACKGROUND = 1 << 6
    BAD_PLATE_QUALITY = 1 << 7
    MULT_EXP_UNMATCHED = 1 << 8
    UNCERTAIN_DATE = 1 << 9
    MULT_EXP_BLEND = 1 << 10
    LARGE_ISO_RMS = 1 << 11
    LARGE_LOCAL_SMOOTH_RMS = 1 << 12
    CLOSE_TO_LIMITING = 1 << 13
    RADIAL_BIN_9 = 1 << 14
    BIN_DRAD_UNKNOWN = 1 << 15
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


class BFlags(Enum):
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


class LocalBinRejectFlags(Enum):
    LARGE_CORRECTION = 1 << 0
    TOO_DIM = 1 << 1
    LARGE_DRAD = 1 << 2


class PlateQualityFlags(Enum):
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


class Lightcurve(Table):
    """
    Cheat sheet:

    - `date` is HJD midpoint
    - `magcal_magdep` is preferred calibrated phot measurement
    """

    pass


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

    # Astrometry: calibrated

    # This appears to be the least-bad way to mask SkyCoords right now.
    # Cf. https://github.com/astropy/astropy/issues/13041
    mask_nans = np.zeros(len(gsc_bin_index))
    mask_nans[mask] = np.nan

    table["pos"] = SkyCoord(
        ra=(input_cols["ra"] + mask_nans) * u.deg,
        dec=(input_cols["dec"] + mask_nans) * u.deg,
        frame="icrs",
    )

    table["gsc_bin_index"] = mc("gsc_bin_index", np.uint32)

    # Astrometry: image-level

    table["image_x"] = mq("X_IMAGE", np.float32, u.pixel)
    table["image_y"] = mq("Y_IMAGE", np.float32, u.pixel)

    # Photometry: calibrated

    table["magcal_iso"] = mq("magcal_iso", np.float32, u.mag)
    table["magcal_iso_rms"] = mq("magcal_iso_rms", np.float32, u.mag)
    table["magcal_local"] = mq("magcal_local", np.float32, u.mag)
    table["magcal_local_rms"] = mq("magcal_local_rms", np.float32, u.mag)
    table["limiting_mag_local"] = all_q("limiting_mag_local", np.float32, u.mag)
    table["magcal_local_error"] = extra_mq(
        "magcal_local_error", np.float32, u.mag, 99.0
    )
    table["magcal_magdep"] = mq("magcal_magdep", np.float32, u.mag)
    table["magcal_magdep_rms"] = extra_mq("magcal_magdep_rms", np.float32, u.mag, 99.0)

    # Photometry: calibration/quality information

    table["drad_rms2"] = extra_mq("dradRMS2", np.float32, u.arcsec, 99.0)
    table["extinction"] = mq("extinction", np.float32, u.mag)
    table["magcor_local"] = extra_mq("magcor_local", np.float32, u.mag, 0.0)
    table["blended_mag"] = extra_mq("Blendedmag", np.float32, u.mag, 0.0)
    table["aflags"] = mc("AFLAGS", np.uint32)
    table["bflags"] = mc("BFLAGS", np.uint32)
    table["a2flags"] = mc("A2FLAGS", np.uint32)
    table["b2flags"] = mc("B2FLAGS", np.uint32)
    table["npoints_local"] = mc("npoints_local", np.uint32)
    table["local_bin_reject_flag"] = mc("rejectFlag", np.uint32)
    table["local_bin_index"] = mc("local_bin_index", np.uint16)
    table["plate_quality_flag"] = all_c("quality", np.uint32)
    table["spatial_bin"] = mc("spatial_bin", np.uint16)
    table["magdep_bin"] = mc("magdep_bin", np.uint16)

    # Photometry: image-level

    table["mag_iso"] = mq("MAG_ISO", np.float32, u.mag)
    table["mag_aper"] = mq("MAG_APER", np.float32, u.mag)
    table["mag_auto"] = mq("MAG_AUTO", np.float32, u.mag)
    table["flux_iso"] = mc("FLUX_ISO", np.float32)
    table["background"] = mc("BACKGROUND", np.float32)
    table["flux_max"] = mc("FLUX_MAX", np.float32)

    # Morphology: calibrated

    table["theta_j2000"] = mq("THETA_J2000", np.float32, u.deg)
    table["ellipticity"] = mc("ELLIPTICITY", np.float32)
    table["iso_area_world"] = mq("ISOAREA_WORLD", np.float32, u.deg**2)
    table["fwhm_world"] = mq("FWHM_WORLD", np.float32, u.deg)

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

    table["date"] = Time(input_cols["Date"], format="jd")
    table["time_accuracy"] = mq("timeAccuracy", np.float32, u.day)

    # Information about the associated catalog source

    table["catalog_ra"] = mq("ra_2", np.float32, u.deg)
    table["catalog_dec"] = mq("dec_2", np.float32, u.deg)
    table["pm_ra_cosdec"] = mq("RaPM", np.float32, u.mas / u.yr)
    table["pm_dec"] = mq("DecPM", np.float32, u.mas / u.yr)

    # Information about the associated plate

    table["series"] = input_cols["series"]
    table["series_id"] = all_c("seriesId", np.uint8)
    table["platenum"] = all_c("plateNumber", np.uint32)
    table["expnum"] = all_c("exposureNumber", np.uint8)
    table["solnum"] = all_c("solutionNumber", np.uint8)
    table["mosnum"] = all_c("mosaicNumber", np.uint8)
    table["plate_dist"] = mq("plate_dist", np.float32, u.deg)

    # SExtractor supporting info

    table["sxt_number"] = mc("NUMBER", np.uint32)

    # DASCH supporting info

    table["dasch_photdb_version_id"] = all_c("versionId", np.uint16)
    table["dasch_plate_version_id"] = all_c("plateVersionId", np.uint8)
    table["dasch_mask_index"] = all_c("maskIndex", np.uint8)

    table.sort(["date"])

    return table
