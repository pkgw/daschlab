# Copyright the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Retrieving and constructing value-added mosaics.
"""

import base64
import gzip
import os.path
from tempfile import NamedTemporaryFile
from typing import List, Optional
import warnings

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.utils.masked import Masked
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from marshmallow.fields import String
import numpy as np
import requests
from tqdm import tqdm

from .timeutil import (
    dasch_time_as_isot,
    dasch_isot_as_astropy,
)


# Results from the DASCH `mosaic_package` API. These are all camelCase; maybe
# dataclasses-json has an equivalent of serde's rename_all?


@dataclass_json
@dataclass
class MosaicData:
    b01Height: int
    b01OrigFileMD5: str
    b01OrigFileSize: int
    b01Width: int
    b16OrigFileMD5: str
    b16OrigFileSize: int
    creationDate: str
    legacyComment: Optional[str]
    legacyRotation: Optional[int]
    mosNum: int
    scanNum: int
    resultId: str
    s3KeyTemplate: str


@dataclass_json
@dataclass
class ExposureData:
    centerSource: Optional[str]
    dateAccDays: Optional[float]
    dateSource: Optional[str]
    decDeg: Optional[float]
    durMin: Optional[float]
    midpointDate: Optional[str]
    number: int
    raDeg: Optional[float]


@dataclass_json
@dataclass
class AstrometryData:
    b01HeaderGz: Optional[str]
    resultId: Optional[str]
    rotationDelta: Optional[int]
    exposures: List[Optional[ExposureData]]


@dataclass_json
@dataclass
class PhotometryData:
    medianColortermApass: Optional[float]
    medianColortermAtlas: Optional[float]
    nSolutionsApass: Optional[int]
    nSolutionsAtlas: Optional[int]
    nMagdepApass: Optional[int]
    nMagdepAtlas: Optional[int]
    resultIdApass: Optional[str]
    resultIdAtlas: Optional[str]


@dataclass_json
@dataclass
class MosaicPackageMetadata:
    series: str
    plateNumber: int
    mosaic: MosaicData
    astrometry: Optional[AstrometryData]
    photometry: Optional[PhotometryData]


@dataclass_json
@dataclass
class MosaicPackageResponse:
    baseFitsUrl: str
    baseFitsSize: int
    metadata: MosaicPackageMetadata


# Results from the StarGlass GET /plates/p/{plate_id} endpoint. Most of this we
# ignore because we prefer the DASCH stuff, but I figured I might as well fully
# type out the response.


@dataclass_json
@dataclass
class PhotoData:
    plate_id: str
    portion: str
    url: str
    thumbnail: bool
    thumbnail_ratio: int
    image_type: str


@dataclass_json
@dataclass
class MosaicData:
    plate_id: str
    bin_factor: str
    flags: str
    rotation: str
    mosaic_num: str


@dataclass_json
@dataclass
class ExposureData:
    crpix1: float
    crpix2: float
    ctr_dec: float
    ctr_ra: float
    delta_dec_x: float
    delta_dec_y: float
    delta_ra_x: float
    delta_ra_y: float
    naxis1: int
    naxis2: int
    datetime: str
    exposure: float
    exposure_num: int
    solution_num: int


@dataclass_json
@dataclass
class CatalogExposureData:
    ctr_dec: float
    ctr_ra: float
    datetime: str
    exposure_num: int
    exposure_length: float


@dataclass_json
@dataclass
class LocationData:
    name: str
    lat: float
    lon: float
    elevation: float


@dataclass_json
@dataclass
class MentionData:
    author: str
    notebook: str
    page_num: int


@dataclass_json
@dataclass
class GetPlateResponse:
    plate_id: str
    jacket_images: List[PhotoData]
    plate_images: List[PhotoData]
    jacket_thumbnail: List[PhotoData]
    plate_thumbnail: List[PhotoData]
    mosaics: List[MosaicData]
    exposures: List[ExposureData]
    catalog_exposures: List[CatalogExposureData]
    telescope: str
    location: LocationData
    has_markings: bool
    markings_cleaned: bool
    class_: Optional[str] = field(
        default=None,
        metadata={"dataclasses_json": {"mm_field": String(data_key="class")}},
    )
    plate_comment: Optional[str] = None
    comment_astronomers: Optional[List[str]] = None
    mentions: Optional[List[MentionData]] = None


def _b64_to_hex(b64text: str) -> str:
    return base64.b64decode(b64text).hex()


def _do_astrometry(binning: int, add_with_comment, b01HeaderGz: str) -> int:
    """
    Take an archived pipeline astrometric solution header and get all of its
    information into our value-add FITS header.
    """

    add = lambda k, v: add_with_comment(k, v, None)
    hdrtext = gzip.decompress(base64.b64decode(b01HeaderGz))
    base_hdr = fits.Header.fromstring(hdrtext, sep="\n")

    # If there's only one solution, it will use the " " WCS tag.
    # If there are multiple solutions, the first one will use the
    # "A" tag, the second will use "B", and so on ... but then
    # the final solution will *also* appear under the " " tag.

    nsol = 1

    if "CTYPE1A" in base_hdr:
        while True:
            next_key = chr(ord("A") + nsol)
            if f"CTYPE1{next_key}" not in base_hdr:
                break

            nsol += 1

        if nsol == 1:
            # As per the above, if there's a CTYPE1A, there should
            # also be a CTYPE1B.
            raise Exception("unexpected astrometry header: CTYPE1A but no CTYPE2B")

    if nsol == 1:
        input_tags = [""]
        add(
            "COMMENT",
            "DASCH astrometric processing found one WCS solution for this plate.",
        )
    else:
        input_tags = [chr(ord("A") + i) for i in range(nsol)]
        add(
            "COMMENT",
            f"DASCH astrometric processing found {nsol} WCS solutions for this plate.",
        )

    for isol, itag in enumerate(input_tags):
        if isol == 0:
            otag = ""
        else:
            otag = chr(ord("A") + isol - 1)

        add("WCSAXES" + otag, 2)
        add("WCSNAME" + otag, f"DASCH astrometric solution #{isol + 1}")
        add("RADESYS" + otag, "ICRS")

        # In order for Astropy to accept our nonstandard distortion headers,
        # we must use this projection type!!
        add("CTYPE1" + otag, "RA---TPV")
        add("CTYPE2" + otag, "DEC--TPV")

        # Thankfully, the distortion terms are defined in such a way that
        # we can use exactly the same values regardless of the binning level.
        # All we need to do is scale CRPIXn and CDn_m.

        add("CUNIT1" + otag, "deg")
        add("CUNIT2" + otag, "deg")
        add("CRVAL1" + otag, base_hdr["CRVAL1" + itag])
        add("CRVAL2" + otag, base_hdr["CRVAL2" + itag])
        add("CRPIX1" + otag, (base_hdr["CRPIX1" + itag] - 0.5) / binning + 0.5)
        add("CRPIX2" + otag, (base_hdr["CRPIX2" + itag] - 0.5) / binning + 0.5)
        add("CD1_1" + otag, base_hdr["CD1_1" + itag] * binning)
        add("CD1_2" + otag, base_hdr["CD1_2" + itag] * binning)
        add("CD2_1" + otag, base_hdr["CD2_1" + itag] * binning)
        add("CD2_2" + otag, base_hdr["CD2_2" + itag] * binning)

        if itag == "":
            check_tail = lambda s: s[-1] in "0123456789"
            get_base = lambda s: s
        else:
            check_tail = lambda s: s.endswith(itag)
            get_base = lambda s: s[:-1]

        for key, value in base_hdr.items():
            if key.startswith("PV") and check_tail(key):
                add(get_base(key) + otag, value)

    return nsol


def _get_mosaic(
    sess: "daschlab.Session",
    plate_id: str,
    binning: int,
):
    from . import __version__, get_version_doi

    if binning == 1:
        is_bin16 = False
    elif binning == 16:
        is_bin16 = True
    else:
        raise ValueError(f"illegal binning {binning!r}")

    # Does the final mosaic exist?

    va_relpath = f"mosaics/{plate_id}_{binning:02d}.fits"
    if os.path.exists(sess.path(va_relpath)):
        return va_relpath

    # Alas, we have some work to do. But maybe the base mosaic exists.

    sess.path("mosaics").mkdir(exist_ok=True)

    base_relpath = f"base_mosaics/{plate_id}_{binning:02d}.fits.fz"
    need_fetch_base = not os.path.exists(sess.path(base_relpath))

    # First API call - Starglass-level metadata

    sg_raw = sess._apiclient.invoke(f"/plates/p/{plate_id}", None, method="get")

    try:
        sg_resp: GetPlateResponse = GetPlateResponse.schema().load(sg_raw)
    except Exception as e:
        # from . import InteractiveError
        # raise InteractiveError(f"GetPlate API request failed: {raw!r}") from e
        raise

    # Second API call - the mosaic_package call that gets us DASCH
    # metadata and the presigned S3 URL.

    payload = {
        "plate_id": plate_id,
        "binning": binning,
    }

    mp_raw = sess._apiclient.invoke("/dasch/dr7/mosaic_package", payload)

    try:
        # Suppress some warnings from dataclasses-json that I think we can ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp: MosaicPackageResponse = MosaicPackageResponse.schema().load(mp_raw)
    except Exception as e:
        # from . import InteractiveError
        # raise InteractiveError(f"mosaic_package API request failed: {raw!r}") from e
        raise

    # Download that base mosaic! Maybe.

    if need_fetch_base:
        sess._info("- Fetching base mosaic file ...")
        sess.path("base_mosaics").mkdir(exist_ok=True)

        with (
            requests.get(resp.baseFitsUrl, stream=True) as bf_stream,
            tqdm(total=resp.baseFitsSize, unit="B", unit_scale=True) as progress,
            NamedTemporaryFile(
                mode="wb",
                dir=sess.path(),
                prefix="tmpbm_",
                suffix=".fits.fz",
                delete=False,
            ) as f_temp,
        ):
            while buf := bf_stream.raw.read(65536):
                f_temp.write(buf)
                progress.update(len(buf))

        os.rename(f_temp.name, sess.path(base_relpath))

    # Generate the value-added file

    md = resp.metadata
    expected_b01_shape = (md.mosaic.b01Height, md.mosaic.b01Width)
    rot_k = 0

    if md.astrometry is not None:
        if md.astrometry.rotationDelta == 90:
            # Sample plate: a01267, overlaps star: Polaris
            rot_k = -1
        elif md.astrometry.rotationDelta == 180 or md.astrometry.rotationDelta == -180:
            # Sample plate for +180: ac12037, overlaps star: Polaris
            # Sample plate for -180: ac01895, overlaps star: Polaris
            rot_k = 2
        elif md.astrometry.rotationDelta == -90:
            # Sample plate: ac01018, overlaps star: Polaris
            rot_k = 1

    if is_bin16:
        expected_shape = (expected_b01_shape[0] // 16, expected_b01_shape[1] // 16)
    else:
        expected_shape = expected_b01_shape

    sess._info("- Generating value-added FITS ...")

    with fits.open(sess.path(base_relpath)) as base_hdul:
        # For the future: it would be nice to avoid loading the whole mosaic
        # into memory if we can. Sometimes we can't avoid it, though. There's
        # some relevant code in daschdata/results.py for recreating "raw"
        # mosaics.

        if len(base_hdul) != 2:
            raise Exception(
                f"unexpected base mosaic structure: {len(base_hdul)} HDUs instead of 2"
            )

        data = base_hdul[1].data
        if data.shape != expected_shape:
            raise Exception(
                f"unexpected base mosaic shape: {data.shape!r} instead of {expected_shape!r}"
            )

        base_header = base_hdul[1].header
        data = np.rot90(data, k=rot_k)

    # An HDU that preserves the headers of the base mosaic:

    orig_header_hdu = fits.ImageHDU()
    orig_header_hdu.name = "ORIGHDR"
    orig_header_hdu.header.update(base_header)
    orig_header_hdu.data = None

    # The new primary HDU:

    prim_hdu = fits.PrimaryHDU()
    prim_hdr = prim_hdu.header

    def add(keyword: str, value, comment: Optional[str]):
        # This method makes sure to always add a new header at the very end
        # of the header list. If we don't do this, Astropy tries to be clever
        # about reordering commentary headers in a way that we don't want.
        card = (keyword, value) if comment is None else (keyword, value, comment)
        prim_hdr.insert(len(prim_hdr), card)

    def maybe(keyword: str, value, comment: Optional[str], truthy: bool = False):
        if value or (not truthy and value is not None):
            add(keyword, value, comment)

    def date(keytail: str, daschdate: str, comment: str):
        aptime = dasch_isot_as_astropy(dasch_time_as_isot(daschdate))
        add("DATE" + keytail, aptime.fits, comment)
        add("MJD" + keytail, aptime.mjd, comment)

    add("DASCHVAM", "DR7", "I'm a DASCH Data Release 7 value-add mosaic")
    add("DASCHLAB", __version__, "Version of daschlab that made this file")
    add("DLABDOI", get_version_doi(), "DOI of this version of daschlab")
    add("DASCHCIT", "https://dasch.cfa.harvard.edu/citing/", "How to cite")
    add("D_PLATE", plate_id, "ID of the plate imaged in this file")
    add("D_SERIES", md.series, "ID of the plate series this plate belongs to")
    add("D_PLNUM", md.plateNumber, "Number of the plate within its series")

    if md.mosaic.legacyComment:
        # Astropy will break this into multiple cards if needed
        add(
            "COMMENT",
            f"DASCH database comment about this plate: {md.mosaic.legacyComment}",
            None,
        )

    if sg_resp.plate_comment:
        add(
            "COMMENT",
            f"Plate Stacks Curator's remark about this plate: {sg_resp.plate_comment}",
            None,
        )

    add(
        "D_MARKED",
        1 if sg_resp.has_markings else 0,
        "Plate has/had handwritten markings",
    )
    add(
        "D_ERASED",
        1 if sg_resp.markings_cleaned else 0,
        "Markings were erased for DASCH scan",
    )
    maybe("D_PCLASS", sg_resp.class_, "Logged plate class ID", truthy=True)
    add("SITE", sg_resp.location.name, "Name of observing site")
    add("TELESCOP", sg_resp.telescope, None)
    add("LATITUDE", sg_resp.location.lat, "[deg] Observatory latitude")
    add(
        "LONGITUD", -sg_resp.location.lon, "[deg] Observatory longitude"
    )  # TODO: double check sign convention
    add("HEIGHT", sg_resp.location.elevation, "[m] Observatory elevation")

    add("D_SCNNUM", md.mosaic.scanNum, "Sequence number of the plate scan")
    add("D_MOSNUM", md.mosaic.mosNum, "Sequence number of the mosaic")
    add("S_OPER", base_header["S_OPER"], "Scanner operator")
    date("-SCN", base_header["DATE-SCN"], "Date of the scan")
    date("-MOS", md.mosaic.creationDate, "Date the mosaic was created")
    add("D_BINLEV", binning, "Amount of pixel binning in this file")
    add(
        "D_ORGMD5",
        _b64_to_hex(md.mosaic.b16OrigFileMD5 if is_bin16 else md.mosaic.b01OrigFileMD5),
        "MD5 of the base mosaic file",
    )
    add(
        "D_ORGSZ",
        md.mosaic.b16OrigFileSize if is_bin16 else md.mosaic.b01OrigFileSize,
        "Size in bytes of the base mosaic file",
    )
    add("D_MOSRID", _b64_to_hex(md.mosaic.resultId), "ID of the mosaicdata result")

    if md.photometry is not None:
        p = md.photometry
        maybe("D_MCTAP", p.medianColortermApass, "median_colorterm_apass")
        maybe("D_MCTAT", p.medianColortermAtlas, "median_colorterm_atlas")
        maybe("D_NSAP", p.nSolutionsApass, "n_solutions_apass")
        maybe("D_NSAT", p.nSolutionsAtlas, "n_solutions_atlas")
        maybe("D_NMDAP", p.nMagdepApass, "n_magdep_apass")
        maybe("D_NMDAT", p.nMagdepAtlas, "n_magdep_atlas")
        maybe("D_RIDAP", _b64_to_hex(p.resultIdApass), "result_id_apass", truthy=True)
        maybe("D_RIDAT", _b64_to_hex(p.resultIdAtlas), "result_id_atlas", truthy=True)

    add("TIMEUNIT", "s", "Time unit for durations")

    if md.astrometry is None:
        exp_hdu = None
    else:
        a = md.astrometry
        maybe(
            "D_ASRID",
            _b64_to_hex(a.resultId),
            "ID of the astromdata result",
            truthy=True,
        )

        if a.b01HeaderGz:
            n_sol = _do_astrometry(binning, add, a.b01HeaderGz)

        n_exp = sum(e is not None for e in a.exposures)
        n_sol_no_exp = sum(e is None for e in a.exposures[:n_sol])
        n_exp_no_sol = len(a.exposures) - n_sol

        if n_exp == 0:
            add(
                "COMMENT",
                "DASCH analysis recovered no information about this plate's exposures",
                None,
            )
        elif n_exp == 1:
            add(
                "COMMENT",
                "DASCH analysis recovered information about one plate exposure",
                None,
            )
        else:
            add(
                "COMMENT",
                f"DASCH analysis recovered information about {n_exp} plate exposures",
                None,
            )

        if n_exp_no_sol:
            add(
                "COMMENT",
                "There are exposure records without corresponding WCS solutions",
                None,
            )

        if n_sol_no_exp:
            add(
                "COMMENT",
                "There are WCS solutions without corresponding exposure records",
                None,
            )

        if n_exp_no_sol == 0 and n_sol_no_exp == 0:
            add("COMMENT", "All exposures are matched to WCS solutions", None)

        add("D_NEXP", n_exp, "Number of exposure records")
        add("D_NSOL", n_sol, "Number of WCS solutions")
        add("D_NENS", n_exp_no_sol, "Number of exposures without matched WCS")
        add("D_NSNE", n_sol_no_exp, "Number of WCS without matched exposures")

        # DATE-OBS and friends. Not quite sure how to best handle multi-exposure
        # plates here. Since the exposures will presumably be made at quite
        # similar times, it seems reasonable to pick one for fields like
        # MJD-OBS, even if you could make an argument that we should leave it
        # blank to force users to reckon with the multi-exposure-ness.
        #
        # We try to find the exposure that has the best metadata for filling in
        # those fields.

        best_exp = None
        best_score = 0
        best_maybesolnum = None

        for maybesolnum, expdata in enumerate(a.exposures):
            if expdata is not None:
                this_score = 0.0

                if expdata.midpointDate:
                    this_score += 50  # most important

                if expdata.dateAccDays is not None:
                    # second-most important; prefer higher accuracy
                    this_score += 30 - expdata.dateAccDays

                if expdata.durMin is not None:
                    this_score += 20

                if maybesolnum < n_sol:
                    # minor proference for exposures with associated WCS
                    this_score += 10

                if this_score > best_score:
                    best_score = this_score
                    best_exp = expdata
                    best_maybesolnum = maybesolnum

        if best_exp is not None:
            add("D_MDEXP", best_exp.number, "Exposure number chosen for metadata")

            if best_maybesolnum < n_sol:
                add("D_MDSOL", best_maybesolnum, "WCS solution associated with D_MDEXP")
            else:
                add(
                    "COMMENT",
                    "Metadata exposure not associated with a WCS solution",
                    None,
                )

            if best_exp.midpointDate:
                date("-OBS", best_exp.midpointDate, "Datetime of this exposure")

            maybe("D_OBSACC", best_exp.dateAccDays, "Uncertainty in MJD-OBS, in days")

            if best_exp.durMin is not None:
                maybe("XPOSURE", best_exp.durMin * 60.0, "[s] Exposure time")

            maybe(
                "D_OBSSRC",
                best_exp.dateSource,
                "Source of exposure datetime",
                truthy=True,
            )

        # Table HDU with detailed multi-exposure information
        #
        # Astropy's support for time columns in FITS tables is quite complex due
        # to efforts to support the FITS+WCS time standards as well as possible.
        # Astropy.Time columns end up getting expressed as pairs of doubles, and
        # I can't get the loading to work nicely. So, blah, just express our
        # midpoint dates as FITS timestamps. They're not incredibly precise,
        # anyway.
        #
        # Expnum and solnum should fit into `int8`, but Astropy then treats them
        # as booleans rather than tiny signed ints.

        if len(a.exposures):
            # Construct unmasked arrays where everything defaults to invalid

            n_t = len(a.exposures)
            t_solnum = np.arange(n_t)
            t_expnum = np.zeros(n_t, dtype=np.int16) - 1
            t_ra = u.Quantity(np.zeros(n_t) + np.nan, u.deg)
            t_dec = u.Quantity(np.zeros(n_t) + np.nan, u.deg)
            t_ctrsrc = [""] * n_t
            t_midpoint = [""] * n_t
            t_dateacc = u.Quantity(np.zeros(n_t) + np.nan, u.day)
            t_datesrc = [""] * n_t
            t_duration = u.Quantity(np.zeros(n_t) + np.nan, u.second)

            # Fill in whatever valid values we may have

            for i, e in enumerate(a.exposures):
                if e is None:
                    continue

                t_expnum[i] = e.number

                if e.raDeg is not None:
                    t_ra[i] = e.raDeg * u.deg

                if e.decDeg is not None:
                    t_dec[i] = e.decDeg * u.deg

                if e.centerSource:
                    t_ctrsrc[i] = e.centerSource

                if e.midpointDate:
                    t_midpoint[i] = dasch_isot_as_astropy(e.midpointDate).fits

                if e.dateAccDays:
                    t_dateacc[i] = e.dateAccDays * u.day

                if e.durMin:
                    t_duration[i] = e.durMin * 60 * u.second

                if e.dateSource:
                    t_datesrc[i] = e.dateSource

            # Now we can mask and assemble into a table

            exp_tbl = Table(masked=True)
            exp_tbl["solnum"] = np.ma.array(
                t_solnum, dtype=np.int16, mask=(t_solnum >= n_sol)
            )
            exp_tbl["expnum"] = np.ma.array(
                t_expnum, dtype=np.int16, mask=(t_expnum < 0)
            )
            exp_tbl["ctr_ra"] = Masked(t_ra, np.isnan(t_ra))
            exp_tbl["ctr_dec"] = Masked(t_dec, np.isnan(t_dec))
            exp_tbl["ctr_src"] = t_ctrsrc
            exp_tbl["date"] = t_midpoint
            exp_tbl["date_src"] = t_datesrc
            exp_tbl["u_date"] = Masked(t_dateacc, np.isnan(t_dateacc))
            exp_tbl["duration"] = Masked(t_duration, np.isnan(t_duration))
            exp_hdu = fits.BinTableHDU(data=exp_tbl, name="EXPOSURS")

    prim_hdu.data = data

    # Table of photographs. Skip the thumbnails since they're derived products.
    # Omit the URLs since they're temporary.

    add("D_NPPHOT", len(sg_resp.plate_images), "Number of DASCH plate photographs")
    add(
        "D_NJPHOT",
        len(sg_resp.jacket_images),
        "Number of DASCH plate-jacket photographs",
    )

    t_phottype = []
    t_portion = []

    for pphot in sg_resp.plate_images:
        t_phottype.append(pphot.image_type)
        t_portion.append(pphot.portion)

    for jphot in sg_resp.jacket_images:
        t_phottype.append(jphot.image_type)
        t_portion.append(jphot.portion)

    photo_tbl = Table({"kind": t_phottype, "portion": t_portion})
    photo_hdu = fits.BinTableHDU(data=photo_tbl, name="PHOTOS")

    # Table of plate mentions

    add(
        "D_NCNAMS",
        len(sg_resp.comment_astronomers or []),
        "Number of names extracted from curator comment",
    )
    add(
        "D_NMNTNS",
        len(sg_resp.mentions or []),
        "Number of plate mentions in PHaEDRA analysis",
    )

    t_name = []
    t_notebook = []
    t_pagenum = []

    for cname in sg_resp.comment_astronomers or []:
        t_name.append(cname)
        t_notebook.append("*curator")
        t_pagenum.append(1)

    for mention in sg_resp.mentions or []:
        t_name.append(mention.author)
        t_notebook.append(mention.notebook)
        t_pagenum.append(mention.page_num)

    people_tbl = Table({"name": t_name, "notebook": t_notebook, "page_num": t_pagenum})
    people_hdu = fits.BinTableHDU(data=people_tbl, name="PEOPLE")

    # Assemble and write, saving via a tempfile just in case something goes wrong.

    va_hdul = fits.HDUList()
    va_hdul.append(prim_hdu)

    if exp_hdu is not None:
        va_hdul.append(exp_hdu)

    va_hdul.append(photo_hdu)
    va_hdul.append(people_hdu)
    va_hdul.append(orig_header_hdu)

    with NamedTemporaryFile(
        mode="wb", dir=sess.path(), prefix="tmpmos_", suffix=".fits", delete=False
    ) as f_temp:
        pass

    va_hdul.writeto(f_temp.name, overwrite=True)
    os.rename(f_temp.name, sess.path(va_relpath))
    sess._info(f"- Saved `{sess.path(va_relpath)}`")
    return va_relpath
