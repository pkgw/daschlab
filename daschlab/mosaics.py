# Copyright the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Retrieving and constructing value-added mosaics.
"""

import os.path
from tempfile import NamedTemporaryFile
from typing import List, Optional
import warnings

from astropy.io import fits
#from astropy import units as u
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import requests
from tqdm import tqdm


# These are all camelCase; maybe dataclasses-json has an equivalent of serde's
# rename_all?

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


def get_mosaic(
    sess: "daschlab.Session",
    plate_id: str,
    binning: int,
):
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

    pass # TODO

    # Second API call - the mosaic_package call that gets us DASCH
    # metadata and the presigned S3 URL.

    payload = {
        "plate_id": plate_id,
        "binning": binning,
    }

    raw = sess._apiclient.invoke("mosaic_package", payload)

    try:
        # Suppress some warnings from dataclasses-json that I think we can ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp: MosaicPackageResponse = MosaicPackageResponse.schema().load(raw)
    except Exception as e:
        # from . import InteractiveError
        # raise InteractiveError(f"mosaic_package API request failed: {raw!r}") from e
        raise

    # Download that base mosaic! Maybe.

    if need_fetch_base:
        print("Fetching base mosaic file ...")
        sess.path("base_mosaics").mkdir(exist_ok=True)

        with (
            requests.get(resp.baseFitsUrl, stream=True) as bf_stream,
            tqdm(total=resp.baseFitsSize, unit='B', unit_scale=True) as progress,
            NamedTemporaryFile(mode="wb", dir=sess.path(), prefix="tmpbm_", suffix=".fits.fz", delete=False) as f_temp
        ):
            while buf := bf_stream.raw.read(65536):
                f_temp.write(buf)
                progress.update(len(buf))

        os.rename(f_temp.name, sess.path(base_relpath))

    # Generate the value-added file

    print(resp) # XXXX

    expected_b01_shape = (resp.metadata.mosaic.b01Height, resp.metadata.mosaic.b01Width)

    if resp.metadata.astrometry is not None:
        if resp.metadata.astrometry.rotationDelta in (-270, -90, 90, 270):
            expected_b01_shape = expected_b01_shape[::-1]

    if is_bin16:
        expected_shape = (expected_b01_shape[0] // 16, expected_b01_shape[1] // 16)
    else:
        expected_shape = expected_b01_shape

    print("Generating value-added FITS ...")
    va_hdul = fits.HDUList()

    with fits.open(sess.path(base_relpath)) as base_hdul:
        # For the future: it would be nice to avoid loading the whole mosaic
        # into memory if we can. Sometimes we can't avoid it, though.

        if len(base_hdul) != 2:
            raise Exception(f"unexpected base mosaic structure: {len(base_hdul)} HDUs instead of 2")

        data = base_hdul[1].data
        if data.shape != expected_shape:
            raise Exception(f"unexpected base mosaic shape: {data.shape!r} instead of {expected_shape!r}")

        base_header = base_hdul[1].header

        # Primary HDU

        prim_hdu = fits.PrimaryHDU()
        prim_hdu.header = base_header # XXXXXXXX
        prim_hdu.data = data

    va_hdul.append(prim_hdu)

    # Finally, save it via a tempfile just in case something goes wrong.

    with (
        NamedTemporaryFile(mode="wb", dir=sess.path(), prefix="tmpmos_", suffix=".fits", delete=False) as f_temp
    ):
        pass

    va_hdul.writeto(f_temp.name, overwrite=True)
    os.rename(f_temp.name, sess.path(va_relpath))
    return va_relpath
