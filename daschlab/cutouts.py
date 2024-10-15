# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Handling DASCH cutouts.

This module currently has no public API. See:

- `daschlab.Session.cutout()`
- `daschlab.exposures.Exposures.show()`
"""

import base64
import gzip

from astropy.coordinates import SkyCoord
import requests


# TODO: genericize
_API_URL = "https://api.dev.starglass.cfa.harvard.edu/public/dasch/dr7/cutout"


def _query_cutout(
    series: str,
    platenum: int,
    solnum: int,
    center: SkyCoord,
) -> bytes:
    url = _API_URL

    payload = {
        "plate_id": f"{series}{platenum:05d}",
        "solution_number": solnum,
        "center_ra_deg": center.ra.deg,
        "center_dec_deg": center.dec.deg,
    }

    # Due to how the AWS APIs work, the return of this API must be JSON content.
    # So, it returns a single JSON string, which is a base64 encoding of the a
    # gzipped FITS file.

    with requests.post(url, json=payload) as resp:
        result = resp.json()

        if not isinstance(result, str):
            raise Exception(
                f"cutout API query for {series}{platenum:05d}/{solnum} "
                f"@ {center} appears to have failed: {result!r}"
            )

        gzdata = base64.b64decode(result)
        fitsdata = gzip.decompress(gzdata)

    return fitsdata
