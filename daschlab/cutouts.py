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

from .apiclient import ApiClient


def _query_cutout(
    client: ApiClient,
    series: str,
    platenum: int,
    solnum: int,
    center: SkyCoord,
) -> bytes:
    payload = {
        "plate_id": f"{series}{platenum:05d}",
        "solution_number": int(solnum),  # `json` will reject `np.int8`
        "center_ra_deg": center.ra.deg,
        "center_dec_deg": center.dec.deg,
    }

    result = client.invoke("/dasch/dr7/cutout", payload)
    if not isinstance(result, str):
        raise Exception(
            f"cutout API query for {series}{platenum:05d}/{solnum} "
            f"@ {center} appears to have failed: {result!r}"
        )

    gzdata = base64.b64decode(result)
    return gzip.decompress(gzdata)
