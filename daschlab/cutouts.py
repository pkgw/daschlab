# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Handling DASCH cutouts.

This module currently has no public API. See:

- `daschlab.Session.cutout()`
- `daschlab.exposures.Exposures.show()`
"""

from urllib.parse import urlencode

from astropy.coordinates import SkyCoord
import requests


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/cutout.php"

# TODO: this needs to support multiple solutions!! The API isn't aware of
# multiple exposures at all, right now, so there's nothing this code can do.


def _query_cutout(
    series: str,
    platenum: int,
    mosnum: int,
    center: SkyCoord,
) -> bytes:
    url = (
        _API_URL
        + "?"
        + urlencode(
            {
                "series": series,
                "platenum": platenum,
                "mosnum": mosnum,
                "cra_deg": center.ra.deg,
                "cdec_deg": center.dec.deg,
            }
        )
    )

    fits = requests.get(url).content

    if len(fits) < 80:
        raise Exception("cutout query failed (plate may not be scanned and WCS-solved)")

    return fits
