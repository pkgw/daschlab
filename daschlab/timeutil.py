# Copyright the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Utilities for dealing with dates and times.
"""

from datetime import datetime
import re
import sys
from typing import Iterable, Optional

from astropy.time import Time
import numpy as np
from pytz import timezone


__all__ = [
    "dasch_time_as_isot",
    "dasch_isot_as_datetime",
    "dasch_isot_as_astropy",
    "dasch_isots_as_time_array",
]


# The exact value here doesn't matter; we just need something accepted by the
# datetime-Time constructor. To be safe we choose a value that can never be
# confused with a date that's relevant to DASCH.
INVALID_DT = datetime(1800, 1, 1)

FRACTIONAL_SECOND_RE = re.compile(r"(.*)T(\d+:\d+:\d+)\.(\d+)(Z?)")


def dasch_time_as_isot(t: str) -> str:
    """
    Convert a "DASCH time", which uses hyphens everywhere, to a proper
    ISO-8601-"T" form.
    """
    bits = t.split("T", 1)
    bits[-1] = bits[-1].replace("-", ":")
    return "T".join(bits)


def dasch_isot_as_datetime(date: str) -> Optional[datetime]:
    """
    Convert a DASCH ISO-8601-"T" datetime to a Python datetime object. DASCH
    ISO-T times may incorrectly end with a seconds field of ":60.0" due to a bug
    in the creation of some database.
    """

    if not date:
        return None

    # Work around timestamp formatting bug in the legacy exposure data table

    if date.endswith(":60.0"):
        date = date[:-4] + "59.9"

    if date.endswith(":60.0Z"):
        date = date[:-5] + "59.9Z"

    # Before Python 3.11, datetime.fromisoformat() can't handle fractional
    # seconds. Lame! This feels worth working around. So, just truncate the
    # fraction if needed. Our timestamps aren't trustworthy to sub-second
    # precision anyway.
    if (sys.version_info.major * 100 + sys.version_info.minor) < 311:
        m = FRACTIONAL_SECOND_RE.match(date)

        if m is not None:
            date = f"{m[1]}T{m[2]}{m[4]}"

    if date.endswith("Z"):
        # expdates are UTC and labeled as such; but Python < 3.11 can't
        # parse the Z extension without help(!)
        date = date[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(date)
    except Exception as e:
        raise Exception(f"failed to parse ISO8601 'T' format {date!r}") from e

    if dt.tzinfo is None:
        # scandates, etc. have no associated timezone info; but we know
        # that all of these are in this timezone:
        tz = timezone("US/Eastern")
        dt = tz.localize(dt)

    return dt


def dasch_isot_as_astropy(date: str) -> Time:
    """
    Convert a single DASCH ISO-8601-"T" date string to an Astropy Time instance.
    Unlike `dasch_isot_as_datetime`, empty inputs are not allowed.
    """

    dt = dasch_isot_as_datetime(date)
    if dt is None:
        raise ValueError(f"illegal DASCH ISO8601T datetime input: {date!r}")

    return Time(dt, format="datetime")


def dasch_isots_as_time_array(dates: Iterable[str]) -> Time:
    """
    Convert an iterable of DASCH ISO-8601-"T" date strings to an Astropy Time
    array. Input days may be missing (None or the empty string), in which case
    they will be masked.
    """

    invalid_indices = []
    dts = []

    for i, dstr in enumerate(dates):
        dt = dasch_isot_as_datetime(dstr)
        if dt is None:
            invalid_indices.append(i)
            dt = INVALID_DT

        dts.append(dt)

    times = Time(dts, format="datetime")

    for i in invalid_indices:
        times[i] = np.ma.masked

    # If we don't do this, Astropy is unable to roundtrip masked times out of
    # the ECSV format.
    times.format = "isot"

    return times
