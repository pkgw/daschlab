# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
The cone "query" defining a daschlab session.

The main class provided by this module is `SessionQuery`. You can obtain an
instance of this with `daschlab.Session.query()`, after defining it with
`daschlab.Session.select_target()`.
"""

from astropy.coordinates import SkyCoord
from astropy import units as u
from dataclasses import dataclass
from dataclasses_json import dataclass_json


__all__ = ["SessionQuery"]


@dataclass_json
@dataclass
class SessionQuery:
    """
    A "query" defining a daschlab session.

    In most cases, the only useful functionality provided by this class is the
    `pos_as_skycoord` method, which obtains the central position of the
    session's area of interest as an `astropy.coordinates.SkyCoord`.
    """

    name: str
    """A textual name of the source corresponding to the query center, or an
    empty string if none is defined."""

    ra_deg: float
    "The ICRS RA of the query center in decimal degrees."

    dec_deg: float
    "The ICRS declination of the query center in decimal degrees."

    def pos_as_skycoord(self) -> SkyCoord:
        """
        Obtain the center position of this query.

        Returns
        =======
        `astropy.coordinates.SkyCoord`
        """
        return SkyCoord(self.ra_deg * u.deg, self.dec_deg * u.deg, frame="icrs")

    @classmethod
    def new_from_name(cls, name: str) -> "SessionQuery":
        """
        Create a new session.

        Parameters
        ==========
        name : `str`
            A source name that is passed to `astropy.coordinates.SkyCoord.from_name`.

        Returns
        =======
        `SessionQuery`
            A new session query object.
        """
        c = SkyCoord.from_name(name)
        return SessionQuery(name=name, ra_deg=c.ra.deg, dec_deg=c.dec.deg)
