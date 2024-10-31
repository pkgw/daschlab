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
    def new_from_coords(cls, coords: SkyCoord) -> "SessionQuery":
        """
        Create a new session, targeting specified celestial coordinates.

        Parameters
        ==========
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the session target location.

        Returns
        =======
        `SessionQuery`
            A new session query object.

        See Also
        ========
        new_from_name : convenience function to do a name-based lookup
        new_from_radec : convenience function for equatorial coordinates
        """
        return SessionQuery(name="", ra_deg=coords.ra.deg, dec_deg=coords.dec.deg)

    @classmethod
    def new_from_name(cls, name: str) -> "SessionQuery":
        """
        Create a new session, targeting a named astronomical source.

        Parameters
        ==========
        name : `str`
            A source name that is passed to `astropy.coordinates.SkyCoord.from_name`.

        Returns
        =======
        `SessionQuery`
            A new session query object.

        See Also
        ========
        new_from_coords : create a session using generic Astropy coordinates
        new_from_radec : convenience function for equatorial coordinates
        """
        c = SkyCoord.from_name(name)
        return SessionQuery(name=name, ra_deg=c.ra.deg, dec_deg=c.dec.deg)

    @classmethod
    def new_from_radec(cls, ra_deg: float, dec_deg: float) -> "SessionQuery":
        """
        Create a new session, targeting specified equatorial coordinates.

        Parameters
        ==========
        ra_deg : `float`
            The right ascension of the session target location, in degrees.
        dec_deg : `float`
            The declination of the session target location, in degrees.

        Returns
        =======
        `SessionQuery`
            A new session query object.

        See Also
        ========
        new_from_coords : create a session using generic Astropy coordinates
        new_from_name : convenience function to do a name-based lookup
        """
        return SessionQuery(name="", ra_deg=ra_deg, dec_deg=dec_deg)
