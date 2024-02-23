# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
The cone "query" defining a daschlab session.
"""

from astropy.coordinates import SkyCoord
from astropy import units as u
from dataclasses import dataclass
from dataclasses_json import dataclass_json


__all__ = ["SessionQuery"]


@dataclass_json
@dataclass
class SessionQuery:
    """A query"""

    name: str  # empty if none associated
    ra_deg: float
    dec_deg: float

    def pos_as_skycoord(self) -> SkyCoord:
        return SkyCoord(self.ra_deg * u.deg, self.dec_deg * u.deg, frame="icrs")

    @classmethod
    def new_from_name(cls, name: str) -> "SessionQuery":
        c = SkyCoord.from_name(name)
        return SessionQuery(name=name, ra_deg=c.ra.deg, dec_deg=c.dec.deg)
