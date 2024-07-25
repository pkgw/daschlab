# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Information about the various plate series encountered in DASCH.

You will probably not need to use this module directly. It provides data that
are used to fill in other tables such as those provided by
`daschlab.exposures.Exposures.series_info()`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np

__all__ = ["SeriesInfo", "SeriesKind"]


class SeriesKind(Enum):
    """
    A classification of different series by plate scale (spatial resolution).
    """

    NARROW = 0
    """\"Narrow\" series have plate scales of 400 arcsec/mm or lower."""

    PATROL = 1
    """\"Patrol\" series have plate scales between 400 arcsec/mm and 900 arcsec/mm."""

    METEOR = 2
    """\"Meteor\" series have plate scales of 900 arcsec/mm or larger."""


@dataclass
class SeriesInfo:
    """
    General information about a plate series.
    """

    series: str
    "The series identifier"

    description: str
    "A textual description of the series."

    kind: SeriesKind
    'The series "kind", in terms of plate scale / spatial resolution.'

    plate_scale: float
    "The characteristic plate scale of plates in this series, in arcsec/mm."

    aperture: float
    "The characteristic telescope aperture used in plates in this series, in meters."


SERIES: Dict[str, SeriesInfo] = {}


def _add(
    series: str, kind: SeriesKind, plate_scale: float, aperture: float, description: str
):
    SERIES[series] = SeriesInfo(series, description, kind, plate_scale, aperture)


_add("a", SeriesKind.NARROW, 60, 0.6, "24-inch Bruce Doublet")
_add("ab", SeriesKind.PATROL, 590, 0.0635, "2.5-inch Ross Portrait Lens")
_add("ac", SeriesKind.PATROL, 606.4, 0.038, "1.5-inch Cooke Lenses")
_add(
    "aco",
    SeriesKind.PATROL,
    611.3,
    0.0254,
    "1 in Cook Lens #832 Series renamed from ac-a",
)
_add("ad", SeriesKind.METEOR, np.nan, np.nan, "Various Meteor Cameras")
_add(
    "adh",
    SeriesKind.NARROW,
    68,
    0.76,
    "32-36 inch BakerSchmidt 10 1/2 inch round Armagh-Dunsink-Harvard",
)
_add("ai", SeriesKind.METEOR, 1360, 0.038, "Meteor Patrol cameras")
_add("ak", SeriesKind.PATROL, 614.5, 0.038, '1.5 in Cooke "Long Focus"')
_add("al", SeriesKind.METEOR, 1200, 0.038, "Cooke Short Focus")
_add("am", SeriesKind.PATROL, 610.8, 0.038, "1-inch, 1.5-inch Cooke Lenses")
_add("an", SeriesKind.PATROL, 574, np.nan, "New Cooke Lens")
_add("ax", SeriesKind.PATROL, 695.7, 0.076, "3 inch Ross-Tessar Lens")
_add("ay", SeriesKind.PATROL, 694.2, 0.066, "2.6-inch Zeiss-Tessar")
_add("ayroe", SeriesKind.NARROW, np.nan, 0.152, "Roe 6-inch")
_add(
    "b",
    SeriesKind.NARROW,
    179.4,
    0.2,
    "8-inch Bache Doublet, Voigtlander, reworked by Clark",
)
_add("bc", SeriesKind.NARROW, np.nan, 0.406, "16-inch Boller & Chivens")
_add("bi", SeriesKind.METEOR, 1446, 0.038, "1.5 inch Ross (short focus)")
_add("bm", SeriesKind.NARROW, 384, 0.075, "3-inch Ross")
_add(
    "bo",
    SeriesKind.PATROL,
    800,
    0.0635,
    '2.5 inch Voigtlander (Little Bache or "Bachito")',
)
_add("br", SeriesKind.NARROW, 204, 0.2, "8-inch Brashear Lens")
_add("c", SeriesKind.NARROW, 52.56, 0.3, "11-inch Draper Refractor")
_add("ca", SeriesKind.PATROL, 596, 0.0635, "2.5 inch Cooke Lens")
_add("ctio", SeriesKind.NARROW, 18, 4, "Cerro Tololo 4 meter")
# _add("d", SeriesKind., NULL |              NULL |     NULL | Positives")
_add("darnor", SeriesKind.METEOR, 890, np.nan, "Darlot Lens North")
_add("darsou", SeriesKind.METEOR, 890, np.nan, "Darlot Lens South")
_add("dnb", SeriesKind.PATROL, 577.3, 0.042, "Damons North Blue")
_add("dnr", SeriesKind.PATROL, 579.7, 0.042, "Damons North Red")
_add("dny", SeriesKind.PATROL, 576.1, 0.042, "Damons North Yellow")
_add("dsb", SeriesKind.PATROL, 574.5, 0.042, "Damons South Blue")
_add("dsr", SeriesKind.PATROL, 579.7, 0.042, "Damons South Red")
_add("dsy", SeriesKind.PATROL, 581.8, 0.042, "Damons South Yellow")
# _add("e", SeriesKind., NULL |              NULL |     NULL | Miscellaneous test plates")
_add("ee", SeriesKind.NARROW, 330, 0.102, "4-inch Voightlander Lens")
_add("er", SeriesKind.NARROW, 390, 0.076, "3-inch Elmer Ross")
_add("fa", SeriesKind.METEOR, 1298, 0.038, "1.5-inch Ross-Xpress")
_add("h", SeriesKind.NARROW, 59.6, 0.6, "24-inch Clark Reflector")
_add("hale", SeriesKind.NARROW, 11.06, 5, "200 inch Hale Telescope")
# _add("hsl", SeriesKind., NULL |              NULL |     NULL | Henrietta Leavitt Logbooks. No plates.")
_add(
    "i",
    SeriesKind.NARROW,
    163.3,
    0.2,
    "8-inch Draper Doublet, Voigtlander Reworked by Clark",
)
_add("ir", SeriesKind.NARROW, 164, 0.2, "8-inch Ross Lundin")
_add("j", SeriesKind.NARROW, 98, 0.6, "24-33 in Jewett Schmidt")
_add("jdar", SeriesKind.PATROL, 560, 0.076, "3-inch Darlot (Series renamed from j)")
_add("ka", SeriesKind.METEOR, 1200, 0.071, "2.8-inch Kodak Aero-Ektar")
_add("kb", SeriesKind.METEOR, 1200, 0.071, "2.8-inch Kodak Aero-Ektar")
_add("kc", SeriesKind.PATROL, 650, 0.127, "K-19 Air Force Camera")
_add("kd", SeriesKind.PATROL, 650, 0.127, "Air Force Camera")
_add(
    "ke",
    SeriesKind.METEOR,
    1160,
    0.071,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel",
)
_add(
    "kf",
    SeriesKind.METEOR,
    1160,
    0.071,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel",
)
_add(
    "kg",
    SeriesKind.METEOR,
    1160,
    0.071,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel Formerly KE (1951)",
)
_add(
    "kge",
    SeriesKind.METEOR,
    1160,
    0.071,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel Formerly KG (1950)",
)
_add("kh", SeriesKind.METEOR, 1160, 0.071, "KE Camera with Installed Rough Focus")
_add("lwla", SeriesKind.NARROW, 36.687, 1.01, "Lowell 40 inch reflector")
_add("m", SeriesKind.NARROW, np.nan, 0.127, "5-in Voightlander Transit Photometer")
_add("ma", SeriesKind.NARROW, 93.7, 0.3, "12-inch Metcalf Doublet")
_add(
    "mb",
    SeriesKind.NARROW,
    390,
    0.1,
    "4-inch Cooke (1-327), 6-inch (328-1776) 3-inch Ross Lundin (2388-2722)",
)
_add(
    "mc", SeriesKind.NARROW, 97.9, 0.4, "16-inch Metcalf Doublet (Refigured after 3500)"
)
_add("md", SeriesKind.NARROW, 193, 0.1, "4-inch Cooke Lens")
_add("me", SeriesKind.PATROL, 600, 0.038, "1.5-inch Cooke Lenses")
_add("meteor", SeriesKind.METEOR, 1200, 0.3, "Various meteor cameras")
_add("mf", SeriesKind.NARROW, 167.3, 0.25, "10-inch Metcalf Triplet")
# _add("misc", SeriesKind., NULL |              NULL |     NULL | Logbook Only. Pages without plates.")
# _add("n", SeriesKind., NULL |              NULL |     NULL | Transit Photometer")
_add(
    "na",
    SeriesKind.NARROW,
    100,
    0.191,
    "7.5-inch Cooke/Clark Refractor at Maria Mitchell Observatory",
)
# _add("nviews", SeriesKind., NULL |              NULL |     NULL | Landscapes and views (Series renamed from n)")
# _add("o", SeriesKind., NULL |              NULL |     NULL | Stellar")
# _add("oa", SeriesKind., NULL |              NULL |     NULL | Stellar")
# _add("p", SeriesKind., NULL |              NULL |     NULL | Eclipse Plates")
_add("pas", SeriesKind.NARROW, 95.64, 0.67, "Asiago Observatory 92/67 cm Schmidt")
_add("poss", SeriesKind.NARROW, 67.19, 1.219, "Palomar Sky Survey (POSS)")
_add("pz", SeriesKind.METEOR, 1553, 0.076, "3 inch Perkin-Zeiss Lens")
# _add("q", SeriesKind., NULL |              NULL |     NULL | Emulsion and Lens Tests")
# _add("qa", SeriesKind., NULL |              NULL |     NULL | Emulsion and Lens Tests")
# _add("qb", SeriesKind., NULL |              NULL |     NULL | Emulsion and Lens Tests")
_add("r", SeriesKind.NARROW, 390, 0.075, "3-inch Ross Fecker")
# _add("ra", SeriesKind., NULL |              NULL |     NULL | Prints of Neg. Post (?) Enlargements")
_add("rb", SeriesKind.NARROW, 395.5, 0.075, "3-inch Ross Fecker")
_add("rh", SeriesKind.NARROW, 391.3, 0.075, "3-inch Ross Fecker")
_add("rl", SeriesKind.NARROW, 290, 0.1, "4-inch Ross Lundin")
_add(
    "ro", SeriesKind.NARROW, 390, 0.075, "Unknown Oak Ridge setup! Aperture is a guess."
)
# _add("rp", SeriesKind., NULL |              NULL |     NULL | NULL")
_add("s", SeriesKind.NARROW, 26.3, 1.52, "60 inch Common")
_add("sb", SeriesKind.NARROW, 26, 1.5, "60-inch Rockefeller Reflector")
_add("sh", SeriesKind.NARROW, 26, 1.5, "61-inch Wyeth Reflector")
# _add("solar", SeriesKind., NULL |              NULL |     NULL | Solar (Series Discarded)")
# _add("sp", SeriesKind., NULL |              NULL |     NULL | Spectrum SH Plates")
# _add("sq", SeriesKind., NULL |              NULL |     NULL | SB Spectrum Plates")
# _add("t", SeriesKind., NULL |              NULL |     NULL | Sky Photometer")
# _add("u", SeriesKind., NULL |              NULL |     NULL | Pole Star Recorder")
# _add("v", SeriesKind., NULL |              NULL |     NULL | Misc Basement")
# _add("vq", SeriesKind., NULL |              NULL |     NULL | Heinrich VQ (Spectrograph)")
# _add("w", SeriesKind., NULL |              NULL |     NULL | Slit Spectrograph")
# _add("wa", SeriesKind., NULL |              NULL |     NULL | Cooke Lens #11366 (Western Association)")
# _add("ww", SeriesKind., NULL |              NULL |     NULL | Heinrich Camera Lens")
_add("x", SeriesKind.NARROW, 42.3, 0.33, "13-inch Boyden Refractor")
_add("y", SeriesKind.NARROW, np.nan, 0.203, "8-inch Clark Doublet")
_add("yb", SeriesKind.NARROW, 55, 0.508, "YSO Double Astrograph")
# _add("z", SeriesKind., NULL |              NULL |     0.33 | 13-inch Boyden Refractor (spectrum plates)")
