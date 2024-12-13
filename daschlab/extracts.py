# Copyright the President and Fellows of Harvard College
# Licensed under the MIT License

"""
DASCH photometry "extracts".
"""

from typing import Dict, Optional

from astropy.coordinates import SkyCoord
from astropy import units as u
from pywwt.layers import TableLayer

from .apiclient import ApiClient
from .photometry import _tabulate_phot_data, _postproc_phot_table, Photometry


__all__ = [
    "Extract",
]


class Extract(Photometry):
    # Row = ...
    # Selector = ...

    def _sess(self) -> "daschlab.Session":
        from . import _lookup_session

        return _lookup_session(self.meta["daschlab_sess_key"])

    def _layers(self) -> Dict[str, TableLayer]:
        return self._sess()._extract_table_layer_cache

    def _exp_id(self) -> str:
        return self.meta["daschlab_exposure_id"]

    def show(
        self, mag_limit: Optional[float] = None, size_vmin_bias: float = 1.0
    ) -> TableLayer:
        """
        Returns
        =======
        `pywwt.layers.TableLayer`
            This is the WWT table layer object corresponding to the displayed
            catalog. You can use it to programmatically control aspects of how
            the data are displayed, such as the which column sets the point
            size.

        Notes
        =====
        In order to use this method, you must first have called
        `daschlab.Session.connect_to_wwt()`.
        """
        tl = self._layers().get(self._exp_id())
        if tl is not None:
            return tl  # TODO: bring it to the top, etc?

        # TODO: pywwt can't handle Astropy tables that use a SkyCoord
        # to hold positional information. That should be fixed, but it
        # will take some time. In the meantime, hack around it.

        compat_table = self.copy()
        compat_table["ra"] = self["pos"].ra.deg
        compat_table["dec"] = self["pos"].dec.deg
        del compat_table["pos"]

        if mag_limit is None:
            mag_limit = compat_table["magcal_magdep"].max()

        wwt = self._sess().wwt()
        tl = wwt.layers.add_table_layer(compat_table)
        self._layers()[self._exp_id()] = tl

        tl.marker_type = "circle"
        # FIXME: the WWT engine's logic to guess column types fails *badly* with
        # these tables, requiring various overrides we apply here. WWT should be
        # fixed to be a lot less cavalier about this.
        tl.lat_att = "dec"
        tl.lon_att = "ra"
        tl.lon_unit = u.deg
        tl.size_att = "magcal_magdep"
        tl.size_vmin = mag_limit + size_vmin_bias
        tl.size_vmax = compat_table["magcal_magdep"].min()
        tl.size_scale = 10.0
        return tl


def _query_extract(
    client: ApiClient,
    refcat: str,
    plate_id: str,
    solnum: int,
    center: SkyCoord,
) -> Extract:
    payload = {
        "refcat": refcat,
        "plate_id": plate_id,
        "solution_number": int(solnum),  # `json` will reject Numpy integer types
        "center_ra_deg": center.ra.deg,
        "center_dec_deg": center.dec.deg,
    }

    data = client.invoke("/dasch/dr7/platephot", payload)
    if not isinstance(data, list):
        from . import InteractiveError

        raise InteractiveError(f"platephot API request failed: {data!r}")

    return _postproc_phot_table(Extract, _tabulate_phot_data(data))
