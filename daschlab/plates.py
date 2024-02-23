# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Lists of plates.
"""

from datetime import datetime
import io
import os
import re
import time
from typing import Dict, Iterable, Optional, Tuple, Union
from urllib.parse import urlencode
import warnings

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table, Row
from astropy.time import Time
from astropy import units as u
from astropy.utils.masked import Masked
from astropy.wcs import WCS
from bokeh.plotting import figure, show
import cairo
import numpy as np
from PIL import Image
from pytz import timezone
from pywwt.layers import ImageLayer
import requests

from .series import SERIES, SeriesKind

__all__ = ["Plates", "PlateReferenceType", "PlateRow", "PlateSelector"]


_API_URL = "http://dasch.rc.fas.harvard.edu/_v2api/queryplates.php"


def _daschtime_to_isot(t: str) -> str:
    outer_bits = t.split("T", 1)
    inner_bits = outer_bits[-1].split("-")
    outer_bits[-1] = ":".join(inner_bits)
    return "T".join(outer_bits)


_PLATE_NAME_REGEX = re.compile(r"^([a-zA-Z]+)([0-9]{1,5})$")


def _parse_plate_name(name: str) -> Tuple[str, int]:
    try:
        m = _PLATE_NAME_REGEX.match(name)
        series = m[1].lower()
        platenum = int(m[2])
    except Exception:
        raise Exception(f"invalid plate name `{name}`")

    return series, platenum


_COLTYPES = {
    "series": str,
    "platenum": int,
    "scannum": int,
    "mosnum": int,
    "expnum": int,
    "solnum": int,
    "class": str,
    "ra": float,
    "dec": float,
    "exptime": float,
    "jd": float,
    # "epoch": float,
    "wcssource": str,
    "scandate": _daschtime_to_isot,
    "mosdate": _daschtime_to_isot,
    "rotation": int,
    "binflags": int,
    "centerdist": float,
    "edgedist": float,
}


class PlateRow(Row):
    def show(self) -> ImageLayer:
        return self._table.show(self)

    def plate_id(self) -> str:
        return f"{self['series']}{self['platenum']:05d}_{self['mosnum']:02d}"


PlateReferenceType = Union[PlateRow, int]


class PlateSelector:
    """
    A magic object to help enable plate-list filtering.
    """

    _plates: "Plates"
    _apply = None

    def __init__(self, plates: "Plates", apply):
        self._plates = plates
        self._apply = apply

    def where(self, row_mask, **kwargs) -> "Plates":
        return self._apply(self, row_mask, **kwargs)

    def rejected(self, **kwargs) -> "Plates":
        m = self._plates["reject"] != 0
        return self._apply(self, m, **kwargs)

    def rejected_with(self, tag: str, strict: bool = False, **kwargs) -> "Plates":
        bitnum0 = None

        if self._plates._rejection_tags is not None:
            bitnum0 = self._plates._rejection_tags.get(tag)

        if bitnum0 is not None:
            m = (self._plates["reject"] & (1 << bitnum0)) != 0
        elif strict:
            raise Exception(f"unknown rejection tag `{tag}`")
        else:
            m = np.zeros(len(self._plates), dtype=bool)

        return self._apply(self, m, **kwargs)

    def local_id(self, local_id: int, **kwargs) -> "Plates":
        m = self._plates["local_id"] == local_id
        return self._apply(self, m, **kwargs)

    def scanned(self, **kwargs) -> "Plates":
        m = ~self._plates["scannum"].mask
        return self._apply(self, m, **kwargs)

    def wcs_solved(self, **kwargs) -> "Plates":
        m = self._plates["wcssource"] == "imwcs"
        return self._apply(self, m, **kwargs)

    def series(self, series: str, **kwargs) -> "Plates":
        m = self._plates["series"] == series
        return self._apply(self, m, **kwargs)

    def narrow(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.NARROW for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def patrol(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.PATROL for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def meteor(self, **kwargs) -> "Plates":
        m = [SERIES[k].kind == SeriesKind.METEOR for k in self._plates["series"]]
        return self._apply(self, m, **kwargs)

    def plate_names(self, names: Iterable[str], **kwargs) -> "Plates":
        # This feels so inefficient, but it's not obvious to me how to do any better
        m = np.zeros(len(self._plates), dtype=bool)

        for name in names:
            series, platenum = _parse_plate_name(name)
            this_one = (self._plates["series"] == series) & (
                self._plates["platenum"] == platenum
            )
            m |= this_one

        return self._apply(self, m, **kwargs)

    def jyear_range(self, jyear_min: float, jyear_max: float, **kwargs) -> "Plates":
        m = (self._plates["obs_date"].jyear >= jyear_min) & (
            self._plates["obs_date"].jyear <= jyear_max
        )
        return self._apply(self, m, **kwargs)


class Plates(Table):
    _sess: "daschlab.Session" = None
    _rejection_tags: Optional[Dict[str, int]] = None
    _layers: Dict[int, ImageLayer] = None
    Row = PlateRow

    # Filtering infrastructure

    def _copy_subset(self, keep, verbose: bool) -> "Plates":
        new = self.copy(True)
        new = new[keep]

        if verbose:
            nn = len(new)
            print(f"Dropped {len(self) - nn} rows; {nn} remaining")

        new._sess = self._sess
        new._rejection_tags = self._rejection_tags
        return new

    @property
    def match(self) -> PlateSelector:
        return PlateSelector(self, lambda _sel, m: m)

    @property
    def count(self) -> PlateSelector:
        return PlateSelector(self, lambda _sel, m: m.sum())

    def _apply_keep_only(self, _selector, flags, verbose=True) -> "Plates":
        return self._copy_subset(flags, verbose)

    @property
    def keep_only(self) -> PlateSelector:
        return PlateSelector(self, self._apply_keep_only)

    def _apply_drop(self, _selector, flags, verbose=True) -> "Plates":
        return self._copy_subset(~flags, verbose)

    @property
    def drop(self) -> PlateSelector:
        return PlateSelector(self, self._apply_drop)

    def _make_reject_selector(
        self, tag: str, verbose: bool, apply_func
    ) -> PlateSelector:
        if self._rejection_tags is None:
            self._rejection_tags = {}

        bitnum0 = self._rejection_tags.get(tag)

        if bitnum0 is None:
            bitnum0 = len(self._rejection_tags)
            if bitnum0 > 63:
                raise Exception("you cannot have more than 64 distinct rejection tags")

            if verbose:
                print(f"Assigned rejection tag `{tag}` to bit number {bitnum0 + 1}")

            self._rejection_tags[tag] = bitnum0

        selector = PlateSelector(self, apply_func)
        selector._bitnum0 = bitnum0
        return selector

    def _apply_reject(self, selector, flags, verbose: bool = True):
        n_before = (self["reject"] != 0).sum()
        self["reject"][flags] |= 1 << selector._bitnum0
        n_after = (self["reject"] != 0).sum()

        if verbose:
            print(
                f"Marked {n_after - n_before} new rows as rejected; {n_after} total are rejected"
            )

    def reject(self, tag: str, verbose: bool = True) -> PlateSelector:
        return self._make_reject_selector(tag, verbose, self._apply_reject)

    def _apply_reject_unless(self, selector, flags, verbose: bool = True):
        return self._apply_reject(selector, ~flags, verbose)

    def reject_unless(self, tag: str, verbose: bool = True) -> PlateSelector:
        return self._make_reject_selector(tag, verbose, self._apply_reject_unless)

    # Non-filtering actions on this list

    def series_info(self) -> Table:
        g = self.drop.rejected(verbose=False).group_by("series")

        t = Table()
        t["series"] = [t[0] for t in g.groups.keys]
        t["count"] = np.diff(g.groups.indices)
        t["kind"] = [SERIES[t[0]].kind for t in g.groups.keys]
        t["plate_scale"] = (
            np.array([SERIES[t[0]].plate_scale for t in g.groups.keys])
            * u.arcsec
            / u.mm
        )
        t["aperture"] = np.array([SERIES[t[0]].aperture for t in g.groups.keys]) * u.m
        t["description"] = [SERIES[t[0]].description for t in g.groups.keys]

        t.sort(["count"], reverse=True)

        return t

    def time_coverage(self) -> figure:
        from scipy.stats import gaussian_kde

        plot_years = np.linspace(1875, 1995, 200)
        plate_years = self.drop.rejected(verbose=False)["obs_date"].jyear

        kde = gaussian_kde(plate_years, bw_method="scott")
        plate_years_smoothed = kde(plot_years)

        # try to get the normalization right: integral of the
        # curve is equal to the number of plates, so that the
        # Y axis is plates per year.

        integral = plate_years_smoothed.sum() * (plot_years[1] - plot_years[0])
        plate_years_smoothed *= len(self) / integral

        p = figure(
            x_axis_label="Year",
            y_axis_label="Plates per year (smoothed)",
        )
        p.line(plot_years, plate_years_smoothed)
        show(p)
        return p

    def show(self, plate_ref: PlateReferenceType) -> ImageLayer:
        plate = self._sess._resolve_plate_reference(plate_ref)

        if self._layers is None:
            self._layers = {}

        local_id = plate["local_id"]
        il = self._layers.get(local_id)

        if il is not None:
            return il  # TODO: bring it to the top, etc?

        fits_relpath = self._sess.cutout(plate)
        if fits_relpath is None:
            from . import InteractiveError

            raise InteractiveError(
                f"cannot create a cutout for plate #{local_id:05d} ({plate.plate_id()})"
            )

        il = self._sess.wwt().layers.add_image_layer(str(self._sess.path(fits_relpath)))
        self._layers[local_id] = il
        return il

    def export_cutouts_to_pdf(self, pdfpath: str, **kwargs):
        _pdf_export(pdfpath, self._sess, self, **kwargs)


def _query_plates(
    center: SkyCoord,
    radius: u.Quantity,
) -> Plates:
    return _postproc_plates(_get_plate_cols(center, radius))


def _get_plate_cols(
    center: SkyCoord,
    radius: u.Quantity,
) -> Plates:
    radius = Angle(radius)

    # API-based query

    url = (
        _API_URL
        + "?"
        + urlencode(
            {
                "cra_deg": center.ra.deg,
                "cdec_deg": center.dec.deg,
                "radius_arcsec": radius.arcsec,
            }
        )
    )

    colnames = None
    coltypes = None
    coldata = None
    saw_sep = False

    with requests.get(url, stream=True) as resp:
        for line in resp.iter_lines():
            line = line.decode("utf-8")
            pieces = line.rstrip().split("\t")

            if colnames is None:
                colnames = pieces
                coltypes = [_COLTYPES.get(c) for c in colnames]
                coldata = [[] if t is not None else None for t in coltypes]
            elif not saw_sep:
                saw_sep = True
            else:
                for row, ctype, cdata in zip(pieces, coltypes, coldata):
                    if ctype is not None:
                        cdata.append(ctype(row))

    return dict(t for t in zip(colnames, coldata) if t[1] is not None)


def _dasch_date_as_datetime(date: str) -> Optional[datetime]:
    if not date:
        return None

    p = date.split("T")
    p[1] = p[1].replace("-", ":")
    naive = datetime.fromisoformat("T".join(p))
    tz = timezone("US/Eastern")
    return tz.localize(naive)


def _dasch_dates_as_time_array(dates) -> Time:
    """
    Convert an iterable of DASCH dates to an Astropy Time array, with masking of
    missing values.
    """
    # No one had any photographic plates in 1800! But the exact value here
    # doesn't matter; we just need something accepted by the datetime-Time
    # constructor.
    invalid_dt = datetime(1800, 1, 1)

    invalid_indices = []
    dts = []

    for i, dstr in enumerate(dates):
        dt = _dasch_date_as_datetime(dstr)
        if dt is None:
            invalid_indices.append(i)
            dt = invalid_dt

        dts.append(dt)

    times = Time(dts, format="datetime")

    for i in invalid_indices:
        times[i] = np.ma.masked

    # If we don't do this, Astropy is unable to roundtrip masked times out of
    # the ECSV format.
    times.format = "isot"

    return times


def _postproc_plates(input_cols) -> Plates:
    table = Plates(masked=True)

    scannum = np.array(input_cols["scannum"], dtype=np.int8)
    mask = scannum == -1

    # create a unitless (non-Quantity) column, unmasked:
    all_c = lambda c, dt: np.array(input_cols[c], dtype=dt)

    # unitless column, masked:
    mc = lambda c, dt: np.ma.array(input_cols[c], mask=mask, dtype=dt)

    # Quantity column, unmasked:
    all_q = lambda c, dt, unit: u.Quantity(np.array(input_cols[c], dtype=dt), unit)

    # Quantity column, masked:
    mq = lambda c, dt, unit: Masked(
        u.Quantity(np.array(input_cols[c], dtype=dt), unit), mask
    )

    # Quantity column, masked, with extra flag values to mask:
    def extra_mq(c, dt, unit, flagval):
        a = np.array(input_cols[c], dtype=dt)
        extra_mask = mask | (a == flagval)
        return Masked(u.Quantity(a, unit), extra_mask)

    table["series"] = input_cols["series"]
    table["platenum"] = np.array(input_cols["platenum"], dtype=np.uint32)
    table["scannum"] = mc("scannum", np.uint8)
    table["mosnum"] = mc("mosnum", np.uint8)
    table["expnum"] = np.array(input_cols["expnum"], dtype=np.uint8)
    table["solnum"] = mc("solnum", np.uint8)
    table["class"] = input_cols["class"]
    table["pos"] = SkyCoord(
        ra=input_cols["ra"] * u.deg,
        dec=input_cols["dec"] * u.deg,
        frame="icrs",
    )
    table["exptime"] = np.array(input_cols["exptime"], dtype=np.float32) * u.minute
    table["obs_date"] = Time(input_cols["jd"], format="jd")
    table["wcssource"] = input_cols["wcssource"]
    table["scan_date"] = _dasch_dates_as_time_array(input_cols["scandate"])
    table["mos_date"] = _dasch_dates_as_time_array(input_cols["mosdate"])
    table["rotation_deg"] = mc("rotation", np.uint16)
    table["binflags"] = mc("binflags", np.uint8)
    table["center_distance"] = (
        np.array(input_cols["centerdist"], dtype=np.float32) * u.cm
    )
    table["edge_distance"] = np.array(input_cols["edgedist"], dtype=np.float32) * u.cm

    table.sort(["obs_date"])

    table["local_id"] = np.arange(len(table))
    table["reject"] = np.zeros(len(table), dtype=np.uint64)

    return table


# PDF export


PAGE_WIDTH = 612  # US Letter paper size, points
PAGE_HEIGHT = 792
MARGIN = 36  # points; 0.5 inch
CUTOUT_WIDTH = PAGE_WIDTH - 2 * MARGIN
DSF = 4  # downsampling factor
LINE_HEIGHT = 14  # points


def _data_to_image(data: np.array) -> Image:
    """
    Here we implement the grayscale method used by the DASCH website. The input is a
    float array and the output is a PIL/Pillow image.

    1. linearly rescale data clipping bottom 2% and top 1% of pixel values
       (pnmnorm)
    2. scale to 0-255 (pnmdepth)
    3. invert scale (pnminvert)
    4. flip vertically (pnmflip; done prior to this function)
    5. apply a 2.2 gamma scale (pnmgamma)

    We reorder a few steps here to simplify the computations. The results aren't
    exactly identical to what you get across the NetPBM steps but they're
    visually indistinguishable in most cases.
    """

    dmin, dmax = np.percentile(data, [2, 99])

    if dmax == dmin:
        data = np.zeros(data.shape, dtype=np.uint8)
    else:
        data = np.clip(data, dmin, dmax)
        data = (data - dmin) / (dmax - dmin)
        data = 1 - data
        data **= 1 / 2.2
        data *= 255
        data = data.astype(np.uint8)

    return Image.fromarray(data, mode="L")


def _pdf_export(
    pdfpath: str,
    sess: "Session",
    plates: Plates,
    refcat_stdmag_limit: Optional[float] = None,
):
    PLOT_CENTERING_CIRCLE = True
    PLOT_SOURCES = True

    center_pos = sess.query().pos_as_skycoord()
    ra_text = center_pos.ra.to_string(unit=u.hour, sep=":", precision=3)
    dec_text = center_pos.dec.to_string(
        unit=u.degree, sep=":", alwayssign=True, precision=2
    )
    center_text = f"{ra_text} {dec_text} J2000"

    name = sess.query().name
    if name:
        center_text = f"{name} = {center_text}"

    # First thing, ensure that we have all of our cutouts

    n_orig = len(plates)
    plates = plates.keep_only.wcs_solved(verbose=False).drop.rejected(verbose=False)
    n_filtered = len(plates)

    if not n_filtered:
        raise Exception("no plates remain after filtering")

    if n_filtered != n_orig:
        print(
            f"Filtered input list of {n_orig} plates down to {n_filtered} for display"
        )

    print(
        f"Ensuring that we have cutouts for {n_filtered} plates, this may take a while ..."
    )
    t0 = time.time()

    for plate in plates:
        if not sess.cutout(plate):
            raise Exception(
                f"unable to fetch cutout for plate LocalId {plate['local_id']}"
            )

    elapsed = time.time() - t0
    print(f"... completed in {elapsed:.0f} seconds")

    # Set up refcat sources (could add filtering, etc.)

    if PLOT_SOURCES:
        refcat = sess.refcat()

        if refcat_stdmag_limit is not None:
            refcat = refcat[~refcat["stdmag"].mask]
            refcat = refcat[refcat["stdmag"] <= refcat_stdmag_limit]

        cat_pos = refcat["pos"]
        print(f"Overlaying {len(refcat)} refcat sources")

    # We're ready to start rendering pages

    t0 = time.time()
    print("Rendering pages ...")

    n_pages = 0
    label_y0 = None
    wcs = None

    with cairo.PDFSurface(pdfpath, PAGE_WIDTH, PAGE_HEIGHT) as pdf:
        for plate in plates:
            local_id = plate["local_id"]
            series = plate["series"]
            platenum = plate["platenum"]
            mosnum = plate["mosnum"]
            plateclass = plate["class"] or "(none)"
            jd = plate["obs_date"].jd
            epoch = plate["obs_date"].jyear
            # platescale = plate["plate_scale"]
            exptime = plate["exptime"]
            center_distance_cm = plate["center_distance"]
            edge_distance_cm = plate["edge_distance"]

            fits_relpath = sess.cutout(plate)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with fits.open(str(sess.path(fits_relpath))) as hdul:
                    data = hdul[0].data
                    astrom_type = hdul[0].header["D_ASTRTY"]

                    if wcs is None:
                        wcs = WCS(hdul[0].header)

                        # Calculate the size of a 20 arcsec circle
                        x0 = data.shape[1] / 2
                        y0 = data.shape[0] / 2
                        c0 = wcs.pixel_to_world(x0, y0)

                        if c0.dec.deg >= 0:
                            c1 = c0.directional_offset_by(180 * u.deg, 1 * u.arcmin)
                        else:
                            c1 = c0.directional_offset_by(0 * u.deg, 1 * u.arcmin)

                        x1, y1 = wcs.world_to_pixel(c1)
                        ref_radius = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                        ref_radius /= DSF  # account for the downsampling

            # downsample to reduce data volume
            height = data.shape[0] // DSF
            width = data.shape[1] // DSF
            h_trunc = height * DSF
            w_trunc = width * DSF
            data = data[:h_trunc, :w_trunc].reshape((height, DSF, width, DSF))
            data = data.mean(axis=(1, 3))

            # Important! FITS is rendered "bottoms-up" so we must flip vertically for
            # the RGB imagery, while will be top-down!
            data = data[::-1]

            # Our data volumes get out of control unless we JPEG-compress
            pil_image = _data_to_image(data)
            jpeg_encoded = io.BytesIO()
            pil_image.save(jpeg_encoded, format="JPEG")
            jpeg_encoded.seek(0)
            jpeg_encoded = jpeg_encoded.read()

            img = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
            img.set_mime_data("image/jpeg", jpeg_encoded)

            cr = cairo.Context(pdf)
            cr.save()
            cr.translate(MARGIN, MARGIN)
            f = CUTOUT_WIDTH / width
            cr.scale(f, f)
            cr.set_source_surface(img, 0, 0)
            cr.paint()
            cr.restore()

            # reference circle overlay

            if PLOT_CENTERING_CIRCLE:
                cr.save()
                cr.set_source_rgb(0.5, 0.5, 0.9)
                cr.set_line_width(1)
                cr.arc(
                    CUTOUT_WIDTH / 2 + MARGIN,  # x
                    CUTOUT_WIDTH / 2 + MARGIN,  # y -- hardcoding square aspect ratio
                    ref_radius * f,  # radius
                    0.0,  # angle 1
                    2 * np.pi,  # angle 2
                )
                cr.stroke()
                cr.restore()

            # Source overlay

            if PLOT_SOURCES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    this_pos = cat_pos.apply_space_motion(
                        Time(epoch, format="decimalyear")
                    )

                this_px = np.array(wcs.world_to_pixel(this_pos))
                this_px /= DSF  # account for our downsampling
                x = this_px[0]
                y = this_px[1]
                y = height - y  # account for FITS/JPEG y axis parity flip
                ok = (x > -0.5) & (x < width - 0.5) & (y > -0.5) & (y < height - 0.5)

                cr.save()
                cr.set_source_rgb(0, 1, 0)
                cr.set_line_width(1)

                for i in range(len(ok)):
                    if not ok[i]:
                        continue

                    ix = x[i] * f + MARGIN
                    iy = y[i] * f + MARGIN
                    cr.move_to(ix - 4, iy - 4)
                    cr.line_to(ix + 4, iy + 4)
                    cr.stroke()
                    cr.move_to(ix - 4, iy + 4)
                    cr.line_to(ix + 4, iy - 4)
                    cr.stroke()

                cr.restore()

            # Labeling

            if label_y0 is None:
                label_y0 = 2 * MARGIN + height * f

            with warnings.catch_warnings():
                # Lots of ERFA complaints for our old years
                warnings.simplefilter("ignore")
                t_iso = Time(jd, format="jd").isot

            cr.save()
            cr.select_font_face(
                "mono", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            linenum = 0

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.save()
            cr.set_font_size(18)
            cr.show_text(
                f"Plate {series.upper()}{platenum:05d} - localId {local_id:5d} - {epoch:.5f}"
            )
            cr.restore()
            linenum += 1.5

            # XXX hardcoding params
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(f"Image: center {center_text}")
            linenum += 1
            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text("  halfsize 600 arcsec, orientation N up, E left")
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(
                f"Mosaic {series:>4}{platenum:05d}_{mosnum:02d}: "
                f"astrom {astrom_type.upper():3} "
                f"class {plateclass:12} "
                f"exp {exptime:5.1f} (min) "
                # f"scale {platescale:7.2f} (arcsec/mm)"
            )
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            url = f"http://dasch.rc.fas.harvard.edu/showplate.php?series={series}&plateNumber={platenum}"
            cr.tag_begin(cairo.TAG_LINK, f"uri='{url}'")
            cr.set_source_rgb(0, 0, 1)
            cr.show_text(url)
            cr.set_source_rgb(0, 0, 0)
            cr.tag_end(cairo.TAG_LINK)
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(f"Midpoint: epoch {epoch:.5f} = HJD {jd:.6f} = {t_iso}")
            linenum += 1

            cr.move_to(MARGIN, label_y0 + linenum * LINE_HEIGHT)
            cr.show_text(
                f"Cutout location: {center_distance_cm:4.1f} cm from center, {edge_distance_cm:4.1f} cm from edge"
            )
            linenum += 1
            cr.restore()

            pdf.show_page()
            n_pages += 1

            if n_pages % 100 == 0:
                print(f"- page {n_pages} - {fits_relpath}")

    elapsed = time.time() - t0
    info = os.stat(pdfpath)
    print(
        f"... completed in {elapsed:.0f} seconds and saved to `{pdfpath}` ({info.st_size / 1024**2:.1f} MiB)"
    )
