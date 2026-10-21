"""Shared RTKLIB-based GNSS post-processing tools."""

from importlib.metadata import version

from .config import RtkConfig, navsys_mask
from .geodesy import GRS80, WGS84, Ellipsoid
from .pos import read_header, read_pos
from .rinex import crop_rinex, header_llh, rinex_span
from .rnx2rtkp import run_rnx2rtkp, solve_static
from .timesys import GPS_EPOCH, GPST_UTC_OFFSET_S, utc_to_gpst

__version__ = version("gnsskit")
__all__ = [
    "GPST_UTC_OFFSET_S",
    "Ellipsoid",
    "GPS_EPOCH",
    "RtkConfig",
    "WGS84",
    "crop_rinex",
    "header_llh",
    "navsys_mask",
    "read_header",
    "read_pos",
    "rinex_span",
    "run_rnx2rtkp",
    "solve_static",
    "utc_to_gpst",
]
