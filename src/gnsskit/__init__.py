"""Shared RTKLIB-based GNSS post-processing tools."""

from importlib.metadata import version

from .config import RtkConfig, navsys_mask
from .pos import read_pos
from .rinex import rinex_span
from .rnx2rtkp import run_rnx2rtkp, solve_static
from .timesys import utc_to_gpst

__version__ = version("gnsskit")
__all__ = [
    "RtkConfig",
    "navsys_mask",
    "read_pos",
    "rinex_span",
    "run_rnx2rtkp",
    "solve_static",
    "utc_to_gpst",
]
