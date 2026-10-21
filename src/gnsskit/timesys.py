"""Time-system helpers (GPST, UTC)."""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

GPS_EPOCH = pd.Timestamp("1980-01-06")
GPST_UTC_OFFSET_S = 18  # GPST - UTC leap seconds (valid since 2017)

T = TypeVar("T", pd.Timestamp, pd.Series)


def utc_to_gpst(t: T, leap_s: int = GPST_UTC_OFFSET_S) -> T:
    """Shift UTC time(s) to GPST.

    Args:
        t: Timestamp or Series of UTC times.
        leap_s: GPST - UTC [s].

    Returns:
        The shifted time(s), same type as `t`.
    """
    return t + pd.Timedelta(seconds=leap_s)


def gps_seconds(t: pd.Timestamp | pd.Series) -> float | pd.Series:
    """Seconds since the GPS epoch (1980-01-06), which avoids week roll-over problems.

    Args:
        t: Timestamp or Series of GPST times.

    Returns:
        Seconds as a float, or a Series of floats.
    """
    return (t - GPS_EPOCH) / pd.Timedelta(seconds=1)
