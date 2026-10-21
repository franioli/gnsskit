"""Sampling a kinematic trajectory (`read_pos` output): at given times or inside windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geodesy import enu_diff
from .pos import POS_COORD_COLS, POS_SD_COLS
from .timesys import GPS_EPOCH


def interpolate(pos: pd.DataFrame, gps_s: np.ndarray) -> pd.DataFrame:
    """Linear interpolation of the trajectory at the given times.

    Q is the worse of the two neighbouring epochs, ns the smaller. Times outside the
    trajectory are not allowed.

    Args:
        pos: Trajectory from `read_pos`, with at least two epochs.
        gps_s: Times to interpolate at, seconds since the GPS epoch.

    Returns:
        DataFrame in the format of `pos`, one row per requested time.
    """
    fmt = pos.attrs["format"]
    t = pos["gps_s"].to_numpy()
    idx = np.searchsorted(t, gps_s).clip(1, len(pos) - 1)
    lo, hi = idx - 1, idx
    q, ratio = pos["Q"].to_numpy(), pos["ratio"].to_numpy()
    cols = POS_COORD_COLS[fmt] + POS_SD_COLS[fmt]
    out = pd.DataFrame({c: np.interp(gps_s, t, pos[c]) for c in cols})
    out["Q"] = np.maximum(q[lo], q[hi])
    out["ns"] = np.minimum(pos["ns"].to_numpy()[lo], pos["ns"].to_numpy()[hi])
    out["ratio"] = np.where(q[lo] == 1, ratio[lo], ratio[hi])
    out["age"] = 0.0
    out["gps_s"] = gps_s
    out["t"] = GPS_EPOCH + pd.to_timedelta(gps_s, unit="s")
    out.attrs["format"] = fmt
    return out


def average(
    pos: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, fix_only: bool = True
) -> dict[str, float | int | bool]:
    """Mean position of the llh epochs in [start, end] with the scatter in E, N, U.

    Args:
        pos: llh trajectory from `read_pos`.
        start: Window start, GPST.
        end: Window end, GPST.
        fix_only: Average the fixed epochs (Q == 1) only.

    Returns:
        Dict with n_epochs, fix_pct, n_used and, if any epoch is used, fixed, lat, lon, h
        [deg, deg, m] and sd_e, sd_n, sd_u [m] (sample standard deviations).

    Raises:
        ValueError: If `pos` is not in llh format.
    """
    if pos.attrs.get("format") != "llh":
        raise ValueError("average needs a llh trajectory (out-solformat=llh)")
    seg = pos[(pos.t >= start) & (pos.t <= end)]
    n_all = len(seg)
    row: dict[str, float | int | bool] = {
        "n_epochs": n_all,
        "fix_pct": 100.0 * (seg.Q == 1).sum() / n_all if n_all else 0.0,
    }
    if fix_only:
        seg = seg[seg.Q == 1]
    row["n_used"] = len(seg)
    if len(seg):
        lat, lon, h = seg.lat.mean(), seg.lon.mean(), seg.h.mean()
        e, n, u = enu_diff(seg.lat, seg.lon, seg.h, lat, lon, h).T
        ddof = 1 if len(seg) > 1 else 0
        row |= {
            "fixed": bool((seg.Q == 1).all()),
            "lat": lat,
            "lon": lon,
            "h": h,
            "sd_e": e.std(ddof=ddof),
            "sd_n": n.std(ddof=ddof),
            "sd_u": u.std(ddof=ddof),
        }
    return row
