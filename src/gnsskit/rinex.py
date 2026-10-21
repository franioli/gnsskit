"""RINEX observation file helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .geodesy import ecef_to_llh


def rinex_span(obs: Path | str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last epoch (GPST) of a RINEX 3 observation file.

    Event records without a timestamp are skipped; raises ValueError if no epoch is found.
    """
    first = last = None
    with Path(obs).open() as f:
        for line in f:
            if not line.startswith(">"):
                continue
            try:
                y, mo, d, h, mi, s = line[1:].split()[:6]
                last = pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi)) + pd.Timedelta(
                    seconds=float(s)
                )
            except ValueError:
                continue
            if first is None:
                first = last
    if first is None or last is None:
        raise ValueError(f"no RINEX 3 epochs found in {obs}")
    return first, last


def header_llh(obs: Path | str) -> tuple[float, float, float]:
    """WGS84 lat, lon [deg] and ellipsoidal height [m] of the antenna from the RINEX header.

    Uses `APPROX POSITION XYZ` plus the vertical `ANTENNA: DELTA H/E/N`, if present.
    """
    xyz, dh = None, 0.0
    with Path(obs).open() as f:
        for line in f:
            if "APPROX POSITION XYZ" in line:
                xyz = tuple(map(float, line[:42].split()))
            elif "ANTENNA: DELTA H/E/N" in line:
                dh = float(line[:14])
            elif "END OF HEADER" in line:
                break
    if xyz is None or xyz == (0.0, 0.0, 0.0):
        raise ValueError(f"no valid APPROX POSITION XYZ in {obs}")
    lat, lon, h = ecef_to_llh(*xyz)
    return float(lat), float(lon), float(h) + dh
