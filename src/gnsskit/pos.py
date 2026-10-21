"""Reading RTKLIB .pos solution files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .timesys import GPST_UTC_OFFSET_S, gps_seconds, utc_to_gpst

# .pos output formats (`out-solformat`). Column names for the coordinates and for the
# 6 covariance terms (3 standard deviations + 3 covariances), as printed by RTKLIB.
POS_COORD_COLS = {
    "llh": ["lat", "lon", "h"],
    "xyz": ["x", "y", "z"],
    "enu": ["e", "n", "u"],
}
POS_SD_COLS = {
    "llh": ["sdn", "sde", "sdu", "sdne", "sdeu", "sdun"],
    "xyz": ["sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx"],
    "enu": ["sde", "sdn", "sdu", "sden", "sdnu", "sdue"],
}
# First coordinate label in the .pos column header -> format
POS_FORMAT_BY_HEADER = {
    "latitude(deg)": "llh",
    "x-ecef(m)": "xyz",
    "e-baseline(m)": "enu",
}


def _pos_header(path: Path) -> tuple[str, str]:
    """Time system ('GPST' or 'UTC') and coordinate format ('llh', 'xyz', 'enu') of a .pos.

    Raises ValueError for unsupported formats (e.g. degrees-minutes-seconds, JST). Without
    a column header line (out-outhead=off) the RTKLIB default is assumed: GPST, llh in deg.
    """
    with path.open() as f:
        for line in f:
            if not line.startswith("%"):
                break
            tokens = line.lstrip("% ").split()
            # The column header is the only comment line that starts with a time system
            if len(tokens) < 2 or tokens[0].upper() not in ("GPST", "UTC", "JST"):
                continue
            tsys = tokens[0].upper()
            if tsys == "JST":
                raise ValueError(f"{path}: unsupported time system JST, use GPST or UTC")
            if tokens[1] not in POS_FORMAT_BY_HEADER:
                raise ValueError(
                    f"{path}: unsupported coordinate format '{tokens[1]}', "
                    "use llh (deg), xyz or enu"
                )
            return tsys, POS_FORMAT_BY_HEADER[tokens[1]]
    return "GPST", "llh"


def read_pos(
    paths: Path | str | Sequence[Path | str], leap_s: int = GPST_UTC_OFFSET_S
) -> pd.DataFrame:
    """Read one or more RTKLIB .pos files into a time-sorted DataFrame.

    Columns `t` (GPST) and `gps_s` (seconds since the GPS epoch) are added. The coordinate
    format is detected from the header: llh (lat, lon, h), xyz (x, y, z) or enu (e, n, u),
    with the matching covariance columns (see POS_SD_COLS); it is stored in
    `df.attrs["format"]`. Only `out-timeform=hms` is supported. UTC files are shifted to
    GPST by `leap_s`. Mixing formats, or reading dms/JST files, raises ValueError. Files
    with no solution rows are skipped (the result may be empty).
    """
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    headers = {p: _pos_header(p) for p in paths}
    formats = {fmt for _, fmt in headers.values()}
    if len(formats) > 1:
        raise ValueError(f"cannot mix .pos coordinate formats: {sorted(formats)}")
    fmt = formats.pop() if formats else "llh"
    names = [
        "date", "time", *POS_COORD_COLS[fmt], "Q", "ns", *POS_SD_COLS[fmt], "age", "ratio"
    ]  # fmt: skip

    frames = []
    for p, (tsys, _) in headers.items():
        try:
            df = pd.read_csv(p, comment="%", sep=r"\s+", header=None, names=names)
        except pd.errors.EmptyDataError:
            continue
        df["t"] = pd.to_datetime(df.date + " " + df.time)
        if tsys == "UTC":
            df["t"] = utc_to_gpst(df["t"], leap_s)
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
        out["gps_s"] = gps_seconds(out["t"])
    else:
        out = pd.DataFrame(columns=[*names, "t", "gps_s"])
    out.attrs["format"] = fmt
    return out


def read_header(path: Path | str) -> list[str]:
    """The `%` header lines of a .pos file (without line endings)."""
    return [
        ln.rstrip("\r\n")
        for ln in Path(path).read_text().splitlines()
        if ln.startswith("%")
    ]


def read_ref_pos(path: Path | str) -> np.ndarray:
    """Base position printed in the `% ref pos` header line of an xyz .pos file."""
    with Path(path).open() as f:
        for line in f:
            if not line.startswith("%"):
                break
            if line.startswith("% ref pos"):
                return np.array(line.split(":", 1)[1].split(), dtype=float)
    raise ValueError(f"{path}: no '% ref pos' header line (needs out-outhead=on)")
