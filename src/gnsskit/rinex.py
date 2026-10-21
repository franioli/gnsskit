"""RINEX observation file helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .geodesy import GRS80, Ellipsoid, ecef_to_llh


def rinex_span(obs: Path | str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last epoch of a RINEX 3 observation file.

    Event records without a timestamp are skipped. The epochs are returned as written in the
    file (GPST for GPS-based files).

    Args:
        obs: RINEX 3 observation file.

    Returns:
        (first, last) epoch.

    Raises:
        ValueError: If no epoch is found.
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


def header_llh(obs: Path | str, ellipsoid: Ellipsoid = GRS80) -> tuple[float, float, float]:
    """Antenna position from the RINEX header, as geodetic lat, lon and ellipsoidal height.

    Uses `APPROX POSITION XYZ` plus the vertical `ANTENNA: DELTA H/E/N`, if present.

    Args:
        obs: RINEX observation file.
        ellipsoid: Reference ellipsoid of the conversion.

    Returns:
        (lat [deg], lon [deg], height [m]).

    Raises:
        ValueError: If there is no valid (non-zero) APPROX POSITION XYZ.
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
    lat, lon, h = ecef_to_llh(*xyz, ellipsoid=ellipsoid)
    return float(lat), float(lon), float(h) + dh


def _epoch_time(line: str) -> pd.Timestamp:
    """Timestamp of a RINEX 3 epoch record (the text after `>`)."""
    y, mo, d, h, mi, s = line.split()[:6]
    return pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi)) + pd.Timedelta(seconds=float(s))


def _header_time(t: pd.Timestamp) -> str:
    sec = t.second + t.microsecond / 1e6
    return f"{t.year:6d}{t.month:6d}{t.day:6d}{t.hour:6d}{t.minute:6d}{sec:13.7f}     GPS"


def _convbin_crop(
    src: Path,
    dst: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    interval: float | None,
) -> bool:
    """Crop with RTKLIB `convbin`; False if it is not installed or fails."""
    exe = shutil.which("convbin")
    if exe is None:
        return False
    with src.open() as f:
        version = f.readline()[:9].strip()
    cmd = [exe, "-r", "rinex", "-v", version, "-od", "-os"]  # keep Doppler and SNR
    if end is not None:
        end += pd.Timedelta(seconds=1)  # convbin excludes the epoch at -te, and truncates it
    for flag, t in (("-ts", start), ("-te", end)):
        if t is not None:
            cmd += [flag, t.strftime("%Y/%m/%d"), t.strftime("%H:%M:%S")]
    if interval is not None:
        cmd += ["-ti", str(interval)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")  # an existing dst is only replaced on success
    proc = subprocess.run([*cmd, "-o", str(tmp), str(src)], capture_output=True)
    ok = proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0
    if ok:
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)
    return ok


def crop_rinex(
    src: Path | str,
    dst: Path | str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    interval: float | None = None,
    use_convbin: bool = False,
) -> Path:
    """Write a RINEX 3 observation file cropped to [start, end] and decimated.

    The pure-Python path drops event records without a timestamp, updates the `INTERVAL`,
    `TIME OF FIRST OBS` and `TIME OF LAST OBS` header lines and copies the rest of the
    header unchanged. RTKLIB `convbin` writes its own header instead, drops QZSS, and
    excludes the epoch at `end`; it is run with `end` + 1 s, so with data above 1 Hz and no
    `interval` it may keep epochs up to 1 s after `end`.

    Args:
        src: Input RINEX 3 observation file.
        dst: Output file; parent directories are created.
        start: First epoch to keep, GPST. None for the start of the file.
        end: Last epoch to keep, GPST. None for the end of the file.
        interval: Keep the epochs whose time within the day is a multiple of this [s].
            None keeps every epoch.
        use_convbin: Try RTKLIB `convbin` first (it must be on PATH) and fall back to the
            Python implementation if it is missing or fails.

    Returns:
        The output path.

    Raises:
        ValueError: If no epoch is kept.
    """
    if use_convbin and _convbin_crop(Path(src), Path(dst), start, end, interval):
        return Path(dst)
    step_ms = None if interval is None else round(interval * 1000)
    header: list[str] = []
    body: list[str] = []
    epochs: list[pd.Timestamp] = []
    keep = False
    with Path(src).open() as f:
        in_header = True
        for line in f:
            if in_header:
                header.append(line)
                in_header = "END OF HEADER" not in line
                continue
            if line.startswith(">"):
                try:
                    t = _epoch_time(line[1:])
                except ValueError:  # event record: dropped together with its lines
                    keep = False
                    continue
                ms = round((t - t.normalize()).total_seconds() * 1000)
                keep = (
                    (start is None or t >= start)
                    and (end is None or t <= end)
                    and (step_ms is None or ms % step_ms == 0)
                )
                if keep:
                    epochs.append(t)
            if keep:
                body.append(line)
    if not epochs:
        raise ValueError(f"{src}: no epochs kept")

    def field(value: str, label: str) -> str:
        return f"{value:<60}{label}\n"

    for i, line in enumerate(header):
        if interval is not None and "INTERVAL" in line:
            header[i] = field(f"{interval:10.3f}", "INTERVAL")
        elif "TIME OF FIRST OBS" in line:
            header[i] = field(_header_time(epochs[0]), "TIME OF FIRST OBS")
        elif "TIME OF LAST OBS" in line:
            header[i] = field(_header_time(epochs[-1]), "TIME OF LAST OBS")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(header + body))
    return dst
