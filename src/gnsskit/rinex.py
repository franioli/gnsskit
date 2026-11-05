"""RINEX 3 observation files: header and time span, trimming, quality inspection."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .refsys import GRS80, Ellipsoid, ecef_to_llh

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SYSTEMS = "GRECJSI"  # RINEX satellite system letters
OBS_SUFFIX = re.compile(r"\.(\d\d[oO]|obs)$")  # observation files: .26O, .26o, .obs


# =============================================================================
# Record parsing (private)
# =============================================================================


def _epoch_time(line: str) -> pd.Timestamp:
    """Timestamp of a RINEX 3 epoch record (the text after `>`)."""
    y, mo, d, h, mi, s = line.split()[:6]
    return pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi)) + pd.Timedelta(seconds=float(s))


def _header_time(t: pd.Timestamp) -> str:
    sec = t.second + t.microsecond / 1e6
    return f"{t.year:6d}{t.month:6d}{t.day:6d}{t.hour:6d}{t.minute:6d}{sec:13.7f}     GPS"


# =============================================================================
# Header and time span
# =============================================================================


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
                last = _epoch_time(line[1:])
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


# =============================================================================
# Trimming and decimation
# =============================================================================


def _convbin_trim(
    src: Path,
    dst: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    interval: float | None,
) -> bool:
    """Trim with RTKLIB `convbin`; False if it is not installed or fails."""
    exe = shutil.which("convbin")
    if exe is None:
        return False
    with src.open() as f:
        version = f.readline()[:9].strip()
    cmd = [exe, "-r", "rinex", "-v", version, "-od", "-os"]  # keep Doppler and SNR
    if end is not None:
        # convbin excludes the epoch at -te, and truncates it
        end += pd.Timedelta(seconds=1)
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


def trim_rinex(
    src: Path | str,
    dst: Path | str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    interval: float | None = None,
    use_convbin: bool = False,
    marker: str | None = None,
) -> Path:
    """Write a RINEX 3 observation file trimmed to [start, end] and decimated.

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
        marker: New `MARKER NAME` for the header. None keeps the original. Ignored by
            `convbin`, which writes its own header.

    Returns:
        The output path.

    Raises:
        ValueError: If no epoch is kept.
    """
    if use_convbin and _convbin_trim(Path(src), Path(dst), start, end, interval):
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
        elif marker is not None and "MARKER NAME" in line:
            header[i] = field(marker, "MARKER NAME")
        elif "TIME OF FIRST OBS" in line:
            header[i] = field(_header_time(epochs[0]), "TIME OF FIRST OBS")
        elif "TIME OF LAST OBS" in line:
            header[i] = field(_header_time(epochs[-1]), "TIME OF LAST OBS")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(header + body))
    return dst


# =============================================================================
# Quality inspection: satellite visibility, gaps, summary
# =============================================================================


def read_visibility(obs: Path | str) -> pd.DataFrame:
    """Satellites tracked at each epoch of a RINEX 3 observation file.

    A satellite counts as tracked if its record has at least one non-blank observation.
    Event records (epoch flag != 0) are skipped and counted in `attrs["n_events"]`.

    Args:
        obs: RINEX 3 observation file.

    Returns:
        DataFrame with columns `t` (epoch, as written in the file) and `sat` (e.g. "G05"),
        one row per tracked satellite and epoch. `attrs` holds `marker` and `n_events`.
    """
    times: list[pd.Timestamp] = []
    sats: list[str] = []
    marker, n_events = Path(obs).stem, 0
    t = None
    with Path(obs).open() as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if "MARKER NAME" in line[60:] and line[:60].strip():
                marker = line[:60].strip()
        for line in f:
            if line.startswith(">"):
                if line[31:32] != "0":
                    t = None
                    n_events += 1
                    continue
                t = _epoch_time(line[1:])
            elif t is not None and line[:1] in SYSTEMS and line[3:].strip():
                times.append(t)
                sats.append(line[:3])
    df = pd.DataFrame({"t": times, "sat": sats})
    df.attrs.update(marker=marker, n_events=n_events)
    return df


def obs_summary(obs: Path | str, gap_factor: float = 1.5) -> dict[str, object]:
    """Summary of a RINEX 3 observation file: time span, sampling, gaps and satellites.

    Args:
        obs: RINEX 3 observation file.
        gap_factor: An epoch spacing larger than `gap_factor` times the median interval is a gap.

    Returns:
        Dict with `marker`, `first`, `last`, `duration_s`, `interval_s`, `n_epochs`,
        `n_gaps`, `max_gap_s`, `n_events`, `mean_sats` (per epoch) and `n_sats` (unique
        satellites per system).

    Raises:
        ValueError: If the file has no observation epochs.
    """
    vis = read_visibility(obs)
    if vis.empty:
        raise ValueError(f"no RINEX 3 observations found in {obs}")
    epochs = pd.Series(vis.t.unique())
    step = epochs.diff().dropna().dt.total_seconds()
    interval = float(step.median()) if len(step) else float("nan")
    gaps = step[step > gap_factor * interval] if len(step) else step
    return {
        "marker": vis.attrs["marker"],
        "first": epochs.iloc[0],
        "last": epochs.iloc[-1],
        "duration_s": float((epochs.iloc[-1] - epochs.iloc[0]).total_seconds()),
        "interval_s": interval,
        "n_epochs": len(epochs),
        "n_gaps": len(gaps),
        "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "n_events": vis.attrs["n_events"],
        "mean_sats": float(len(vis) / len(epochs)),
        "n_sats": vis.groupby(vis.sat.str[0]).sat.nunique().to_dict(),
    }


def find_rinex_obs(paths: Sequence[Path | str]) -> list[Path]:
    """RINEX observation files from a mix of files and folders.

    Files are kept as given. A folder is replaced by the observation files directly inside it
    (names ending in `.YYo`, `.YYO` or `.obs`, sorted by name; subfolders are not searched), so
    navigation files next to them are ignored.

    Args:
        paths: Observation files and/or folders.

    Returns:
        The observation files, in the order of `paths`.
    """
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if OBS_SUFFIX.search(f.name))
        else:
            files.append(p)
    return files


def read_obs_types(obs: Path | str) -> dict[str, list[str]]:
    """Observation codes of each satellite system, in file order (`SYS / # / OBS TYPES`).

    Args:
        obs: RINEX 3 observation file.

    Returns:
        {system letter: [codes]}, e.g. {"G": ["C1C", "L1C", "D1C", "S1C", ...]}.
    """
    types: dict[str, list[str]] = {}
    system = ""
    with Path(obs).open() as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if line[60:].startswith("SYS / # / OBS TYPES"):
                if line[0] != " ":  # first line of a system; the others continue its list
                    system = line[0]
                    types[system] = []
                types[system] += line[7:60].split()
    return types


def satellite_stats(obs: Path | str) -> pd.DataFrame:
    """Tracking statistics of every satellite of a RINEX 3 observation file.

    Args:
        obs: RINEX 3 observation file.

    Returns:
        DataFrame indexed by satellite with first and last epoch tracked, `n_epochs` (epochs
        tracked), `of_file` (n_epochs / epochs of the file) and `of_arc` (n_epochs / epochs
        between its first and last, which is below 1 if the satellite was lost and regained).
    """
    vis = read_visibility(obs)
    epochs = np.sort(vis.t.unique())
    g = vis.groupby("sat").t
    df = pd.DataFrame({"first": g.min(), "last": g.max(), "n_epochs": g.size()})
    arc = np.searchsorted(epochs, df["last"].to_numpy(), "right") - np.searchsorted(
        epochs, df["first"].to_numpy(), "left"
    )
    df["of_file"] = df.n_epochs / len(epochs)
    df["of_arc"] = df.n_epochs / arc
    return df


def signal_stats(obs: Path | str) -> pd.DataFrame:
    """Completeness and signal strength of every observation code of each system.

    Args:
        obs: RINEX 3 observation file.

    Returns:
        DataFrame indexed by (system, code) with `n_obs` (non-blank values),
        `completeness` (n_obs / satellite records of that system: how often the satellite
        has the code when it is tracked) and `mean_snr` (mean of the S codes in the unit of
        the file, normally dB-Hz; NaN for the other codes).
    """
    types = read_obs_types(obs)
    n_rec = dict.fromkeys(types, 0)
    n_obs = {(s, c): 0 for s, codes in types.items() for c in codes}
    snr = {k: 0.0 for k in n_obs if k[1][0] == "S"}
    valid = False
    with Path(obs).open() as f:
        for line in f:
            if "END OF HEADER" in line:
                break
        for line in f:
            if line.startswith(">"):
                valid = line[31:32] == "0"  # skip event records
            elif valid and line[:1] in types:
                s = line[0]
                n_rec[s] += 1
                for k, code in enumerate(types[s]):
                    field = line[3 + 16 * k : 17 + 16 * k]
                    if field.strip():
                        n_obs[s, code] += 1
                        if code[0] == "S":
                            snr[s, code] += float(field)
    rows = [
        {
            "system": s,
            "code": c,
            "n_obs": n,
            "completeness": n / n_rec[s] if n_rec[s] else np.nan,
            "mean_snr": snr[s, c] / n if (s, c) in snr and n else np.nan,
        }
        for (s, c), n in n_obs.items()
    ]
    return pd.DataFrame(rows).set_index(["system", "code"])


def plot_visibility(
    obs_files: Sequence[Path | str], out: Path | str | None = None, dpi: int = 150
) -> Figure:
    """Satellite visibility plot (as in RTKPLOT), one panel per observation file.

    Needs matplotlib. Colours: GPS blue, GLONASS red, Galileo green, BeiDou orange.

    Args:
        obs_files: RINEX 3 observation files.
        out: If given, the figure is also saved to this path.
        dpi: Resolution of the saved figure.

    Returns:
        The matplotlib figure.
    """
    try:
        import matplotlib.dates as mdates
        from matplotlib.figure import Figure
    except ImportError as e:  # pragma: no cover
        raise ImportError("plot_visibility needs matplotlib: pip install 'gnsskit[plot]'") from e

    colors = {"G": "tab:blue", "R": "tab:red", "E": "tab:green", "C": "tab:orange",
              "J": "tab:purple", "S": "tab:gray", "I": "tab:brown"}  # fmt: skip
    fig = Figure(figsize=(12, 3.5 * len(obs_files)))  # no pyplot: no GUI backend needed
    axes = fig.subplots(len(obs_files), 1, squeeze=False)
    for ax, path in zip(axes[:, 0], obs_files):
        vis = read_visibility(path)
        sats = sorted(vis.sat.unique())
        for i, sat in enumerate(sats):
            t = vis.t[vis.sat == sat]
            ax.plot(t, [i] * len(t), "|", ms=3, color=colors[sat[0]])
        ax.set_yticks(range(len(sats)), sats, fontsize=5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        span = f"{vis.t.min():%H:%M:%S} - {vis.t.max():%H:%M:%S}" if len(vis) else "no data"
        ax.set_title(
            f"{Path(path).name} [{vis.attrs['marker']}]  {span}  ({len(sats)} sats)",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=dpi)
    return fig
