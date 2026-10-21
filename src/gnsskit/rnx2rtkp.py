"""Running the RTKLIB rnx2rtkp executable."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import RtkConfig
from .pos import read_pos

RTKLIB_URL = "https://github.com/rtklibexplorer/RTKLIB"


def install_hint() -> str:
    """How to install rnx2rtkp on the current platform."""
    if sys.platform == "win32":
        return (
            f"Download an RTKLIB release (demo5 recommended: {RTKLIB_URL}/releases), "
            "unzip it, and add the folder containing rnx2rtkp.exe to your PATH, "
            "or pass --exe C:\\path\\to\\rnx2rtkp.exe."
        )
    return (
        f"Build it from source (demo5 recommended):\n"
        f"  git clone {RTKLIB_URL}\n"
        "  cd RTKLIB/app/consapp/rnx2rtkp/gcc && make && sudo make install\n"
        "or copy the binary to a folder on your PATH. Debian/Ubuntu may also offer an "
        "older build (sudo apt install rtklib). Alternatively pass --exe /path/to/rnx2rtkp."
    )


def find_rnx2rtkp(exe: str = "rnx2rtkp") -> str:
    """Resolve the rnx2rtkp executable on PATH.

    Args:
        exe: Executable name or path.

    Returns:
        Absolute path of the executable.

    Raises:
        FileNotFoundError: If the executable is not found (the message has install hints).
    """
    path = shutil.which(exe)
    if path is None:
        raise FileNotFoundError(f"'{exe}' not found on PATH. {install_hint()}")
    return path


def _fmt_time(t: pd.Timestamp) -> list[str]:
    """Timestamp as the [date, time] argument pair rnx2rtkp expects."""
    return [t.strftime("%Y/%m/%d"), t.strftime("%H:%M:%S")]


def build_command(
    conf: Path,
    rover: Path,
    base: Path,
    nav: Sequence[Path],
    out: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exe: str = "rnx2rtkp",
    tint: float | None = None,
) -> list[str]:
    """Build the rnx2rtkp command line.

    Argument order is rover observations, base observations, then navigation files.

    Args:
        conf: RTKLIB configuration file (`-k`).
        rover: Rover RINEX observation file.
        base: Base RINEX observation file.
        nav: Navigation, precise orbit (SP3) and clock (CLK) files; rnx2rtkp tells them apart by file extension.
        out: Output .pos file (`-o`).
        start: Processing start time, GPST (`-ts`). None for the start of the data.
        end: Processing end time, GPST (`-te`). None for the end of the data.
        exe: rnx2rtkp executable.
        tint: Processing interval [s] (`-ti`). None for every epoch.

    Returns:
        The command as a list of arguments.
    """
    cmd = [exe, "-k", str(conf)]
    if start is not None:
        cmd += ["-ts", *_fmt_time(start)]
    if end is not None:
        cmd += ["-te", *_fmt_time(end)]
    if tint is not None:
        cmd += ["-ti", str(tint)]
    return cmd + ["-o", str(out), str(rover), str(base), *map(str, nav)]


def run_rnx2rtkp(
    rover: Path,
    base: Path,
    nav: Sequence[Path],
    config: RtkConfig,
    out: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    tint: float | None = None,
    exe: str = "rnx2rtkp",
) -> Path:
    """Run rnx2rtkp on a rover/base pair and return the output .pos path.

    The config is written to a temporary file for the run. A zero exit code does not guarantee a solution: with e.g. missing precise products rnx2rtkp may write a .pos file with no epochs, which `read_pos` returns as an empty DataFrame.

    Args:
        rover: Rover RINEX observation file.
        base: Base RINEX observation file.
        nav: Navigation files (broadcast), and precise orbit (SP3) and clock (CLK) files when
            `config.sateph` is "precise".
        config: RTKLIB processing options.
        out: Output .pos file; parent directories are created.
        start: Processing start time, GPST. None for the start of the data.
        end: Processing end time, GPST. None for the end of the data.
        tint: Processing interval [s]; decimates the processing, needed when rover and base
            sampling differ. None for every epoch.
        exe: rnx2rtkp executable name or path.

    Returns:
        Path of the written .pos file (`out`).

    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If rnx2rtkp exits with a non-zero code.
    """
    exe = find_rnx2rtkp(exe)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        conf = config.write(Path(tmp) / "rtk.conf")
        cmd = build_command(conf, rover, base, nav, out, start, end, exe, tint)
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rnx2rtkp failed ({proc.returncode}): {proc.stderr.strip()[-500:]}"
        )
    return out


def solve_static(
    rover: Path,
    base: Path,
    nav: Sequence[Path],
    config: RtkConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exe: str = "rnx2rtkp",
    tint: float | None = None,
) -> pd.Series | None:
    """Static solution over [start, end], run in a temporary directory.

    Args:
        rover: Rover RINEX observation file.
        base: Base RINEX observation file.
        nav: Navigation, precise orbit and clock files (see `run_rnx2rtkp`).
        config: RTKLIB processing options.
        start: Processing start time, GPST.
        end: Processing end time, GPST.
        exe: rnx2rtkp executable name or path.
        tint: Processing interval [s]. None for every epoch.

    Returns:
        The last (cumulative) epoch of the forward filter as a row of the `read_pos` table, or None if there is no solution.

    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If rnx2rtkp exits with a non-zero code.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = run_rnx2rtkp(
            rover, base, nav, config, Path(tmp) / "out.pos",
            start=start, end=end, tint=tint, exe=exe,
        )  # fmt: skip
        if not out.exists() or out.stat().st_size == 0:
            return None
        pos = read_pos(out)
    return pos.iloc[-1] if len(pos) else None
