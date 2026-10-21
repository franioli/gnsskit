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
    """Resolve the rnx2rtkp executable; raise FileNotFoundError with install hints."""
    path = shutil.which(exe)
    if path is None:
        raise FileNotFoundError(f"'{exe}' not found on PATH. {install_hint()}")
    return path


def _fmt_time(t: pd.Timestamp) -> list[str]:
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
    """rnx2rtkp command line: rover, then base observations, then navigation files.

    start/end are GPST, tint is the processing interval [s].
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
    """Run rnx2rtkp and return the output .pos path.

    `tint` [s] decimates the processing, needed when rover and base sampling differ.
    """
    exe = find_rnx2rtkp(exe)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        conf = config.write(Path(tmp) / "rtk.conf")
        cmd = build_command(conf, rover, base, nav, out, start, end, exe, tint)
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rnx2rtkp failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
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
    """Static solution over [start, end]: the last (cumulative) epoch of the forward filter."""
    with tempfile.TemporaryDirectory() as tmp:
        out = run_rnx2rtkp(
            rover, base, nav, config, Path(tmp) / "out.pos",
            start=start, end=end, tint=tint, exe=exe,
        )  # fmt: skip
        if not out.exists() or out.stat().st_size == 0:
            return None
        pos = read_pos(out)
    return pos.iloc[-1] if len(pos) else None
