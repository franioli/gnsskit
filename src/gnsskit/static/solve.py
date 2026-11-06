"""Static solution of a rover over a time window (one rnx2rtkp run, last filter epoch)."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..rtklib.config import RtkConfig
from ..rtklib.pos import read_pos
from ..rtklib.runner import run_rnx2rtkp


def solve_static(
    rover: Path | str,
    base: Path | str,
    nav: Sequence[Path | str],
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
        The last (cumulative) epoch of the forward filter as a row of the `read_pos` table,
        or None if there is no solution.

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
