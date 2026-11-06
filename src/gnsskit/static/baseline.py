"""Static GNSS baselines: data model, extraction from RTKLIB solutions, processing, storage.

A baseline is the ECEF vector base -> rover with its 3x3 covariance and the quality
indicators of the solution. `BaselineResult` is independent of RTKLIB: the RTKLIB-specific
part is only how it is built (`baseline_from_solution`, `extract_baseline`) and computed
(`process_baseline`), so other solvers can produce the same objects for `gnsskit.static.network`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..rtklib.config import RtkConfig
from ..rtklib.pos import POS_COORD_COLS, pos_covariance, read_pos, read_ref_pos
from ..rtklib.runner import run_rnx2rtkp

# =============================================================================
# Data model
# =============================================================================

# Upper-triangular entries of the 3x3 covariance -> suffix of the table column
_COV_ENTRIES = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yy": (1, 1),
    "yz": (1, 2),
    "zz": (2, 2),
}


@dataclass(frozen=True)
class BaselineResult:
    """Final static solution of one baseline: the vector base -> rover (ECEF) + quality."""

    rover: str
    base: str
    session: str
    epoch: pd.Timestamp  # epoch of the solution (GPST)
    dxyz: np.ndarray  # (3,) m, X_rover - X_base
    cov: np.ndarray  # (3, 3) m^2, formal covariance of dxyz
    q: int  # 1 = fixed, 2 = float
    ns: int  # satellites used
    ratio: float  # ambiguity validation ratio
    pos_file: Path | None = None

    @property
    def length(self) -> float:
        """3D baseline length [m]."""
        return float(np.linalg.norm(self.dxyz))

    def is_good(self, min_ratio: float = 3.0, min_sats: int = 6) -> bool:
        """Whether the solution is fixed, with enough satellites and a sufficient ratio.

        Args:
            min_ratio: Minimum ambiguity validation ratio.
            min_sats: Minimum number of satellites used.

        Returns:
            True if Q is 1 (fixed), ratio >= `min_ratio` and ns >= `min_sats`.
        """
        return self.q == 1 and self.ratio >= min_ratio and self.ns >= min_sats

    def to_row(self) -> dict[str, Any]:
        """Flat dict of the baseline: one table row (see `baselines_to_df`)."""
        return {
            "session": self.session,
            "base": self.base,
            "rover": self.rover,
            "epoch": self.epoch,
            "dx": self.dxyz[0],
            "dy": self.dxyz[1],
            "dz": self.dxyz[2],
            **{f"cov_{k}": self.cov[i, j] for k, (i, j) in _COV_ENTRIES.items()},
            "length": self.length,
            "Q": self.q,
            "ns": self.ns,
            "ratio": self.ratio,
            "pos_file": str(self.pos_file) if self.pos_file else "",
        }

    @classmethod
    def from_row(cls, row: Mapping[Hashable, Any]) -> BaselineResult:
        """Inverse of `to_row`; `length` is recomputed, not read."""
        cov = np.zeros((3, 3))
        for k, (i, j) in _COV_ENTRIES.items():
            cov[i, j] = cov[j, i] = row[f"cov_{k}"]
        session, pos_file = row.get("session"), row.get("pos_file")
        return cls(
            rover=str(row["rover"]),
            base=str(row["base"]),
            session=session if isinstance(session, str) else "",
            epoch=_timestamp(row["epoch"]),
            dxyz=np.array([row["dx"], row["dy"], row["dz"]], dtype=float),
            cov=cov,
            q=int(row["Q"]),
            ns=int(row["ns"]),
            ratio=float(row["ratio"]),
            pos_file=Path(pos_file) if isinstance(pos_file, str) and pos_file else None,
        )


def _timestamp(value: Any) -> pd.Timestamp:
    """Timestamp from a value, rejecting missing epochs (NaT)."""
    t = pd.Timestamp(value)
    if not isinstance(t, pd.Timestamp):
        raise ValueError(f"invalid baseline epoch: {value!r}")
    return t


# =============================================================================
# Construction from RTKLIB solutions
# =============================================================================


def baseline_from_solution(
    sol: pd.Series | Mapping[str, Any],
    base_xyz: Sequence[float] | np.ndarray,
    rover: str,
    base: str,
    session: str = "",
    pos_file: Path | str | None = None,
) -> BaselineResult:
    """Baseline from one epoch of an xyz `read_pos` table and the base position.

    Args:
        sol: One epoch (row) of an xyz `read_pos` table.
        base_xyz: (3,) ECEF base coordinates [m] used in the processing.
        rover: Rover point name.
        base: Base point name.
        session: Session label stored in the result.
        pos_file: Source .pos file, stored for reference.

    Returns:
        The baseline rover - base. Its covariance is the formal covariance of the rover
        solution, which is also that of the baseline because RTKLIB holds the base fixed;
        it is optimistic (see `gnsskit.static.network.K_S`).
    """
    xyz = np.array([sol[c] for c in POS_COORD_COLS["xyz"]], dtype=float)
    return BaselineResult(
        rover=rover,
        base=base,
        session=session,
        epoch=_timestamp(sol["t"]),
        dxyz=xyz - np.asarray(base_xyz, dtype=float),
        cov=pos_covariance(sol, "xyz"),
        q=int(sol["Q"]),
        ns=int(sol["ns"]),
        ratio=float(sol["ratio"]),
        pos_file=Path(pos_file) if pos_file is not None else None,
    )


def extract_baseline(
    pos_file: Path | str, rover: str, base: str, session: str = ""
) -> BaselineResult | None:
    """Final-epoch baseline from an xyz .pos file.

    The vector is the last solution minus the base position in the `% ref pos` header, so the
    file needs `out-solformat=xyz` and `out-outhead=on`.

    Args:
        pos_file: RTKLIB .pos file.
        rover: Rover point name.
        base: Base point name.
        session: Session label stored in the result.

    Returns:
        The baseline, or None if the file has no solution epoch.

    Raises:
        ValueError: If the file is not in xyz format or has no `% ref pos` header line.
    """
    pos = read_pos(pos_file)
    if pos.attrs["format"] != "xyz":
        raise ValueError(f"{pos_file}: need out-solformat=xyz to extract a baseline")
    if pos.empty:
        return None
    return baseline_from_solution(
        pos.iloc[-1], read_ref_pos(pos_file), rover, base, session, pos_file
    )


# =============================================================================
# Processing with rnx2rtkp
# =============================================================================


def process_baseline(
    rover: str,
    base: str,
    rover_obs: Path | str,
    base_obs: Path | str,
    nav: Sequence[Path | str],
    config: RtkConfig,
    out: Path | str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    tint: float | None = None,
    session: str = "",
    exe: str = "rnx2rtkp",
) -> BaselineResult | None:
    """Run rnx2rtkp for one baseline, keep the .pos file and extract the result.

    Args:
        rover: Rover point name.
        base: Base point name.
        rover_obs: Rover RINEX observation file.
        base_obs: Base RINEX observation file.
        nav: Navigation, precise orbit and clock files (see `run_rnx2rtkp`).
        config: RTKLIB options; needs `solformat="xyz"`, `outhead=True` and a base position
            (e.g. `base_postype="rinexhead"`), which is written in the `% ref pos` header.
        out: Output .pos file, kept on disk.
        start: Processing start time, GPST.
        end: Processing end time, GPST.
        tint: Processing interval [s]. None for every epoch.
        session: Session label stored in the result.
        exe: rnx2rtkp executable name or path.

    Returns:
        The baseline, or None if rnx2rtkp produced no solution.

    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If rnx2rtkp exits with a non-zero code.
        ValueError: If the .pos file is not in xyz format.
    """
    run_rnx2rtkp(rover_obs, base_obs, nav, config, out, start=start, end=end, tint=tint, exe=exe)
    return extract_baseline(out, rover, base, session)


# =============================================================================
# Tables and storage
# =============================================================================


def baselines_to_df(results: Sequence[BaselineResult]) -> pd.DataFrame:
    """Tabulate baselines, one row per baseline.

    Args:
        results: Baselines to tabulate.

    Returns:
        DataFrame with columns session, base, rover, epoch, dx, dy, dz, the covariance terms
        cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz [m^2], length, Q, ns, ratio and
        pos_file (empty string if unknown).
    """
    return pd.DataFrame([b.to_row() for b in results])


def df_to_baselines(df: pd.DataFrame) -> list[BaselineResult]:
    """Inverse of `baselines_to_df`.

    Args:
        df: Table with the columns written by `baselines_to_df`.

    Returns:
        One `BaselineResult` per row.
    """
    return [BaselineResult.from_row(r) for r in df.to_dict("records")]


def write_baselines(results: Sequence[BaselineResult], path: Path | str) -> Path:
    """Save the baselines as CSV (see `baselines_to_df`).

    Args:
        results: Baselines to save.
        path: Output CSV file; parent directories are created.

    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines_to_df(results).to_csv(path, index=False)
    return path


def read_baselines(path: Path | str) -> list[BaselineResult]:
    """Load baselines saved with `write_baselines`.

    Args:
        path: CSV file.

    Returns:
        The baselines; point and session names stay strings (e.g. '063604').
    """
    df = pd.read_csv(path, dtype={"session": str, "base": str, "rover": str}, keep_default_na=False)
    return df_to_baselines(df)
