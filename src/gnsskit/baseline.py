"""Static baselines from RTKLIB: extraction of the final vector + covariance, tables, storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RtkConfig
from .pos import POS_COORD_COLS, POS_SD_COLS, read_pos, read_ref_pos
from .rnx2rtkp import run_rnx2rtkp

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
    epoch: pd.Timestamp  # last epoch of the forward solution (GPST)
    dxyz: np.ndarray  # (3,) m, X_rover - X_base
    cov: np.ndarray  # (3, 3) m^2, formal covariance of dxyz
    q: int  # 1 = fixed, 2 = float
    ns: int  # satellites used
    ratio: float  # ambiguity validation ratio
    pos_file: Path | None = None

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.dxyz))

    def is_good(self, min_ratio: float = 3.0, min_sats: int = 6) -> bool:
        """Fixed solution with enough satellites and a sufficient ratio."""
        return self.q == 1 and self.ratio >= min_ratio and self.ns >= min_sats


def _cov_from_sd(row: Mapping[str, float]) -> np.ndarray:
    """3x3 covariance from the xyz .pos columns sdx, sdy, sdz, sdxy, sdyz, sdzx.

    RTKLIB prints the off-diagonal terms as sign(cov) * sqrt(|cov|), so they are
    squared back here.
    """
    sdx, sdy, sdz, sdxy, sdyz, sdzx = (row[c] for c in POS_SD_COLS["xyz"])
    cxy, cyz, czx = (np.sign(v) * v**2 for v in (sdxy, sdyz, sdzx))
    return np.array([[sdx**2, cxy, czx], [cxy, sdy**2, cyz], [czx, cyz, sdz**2]])


def extract_baseline(
    pos_file: Path | str, rover: str, base: str, session: str = ""
) -> BaselineResult | None:
    """Final-epoch baseline from an xyz .pos file; None if the file has no solution."""
    pos = read_pos(pos_file)
    if pos.attrs["format"] != "xyz":
        raise ValueError(f"{pos_file}: need out-solformat=xyz to extract a baseline")
    if pos.empty:
        return None
    last = {str(k): v for k, v in pos.iloc[-1].items()}
    xyz = np.array([last[c] for c in POS_COORD_COLS["xyz"]], dtype=float)
    return BaselineResult(
        rover=rover,
        base=base,
        session=session,
        epoch=last["t"],
        dxyz=xyz - read_ref_pos(pos_file),
        cov=_cov_from_sd(last),
        q=int(last["Q"]),
        ns=int(last["ns"]),
        ratio=float(last["ratio"]),
        pos_file=Path(pos_file),
    )


def process_baseline(
    rover: str,
    base: str,
    rover_obs: Path,
    base_obs: Path,
    nav: Sequence[Path],
    config: RtkConfig,
    out: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tint: float | None = None,
    session: str = "",
    exe: str = "rnx2rtkp",
) -> BaselineResult | None:
    """Run rnx2rtkp for one baseline (times in GPST), keep the .pos, extract the result."""
    run_rnx2rtkp(rover_obs, base_obs, nav, config, out, start=start, end=end, tint=tint, exe=exe)
    return extract_baseline(out, rover, base, session)


def baselines_to_df(results: Sequence[BaselineResult]) -> pd.DataFrame:
    """One row per baseline: vector, covariance (cov_xx..cov_zz), quality, source file."""
    rows = [
        {
            "session": b.session,
            "base": b.base,
            "rover": b.rover,
            "epoch": b.epoch,
            "dx": b.dxyz[0],
            "dy": b.dxyz[1],
            "dz": b.dxyz[2],
            **{f"cov_{k}": b.cov[i, j] for k, (i, j) in _COV_ENTRIES.items()},
            "length": b.length,
            "Q": b.q,
            "ns": b.ns,
            "ratio": b.ratio,
            "pos_file": str(b.pos_file) if b.pos_file else "",
        }
        for b in results
    ]
    return pd.DataFrame(rows)


def df_to_baselines(df: pd.DataFrame) -> list[BaselineResult]:
    """Inverse of `baselines_to_df`."""
    results = []
    for r in df.itertuples(index=False):
        cov = np.zeros((3, 3))
        for k, (i, j) in _COV_ENTRIES.items():
            cov[i, j] = cov[j, i] = getattr(r, f"cov_{k}")
        pos_file = r.pos_file if isinstance(r.pos_file, str) and r.pos_file else None
        results.append(
            BaselineResult(
                rover=str(r.rover),
                base=str(r.base),
                session=str(r.session) if pd.notna(r.session) else "",
                epoch=pd.Timestamp(r.epoch),
                dxyz=np.array([r.dx, r.dy, r.dz]),
                cov=cov,
                q=int(r.Q),
                ns=int(r.ns),
                ratio=float(r.ratio),
                pos_file=Path(pos_file) if pos_file else None,
            )
        )
    return results

def write_baselines(results: Sequence[BaselineResult], path: Path | str) -> Path:
    """Save the baselines as CSV (see `baselines_to_df`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines_to_df(results).to_csv(path, index=False)
    return path


def read_baselines(path: Path | str) -> list[BaselineResult]:
    """Load baselines saved with `write_baselines`; point names stay strings (e.g. '063604')."""
    df = pd.read_csv(
        path, dtype={"session": str, "base": str, "rover": str}, keep_default_na=False
    )
    return df_to_baselines(df)

