from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnsskit.rtklib.pos import pos_covariance, read_pos
from gnsskit.static.baseline import (
    BaselineResult, baseline_from_solution, extract_baseline, read_baselines, write_baselines,
)  # fmt: skip

from conftest import DATA, llh_header, llh_line


def test_pos_covariance_restores_signs() -> None:
    cov = np.array([[4.0, -1.0, 0.5], [-1.0, 9.0, 0.0], [0.5, 0.0, 1.0]]) * 1e-6
    sgn = lambda c: np.sign(c) * np.sqrt(abs(c))  # noqa: E731  RTKLIB's printing
    row = dict(zip(["sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx"],
                   [*np.sqrt(np.diag(cov)), sgn(cov[0, 1]), sgn(cov[1, 2]), sgn(cov[2, 0])]))  # fmt: skip
    assert np.allclose(pos_covariance(row), cov)


def test_extract_baseline_takes_last_epoch() -> None:
    b = extract_baseline(DATA / "xyz_static.pos", "1005", "063604", "S1")
    assert b is not None and (b.q, b.ns, b.ratio) == (1, 11, 3.6)
    assert np.allclose(
        b.dxyz,
        [4428458.8882 - 4428814.9484, 857020.7700 - 859166.7316, 4494395.6355 - 4493637.0158],
    )
    assert b.is_good() and b.cov[0, 0] == pytest.approx(0.0017**2)
    pos = read_pos(DATA / "xyz_static.pos")  # statistics over all the epochs
    assert b.n_epochs == len(pos) > 1
    assert b.fix_pct == pytest.approx(100 * (pos.Q == 1).mean())
    assert b.ratio_median == pytest.approx(pos.ratio.median())


def test_baseline_from_solution() -> None:
    sol = {"t": pd.Timestamp("2026-11-04 10:00"), "x": 11.0, "y": 22.0, "z": 33.0, "Q": 2, "ns": 9,
           "ratio": 0.0, "sdx": 0.01, "sdy": 0.02, "sdz": 0.03, "sdxy": 0.0, "sdyz": 0.0, "sdzx": 0.0}  # fmt: skip
    b = baseline_from_solution(sol, np.array([10.0, 20.0, 30.0]), "R", "B")
    assert np.allclose(b.dxyz, [1, 2, 3]) and b.cov[2, 2] == pytest.approx(9e-4) and not b.is_good()


def test_extract_baseline_errors(tmp_path: Path) -> None:
    llh = tmp_path / "l.pos"
    llh.write_text(llh_header() + llh_line("10:00:00.000", 46, 9, 1, 1))
    with pytest.raises(ValueError, match="xyz"):
        extract_baseline(llh, "a", "b")
    empty = tmp_path / "e.pos"
    empty.write_text((DATA / "xyz_static.pos").read_text().split("\n2026")[0] + "\n")
    assert extract_baseline(empty, "a", "b") is None


def test_write_read_round_trip(tmp_path: Path) -> None:
    cov = np.diag([1e-6, 2e-6, 3e-6])
    cov[0, 1] = cov[1, 0] = -1e-7
    b = BaselineResult("1005", "063604", "S1", pd.Timestamp("2026-10-07 11:22:15"),
                       np.array([1.0, 2.0, 3.0]), cov, 1, 10, 5.0)  # fmt: skip
    back = read_baselines(write_baselines([b], tmp_path / "x" / "b.csv"))[0]
    assert (back.base, back.rover, back.session) == ("063604", "1005", "S1")
    assert np.allclose(back.cov, cov) and np.allclose(back.dxyz, b.dxyz) and back.epoch == b.epoch
    assert back.n_epochs == 0 and np.isnan(back.fix_pct)  # unknown statistics stay unknown
    old = read_baselines(DATA / "baselines.csv")  # written before the statistics existed
    assert old[0].n_epochs == 0 and np.isnan(old[0].ratio_median)
