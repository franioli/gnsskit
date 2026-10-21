import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

from gnsskit.baseline import BaselineResult, read_baselines
from gnsskit.network import adjust_network, coordinate_table, residual_table

from conftest import DATA

TRUE = {
    "A": np.array([4428816.0, 859169.0, 4493638.0]),
    "B": np.array([4428543.0, 851920.0, 4495267.0]),
    "C": np.array([4428458.0, 857020.0, 4494395.0]),
    "D": np.array([4428805.0, 857145.0, 4494032.0]),
}
PAIRS = [("A", "C"), ("C", "B"), ("A", "B"), ("C", "D"), ("A", "D"), ("D", "B")]  # base, rover


def _baselines(noise: float = 0.0, seed: int = 0) -> list[BaselineResult]:
    rng = np.random.default_rng(seed)
    cov = np.diag([2e-3, 1e-3, 2e-3]) ** 2
    return [
        BaselineResult(
            rover=r, base=b, session="S", epoch=pd.Timestamp("2026-10-07"),
            dxyz=TRUE[r] - TRUE[b] + rng.normal(0, noise, 3), cov=cov, q=1, ns=10, ratio=9.0,
        )
        for b, r in PAIRS
    ]  # fmt: skip


def test_recovers_truth_without_noise() -> None:
    adj = adjust_network(_baselines(), {"A": TRUE["A"]})
    assert adj.dof == 9
    for p, xyz in zip(adj.unknowns, adj.xyz):
        assert np.allclose(xyz, TRUE[p], atol=1e-6)
    assert np.allclose(adj.v_hat, 0, atol=1e-6)
    assert np.isclose(np.trace(adj.R), adj.dof)


def test_noise_gives_consistent_variance_factor() -> None:
    adj = adjust_network(_baselines(noise=2e-3, seed=3), {"A": TRUE["A"]})
    lo, hi = adj.sigma_v_hat_bounds
    assert 0 < adj.sigma_v_hat < 10 * hi
    assert len(residual_table(adj)) == len(PAIRS)


def test_coordinate_table_lists_fixed_point_with_zero_sd() -> None:
    adj = adjust_network(_baselines(noise=2e-3), {"A": TRUE["A"]})
    table = coordinate_table(adj).set_index("point")
    assert table.loc["A", ["sd_E", "sd_N", "sd_h"]].eq(0).all()
    assert (table.loc[["B", "C", "D"], "sd_E"] > 0).all()
    assert abs(table.loc["B", "lat"] - 45.1) < 0.1


def test_rtklib_covariance_option_and_bad_option() -> None:
    adj = adjust_network(_baselines(), {"A": TRUE["A"]}, stochastic="rtklib")
    for p, xyz in zip(adj.unknowns, adj.xyz):
        assert np.allclose(xyz, TRUE[p], atol=1e-6)
    with pytest.raises(ValueError):
        adjust_network(_baselines(), {"A": TRUE["A"]}, stochastic="foo")  # type: ignore[arg-type]


def test_matches_scipy_least_squares_on_whitened_residuals() -> None:
    bl = _baselines(noise=3e-3, seed=1)
    adj = adjust_network(bl, {"A": TRUE["A"]})
    L = np.linalg.cholesky(np.linalg.inv(adj.Q))
    res = least_squares(lambda x: L.T @ (adj.y0 - adj.b - adj.A @ x), adj.X_hat + 0.5, xtol=1e-14, ftol=1e-14)
    assert np.allclose(res.x, adj.X_hat, atol=1e-6)


def test_course_network_regression() -> None:
    """Frozen output on the 2026-10-07 Mantova data (values from the course notebook run)."""
    fixed = {"063604": [4428814.9484, 859166.7316, 4493637.0158]}
    adj = adjust_network(read_baselines(DATA / "baselines.csv"), fixed)
    assert adj.dof == 9 and adj.tau.max() < 3.29
    lo, hi = adj.sigma_v_hat_bounds
    assert lo < adj.sigma_v_hat < hi
    utm = coordinate_table(adj).set_index("point")
    assert adj.sigma_v_hat == pytest.approx(0.5188, abs=1e-4)
    expected = {
        "1005": (653672.4828, 4994710.5228, 64.8366),
        "1002": (648621.1383, 4995831.5506, 58.9444),
        "1006": (653741.0402, 4994198.1557, 64.5235),
    }
    for p, enh in expected.items():
        assert utm.loc[p, ["E", "N", "h_ell"]].to_numpy(float) == pytest.approx(enh, abs=1e-4)
