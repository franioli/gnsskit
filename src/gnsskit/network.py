"""Least-squares adjustment of a GNSS baseline network with fixed points.

Notation (as in the course notebooks):
    y = A x + b                  baselines  X_rover - X_base = dxyz, fixed points in b
    y0 = y + v,  v ~ N(0, Q)     observations y0 (n values), parameters x (m values)
    N = A^T Q^-1 A
    X_hat = N^-1 A^T Q^-1 (y0 - b),   Y_hat = A X_hat + b,   v_hat = y0 - Y_hat
    sigma_v_hat = v_hat^T Q^-1 v_hat / (n - m)            a-posteriori variance factor
    Cxx = sigma_v_hat N^-1,  Cyy = sigma_v_hat A N^-1 A^T,  Cvv = sigma_v_hat (Q - A N^-1 A^T)

Stochastic model of a baseline (`stochastic` argument of `adjust_network`):
    "apriori": east/north/up errors, uncorrelated, with sd = a + b[ppm] * length;
               realistic for 10 min of static data.
    "rtklib":  formal covariance of the RTKLIB solution (optimistic, see MIN_VARIANCE).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .baseline import BaselineResult
from .geodesy import ecef_to_enu_matrix, ecef_to_llh

try:
    from pyproj import Transformer
    from scipy.stats import chi2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gnsskit.network needs the 'network' extra: pip install 'gnsskit[network]'"
    ) from e

# Default a-priori baseline precision as (constant [m], ppm of the length)
SIGMA_H = (5e-3, 1.0)  # horizontal: 5 mm + 1 ppm
SIGMA_V = (10e-3, 2.0)  # vertical: 10 mm + 2 ppm

# RTKLIB prints standard deviations rounded to 0.1 mm, which can make a 3x3 covariance
# non-positive-definite; its eigenvalues are floored at this variance before inversion.
MIN_VARIANCE = 1e-3**2  # m^2
TAU_CRITICAL = 3.29  # two-sided 0.1 % level of a standard normal (Baarda data snooping)


@dataclass(frozen=True)
class Adjustment:
    """Result of `adjust_network`. Observations are ordered baseline by baseline (dx, dy, dz),
    parameters point by point (X, Y, Z), the points following `unknowns`."""

    unknowns: list[str]  # adjusted (free) points
    fixed: dict[str, np.ndarray]  # fixed points, ECEF m
    baselines: list[BaselineResult]
    y0: np.ndarray  # (n,) observations
    b: np.ndarray  # (n,) known terms (fixed points)
    A: np.ndarray  # (n, m) design matrix
    Q: np.ndarray  # (n, n) covariance of the observations
    X_hat: np.ndarray  # (m,) estimated coordinates
    Y_hat: np.ndarray  # (n,) adjusted observations
    v_hat: np.ndarray  # (n,) residuals
    sigma_v_hat: float  # a-posteriori variance factor, expected ~1
    Cxx: np.ndarray  # (m, m) covariance of the parameters
    Cyy: np.ndarray  # (n, n) covariance of the adjusted observations
    Cvv: np.ndarray  # (n, n) covariance of the residuals
    R: np.ndarray  # (n, n) redundancy matrix, trace = n - m
    tau: np.ndarray  # (n,) normalised residuals

    @property
    def n(self) -> int:
        """Number of observations."""
        return self.y0.size

    @property
    def m(self) -> int:
        """Number of estimated parameters."""
        return self.X_hat.size

    @property
    def dof(self) -> int:
        """Degrees of freedom, n - m."""
        return self.n - self.m

    @property
    def xyz(self) -> np.ndarray:
        """(len(unknowns), 3) adjusted ECEF coordinates."""
        return self.X_hat.reshape(-1, 3)

    @property
    def sigma_v_hat_bounds(self) -> tuple[float, float]:
        """95 % acceptance interval of sigma_v_hat under the chi-square test."""
        return chi2.ppf(0.025, self.dof) / self.dof, chi2.ppf(0.975, self.dof) / self.dof


def _make_pd(cov: np.ndarray) -> np.ndarray:
    """Covariance with eigenvalues floored at MIN_VARIANCE."""
    w, vec = np.linalg.eigh(cov)
    return (vec * np.maximum(w, MIN_VARIANCE)) @ vec.T


def _apriori_block(
    length: float,
    R_enu: np.ndarray,
    sigma_h: tuple[float, float],
    sigma_v: tuple[float, float],
) -> np.ndarray:
    """ECEF 3x3 covariance of a baseline: east/north/up errors a + b[ppm] * length."""
    sd_h = sigma_h[0] + sigma_h[1] * 1e-6 * length
    sd_v = sigma_v[0] + sigma_v[1] * 1e-6 * length
    Q_enu = np.diag([sd_h**2, sd_h**2, sd_v**2])
    return R_enu.T @ Q_enu @ R_enu


def _observation_covariance(
    baselines: Sequence[BaselineResult],
    fixed: Mapping[str, np.ndarray],
    stochastic: str,
    sigma_h: tuple[float, float],
    sigma_v: tuple[float, float],
) -> np.ndarray:
    """Block-diagonal covariance Q of the observations, one 3x3 block per baseline."""
    # One east-north-up rotation, taken at the first fixed point, serves a small network
    lat0, lon0, _ = ecef_to_llh(*next(iter(fixed.values())))
    R_enu = ecef_to_enu_matrix(lat0, lon0)

    Q = np.zeros((3 * len(baselines),) * 2)
    for k, bl in enumerate(baselines):
        if stochastic == "apriori":
            block = _apriori_block(bl.length, R_enu, sigma_h, sigma_v)
        else:
            block = _make_pd(bl.cov)
        Q[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = block
    return Q


def _design_matrix(
    baselines: Sequence[BaselineResult],
    unknowns: Sequence[str],
    fixed: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix A and known terms b of y = A x + b.

    A has +I (3x3) in the columns of the rover and -I in those of the base. When the rover
    or the base is a fixed point, its coordinates are moved to the known terms b.
    """
    A = np.zeros((3 * len(baselines), 3 * len(unknowns)))
    b = np.zeros(3 * len(baselines))
    for k, bl in enumerate(baselines):
        rows = slice(3 * k, 3 * k + 3)
        for point, sign in ((bl.rover, +1), (bl.base, -1)):
            if point in unknowns:
                j = list(unknowns).index(point)
                A[rows, 3 * j : 3 * j + 3] = sign * np.eye(3)
            else:
                b[rows] += sign * fixed[point]
    return A, b


def adjust_network(
    baselines: Sequence[BaselineResult],
    fixed: Mapping[str, Sequence[float]],
    stochastic: Literal["apriori", "rtklib"] = "apriori",
    sigma_h: tuple[float, float] = SIGMA_H,
    sigma_v: tuple[float, float] = SIGMA_V,
) -> Adjustment:
    """Weighted least squares of the baseline vectors with the `fixed` points held.

    Args:
        baselines: Baselines (base -> rover vectors) forming the network.
        fixed: Point name -> ECEF coordinates [m] of the held points.
        stochastic: Baseline error model, "apriori" (ENU errors from `sigma_h`/`sigma_v`)
            or "rtklib" (formal RTKLIB covariance); see the module docstring.
        sigma_h: A-priori horizontal precision as (constant [m], ppm of the length).
            Used by the "apriori" model only.
        sigma_v: A-priori vertical precision, same form. "apriori" only.

    Returns:
        The adjustment, with parameters for the points not in `fixed`.

    Raises:
        ValueError: If `stochastic` is not "apriori" or "rtklib".
    """
    if stochastic not in ("apriori", "rtklib"):
        raise ValueError(f"stochastic must be 'apriori' or 'rtklib', not {stochastic!r}")
    fixed = {p: np.asarray(x, dtype=float) for p, x in fixed.items()}
    points = dict.fromkeys(p for bl in baselines for p in (bl.base, bl.rover))
    unknowns = [p for p in points if p not in fixed]

    y0 = np.concatenate([bl.dxyz for bl in baselines])
    Q = _observation_covariance(baselines, fixed, stochastic, sigma_h, sigma_v)
    A, b = _design_matrix(baselines, unknowns, fixed)
    n, m = A.shape

    Q_inv = np.linalg.inv(Q)  # weight matrix
    N = A.T @ Q_inv @ A  # normal matrix
    X_hat = np.linalg.solve(N, A.T @ Q_inv @ (y0 - b))  # estimated parameters
    Y_hat = A @ X_hat + b  # adjusted observations
    v_hat = y0 - Y_hat  # residuals
    sigma_v_hat = float(v_hat @ Q_inv @ v_hat / (n - m))  # a-posteriori variance factor

    N_inv = np.linalg.inv(N)
    Cxx = sigma_v_hat * N_inv
    Cyy = sigma_v_hat * (A @ N_inv @ A.T)
    Cvv = sigma_v_hat * (Q - A @ N_inv @ A.T)
    R = (Q - A @ N_inv @ A.T) @ Q_inv  # redundancy matrix
    tau = np.abs(v_hat) / np.sqrt(np.maximum(np.diag(Cvv), 1e-30))  # normalised residuals

    return Adjustment(
        unknowns=unknowns,
        fixed=fixed,
        baselines=list(baselines),
        y0=y0,
        b=b,
        A=A,
        Q=Q,
        X_hat=X_hat,
        Y_hat=Y_hat,
        v_hat=v_hat,
        sigma_v_hat=sigma_v_hat,
        Cxx=Cxx,
        Cyy=Cyy,
        Cvv=Cvv,
        R=R,
        tau=tau,
    )


# ------------------------------------------------------------------------------
# Result tables
# ------------------------------------------------------------------------------


def residual_table(adj: Adjustment) -> pd.DataFrame:
    """Residuals, redundancy number and normalised residual of every baseline.

    Args:
        adj: Result of `adjust_network`.

    Returns:
        One row per baseline: session, base, rover, vx_mm, vy_mm, vz_mm [mm], redundancy,
        tau (largest of the three normalised residuals) and outlier (tau > TAU_CRITICAL).
    """
    tau = adj.tau.reshape(-1, 3).max(axis=1)
    return pd.DataFrame(
        {
            "session": [bl.session for bl in adj.baselines],
            "base": [bl.base for bl in adj.baselines],
            "rover": [bl.rover for bl in adj.baselines],
            "vx_mm": adj.v_hat[0::3] * 1e3,
            "vy_mm": adj.v_hat[1::3] * 1e3,
            "vz_mm": adj.v_hat[2::3] * 1e3,
            "redundancy": np.diag(adj.R).reshape(-1, 3).sum(axis=1),
            "tau": tau,
            "outlier": tau > TAU_CRITICAL,
        }
    )


def ecef_table(adj: Adjustment) -> pd.DataFrame:
    """Adjusted ECEF coordinates and standard deviations of the unknown points.

    Args:
        adj: Result of `adjust_network`.

    Returns:
        DataFrame indexed by point with X, Y, Z, sd_X, sd_Y, sd_Z [m].
    """
    sd = np.sqrt(np.diag(adj.Cxx))
    return pd.DataFrame(
        {
            "point": adj.unknowns,
            "X": adj.X_hat[0::3], "Y": adj.X_hat[1::3], "Z": adj.X_hat[2::3],
            "sd_X": sd[0::3], "sd_Y": sd[1::3], "sd_Z": sd[2::3],
        }
    ).set_index("point")  # fmt: skip


def _ecef_to_utm(epsg: int) -> Callable[[np.ndarray], np.ndarray]:
    """Function ECEF -> (E, N, ellipsoidal height) of the UTM zone `epsg`."""
    to_utm = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    return lambda xyz: np.array(to_utm.transform(*xyz))


def utm_table(adj: Adjustment, utm_epsg: int = 32632) -> pd.DataFrame:
    """UTM coordinates (ellipsoidal height) with propagated standard deviations.

    The covariance is propagated as C_utm = J C J^T, with J = d(E, N, h) / d(X, Y, Z)
    computed by central differences. Fixed points are listed with zero standard deviation.
    The input ECEF coordinates are treated as WGS84, which equals ETRF2000 / RDN2008 within
    the accuracy of GNSS; for those frames pass their UTM code (e.g. 7791 for RDN2008 UTM 32N).

    Args:
        adj: Result of `adjust_network`.
        utm_epsg: EPSG code of the UTM zone (32632 = WGS84 UTM 32N, 7791 = RDN2008 UTM 32N).

    Returns:
        DataFrame with point, E, N, h_ell, sd_E, sd_N, sd_h [m]; unknown points first.
    """
    ecef_to_utm = _ecef_to_utm(utm_epsg)
    rows = []
    for j, point in enumerate(adj.unknowns):
        X = adj.X_hat[3 * j : 3 * j + 3]  # adjusted ECEF coordinates
        C = adj.Cxx[3 * j : 3 * j + 3, 3 * j : 3 * j + 3]  # their 3x3 covariance

        east, north, h = ecef_to_utm(X)
        step = 1.0
        J = np.column_stack(
            [(ecef_to_utm(X + step * e) - ecef_to_utm(X - step * e)) / (2 * step) for e in np.eye(3)]
        )
        sd_E, sd_N, sd_h = np.sqrt(np.diag(J @ C @ J.T))
        rows.append([point, east, north, h, sd_E, sd_N, sd_h])

    for point, X in adj.fixed.items():
        east, north, h = ecef_to_utm(X)
        rows.append([point, east, north, h, 0.0, 0.0, 0.0])
    return pd.DataFrame(rows, columns=["point", "E", "N", "h_ell", "sd_E", "sd_N", "sd_h"])


def coordinate_table(adj: Adjustment, utm_epsg: int = 32632) -> pd.DataFrame:
    """All adjusted points (fixed ones last): ECEF, geographic and UTM, with standard deviations.

    Geographic coordinates are on GRS80 (ellipsoidal height), UTM coordinates in `utm_epsg`.

    Args:
        adj: Result of `adjust_network`.
        utm_epsg: EPSG code of the UTM zone (32632 = WGS84 UTM 32N, 7791 = RDN2008 UTM 32N).

    Returns:
        DataFrame indexed by point, with the ECEF, geographic and UTM columns.
    """
    utm = utm_table(adj, utm_epsg).set_index("point")
    ecef = ecef_table(adj)
    fixed = pd.DataFrame(
        [[*X, 0.0, 0.0, 0.0] for X in adj.fixed.values()],
        index=pd.Index(list(adj.fixed), name="point"),
        columns=ecef.columns,
    )
    table = pd.concat([ecef, fixed]).join(utm.drop(columns="h_ell"))
    lat, lon, h = ecef_to_llh(*table[["X", "Y", "Z"]].to_numpy().T)
    table.insert(6, "lat", lat)
    table.insert(7, "lon", lon)
    table.insert(8, "h_ell", h)
    table.insert(0, "fixed", table.index.isin(list(adj.fixed)))
    return table.reset_index()
