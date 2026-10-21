"""Small geodetic helpers on the WGS84 ellipsoid (numpy only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

ArrayLike = float | np.ndarray | pd.Series


def llh_to_ecef(lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> np.ndarray:
    """ECEF coordinates [m], shape (..., 3), from lat/lon [deg] and ellipsoidal height [m]."""
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
    h = np.asarray(h, float)
    return np.stack(
        [
            (n + h) * np.cos(phi) * np.cos(lam),
            (n + h) * np.cos(phi) * np.sin(lam),
            (n * (1 - WGS84_E2) + h) * np.sin(phi),
        ],
        axis=-1,
    )


def ecef_to_llh(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lat, lon [deg] and ellipsoidal height [m] from ECEF coordinates [m]."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(10):  # fixed-point iteration, converges to sub-mm in a few steps
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def ecef_to_enu_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation from ECEF to local east-north-up, lat/lon in degrees."""
    phi, lam = np.radians(lat), np.radians(lon)
    return np.array(
        [
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ]
    )


def radii(lat_deg: ArrayLike, h: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Meridian and prime-vertical radii plus height, in metres: (Rm + h, (Rn + h) cos(lat))."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    w = np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
    h = np.asarray(h, dtype=float)
    return WGS84_A * (1 - WGS84_E2) / w**3 + h, (WGS84_A / w + h) * np.cos(phi)


def enu_diff(
    lat: ArrayLike, lon: ArrayLike, h: ArrayLike, lat0: ArrayLike, lon0: ArrayLike, h0: ArrayLike
) -> np.ndarray:
    """(E, N, U) offsets in metres of points (lat, lon, h) from reference points (lat0, lon0, h0)."""
    rm, rn = radii(lat0, h0)
    e = np.radians(np.asarray(lon, dtype=float) - np.asarray(lon0, dtype=float)) * rn
    n = np.radians(np.asarray(lat, dtype=float) - np.asarray(lat0, dtype=float)) * rm
    u = np.asarray(h, dtype=float) - np.asarray(h0, dtype=float)
    return np.column_stack([e, n, u])
