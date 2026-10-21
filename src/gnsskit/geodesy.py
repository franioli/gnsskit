"""Small geodetic helpers on a reference ellipsoid (numpy only).

The default is GRS80, the ellipsoid of ITRS, ETRS89/ETRF2000 and the Italian RDN2008. WGS84 has
the same semi-major axis and a flattening that differs in the 11th digit: the two give the same
coordinates to about 0.1 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ellipsoid:
    """Reference ellipsoid of revolution."""

    a: float  # semi-major axis [m]
    inv_f: float  # inverse flattening

    @property
    def e2(self) -> float:
        """First eccentricity squared."""
        f = 1 / self.inv_f
        return f * (2 - f)


GRS80 = Ellipsoid(6378137.0, 298.257222101)
WGS84 = Ellipsoid(6378137.0, 298.257223563)

ArrayLike = float | np.ndarray | pd.Series


def llh_to_ecef(
    lat: ArrayLike, lon: ArrayLike, h: ArrayLike, ellipsoid: Ellipsoid = GRS80
) -> np.ndarray:
    """ECEF coordinates from geodetic coordinates.

    Args:
        lat: Latitude [deg].
        lon: Longitude [deg].
        h: Ellipsoidal height [m].
        ellipsoid: Reference ellipsoid.

    Returns:
        Array of shape (..., 3) with X, Y, Z [m].
    """
    a, e2 = ellipsoid.a, ellipsoid.e2
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    h = np.asarray(h, float)
    return np.stack(
        [
            (n + h) * np.cos(phi) * np.cos(lam),
            (n + h) * np.cos(phi) * np.sin(lam),
            (n * (1 - e2) + h) * np.sin(phi),
        ],
        axis=-1,
    )


def ecef_to_llh(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, ellipsoid: Ellipsoid = GRS80
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geodetic coordinates from ECEF coordinates.

    Args:
        x: ECEF X [m].
        y: ECEF Y [m].
        z: ECEF Z [m].
        ellipsoid: Reference ellipsoid.

    Returns:
        (lat [deg], lon [deg], ellipsoidal height [m]).
    """
    a, e2 = ellipsoid.a, ellipsoid.e2
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):  # fixed-point iteration, converges to sub-mm in a few steps
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def ecef_to_enu_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation matrix from ECEF to local east-north-up.

    Args:
        lat: Latitude of the local origin [deg].
        lon: Longitude of the local origin [deg].

    Returns:
        (3, 3) matrix R, with enu = R @ (ecef difference).
    """
    phi, lam = np.radians(lat), np.radians(lon)
    return np.array(
        [
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ]
    )


def radii(
    lat_deg: ArrayLike, h: ArrayLike, ellipsoid: Ellipsoid = GRS80
) -> tuple[np.ndarray, np.ndarray]:
    """Meridian and prime-vertical radii plus height: (Rm + h, (Rn + h) cos(lat)).

    Multiplying them by a latitude / longitude difference in radians gives metres.

    Args:
        lat_deg: Latitude [deg].
        h: Ellipsoidal height [m].
        ellipsoid: Reference ellipsoid.

    Returns:
        (Rm + h, (Rn + h) cos(lat)) [m].
    """
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    a, e2 = ellipsoid.a, ellipsoid.e2
    w = np.sqrt(1 - e2 * np.sin(phi) ** 2)
    h = np.asarray(h, dtype=float)
    return a * (1 - e2) / w**3 + h, (a / w + h) * np.cos(phi)


def enu_diff(
    lat: ArrayLike,
    lon: ArrayLike,
    h: ArrayLike,
    lat0: ArrayLike,
    lon0: ArrayLike,
    h0: ArrayLike,
    ellipsoid: Ellipsoid = GRS80,
) -> np.ndarray:
    """(E, N, U) offsets of points from reference points, by small-difference approximation.

    Args:
        lat: Latitude of the points [deg].
        lon: Longitude of the points [deg].
        h: Ellipsoidal height of the points [m].
        lat0: Latitude of the reference points [deg].
        lon0: Longitude of the reference points [deg].
        h0: Ellipsoidal height of the reference points [m].
        ellipsoid: Reference ellipsoid.

    Returns:
        Array of shape (n, 3) with E, N, U [m].
    """
    rm, rn = radii(lat0, h0, ellipsoid)
    e = np.radians(np.asarray(lon, dtype=float) - np.asarray(lon0, dtype=float)) * rn
    n = np.radians(np.asarray(lat, dtype=float) - np.asarray(lat0, dtype=float)) * rm
    u = np.asarray(h, dtype=float) - np.asarray(h0, dtype=float)
    return np.column_stack([e, n, u])
