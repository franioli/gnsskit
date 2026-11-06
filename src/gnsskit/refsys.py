"""Reference systems: time (GPST, UTC) and space (ellipsoid, ECEF, geodetic, local ENU).

The default ellipsoid is GRS80, the ellipsoid of ITRS, ETRS89/ETRF2000 and the Italian
RDN2008. WGS84 has the same semi-major axis and a flattening that differs in the 11th digit:
the two give the same coordinates to about 0.1 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar

import numpy as np
import pandas as pd

# =============================================================================
# Time systems (GPST, UTC)
# =============================================================================

GPS_EPOCH = pd.Timestamp("1980-01-06")
GPST_UTC_OFFSET_S = 18  # GPST - UTC leap seconds (valid since 2017)

T = TypeVar("T", pd.Timestamp, pd.Series)


def utc_to_gpst(t: T, leap_s: int = GPST_UTC_OFFSET_S) -> T:
    """Shift UTC time(s) to GPST.

    Args:
        t: Timestamp or Series of UTC times.
        leap_s: GPST - UTC [s].

    Returns:
        The shifted time(s), same type as `t`.
    """
    return t + pd.Timedelta(seconds=leap_s)


def gpst_to_local(t: T, tz: str = "Europe/Rome", leap_s: int = GPST_UTC_OFFSET_S) -> T:
    """Convert GPST time(s) to local civil time, with daylight saving time.

    Args:
        t: Timestamp or Series of GPST times (naive).
        tz: IANA time zone of the result.
        leap_s: GPST - UTC [s].

    Returns:
        The local time(s) as naive datetimes, same type as `t`.
    """
    utc = t - pd.Timedelta(seconds=leap_s)
    if isinstance(utc, pd.Series):
        return utc.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    return utc.tz_localize("UTC").tz_convert(tz).tz_localize(None)


def gps_seconds(t: pd.Timestamp | pd.Series) -> float | pd.Series:
    """Seconds since the GPS epoch (1980-01-06), which avoids week roll-over problems.

    Args:
        t: Timestamp or Series of GPST times.

    Returns:
        Seconds as a float, or a Series of floats.
    """
    return (t - GPS_EPOCH) / pd.Timedelta(seconds=1)


# =============================================================================
# Reference ellipsoids
# =============================================================================


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


# =============================================================================
# Global coordinates: geodetic <-> ECEF
# =============================================================================


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


# =============================================================================
# Local frames: ENU rotation, ENU frame and small-difference offsets
# =============================================================================


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


@dataclass(frozen=True, eq=False)
class ENUFrame:
    """Local east-north-up frame with origin at a point, tangent to the ellipsoid there.

    Points are shifted to the origin and rotated; vectors (baselines) and covariance matrices
    are only rotated: enu = R (xyz - origin), denu = R dxyz, C_enu = R C_xyz R^T.
    Methods take one (3,) array or a stack (n, 3); covariances (3, 3) or (n, 3, 3).
    """

    origin: np.ndarray  # (3,) ECEF of the origin [m]
    ellipsoid: Ellipsoid = GRS80

    @classmethod
    def from_ecef(cls, xyz: ArrayLike, ellipsoid: Ellipsoid = GRS80) -> ENUFrame:
        """Frame with origin at an ECEF point [m]."""
        return cls(np.asarray(xyz, float).reshape(3), ellipsoid)

    @classmethod
    def from_llh(cls, lat: float, lon: float, h: float, ellipsoid: Ellipsoid = GRS80) -> ENUFrame:
        """Frame with origin at a geodetic point (lat, lon [deg], ellipsoidal h [m])."""
        return cls(llh_to_ecef(lat, lon, h, ellipsoid), ellipsoid)

    @cached_property
    def llh(self) -> tuple[float, float, float]:
        """Geodetic coordinates of the origin: lat, lon [deg], h [m]."""
        lat, lon, h = ecef_to_llh(*self.origin, ellipsoid=self.ellipsoid)
        return float(lat), float(lon), float(h)

    @cached_property
    def rotation(self) -> np.ndarray:
        """(3, 3) rotation R from ECEF to ENU; its inverse is R^T."""
        lat, lon, _ = self.llh
        return ecef_to_enu_matrix(lat, lon)

    # --- points: shift to the origin and rotate ---------------------------------

    def to_enu(self, xyz: ArrayLike) -> np.ndarray:
        """ENU coordinates [m] of ECEF points [m]."""
        return self.vector_to_enu(np.asarray(xyz, float) - self.origin)

    def to_ecef(self, enu: ArrayLike) -> np.ndarray:
        """ECEF coordinates [m] of ENU points [m]."""
        return self.origin + self.vector_to_ecef(enu)

    def llh_to_enu(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> np.ndarray:
        """ENU coordinates [m] of geodetic points (lat, lon [deg], h [m])."""
        return self.to_enu(llh_to_ecef(lat, lon, h, self.ellipsoid))

    def enu_to_llh(self, enu: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Geodetic coordinates (lat, lon [deg], h [m]) of ENU points [m]."""
        xyz = self.to_ecef(enu)
        return ecef_to_llh(xyz[..., 0], xyz[..., 1], xyz[..., 2], self.ellipsoid)

    # --- vectors and covariances: rotation only ---------------------------------

    def vector_to_enu(self, dxyz: ArrayLike) -> np.ndarray:
        """ENU components of ECEF vectors (e.g. baselines)."""
        return np.asarray(dxyz, float) @ self.rotation.T

    def vector_to_ecef(self, denu: ArrayLike) -> np.ndarray:
        """ECEF components of ENU vectors."""
        return np.asarray(denu, float) @ self.rotation

    def cov_to_enu(self, cov: ArrayLike) -> np.ndarray:
        """ENU covariances R C R^T of ECEF covariances."""
        r = self.rotation
        return r @ np.asarray(cov, float) @ r.T

    def cov_to_ecef(self, cov: ArrayLike) -> np.ndarray:
        """ECEF covariances R^T C R of ENU covariances."""
        r = self.rotation
        return r.T @ np.asarray(cov, float) @ r


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
