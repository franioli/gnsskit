import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from gnsskit import refsys
from gnsskit.refsys import GPS_EPOCH, gps_seconds, gpst_to_local, utc_to_gpst

P063604 = (4428814.9484, 859166.7316, 4493637.0158)


def test_time_helpers() -> None:
    t = pd.Timestamp("2026-10-07 12:00:00")
    assert utc_to_gpst(t) == t + pd.Timedelta(seconds=18)
    assert gps_seconds(GPS_EPOCH + pd.Timedelta(weeks=1)) == 604800
    assert list(gps_seconds(pd.Series([GPS_EPOCH]))) == [0]


def test_ecef_to_llh_matches_pyproj() -> None:
    lat, lon, h = refsys.ecef_to_llh(*P063604)
    lon_p, lat_p, h_p = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True).transform(
        *P063604
    )
    assert lat == pytest.approx(lat_p, abs=1e-9)
    assert lon == pytest.approx(lon_p, abs=1e-9)
    assert h == pytest.approx(h_p, abs=1e-4)


def test_ellipsoids_differ_by_about_a_tenth_of_a_millimetre() -> None:
    d = refsys.llh_to_ecef(46.3, 9.8, 2300.0) - refsys.llh_to_ecef(46.3, 9.8, 2300.0, refsys.WGS84)
    assert 1e-5 < np.linalg.norm(d) < 3e-4
    lat, lon, h = refsys.ecef_to_llh(
        *refsys.llh_to_ecef(46.3, 9.8, 2300.0, refsys.WGS84), ellipsoid=refsys.WGS84
    )
    assert (lat, lon, h) == pytest.approx((46.3, 9.8, 2300.0), abs=1e-9)


def test_llh_round_trip() -> None:
    xyz = refsys.llh_to_ecef(45.1, 10.8, 62.6)
    assert np.allclose(refsys.ecef_to_llh(*xyz), (45.1, 10.8, 62.6), atol=1e-9)


def test_enu_matrix_is_a_rotation_with_up_vertical() -> None:
    R = refsys.ecef_to_enu_matrix(45.1, 10.8)
    assert np.allclose(R @ R.T, np.eye(3))
    up = refsys.llh_to_ecef(45.1, 10.8, 1.0) - refsys.llh_to_ecef(45.1, 10.8, 0.0)
    assert np.allclose(R @ up, [0, 0, 1], atol=1e-9)


def test_enu_diff_north() -> None:
    e, n, u = refsys.enu_diff(45.00001, 10.0, 0.0, 45.0, 10.0, 0.0)[0]
    assert n == pytest.approx(1.11, abs=0.01)
    assert e == 0 and u == 0


def test_gpst_to_local_follows_daylight_saving() -> None:
    winter, summer = pd.Timestamp("2026-11-04 09:50:02"), pd.Timestamp("2026-07-01 12:00:18")
    assert gpst_to_local(winter) == pd.Timestamp("2026-11-04 10:49:44")  # CET, UTC+1
    assert gpst_to_local(summer) == pd.Timestamp("2026-07-01 14:00:00")  # CEST, UTC+2
    both = gpst_to_local(pd.Series([winter, summer]))
    assert both.tolist() == [pd.Timestamp("2026-11-04 10:49:44"), pd.Timestamp("2026-07-01 14:00:00")]
    assert utc_to_gpst(gpst_to_local(winter, "UTC", 0)) == winter + pd.Timedelta(seconds=18)


def test_enu_frame_points_vectors_and_covariances() -> None:
    frame = refsys.ENUFrame.from_llh(45.48, 9.23, 150.0)
    assert np.allclose(frame.to_enu(frame.origin), 0)
    assert np.allclose(frame.llh_to_enu(45.48, 9.23, 250.0), [0, 0, 100], atol=1e-6)
    pts = frame.origin + np.array([[300.0, -200.0, 50.0], [-10.0, 700.0, 5.0]])
    assert np.allclose(frame.to_ecef(frame.to_enu(pts)), pts)
    lat, lon, h = frame.enu_to_llh(frame.to_enu(pts))
    assert np.allclose(refsys.llh_to_ecef(lat, lon, h), pts, atol=1e-6)
    # a vector is only rotated: the difference of two transformed points
    assert np.allclose(frame.vector_to_enu(pts[1] - pts[0]), frame.to_enu(pts[1]) - frame.to_enu(pts[0]))
    cov = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]])
    assert np.allclose(frame.cov_to_ecef(frame.cov_to_enu(cov)), cov)
    assert np.trace(frame.cov_to_enu(cov)) == pytest.approx(np.trace(cov))  # rotation invariant
    # independent reference: PROJ topocentric conversion (exact, like ENUFrame)
    topo = Transformer.from_pipeline(
        "+proj=pipeline +step +proj=cart +ellps=GRS80 "
        "+step +proj=topocentric +ellps=GRS80 +lat_0=45.48 +lon_0=9.23 +h_0=150"
    )
    lat1, lon1, h1 = 45.486, 9.238, 160.0
    assert np.allclose(frame.llh_to_enu(lat1, lon1, h1), topo.transform(lon1, lat1, h1), atol=1e-6)
