import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from gnsskit import geodesy
from gnsskit.timesys import GPS_EPOCH, gps_seconds, utc_to_gpst

P063604 = (4428814.9484, 859166.7316, 4493637.0158)


def test_time_helpers() -> None:
    t = pd.Timestamp("2026-10-07 12:00:00")
    assert utc_to_gpst(t) == t + pd.Timedelta(seconds=18)
    assert gps_seconds(GPS_EPOCH + pd.Timedelta(weeks=1)) == 604800
    assert list(gps_seconds(pd.Series([GPS_EPOCH]))) == [0]


def test_ecef_to_llh_matches_pyproj() -> None:
    lat, lon, h = geodesy.ecef_to_llh(*P063604)
    lon_p, lat_p, h_p = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True).transform(*P063604)
    assert lat == pytest.approx(lat_p, abs=1e-9)
    assert lon == pytest.approx(lon_p, abs=1e-9)
    assert h == pytest.approx(h_p, abs=1e-4)


def test_ellipsoids_differ_by_about_a_tenth_of_a_millimetre() -> None:
    d = geodesy.llh_to_ecef(46.3, 9.8, 2300.0) - geodesy.llh_to_ecef(46.3, 9.8, 2300.0, geodesy.WGS84)
    assert 1e-5 < np.linalg.norm(d) < 3e-4
    lat, lon, h = geodesy.ecef_to_llh(*geodesy.llh_to_ecef(46.3, 9.8, 2300.0, geodesy.WGS84), ellipsoid=geodesy.WGS84)
    assert (lat, lon, h) == pytest.approx((46.3, 9.8, 2300.0), abs=1e-9)


def test_llh_round_trip() -> None:
    xyz = geodesy.llh_to_ecef(45.1, 10.8, 62.6)
    assert np.allclose(geodesy.ecef_to_llh(*xyz), (45.1, 10.8, 62.6), atol=1e-9)


def test_enu_matrix_is_a_rotation_with_up_vertical() -> None:
    R = geodesy.ecef_to_enu_matrix(45.1, 10.8)
    assert np.allclose(R @ R.T, np.eye(3))
    up = geodesy.llh_to_ecef(45.1, 10.8, 1.0) - geodesy.llh_to_ecef(45.1, 10.8, 0.0)
    assert np.allclose(R @ up, [0, 0, 1], atol=1e-9)


def test_enu_diff_north() -> None:
    e, n, u = geodesy.enu_diff(45.00001, 10.0, 0.0, 45.0, 10.0, 0.0)[0]
    assert n == pytest.approx(1.11, abs=0.01)
    assert e == 0 and u == 0
