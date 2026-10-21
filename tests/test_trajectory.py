import pandas as pd
import pytest

from gnsskit import read_pos
from gnsskit.trajectory import average, interpolate

from conftest import llh_header, llh_line


@pytest.fixture
def pos(tmp_path):
    f = tmp_path / "t.pos"
    rows = [("12:00:00.000", 46.0, 1), ("12:00:05.000", 46.00001, 1), ("12:00:10.000", 46.00002, 2)]
    f.write_text(llh_header() + "".join(llh_line(t, lat, 9.0, 2000.0, q) for t, lat, q in rows))
    return read_pos(f)


def test_interpolate(pos) -> None:
    t = pos.gps_s.iloc[0] + 2.5
    ev = interpolate(pos, [t])
    assert ev.lat[0] == pytest.approx(46.000005, abs=1e-12)
    assert ev.Q[0] == 1 and ev.attrs["format"] == "llh"
    assert interpolate(pos, [pos.gps_s.iloc[0] + 7.5]).Q[0] == 2  # worse of the neighbours


def test_average_fix_only(pos) -> None:
    t0, t1 = pos.t.iloc[0], pos.t.iloc[-1]
    a = average(pos, t0, t1)
    assert (a["n_epochs"], a["n_used"], a["fixed"]) == (3, 2, True)
    assert a["fix_pct"] == pytest.approx(200 / 3)
    assert a["lat"] == pytest.approx(46.000005) and a["sd_n"] > 0
    assert average(pos, t0, t1, fix_only=False)["n_used"] == 3
    empty = average(pos, t1 + pd.Timedelta(hours=1), t1 + pd.Timedelta(hours=2))
    assert empty["n_used"] == 0 and "lat" not in empty
