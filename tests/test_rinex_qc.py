from pathlib import Path

import pandas as pd
import pytest

from gnsskit.rinex import (
    find_rinex_obs,
    obs_summary,
    plot_visibility,
    read_obs_types,
    read_visibility,
    satellite_stats,
    signal_stats,
)


@pytest.fixture
def rnx(tmp_path: Path) -> Path:
    """3 epochs at 1 s, a 4 s gap, an event record and a blank satellite record."""
    f = tmp_path / "o.26O"
    f.write_text(
        "TEST                                                        MARKER NAME\n"
        "                                                            END OF HEADER\n"
        "> 2026 11 04 09 00  0.0000000  0  2\nG01  1.0\nR02  2.0\n"
        "> 2026 11 04 09 00  1.0000000  0  2\nG01  1.0\nE03  3.0\n"
        "> 2026 11 04 09 00  1.5000000  3  1\nCOMMENT\n"
        "> 2026 11 04 09 00  2.0000000  0  2\nG01  1.0\nG02\n"
        "> 2026 11 04 09 00  6.0000000  0  1\nG01  1.0\n"
    )
    return f


def test_read_visibility(rnx: Path) -> None:
    vis = read_visibility(rnx)
    assert len(vis) == 6 and set(vis.sat) == {"G01", "R02", "E03"}  # blank G02 dropped
    assert vis.attrs == {"marker": "TEST", "n_events": 1}


def test_obs_summary(rnx: Path) -> None:
    s = obs_summary(rnx)
    assert s["n_epochs"] == 4 and s["duration_s"] == 6 and s["interval_s"] == 1
    assert (
        s["n_gaps"] == 1 and s["max_gap_s"] == 4 and s["first"] == pd.Timestamp("2026-11-04 09:00")
    )
    assert s["n_sats"] == {"G": 1, "R": 1, "E": 1}


def test_plot_visibility(rnx: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "vis.png"
    plot_visibility([rnx, rnx], out)
    assert out.stat().st_size > 0


@pytest.fixture
def rnx_obs(tmp_path: Path) -> Path:
    """GPS (C1C, S1C) and GLONASS (C1C) records; G02 lacks C1C in the second epoch, G01 is lost."""
    f = tmp_path / "s.26O"
    f.write_text(
        "G    2 C1C S1C                                              SYS / # / OBS TYPES\n"
        "R    1 C1C                                                  SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
        "> 2026 11 04 09 00  0.0000000  0  3\n"
        "G01  20000000.000 7        40.000 7\n"
        "G02  21000000.000 7        50.000 7\n"
        "R03  22000000.000 7\n"
        "> 2026 11 04 09 00  1.0000000  0  2\n"
        "G02                         30.000 7\n"
        "R03  22000001.000 7\n"
        "> 2026 11 04 09 00  2.0000000  0  1\n"
        "G01  20000002.000 7        60.000 7\n"
    )
    return f


def test_read_obs_types_with_continuation(tmp_path: Path) -> None:
    f = tmp_path / "t.26O"
    f.write_text(
        "G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES\n"
        "       C2W                                                  SYS / # / OBS TYPES\n"
        "E    1 C1C                                                  SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
    )
    assert read_obs_types(f) == {"G": ["C1C", "L1C", "D1C", "S1C", "C2W"], "E": ["C1C"]}


def test_satellite_stats(rnx_obs: Path) -> None:
    st = satellite_stats(rnx_obs)
    assert st.n_epochs.to_dict() == {"G01": 2, "G02": 2, "R03": 2}
    assert st.of_file["G01"] == pytest.approx(2 / 3)
    assert st.of_arc["G01"] == pytest.approx(2 / 3)  # lost at the second of three epochs
    assert st.of_arc["R03"] == 1.0


def test_signal_stats(rnx_obs: Path) -> None:
    st = signal_stats(rnx_obs)
    assert st.loc[("G", "C1C"), "n_obs"] == 3 and st.loc[("G", "C1C"), "completeness"] == pytest.approx(3 / 4)
    assert st.loc[("G", "S1C"), "completeness"] == 1.0
    assert st.loc[("G", "S1C"), "mean_snr"] == pytest.approx((40 + 50 + 30 + 60) / 4)
    assert pd.isna(st.loc[("G", "C1C"), "mean_snr"]) and st.loc[("R", "C1C"), "completeness"] == 1.0


def test_find_rinex_obs(tmp_path: Path) -> None:
    for n in ("a.26O", "b.26o", "c.obs", "a.26N", "a.26G", "readme.txt"):
        (tmp_path / n).touch()
    other = tmp_path / "x.rnx"
    other.touch()
    found = find_rinex_obs([tmp_path, other])
    assert [f.name for f in found] == ["a.26O", "b.26o", "c.obs", "x.rnx"]
