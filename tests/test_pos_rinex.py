from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnsskit import read_pos, rinex_span
from gnsskit.pos import read_header, read_ref_pos
from gnsskit.rinex import header_llh

from conftest import DATA, llh_header, llh_line


def test_read_pos_sorted_concat_and_gps_seconds(tmp_path: Path) -> None:
    a, b = tmp_path / "a.pos", tmp_path / "b.pos"
    a.write_text(llh_header() + llh_line("10:00:10.000", 46, 9, 1, 1))
    b.write_text(llh_header() + llh_line("10:00:00.000", 46, 9, 1, 2))
    df = read_pos([a, b])
    assert df.Q.tolist() == [2, 1] and df.attrs["format"] == "llh"
    assert df.gps_s.diff().iloc[1] == 10


def test_utc_file_is_shifted(tmp_path: Path) -> None:
    f = tmp_path / "u.pos"
    f.write_text(llh_header("UTC") + llh_line("10:00:00.000", 46, 9, 1, 1))
    assert read_pos(f).t.iloc[0] == pd.Timestamp("2026-09-06 10:00:18")


def test_xyz_format_and_ref_pos() -> None:
    df = read_pos(DATA / "xyz_static.pos")
    assert df.attrs["format"] == "xyz" and {"x", "sdxy"} <= set(df.columns)
    assert np.allclose(read_ref_pos(DATA / "xyz_static.pos"), [4428814.9484, 859166.7316, 4493637.0158])
    assert read_header(DATA / "xyz_static.pos")[0].startswith("% program")


def test_mixed_and_unsupported_formats(tmp_path: Path) -> None:
    llh = tmp_path / "l.pos"
    llh.write_text(llh_header() + llh_line("10:00:00.000", 46, 9, 1, 1))
    with pytest.raises(ValueError, match="mix"):
        read_pos([llh, DATA / "xyz_static.pos"])
    dms = tmp_path / "d.pos"
    dms.write_text("%  GPST  latitude(d'")
    with pytest.raises(ValueError, match="unsupported"):
        read_pos(dms)


def test_header_only_and_no_header(tmp_path: Path) -> None:
    f = tmp_path / "h.pos"
    f.write_text(llh_header())
    df = read_pos(f)
    assert df.empty and df.attrs["format"] == "llh"
    g = tmp_path / "n.pos"
    g.write_text(llh_line("10:00:00.000", 46, 9, 1, 1))
    assert len(read_pos(g)) == 1
    with pytest.raises(ValueError, match="ref pos"):
        read_ref_pos(f.with_name("n.pos"))


def test_rinex_span_skips_event_records(tmp_path: Path) -> None:
    f = tmp_path / "o.26O"
    f.write_text(
        "> 2026 09 06 08 48 40.0000000  0  1\nG01 1\n"
        "> 2026 09 06 08 48 40.5000000  3  1\n"
        "> 2026 09 06 08 48 41.5000000  0  1\n"
    )
    assert rinex_span(f) == (pd.Timestamp("2026-09-06 08:48:40"), pd.Timestamp("2026-09-06 08:48:41.5"))
    empty = tmp_path / "e.26O"
    empty.write_text("header only\n")
    with pytest.raises(ValueError):
        rinex_span(empty)


def test_header_llh(tmp_path: Path) -> None:
    f = tmp_path / "b.23O"
    f.write_text(
        "  4355294.7798   738007.0719  4589832.7496                  APPROX POSITION XYZ \n"
        "        0.0500        0.0000        0.0000                  ANTENNA: DELTA H/E/N\n"
        "                                                            END OF HEADER       \n"
    )
    lat, lon, h = header_llh(f)
    assert lat == pytest.approx(46.28901474, abs=1e-8)
    assert lon == pytest.approx(9.61744532, abs=1e-8)
    assert h == pytest.approx(3220.0429 + 0.05, abs=1e-3)
