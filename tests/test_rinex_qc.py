from pathlib import Path

import pandas as pd
import pytest

from gnsskit.rinex import obs_summary, plot_visibility, read_visibility


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
    assert s["n_gaps"] == 1 and s["max_gap_s"] == 4 and s["first"] == pd.Timestamp("2026-11-04 09:00")
    assert s["n_sats"] == {"G": 1, "R": 1, "E": 1}


def test_plot_visibility(rnx: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "vis.png"
    plot_visibility([rnx, rnx], out)
    assert out.stat().st_size > 0
