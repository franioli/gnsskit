import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gnsskit import RtkConfig, rnx2rtkp, solve_static
from gnsskit.rnx2rtkp import build_command, find_rnx2rtkp, run_rnx2rtkp


def test_build_command() -> None:
    t0, t1 = pd.Timestamp("2026-09-06 08:00:00"), pd.Timestamp("2026-09-06 08:10:00")
    cmd = build_command(Path("c"), Path("r"), Path("b"), [Path("n1")], Path("o"), t0, t1, tint=5)
    assert cmd == ["rnx2rtkp", "-k", "c", "-ts", "2026/09/06", "08:00:00", "-te", "2026/09/06",
                   "08:10:00", "-ti", "5", "-o", "o", "r", "b", "n1"]  # fmt: skip
    assert not {"-ts", "-te", "-ti"} & set(build_command(Path("c"), Path("r"), Path("b"), [], Path("o")))


def test_run_writes_conf_and_output(tmp_path: Path, fake_rnx2rtkp: str) -> None:
    cfg = RtkConfig(elmask=12.0)
    out = run_rnx2rtkp(Path("r"), Path("b"), [], cfg, tmp_path / "new" / "o.pos", exe=fake_rnx2rtkp)
    assert out.exists()
    assert (tmp_path / "new" / "o.pos.conf").read_text() == cfg.to_text()


def test_run_raises_on_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="rnx2rtkp failed"):
        run_rnx2rtkp(Path("r"), Path("b"), [], RtkConfig(), tmp_path / "o.pos", exe="false")


def test_options_are_keyword_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_rnx2rtkp(Path("r"), Path("b"), [], RtkConfig(), tmp_path / "o.pos", "rnx2rtkp")  # type: ignore[misc]


def test_solve_static_returns_last_epoch(fake_rnx2rtkp: str) -> None:
    sol = solve_static(Path("r"), Path("b"), [], RtkConfig(), pd.Timestamp(0), pd.Timestamp(1), exe=fake_rnx2rtkp)
    assert sol is not None and sol.Q == 1 and sol.h == 2000.0


@pytest.mark.parametrize("platform, expected", [("linux", "make install"), ("win32", "PATH")])
def test_missing_exe_gives_install_hint(monkeypatch: pytest.MonkeyPatch, platform: str, expected: str) -> None:
    monkeypatch.setattr(rnx2rtkp, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(rnx2rtkp.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError) as e:
        find_rnx2rtkp()
    assert expected in str(e.value) and rnx2rtkp.RTKLIB_URL in str(e.value)


@pytest.mark.skipif(shutil.which("rnx2rtkp") is None, reason="rnx2rtkp not installed")
def test_rnx2rtkp_installed() -> None:
    assert Path(find_rnx2rtkp()).exists()
