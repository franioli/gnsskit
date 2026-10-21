from pathlib import Path

import pytest

from gnsskit import RtkConfig, navsys_mask


def test_navsys_mask() -> None:
    assert navsys_mask("GEC") == 41
    assert navsys_mask("gregc") == 45
    with pytest.raises(ValueError):
        navsys_mask("GX")


def test_options_rendering() -> None:
    cfg = RtkConfig(dynamics=True, base_pos=(1.0, 2.0, 3.0), phase_windup=None)
    opts = cfg.options
    assert opts["pos1-navsys"] == "9"
    assert opts["pos1-dynamics"] == "on"
    assert (opts["ant2-pos1"], opts["ant2-pos3"]) == ("1.0", "3.0")
    assert "pos1-posopt3" not in opts


def test_bad_literal_raises() -> None:
    with pytest.raises(ValueError):
        RtkConfig(posmode="foo")  # type: ignore[arg-type]


def test_from_file_and_overrides(tmp_path: Path) -> None:
    f = tmp_path / "a.conf"
    f.write_text("pos1-elmask = 10  # deg\nfoo-bar=1\n# comment\n")
    cfg = RtkConfig.from_file(f)
    assert cfg.options["pos1-elmask"] == "10" and cfg.options["pos1-posmode"] == "static"
    only = RtkConfig.from_file(f, RtkConfig.empty())
    assert only.options.keys() == {"pos1-elmask", "foo-bar"}
    assert cfg.with_options({"pos1-elmask": 5}).options["pos1-elmask"] == "5"
    assert cfg.updated(elmask=20).options["pos1-elmask"] == "10"  # extra overrides fields


def test_base_and_presets() -> None:
    cfg = RtkConfig().with_base_llh(46.0, 9.0, 100.0, antenna_h=1.5)
    assert cfg.has_base_position and not RtkConfig().has_base_position
    assert cfg.options["ant2-postype"] == "llh" and cfg.options["ant2-antdelu"] == "1.5"
    kin = RtkConfig.kinematic().options
    assert kin["pos1-posmode"] == "kinematic" and kin["misc-timeinterp"] == "on"


def test_write(tmp_path: Path) -> None:
    p = RtkConfig(elmask=12.0).write(tmp_path / "sub" / "x.conf")
    assert "pos1-elmask        =12.0\n" in p.read_text()
