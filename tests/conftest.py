import stat
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"

LLH_COLS = (
    "latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)"
    "  sdeu(m)  sdun(m) age(s)  ratio"
)


def llh_header(tsys: str = "GPST") -> str:
    return f"% program   : RTKLIB test\n% ref pos   : 46.0 9.0 2000.0\n%\n%  {tsys}  {LLH_COLS}\n"


def llh_line(t: str, lat: float, lon: float, h: float, q: int, ratio: float = 10.0) -> str:
    return (
        f"2026/09/06 {t}   {lat:.9f}  {lon:.9f}  {h:.4f}   {q}  15   0.0040   0.0030"
        f"   0.0090  -0.0010  -0.0010   0.0010   0.00  {ratio:.1f}\n"
    )


@pytest.fixture
def fake_rnx2rtkp(tmp_path: Path) -> str:
    """An 'rnx2rtkp' that writes a one-epoch llh .pos to the -o path and the -k conf next to it."""
    pos = tmp_path / "fake.pos"
    pos.write_text(llh_header() + llh_line("12:00:00.000", 46.0, 9.0, 2000.0, 1))
    exe = tmp_path / "fake_rnx2rtkp"
    exe.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; [ "$1" = "-k" ] && conf="$2"; shift; done\n'
        f'cp "{pos}" "$out"\ncp "$conf" "$out.conf"\n'
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)
