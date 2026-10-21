"""Crop and decimate the raw RINEX files into the small `data/` folder used by the example.

Usage:
    python prepare_data.py RAW_DIR

RAW_DIR holds `sonp249g00.rnx/sonp249g00.26o` (SPIN3 SONP, 1 s) and
`Reach_raw_20260906082522_RINEX_3_03/` (Emlid Reach M2, 1 s). Base and rover are cropped to
their common time span and decimated to 30 s, with RTKLIB `convbin` if it is on PATH and
with the Python implementation otherwise.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from gnsskit import crop_rinex, rinex_span

HERE = Path(__file__).parent
INTERVAL = 30.0  # s


def main(raw: Path) -> None:
    """Write the reduced files to `data/`."""
    out = HERE / "data"
    reach = raw / "Reach_raw_20260906082522_RINEX_3_03"
    rover = reach / "Reach_raw_20260906082522.26O"
    base = raw / "sonp249g00.rnx/sonp249g00.26o"

    spans = [rinex_span(rover), rinex_span(base)]
    start = max(s for s, _ in spans)
    end = min(e for _, e in spans)
    print(f"common window: {start} -> {end} GPST")

    crop_rinex(base, out / "sonp249g00.26o", start, end, INTERVAL, use_convbin=True)
    crop_rinex(rover, out / "reach_m2.26o", start, end, INTERVAL, use_convbin=True)
    shutil.copy(reach / "Reach_raw_20260906082522.26P", out / "reach_m2.26p")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
