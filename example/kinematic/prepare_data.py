"""Copy the drone files and crop the drone and base observations to a common window.

Usage:
    python prepare_data.py RAW_DIR

RAW_DIR holds `DJI_202609061306_001_ventina/` (Mavic 3E PPK files) and
`Reach_raw_20260906082522_RINEX_3_03/` (base, 1 s). Drone and base are cropped to the time span
of the flight; the drone observations are decimated to 1 Hz (they are logged at 5 Hz) to
keep the files small. RTKLIB `convbin` is used if it is on PATH, the Python implementation
otherwise.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from gnsskit import crop_rinex, rinex_span

HERE = Path(__file__).parent
FLIGHT = "DJI_202609061306_001_ventina"
DRONE_INTERVAL = 1.0  # s


def main(raw: Path) -> None:
    """Write the reduced files to `data/`."""
    out = HERE / "data"
    flight = raw / FLIGHT
    reach = raw / "Reach_raw_20260906082522_RINEX_3_03"
    drone_obs = flight / f"{FLIGHT}_PPKOBS.obs"
    base_obs = reach / "Reach_raw_20260906082522.26O"

    start, end = rinex_span(drone_obs)
    print(f"common window: {start} -> {end} GPST")
    crop_rinex(base_obs, out / "reach_m2.26o", start, end, use_convbin=True)
    crop_rinex(drone_obs, out / "drone.obs", start, end, DRONE_INTERVAL, use_convbin=True)
    shutil.copy(reach / "Reach_raw_20260906082522.26P", out / "reach_m2.26p")
    shutil.copy(flight / f"{FLIGHT}_PPKNAV.nav", out / "drone.nav")
    shutil.copy(flight / f"{FLIGHT}_Timestamp.MRK", out / "drone.mrk")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
