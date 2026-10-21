"""Kinematic (PPK) processing: DJI Mavic 3E -> Emlid Reach M2 base.

The base position is the static solution of `../static/static_processing.py`. The drone
trajectory is computed with rnx2rtkp (forward + backward combined), then interpolated at
the photo times listed in the DJI `.MRK` file and compared with the positions stored by DJI.

Usage:
    python kinematic_processing.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnsskit import RtkConfig, run_rnx2rtkp
from gnsskit.geodesy import enu_diff
from gnsskit.pos import read_pos
from gnsskit.timesys import GPS_EPOCH
from gnsskit.trajectory import interpolate

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"
BASE_POSITION = HERE.parent / "static" / "output" / "reach_m2_position.csv"

# MRK line: n  TOW[s]  [week]  dN,N dE,E dV,V  lat,Lat lon,Lon h,Ellh  sd sd sd  Q,Q
MRK_LINE = re.compile(
    r"(\d+)\s+([\d.]+)\s+\[(\d+)\].*?([\d.]+),Lat\s+([\d.]+),Lon\s+([\d.]+),Ellh"
)

config = RtkConfig.kinematic().updated(
    frequency="l1+l2",
    elmask=15.0,
    ionoopt="brdc",
    tropopt="saas",
    navsys="GEC",  # GPS, Galileo, BeiDou
    gloarmode="off",
    bdsarmode=True,
    solformat="llh",
)


def read_mrk(path: Path) -> pd.DataFrame:
    """Photo events of a DJI MRK file: photo number, GPST time, and DJI lat, lon, h."""
    rows = []
    for line in path.read_text().splitlines():
        m = MRK_LINE.search(line)
        if m:
            n, tow, week, lat, lon, h = m.groups()
            t = GPS_EPOCH + pd.Timedelta(weeks=int(week), seconds=float(tow))
            rows.append((int(n), t, float(lat), float(lon), float(h)))
    return pd.DataFrame(rows, columns=["photo", "t", "lat", "lon", "h"])


def main() -> None:
    """Run rnx2rtkp, summarise the trajectory and compare it with the MRK positions."""
    if not BASE_POSITION.exists():
        raise SystemExit(f"{BASE_POSITION} not found: run ../static/static_processing.py first")
    base = pd.read_csv(BASE_POSITION).iloc[0]
    cfg = config.with_base_llh(base.lat, base.lon, base.height)

    pos_file = run_rnx2rtkp(
        DATA / "drone.obs", DATA / "reach_m2.26o", [DATA / "drone.nav", DATA / "reach_m2.26p"],
        cfg, OUT / "drone.pos",
    )  # fmt: skip
    pos_file.with_name(f"{pos_file.stem}_events.pos").unlink(
        missing_ok=True
    )  # empty events
    pos = read_pos(pos_file)
    if pos.empty:
        raise SystemExit(
            "no solution: check the base position and the navigation files"
        )

    print(f"epochs: {len(pos)}  {pos.t.iloc[0]} -> {pos.t.iloc[-1]} GPST")
    print(
        "Q counts (1 = fixed, 2 = float, 5 = single):",
        pos.Q.value_counts().sort_index().to_dict(),
    )
    fixed = pos[pos.Q == 1]
    print(f"fixed: {len(fixed) / len(pos) * 100:.1f} %, median sd n/e/u [mm]: "
          f"{fixed.sdn.median() * 1e3:.1f} / {fixed.sde.median() * 1e3:.1f} / "
          f"{fixed.sdu.median() * 1e3:.1f}")  # fmt: skip

    photos = read_mrk(DATA / "drone.mrk")
    inside = photos[(photos.t >= pos.t.iloc[0]) & (photos.t <= pos.t.iloc[-1])]
    at_photos = interpolate(
        pos, inside.t.map(lambda t: (t - GPS_EPOCH).total_seconds()).to_numpy()
    )
    at_photos.insert(0, "photo", inside.photo.to_numpy())
    at_photos.drop(columns=["age", "gps_s"]).to_csv(
        OUT / "photo_positions.csv", index=False
    )
    print(f"{len(at_photos)} photo positions -> {OUT / 'photo_positions.csv'}")

    # DJI stores its own positions, computed against another base: a constant offset is expected
    d = enu_diff(
        at_photos.lat, at_photos.lon, at_photos.h, inside.lat, inside.lon, inside.h
    )
    print("PPK - MRK [m]   mean / std")
    for name, col in zip("ENU", d.T, strict=True):
        print(f"  d{name}: {col.mean():+.3f} / {np.std(col):.3f}")


if __name__ == "__main__":
    main()
