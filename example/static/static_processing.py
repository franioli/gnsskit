"""Static processing: SPIN3 base SONP -> Emlid Reach M2 (13 km baseline, about 5 h).

The base coordinates come from the RINEX header of SONP (`base_postype="rinexhead"`), so
the Reach position is in the frame of that header. The result is written to
`output/reach_m2_position.csv` and used as base position by the kinematic example.

Usage:
    python static_processing.py
"""

from __future__ import annotations

from pathlib import Path

from gnsskit import RtkConfig, rinex_span, run_rnx2rtkp
from gnsskit.pos import read_pos

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"

config = RtkConfig(
    posmode="static",
    frequency="l1+l2",
    soltype="forward",  # the last epoch is the solution of the whole session
    elmask=15.0,
    ionoopt="brdc",
    tropopt="est-ztd",  # 2000 m height difference to the base: estimate the zenith delay
    sateph="brdc",
    navsys="GREC",  # GPS, GLONASS, Galileo, BeiDou
    base_postype="rinexhead",
    solformat="llh",
)


def main() -> None:
    """Run rnx2rtkp and save the final Reach M2 position."""
    rover = DATA / "reach_m2.26o"
    base = DATA / "sonp249g00.26o"
    start, end = rinex_span(rover)
    print(f"session: {start} -> {end} GPST")

    pos_file = run_rnx2rtkp(
        rover, base, [DATA / "reach_m2.26p"], config, OUT / "reach_m2.pos",
        start=start, end=end, tint=30,
    )  # fmt: skip
    pos_file.with_name(f"{pos_file.stem}_events.pos").unlink(missing_ok=True)  # empty events
    pos = read_pos(pos_file)
    if pos.empty:
        raise SystemExit("no solution: check the navigation file and the time span")

    fixed = (pos.Q == 1).mean() * 100
    last = pos.iloc[-1]
    print(f"epochs: {len(pos)}, fixed: {fixed:.0f} %")
    print(f"final epoch: Q={last.Q}, ns={last.ns}, ratio={last.ratio:.1f}")
    print(f"Reach M2: lat {last.lat:.9f}, lon {last.lon:.9f}, h {last.h:.4f} m (ellipsoidal)")
    print(f"sd n/e/u: {last.sdn * 1e3:.1f} / {last.sde * 1e3:.1f} / {last.sdu * 1e3:.1f} mm")
    if last.Q != 1:
        print("warning: the final solution is not fixed")

    csv = OUT / "reach_m2_position.csv"
    csv.write_text(f"lat,lon,height\n{last.lat:.10f},{last.lon:.10f},{last.h:.5f}\n")
    print(f"written {csv}")


if __name__ == "__main__":
    main()
