# Example: static baselines and network adjustment

Four GNSS receivers observed on 2026-10-07 (Leica, RINEX 3). Point `063604` has known
coordinates and is held fixed; the other points (`1002`, `1005`, `1006`) are computed.
Two 10-minute sessions give six baselines, which are then adjusted as a network.

## Notebooks

Run them in this order, from this folder, with `gnsskit` installed
(`pip install -e '.[network]'` from the repository root).

1. [newtork_baseline_computation.ipynb](newtork_baseline_computation.ipynb) runs RTKLIB
   (`rnx2rtkp`) on each baseline of each session with precise orbits and clocks, and writes
   `output/baselines.csv` plus one `.pos` file per baseline.
2. [network_adjustment.ipynb](network_adjustment.ipynb) reads the CSV and does a weighted
   least-squares adjustment with `063604` fixed. It gives adjusted coordinates (ECEF, geographic,
   UTM 32N), residuals, redundancy numbers and outlier tests.

`rnx2rtkp` must be on your PATH. If it is not, `run_rnx2rtkp` raises an error with
installation instructions (RTKLIB demo5 is recommended).

## Data

| Folder | Content |
|---|---|
| `rinex/` | Observation (`.26o`) and broadcast navigation (`.26n` GPS, `.26g` GLONASS, `.26l` Galileo) files, one set per point |
| `aux/` | CODE final MGEX products for day 280: `*_ORB.SP3` orbits, `*_CLK.CLK` clocks, and the `igs20.atx` antenna models |
| `rtklib.conf` | RTKLIB options: static, GPS+GLONASS+Galileo, L1+L2, `sateph = precise` |
| `output/` | Output of the notebooks (not in the repository) |

`sateph = precise` means the SP3 and CLK files in `aux/` **must** be passed to `rnx2rtkp`
together with the broadcast files. Without them it finds no satellite positions and writes an
empty `.pos` file.