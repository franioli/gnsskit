# Example: kinematic (PPK) processing of a drone, base Emlid Reach M2

Post-processed kinematic positioning of the GNSS receiver of a DJI Mavic 3 Enterprise (RTK
version) with respect to the Emlid Reach M2 that stood on the Ventina glacier. The base
coordinates are the static solution of [../static](../static), so run that example first (its
`output/reach_m2_position.csv` is already included). The flight was on 2026-09-06, 11:14 to 11:33 GPST,
about 19 minutes with 286 photos. The drone stayed within 1.3 km of the base.

## Run

```bash
python kinematic_processing.py
```

It runs `rnx2rtkp` (forward and backward filters combined), prints the quality of the trajectory,
and writes:

- `output/drone.pos`: the trajectory (lat, lon, ellipsoidal height, standard deviations, Q, ratio);
- `output/photo_positions.csv`: the trajectory interpolated at the time of each photo.

It then compares the photo positions with those stored by DJI in the MRK file.

## Data (`data/`)

| File | Content |
|---|---|
| `drone.obs` | Mavic 3E observations (`*_PPKOBS.obs`, RINEX 3.05): GPS, GLONASS, Galileo and BeiDou, L1 + L2 (E5b, B2I). **Rover** |
| `drone.nav` | Broadcast navigation messages of the drone (`*_PPKNAV.nav`) |
| `drone.mrk` | Photo events (`*_Timestamp.MRK`): number, GPS week and time of week, camera offsets, DJI position |
| `reach_m2.26o` | Reach M2 observations, 1 s, cropped to 11:10 to 11:38 GPST. **Base** |
| `reach_m2.26p` | Navigation messages recorded by the Reach M2 |

The drone logs at 5 Hz. The example uses **1 Hz** to keep the files small (the raw file is 23 MB); the
photo positions are then linear interpolations between epochs a second apart. For the final
products use the full 5 Hz data. `prepare_data.py RAW_DIR` repeats the reduction from the raw files.

## Processing choices

| Option | Value | Why |
|---|---|---|
| `posmode` | `kinematic` | the rover moves |
| `soltype` | `combined` | forward and backward filters combined |
| `dynamics` | off | a drone accelerates freely |
| `navsys` | `GEC` | GPS, Galileo, BeiDou. GLONASS is left out because its ambiguities are not fixed (`gloarmode=off`) |
| `ionoopt`, `tropopt` | `brdc`, `saas` | short baseline, the residual atmosphere cancels in the differences |
| base position | static solution | ellipsoidal WGS84, with antenna height 0 (the position is that of the antenna reference of the Reach) |

The trajectory is the path of the drone **antenna**. The camera is a few centimetres away, and its offsets are
in the MRK file, but they are not applied here.

## Result

1136 of 1160 epochs (97.9 %) are fixed, the others are float. The median standard deviation of the fixed
epochs is 3 mm (north), 2 mm (east), 9 mm (up).

The photo positions differ from the DJI ones by a nearly constant vector (mean E -0.17 m,
N -2.27 m, U +1.48 m, standard deviations of 3 to 5 cm). This is expected: DJI used the coordinates of another base, so the whole
flight is shifted, and the shift tells you nothing about the quality of this solution. The scatter comes
from the interpolation of the 1 Hz data and from the camera offsets that are not removed.

The height of the base is the weakest part of the chain: the static solution may be a few centimetres off in height,
and the whole trajectory follows it.
