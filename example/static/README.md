# Example: static processing, SPIN3 base to Emlid Reach M2

Static positioning of an Emlid Reach M2 receiver against the SPIN3 permanent station SONP
(Sondrio), baseline length about 14 km, session of about 5 hours. The Reach M2 stayed on the same
point at the Ventina glacier (Valmalenco) on 2026-09-06 and later served as base station for a
drone flight (see [../kinematic](../kinematic)).

## Run

```bash
python static_processing.py
```

It runs `rnx2rtkp` (RTKLIB demo5 or similar must be on your PATH), prints the fix rate and the final
position, and writes `output/reach_m2.pos` (all epochs) and `output/reach_m2_position.csv`
(final position, the base of the kinematic example). `rnx2rtkp` needs the `gnsskit` package; run
`pip install -e .` from the repository root.

## Data (`data/`)

| File | Content |
|---|---|
| `sonp249g00.26o` | SPIN3 SONP observations, RINEX 3.04, GPS + GLONASS + Galileo + BeiDou. Stonex SC2200 receiver, antenna STXSA1200 STXR (height 0.05 m, read from the header). Day 249 of 2026. **Base** |
| `reach_m2.26o` | Emlid Reach M2 observations, RINEX 3.03, GPS, GLONASS, Galileo, QZSS and BeiDou. L1 + L2 (E5b for Galileo, B2I for BeiDou). Antenna type and height not in the header. **Rover** |
| `reach_m2.26p` | Broadcast navigation messages (all systems) recorded by the Reach M2, RINEX 3.03 |

Both observation files are cropped to the Reach session (08:26 to 13:20 GPST) and **decimated from
1 s to 30 s** to keep the example small (the raw files are 160 MB and 52 MB).
`prepare_data.py RAW_DIR` repeats this from the raw files with `gnsskit.crop_rinex`.

The SONP navigation files are not part of the example: the Reach navigation file is used instead.

## Processing choices

| Option | Value | Why |
|---|---|---|
| `posmode` | `static` | one fixed point |
| `frequency` | `l1+l2` | dual frequency, needed to fix ambiguities over 14 km |
| `soltype` | `forward` | the last epoch is the estimate from the whole session |
| `ionoopt` | `brdc` | broadcast ionosphere model; `dual-freq` (iono-free) did not fix in a quick test |
| `tropopt` | `est-ztd` | the height difference to the base is 1930 m; estimating the zenith delay moved the height by 7 cm compared with `saas` |
| `navsys` | `GREC` | GPS, GLONASS, Galileo, BeiDou |
| base position | RINEX header | the SONP header coordinates, the same as in the Emlid Studio solution used for comparison (not checked against the SPIN3 published coordinates) |

No antenna calibration file is used. The SONP antenna is not in `igs20.atx`, and the Reach antenna is unknown.

## Result

About 83 % of the epochs are fixed, the final epoch is fixed (ratio 7.0, 12 satellites) with a
formal standard deviation below 3 mm. Reach M2 position:

```
lat 46.279077742   lon 9.780507778   h 2302.574 m  (WGS84 ellipsoidal)
```

The same data processed with Emlid Studio gave h = 2302.622 m, with horizontal differences of
about 2 cm. The formal standard deviation is optimistic. Plan on a few centimetres of
uncertainty, mostly in height, caused by the troposphere and by the unmodelled antenna
(not verified against an independent reference).
