# gnsskit

Shared tools for post-processing GNSS data with [RTKLIB](https://www.rtklib.com/) `rnx2rtkp`, used by
[djippk](https://github.com/franioli/djippk) (drone PPK) and [stopgo](https://github.com/franioli/stopgo)
(stop-and-go surveys).

- **RTKLIB wrappers** (`gnsskit.rtklib`): `RtkConfig`, a typed, immutable set of `rnx2rtkp` options
  (`from_file`, `updated`, `with_options`, `with_base_llh`, `write`); `run_rnx2rtkp` and `find_rnx2rtkp`;
  `read_pos` (llh, xyz and enu `.pos` files).
- **RINEX** (`gnsskit.rinex`): `rinex_span`, `header_llh`, `trim_rinex`; inspection with `obs_summary`
  (span, sampling, gaps, satellites), `read_visibility` and `plot_visibility` (RTKPLOT-style satellite
  visibility, needs the `plot` extra).
- **Static** (`gnsskit.static`): `solve_static` (one solution over a time window), `process_baseline`
  (static baselines with covariance), baseline IO and `loop_closures` (misclosure of every triangle of baselines); least-squares adjustment of a redundant baseline
  network in `gnsskit.static.network` (needs the `network` extra).
- **Kinematic** (`gnsskit.kinematic`): trajectory interpolation and averaging.
- **Reference systems** (`gnsskit.refsys`): GPST/UTC, ellipsoids, ECEF/geodetic conversions and
  `ENUFrame`, a local east-north-up frame at a point (points, vectors and covariances, both ways).

## Install

```sh
pip install gnsskit                  # library
pip install "gnsskit[network]"       # + network adjustment (scipy, pyproj)
pip install "gnsskit[network,plot]"  # + satellite visibility plots (matplotlib)
pip install git+https://github.com/franioli/gnsskit   # development version
```

`rnx2rtkp` is not installed by pip. Build the [demo5 fork](https://github.com/rtklibexplorer/RTKLIB) and put it on
your `PATH`; if it is missing, `run_rnx2rtkp` raises an error with instructions. Developed with `rnx2rtkp ver.EX 2.5.1`.

## Example

```python
from gnsskit import RtkConfig, read_pos, run_rnx2rtkp

config = RtkConfig(posmode="static", navsys="GEC", solformat="llh").with_base_llh(46.2, 9.8, 2300.0)
pos_file = run_rnx2rtkp(rover_obs, base_obs, [nav], config, "out.pos")
pos = read_pos(pos_file)
```

See [example/](https://github.com/franioli/gnsskit/tree/main/example) for complete static, kinematic and network adjustment examples.
