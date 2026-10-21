# gnsskit

Shared tools for post-processing GNSS data with [RTKLIB](https://www.rtklib.com/) `rnx2rtkp`, used by
[djippk](https://github.com/franioli/djippk) (drone PPK) and [stopgo](https://github.com/franioli/stopgo)
(stop-and-go surveys).

- **Configuration**: `RtkConfig`, a typed, immutable set of `rnx2rtkp` options (`from_file`, `updated`,
  `with_options`, `with_base_llh`, `write`).
- **Running**: `run_rnx2rtkp`, `solve_static`, `process_baseline` (static baselines with covariance).
- **Reading**: `read_pos` (llh, xyz and enu `.pos` files), `rinex_span`, `header_llh`, `crop_rinex`.
- **Analysis**: trajectory interpolation and averaging, and a least-squares adjustment of a baseline network
  (`gnsskit.network`, needs the `network` extra).

## Install

```sh
pip install git+https://github.com/franioli/gnsskit                # library
pip install 'gnsskit[network] @ git+https://github.com/franioli/gnsskit'   # + network adjustment
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

See [example/](example/) for complete static, kinematic and network adjustment examples.
