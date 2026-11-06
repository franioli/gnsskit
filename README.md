# gnsskit

Shared tools for post-processing GNSS data with [RTKLIB](https://www.rtklib.com/) `rnx2rtkp`, used by
[djippk](https://github.com/franioli/djippk) (drone PPK) and [stopgo](https://github.com/franioli/stopgo)
(stop-and-go surveys).

> [!IMPORTANT]
> **RTKLIB is not included and must be installed separately.** gnsskit runs the RTKLIB
> command-line program `rnx2rtkp` (the CLI of RTKPOST) for the GNSS processing; `pip`/`uv` do
> not install it. See [RTKLIB](#rtklib-required-separately) below.

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

With [uv](https://docs.astral.sh/uv/) (recommended), in your project:

```sh
uv add gnsskit                  # library
uv add "gnsskit[network]"       # + network adjustment (scipy, pyproj)
uv add "gnsskit[network,plot]"  # + satellite visibility plots (matplotlib)
uv add git+https://github.com/franioli/gnsskit   # development version
```

or in an existing environment, `uv pip install gnsskit` (same extras).

With pip:

```sh
pip install gnsskit
pip install "gnsskit[network,plot]"
```

### RTKLIB (required separately)

The processing functions (`run_rnx2rtkp`, `process_baseline`, `solve_static`) call the RTKLIB
command-line program **`rnx2rtkp`**, which must be installed on your system and found on your
`PATH` (or passed with `exe=`); `trim_rinex(use_convbin=True)` also needs RTKLIB `convbin`.
The [demo5 fork](https://github.com/rtklibexplorer/RTKLIB) is
recommended; gnsskit is developed with `rnx2rtkp ver.EX 2.5.1`.

- **Windows**: download a [demo5 release](https://github.com/rtklibexplorer/RTKLIB/releases),
  unzip it and add the folder with `rnx2rtkp.exe` to your `PATH`.
- **Linux / macOS**: build it from source:
  ```sh
  git clone https://github.com/rtklibexplorer/RTKLIB
  cd RTKLIB/app/consapp/rnx2rtkp/gcc && make && sudo make install
  ```

Check it with `rnx2rtkp -?`. If it is missing, `run_rnx2rtkp` raises an error with these
instructions. The rest of gnsskit (RINEX tools, `read_pos`, loop closures, network adjustment,
reference systems) is pure Python and works without RTKLIB, e.g. on baselines computed elsewhere.

## Example

```python
from gnsskit import RtkConfig, read_pos, run_rnx2rtkp

config = RtkConfig(posmode="static", navsys="GEC", solformat="llh").with_base_llh(46.2, 9.8, 2300.0)
pos_file = run_rnx2rtkp(rover_obs, base_obs, [nav], config, "out.pos")
pos = read_pos(pos_file)
```

See [example/](https://github.com/franioli/gnsskit/tree/main/example) for complete static, kinematic and network adjustment examples.

## Contributing

Issues are welcome on [GitHub](https://github.com/franioli/gnsskit/issues). To contribute code:

1. Fork the repository on GitHub and clone your fork:
   ```sh
   git clone https://github.com/<your-username>/gnsskit && cd gnsskit
   git checkout -b my-feature
   ```
2. Set up the environment, make your changes and check them:
   ```sh
   uv sync                      # environment with the dev dependencies (incl. ruff)
   uv run pytest                # tests
   uv run ruff format .         # format the code
   uv run ruff check --fix .    # lint, fixing what can be fixed automatically
   ```

3. Push the branch to your fork and open a pull request against `main` of
   `franioli/gnsskit`.

## Releasing a new version

For the current maintainers only (it needs the PyPI credentials of the project).

1. Bump `version` in `pyproject.toml` (PyPI never accepts the same version twice), run the
   tests (`uv run pytest`), then commit and tag:
   ```sh
   git commit -am "Release X.Y.Z" && git tag -a vX.Y.Z -m "gnsskit X.Y.Z"
   git push origin main --tags
   ```
2. Build and upload, with a PyPI API token scoped to the `gnsskit` project kept outside the
   repository (here in `~/.config/pypi/gnsskit.token`, `chmod 600`):
   ```sh
   rm -rf dist && uv build
   UV_PUBLISH_TOKEN=$(cat ~/.config/pypi/gnsskit.token) uv publish
   ```
