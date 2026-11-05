"""Static processing: single solutions over a window, baselines and their network adjustment.

The least-squares adjustment lives in `gnsskit.static.network` and is not imported here,
because it needs the `network` extra (scipy, pyproj).
"""

from .baseline import (
    BaselineResult,
    baseline_from_solution,
    baselines_to_df,
    df_to_baselines,
    extract_baseline,
    process_baseline,
    read_baselines,
    write_baselines,
)
from .solve import solve_static

__all__ = [
    "BaselineResult",
    "baseline_from_solution",
    "baselines_to_df",
    "df_to_baselines",
    "extract_baseline",
    "process_baseline",
    "read_baselines",
    "solve_static",
    "write_baselines",
]
