"""Shared RTKLIB-based GNSS post-processing tools."""

from importlib.metadata import version

from .kinematic import interpolate
from .refsys import (
    GPS_EPOCH,
    GPST_UTC_OFFSET_S,
    GRS80,
    WGS84,
    Ellipsoid,
    enu_diff,
    gpst_to_local,
    utc_to_gpst,
)
from .rinex import (
    find_rinex_obs,
    header_llh,
    obs_summary,
    plot_visibility,
    read_visibility,
    rinex_span,
    satellite_stats,
    signal_stats,
    trim_rinex,
)
from .rtklib import (
    RtkConfig,
    navsys_mask,
    read_header,
    read_pos,
    read_ref_pos,
    run_rnx2rtkp,
)
from .static import (
    baselines_to_df,
    df_to_baselines,
    loop_closures,
    process_baseline,
    solve_static,
)

__version__ = version("gnsskit")
__all__ = [
    "GPST_UTC_OFFSET_S",
    "GPS_EPOCH",
    "GRS80",
    "WGS84",
    "Ellipsoid",
    "RtkConfig",
    "baselines_to_df",
    "df_to_baselines",
    "enu_diff",
    "find_rinex_obs",
    "gpst_to_local",
    "header_llh",
    "interpolate",
    "loop_closures",
    "navsys_mask",
    "obs_summary",
    "plot_visibility",
    "process_baseline",
    "read_header",
    "read_pos",
    "read_ref_pos",
    "read_visibility",
    "rinex_span",
    "run_rnx2rtkp",
    "satellite_stats",
    "signal_stats",
    "solve_static",
    "trim_rinex",
    "utc_to_gpst",
]
