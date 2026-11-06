"""Loop closures of a baseline network: a quick consistency check before the adjustment.

Three points A, B, C joined by the baselines A->B, B->C and C->A close on themselves:
the sum of the three vectors must be zero. The misclosure shows blunders (wrong fixes,
wrong antenna heights, wrong point names) without any stochastic model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
import pandas as pd

from ..refsys import ecef_to_enu_matrix, ecef_to_llh
from .baseline import BaselineResult

COLUMNS = ["loop", "baselines", "dx", "dy", "dz", "closure", "perimeter", "ppm"]


def _vector(bl: BaselineResult, start: str, end: str) -> np.ndarray:
    """Vector start -> end of a baseline, reversed if it was measured the other way round."""
    return bl.dxyz if (bl.base, bl.rover) == (start, end) else -bl.dxyz


def loop_closures(
    baselines: Sequence[BaselineResult], origin: Sequence[float] | None = None
) -> pd.DataFrame:
    """Misclosure of every triangle that can be formed with the baselines.

    A triangle is any three points that are pairwise joined by at least one baseline. If a pair has several baselines (repeated sessions), every combination is reported.

    Args:
        baselines: Baselines of the network.
        origin: ECEF coordinates [m] of a point of the network (e.g. the fixed one). If given, the misclosure is also expressed in the local east, north, up frame.

    Returns:
        DataFrame, largest misclosure first, with one row per loop: `loop` (points), `baselines` ("session:base>rover" of the three sides), the misclosure `dx`, `dy`, `dz`[m], its norm `closure` [m], the `perimeter` [m], `ppm` (closure / perimeter) and, with `origin`, `dE`, `dN`, `dU` and `dH` (horizontal norm) [m].
    """
    sides_of: dict[frozenset[str], list[BaselineResult]] = defaultdict(list)
    for bl in baselines:
        sides_of[frozenset((bl.base, bl.rover))].append(bl)
    points = sorted({p for pair in sides_of for p in pair})

    rot = None
    if origin is not None:
        lat, lon, _ = ecef_to_llh(origin[0], origin[1], origin[2])
        rot = ecef_to_enu_matrix(float(lat), float(lon))

    rows = []
    for a, b, c in combinations(points, 3):
        sides = [(a, b), (b, c), (c, a)]
        if not all(frozenset(s) in sides_of for s in sides):
            continue
        for combo in product(*(sides_of[frozenset(s)] for s in sides)):
            d = np.sum([_vector(bl, *side) for bl, side in zip(combo, sides)], axis=0)
            perimeter = sum(bl.length for bl in combo)
            row = {
                "loop": f"{a}-{b}-{c}",
                "baselines": " ".join(f"{bl.session}:{bl.base}>{bl.rover}" for bl in combo),
                "dx": d[0], "dy": d[1], "dz": d[2],
                "closure": float(np.linalg.norm(d)),
                "perimeter": perimeter,
                "ppm": 1e6 * float(np.linalg.norm(d)) / perimeter,
            }  # fmt: skip
            if rot is not None:
                e, n, u = rot @ d
                row |= {"dE": e, "dN": n, "dU": u, "dH": float(np.hypot(e, n))}
            rows.append(row)
    cols = COLUMNS + (["dE", "dN", "dU", "dH"] if rot is not None else [])
    return pd.DataFrame(rows, columns=cols).sort_values(
        "closure", ascending=False, ignore_index=True
    )
