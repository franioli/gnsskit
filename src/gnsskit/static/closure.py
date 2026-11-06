"""Loop closures of a baseline network: a quick consistency check before the adjustment.

Three points A, B, C joined by the baselines A->B, B->C and C->A close on themselves:
the sum of the three vectors must be zero. The misclosure shows blunders (wrong fixes,
wrong antenna heights, wrong point names) without any stochastic model.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .baseline import BaselineResult


def _vector(bl: BaselineResult, start: str, end: str) -> np.ndarray:
    """Vector start -> end of a baseline, reversed if it was measured the other way round."""
    return bl.dxyz if (bl.base, bl.rover) == (start, end) else -bl.dxyz


def loop_closures(
    baselines: Sequence[BaselineResult], axes: Sequence[str] = ("x", "y", "z")
) -> pd.DataFrame:
    """Misclosure of every triangle that can be formed with the baselines.

    A triangle is any three points that are pairwise joined by a baseline. If a pair has
    several baselines (repeated sessions), every combination is reported. The sides are taken
    in the direction a -> b -> c -> a, reversing the baselines measured the other way round,
    so the misclosure is the sum of the three vectors. It is computed in whatever Cartesian
    frame the baselines are given in (ECEF, east-north-up, ...): rotate them beforehand.

    Args:
        baselines: Baselines of the network, all in the same frame (`dxyz` is the vector in
            that frame).
        axes: Names of the three axes of that frame, used in the column names.

    Returns:
        DataFrame, largest misclosure first, one row per loop: `loop` (points), `baselines`
        ("session:base>rover" of the three sides), the misclosure along the three axes in
        columns `d<axis>` (`dx`, `dy`, `dz` by default) [m] and its 3D norm `closure` [m].
        Empty if there is no triangle.
    """
    rows = []
    for trio in combinations(baselines, 3):
        pairs = [frozenset((bl.base, bl.rover)) for bl in trio]
        points = sorted(frozenset.union(*pairs))
        if len(points) != 3 or len(set(pairs)) != 3:  # not a triangle
            continue
        a, b, c = points
        by_pair = dict(zip(pairs, trio))
        sides = [(a, b), (b, c), (c, a)]
        loop = [by_pair[frozenset(side)] for side in sides]
        d = np.sum([_vector(bl, *side) for bl, side in zip(loop, sides)], axis=0)
        rows.append({
            "loop": f"{a}-{b}-{c}",
            "baselines": " ".join(f"{bl.session}:{bl.base}>{bl.rover}" for bl in loop),
            **{f"d{axis}": v for axis, v in zip(axes, d)},
            "closure": float(np.linalg.norm(d)),
        })  # fmt: skip
    cols = ["loop", "baselines", *(f"d{axis}" for axis in axes), "closure"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        "closure", ascending=False, ignore_index=True
    )
