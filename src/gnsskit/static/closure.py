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

from ..refsys import ENUFrame
from .baseline import BaselineResult

COLUMNS = ["loop", "baselines", "dx", "dy", "dz", "closure", "perimeter", "ppm"]


def _vector(bl: BaselineResult, start: str, end: str) -> np.ndarray:
    """Vector start -> end of a baseline, reversed if it was measured the other way round."""
    return bl.dxyz if (bl.base, bl.rover) == (start, end) else -bl.dxyz


def loop_closures(
    baselines: Sequence[BaselineResult], origin: Sequence[float] | np.ndarray | None = None
) -> pd.DataFrame:
    """Misclosure of every triangle that can be formed with the baselines.

    A triangle is any three points that are pairwise joined by at least one baseline. If a
    pair has several baselines (repeated sessions), every combination is reported. Each side
    is taken in the direction a -> b -> c -> a, reversing the baselines measured the other
    way round, so the misclosure is the sum of the three vectors.

    Args:
        baselines: Baselines of the network.
        origin: ECEF coordinates [m] of a point of the network (e.g. the fixed one). If
            given, the misclosure is also expressed in the east, north, up frame there.

    Returns:
        DataFrame, largest misclosure first, one row per loop: `loop` (points), `baselines`
        ("session:base>rover" of the three sides), the misclosure `dx`, `dy`, `dz` [m], its
        norm `closure` [m], the `perimeter` [m], `ppm` (closure / perimeter) and, with
        `origin`, `dE`, `dN`, `dU` and `dH` (horizontal norm) [m]. Empty if no triangle.
    """
    sides_of: dict[frozenset[str], list[BaselineResult]] = defaultdict(list)
    for bl in baselines:
        sides_of[frozenset((bl.base, bl.rover))].append(bl)
    points = sorted({p for pair in sides_of for p in pair})

    frame = ENUFrame.from_ecef(origin) if origin is not None else None

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
            if frame is not None:
                e, n, u = frame.vector_to_enu(d)
                row |= {"dE": e, "dN": n, "dU": u, "dH": float(np.hypot(e, n))}
            rows.append(row)
    cols = COLUMNS + (["dE", "dN", "dU", "dH"] if frame is not None else [])
    return pd.DataFrame(rows, columns=cols).sort_values(
        "closure", ascending=False, ignore_index=True
    )
