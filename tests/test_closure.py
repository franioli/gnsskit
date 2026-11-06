from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gnsskit.refsys import ENUFrame, llh_to_ecef
from gnsskit.static import BaselineResult, loop_closures

T = pd.Timestamp("2026-11-04 10:00")
ORIGIN = llh_to_ecef(45.4, 9.2, 150.0)  # a point near Milan
XYZ = {"A": np.zeros(3), "B": np.array([100.0, 50.0, -20.0]), "C": np.array([-30.0, 200.0, 60.0])}


def bl(base: str, rover: str, session: str = "s1", error=(0.0, 0.0, 0.0)) -> BaselineResult:
    return BaselineResult(
        rover, base, session, T, XYZ[rover] - XYZ[base] + error, np.eye(3) * 1e-6, 1, 10, 5.0
    )


def test_consistent_triangle_closes_whatever_the_direction() -> None:
    # A->B, C->B (reversed), C->A (the three sides are measured in different directions)
    df = loop_closures([bl("A", "B"), bl("C", "B"), bl("C", "A")])
    assert len(df) == 1 and df.loop[0] == "A-B-C"
    assert df.closure[0] == pytest.approx(0.0, abs=1e-9)


def test_error_in_one_baseline_is_the_misclosure_in_any_frame() -> None:
    err = np.array([0.03, -0.04, 0.01])
    baselines = [bl("A", "B", error=err), bl("B", "C"), bl("A", "C")]
    df = loop_closures(baselines)
    assert np.allclose(df.loc[0, ["dx", "dy", "dz"]].to_numpy(float), err)
    assert df.closure[0] == pytest.approx(np.linalg.norm(err))
    # the same baselines rotated to east, north, up: the misclosure is rotated, not changed
    frame = ENUFrame.from_ecef(ORIGIN)
    enu = [replace(b, dxyz=frame.vector_to_enu(b.dxyz)) for b in baselines]
    df_enu = loop_closures(enu, axes=("E", "N", "U"))
    assert np.allclose(df_enu.loc[0, ["dE", "dN", "dU"]].to_numpy(float), frame.vector_to_enu(err))
    assert df_enu.closure[0] == pytest.approx(df.closure[0])


def test_repeated_baseline_gives_one_loop_each_and_open_networks_none() -> None:
    df = loop_closures(
        [bl("A", "B"), bl("B", "C"), bl("A", "C", "s1"), bl("A", "C", "s2", error=(0, 0, 0.1))]
    )
    assert sorted(df.closure.round(3)) == [0.0, 0.1] and set(df.loop) == {"A-B-C"}
    assert loop_closures([bl("A", "B"), bl("B", "C")]).empty
