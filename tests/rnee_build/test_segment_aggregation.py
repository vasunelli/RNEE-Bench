from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SPEC = importlib.util.spec_from_file_location("segments", Path(__file__).parents[2] / "src" / "rnee_build" / "35_build_segments.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_interval_is_split_exactly_at_60s_boundary() -> None:
    rows = pd.DataFrame({"pilot_trip_id": ["t", "t", "t"], "Timestamp(ms)": [0, 55_000, 65_000], "dt_seconds": [0.0, 10.0, 0.0], "dt_valid": [False, True, False]})
    result = MODULE.interval_contributions(rows, 60_000)
    assert result["weight_s"].tolist() == [5.0, 5.0]
    assert result["_segment_index"].tolist() == [0, 1]
    assert result["fraction"].tolist() == [0.5, 0.5]


def test_segment_id_is_stable_and_granularity_specific() -> None:
    first = MODULE.stable_uid("trip", 0, "60s")
    assert first == MODULE.stable_uid("trip", 0, "60s")
    assert first != MODULE.stable_uid("trip", 0, "30s")


def test_transition_count_does_not_create_route_identifier() -> None:
    rows = pd.DataFrame({"_segment_key": ["a"] * 5, "functional_road_class": ["local", "local", "primary", None, "primary"]})
    result = MODULE.transition_counts(rows)
    assert int(result.loc["a"]) == 1
