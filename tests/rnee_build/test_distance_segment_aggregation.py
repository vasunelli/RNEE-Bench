from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SPEC = importlib.util.spec_from_file_location("distance_segments", Path(__file__).parents[2] / "src" / "rnee_build" / "38_build_distance_segments.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_interval_is_split_exactly_at_500m_boundary() -> None:
    rows = pd.DataFrame({
        "pilot_trip_id": ["t", "t"],
        "Timestamp(ms)": [0, 40_000],
        "dt_seconds": [40.0, 0.0],
        "dt_valid": [True, False],
        "Vehicle Speed[km/h]": [54.0, 0.0],
    })
    result = MODULE.distance_contributions(rows, 500.0)
    assert result["distance_m"].round(9).tolist() == [500.0, 100.0]
    assert result["_segment_index"].tolist() == [0, 1]
    assert round(float(result["weight_s"].sum()), 9) == 40.0
    assert round(float(result["fraction"].sum()), 9) == 1.0


def test_stopped_time_is_conserved_in_current_distance_segment() -> None:
    rows = pd.DataFrame({
        "pilot_trip_id": ["t", "t"],
        "Timestamp(ms)": [0, 10_000],
        "dt_seconds": [10.0, 0.0],
        "dt_valid": [True, False],
        "Vehicle Speed[km/h]": [0.0, 0.0],
    })
    result = MODULE.distance_contributions(rows, 500.0)
    assert result["distance_m"].tolist() == [0.0]
    assert result["weight_s"].tolist() == [10.0]
    assert result["fraction"].tolist() == [1.0]
    assert result["_segment_index"].tolist() == [0]


def test_distance_segment_id_is_granularity_specific() -> None:
    first = MODULE.SEG.stable_uid("trip", "0.000000", "500m")
    assert first == MODULE.SEG.stable_uid("trip", "0.000000", "500m")
    assert first != MODULE.SEG.stable_uid("trip", "0.000000", "30s")
