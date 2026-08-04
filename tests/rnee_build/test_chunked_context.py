import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rnee_chunked_context", ROOT / "src" / "rnee_build" / "16_build_context_semantics_chunked.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_trip_chunk_plan_preserves_complete_trips_and_rows():
    frame = pd.DataFrame(
        {
            "source_file": ["a"] * 7 + ["b"] * 4,
            "pilot_trip_id": ["t1"] * 3 + ["t2"] * 4 + ["t3"] * 4,
            "osm_snapshot_id": ["s1"] * 7 + ["s2"] * 4,
        }
    )
    chunks = MODULE.plan_trip_chunks(frame, max_rows=6)
    assert sum(chunk["row_count"] for chunk in chunks) == len(frame)
    assert sum(chunk["trip_count"] for chunk in chunks) == 3
    assigned = [trip["pilot_trip_id"] for chunk in chunks for trip in chunk["trips"]]
    assert sorted(assigned) == ["t1", "t2", "t3"]
    assert len(assigned) == len(set(assigned))
    assert all(chunk["row_count"] <= 6 for chunk in chunks)


def test_chunk_points_matches_declared_partition():
    frame = pd.DataFrame(
        {
            "pilot_trip_id": ["t1", "t1", "t2"],
            "value": [1, 2, 3],
        }
    )
    chunk = {"chunk_id": "chunk_0000", "row_count": 2, "trip_count": 1, "trips": [{"pilot_trip_id": "t1"}]}
    result = MODULE.chunk_points(frame, chunk)
    assert result["value"].tolist() == [1, 2]
