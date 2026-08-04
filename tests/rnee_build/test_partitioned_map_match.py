import importlib.util
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rnee_partitioned_map", ROOT / "src" / "rnee_build" / "20_run_map_match_partitioned.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SPEC_06 = importlib.util.spec_from_file_location(
    "rnee_map_match", ROOT / "src" / "rnee_build" / "06_map_match_trips.py"
)
MODULE_06 = importlib.util.module_from_spec(SPEC_06)
assert SPEC_06.loader is not None
SPEC_06.loader.exec_module(MODULE_06)


def test_partition_plan_is_stable_and_one_file_per_partition():
    plan = MODULE.partition_plan(["b.csv", "a.csv"])
    assert plan == [
        {"partition_id": "partition_0000", "source_file": "b.csv"},
        {"partition_id": "partition_0001", "source_file": "a.csv"},
    ]


def test_partition_plan_rejects_duplicate_sources():
    try:
        MODULE.partition_plan(["a.csv", "a.csv"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate source files must fail")


def test_midpoint_fallback_split_conserves_order_and_points():
    positions = np.arange(7)
    left, right = MODULE_06.midpoint_fallback_split(positions, 3)
    assert left.tolist() == [0, 1, 2]
    assert right.tolist() == [3, 4, 5, 6]
    assert MODULE_06.midpoint_fallback_split(np.arange(5), 3) is None


def test_request_failure_bound_is_inclusive_and_rejects_excess():
    assert MODULE.request_failures_within_bound(1, 200, 2, 0.005)
    assert not MODULE.request_failures_within_bound(2, 200, 2, 0.005)
    assert not MODULE.request_failures_within_bound(3, 1000, 2, 0.005)
