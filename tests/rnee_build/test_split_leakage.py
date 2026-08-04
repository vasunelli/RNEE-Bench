from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[2]
BUILD_SPEC = importlib.util.spec_from_file_location("split_builder", ROOT / "src" / "rnee_build" / "36_build_splits.py")
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(BUILD)
VALIDATE_SPEC = importlib.util.spec_from_file_location("split_validator", ROOT / "src" / "rnee_build" / "37_validate_leakage.py")
VALIDATE = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(VALIDATE)


def miniature_frame() -> pd.DataFrame:
    rows = []
    for vehicle in range(1, 5):
        for trip in range(1, 11):
            trip_uid = f"v{vehicle}-t{trip}"
            for segment in range(2):
                rows.append({"segment_id": f"{trip_uid}-s{segment}", "trip_uid": trip_uid, "VehId": vehicle})
    return pd.DataFrame(rows)


def test_rank_assignment_never_splits_group() -> None:
    frame = miniature_frame()
    eligible = pd.Series(True, index=frame.index)
    roles = BUILD.assign_groups_by_rank(
        frame,
        eligible,
        "trip_uid",
        {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        42,
    )
    result = frame.assign(role=roles).groupby("trip_uid")["role"].nunique()
    assert int(result.max()) == 1


def test_within_vehicle_cold_trip_preserves_trip_blocks() -> None:
    frame = miniature_frame()
    role, _ = BUILD.build_cold_trip(
        frame,
        {"seed": 7, "minimum_trips_per_vehicle": 4, "fractions": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}},
    )
    result = frame.assign(role=role).groupby("trip_uid")["role"].nunique()
    assert int(result.max()) == 1
    assert set(BUILD.ACTIVE_ROLES).issubset(set(role))


def test_row_balanced_vehicle_assignment_preserves_vehicle_blocks() -> None:
    frame = miniature_frame()
    eligible = pd.Series(True, index=frame.index)
    role = BUILD.assign_groups_balanced_by_rows(
        frame,
        eligible,
        "VehId",
        {"train": 0.65, "validation": 0.1, "calibration": 0.1, "test": 0.15},
        9,
    )
    assert int(frame.assign(role=role).groupby("VehId")["role"].nunique().max()) == 1


def test_connected_spatial_selection_and_buffer_geometry() -> None:
    frame = pd.DataFrame({
        "spatial_block_id": ["0_0"] * 5 + ["1_0"] * 4 + ["2_0"] * 3 + ["9_9"],
    })
    selected = BUILD.select_connected_test_blocks(frame, {"test_anchor_block": "0_0", "test_target_fraction": 0.6, "seed": 1})
    assert selected == {"0_0", "1_0"}
    assert VALIDATE.block_within_buffer("2_0", selected, 1)
    assert not VALIDATE.block_within_buffer("9_9", selected, 1)


def test_functional_class_holdout_excludes_same_trip_other_class() -> None:
    frame = pd.DataFrame({
        "trip_uid": ["a", "a", "b", "c", "d", "e"],
        "functional_road_class_dominant": ["motorway", "local", "local", "local", "local", "local"],
    })
    role, _ = BUILD.build_cold_functional_road_class(
        frame,
        {"heldout_class": "motorway", "seed": 3, "remaining_fractions": {"train": 0.8, "validation": 0.1, "calibration": 0.1, "test": 0.0}},
    )
    assert role.iloc[0] == "test"
    assert role.iloc[1] == "excluded_same_test_trip_road_class"
    active_non_test = frame.loc[role.isin(["train", "validation", "calibration"]), "functional_road_class_dominant"]
    assert "motorway" not in set(active_non_test)


def test_block_parser_rejects_missing_and_bad_values() -> None:
    assert BUILD.parse_block("273_4684") == (273, 4684)
    assert BUILD.parse_block(None) is None
    assert BUILD.parse_block("bad") is None
