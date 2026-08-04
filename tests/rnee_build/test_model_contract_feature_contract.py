from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "45_freeze_model_contract.py"
SPEC = importlib.util.spec_from_file_location("model_contract_contract", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_interval_contributions_conserve_valid_duration_and_split_boundary() -> None:
    rows = pd.DataFrame(
        {
            "pilot_trip_id": ["t", "t", "t"],
            "Timestamp(ms)": [0, 59_000, 62_000],
            "dt_seconds": [59.0, 3.0, np.nan],
            "dt_valid": [True, True, False],
        }
    )
    pieces = MODULE.interval_contributions(rows, 60_000)
    assert np.isclose(pieces["weight_s"].sum(), 62.0)
    assert set(pieces["_segment_index"]) == {0, 1}
    assert np.isclose(pieces.loc[pieces["_segment_index"].eq(1), "weight_s"].sum(), 2.0)


def test_fixed_category_mapping_preserves_missing_and_unknown() -> None:
    mapping = {"asphalt": "paved", "dirt": "unpaved"}
    assert MODULE.normalize_category("asphalt", mapping, "missing") == "paved"
    assert MODULE.normalize_category("WOOD", mapping, "missing") == "other"
    assert MODULE.normalize_category(pd.NA, mapping, "missing") == "missing"


def test_forbidden_feature_detector_catches_ids_and_coordinates() -> None:
    config = {
        "forbidden_features": {
            "exact": ["segment_id"],
            "tokens": ["latitude", "road_id"],
        }
    }
    hits = MODULE.forbidden_hits(["speed_mean", "segment_id", "matched_latitude"], config)
    assert {item["feature"] for item in hits} == {"segment_id", "matched_latitude"}


def test_support_classification_enforces_all_four_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "split_role": ["train", "test", "test"],
            "vehicle_count": [1, 20, 20],
            "trip_count": [1, 200, 200],
            "segment_count": [1, 2000, 2000],
            "max_vehicle_segment_share": [1.0, 0.15, 0.151],
        }
    )
    thresholds = {
        "minimum_vehicles": 20,
        "minimum_trips": 200,
        "minimum_segments": 2000,
        "maximum_vehicle_segment_share": 0.15,
    }
    assert MODULE.classify_test_support(frame, thresholds).tolist() == [
        "NOT_APPLICABLE",
        "PRIMARY_SUPPORTED",
        "DOWNGRADED_UNSUPPORTED",
    ]


def test_freeze_target_membership_matches_exact_common_support(tmp_path: Path) -> None:
    split_root = tmp_path / "splits"
    family = split_root / "random_trip_blocked"
    family.mkdir(parents=True)
    pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "split_role": ["train", "validation", "calibration", "test"],
        }
    ).to_parquet(family / "model_membership.parquet", index=False)
    segments = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "VehId": [1, 2, 3, 4],
            "trip_uid": ["t1", "t2", "t3", "t4"],
            "engine_type": ["ICE", "ICE", "ICE", "HEV"],
            "prediction_usable": [True, True, True, True],
            "road_semantics_model_eligible": [True, False, True, True],
            "fuel_target_coverage": [1.0, 1.0, 0.90, 1.0],
            "battery_target_coverage": [0.0, 0.0, 0.0, 0.0],
        }
    )
    config = {
        "split_families": ["random_trip_blocked"],
        "targets": {
            "ICE_fuel_L": {
                "engine_type": "ICE",
                "channel": "fuel",
                "unit": "L",
                "coverage_column": "fuel_target_coverage",
            }
        },
        "target_minimum_coverage": 0.95,
        "support_thresholds": {
            "minimum_vehicles": 1,
            "minimum_trips": 1,
            "minimum_segments": 1,
            "maximum_vehicle_segment_share": 1.0,
        },
    }
    support, artifacts = MODULE.freeze_target_memberships(
        segments, split_root, tmp_path / "output", config
    )
    frozen = pd.read_parquet(artifacts[0]["path"])
    assert frozen.to_dict(orient="records") == [{"segment_id": "a", "split_role": "train"}]
    assert len(support) == 4
