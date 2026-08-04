import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "rnee_build" / "09_analyze_quality_check_remediation.py"
SPEC = importlib.util.spec_from_file_location("rnee_quality_1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raw_gps_triggered_incorrect_is_split_not_deleted():
    frame = pd.DataFrame(
        {
            "point_index": [0, 1],
            "is_coordinate_update": [True, True],
            "timestamp_ms_raw": [0, 1000],
            "latitude_raw": [42.0, 42.01],
            "longitude_raw": [-83.0, -83.0],
        }
    )
    config = {
        "raw_gps_break_rules": {
            "maximum_coordinate_update_speed_kmh": 200,
            "maximum_coordinate_update_jump_m": 500,
        }
    }
    result = MODULE.classify_trip(frame, "incorrect", config)
    assert result["classification"] == "raw_gps_triggered_incorrect"
    assert result["recommended_action"] == "split_at_flagged_update_transition"
    assert result["delete_original_trip"] is False


def test_incorrect_without_raw_trigger_remains_match_error_candidate():
    frame = pd.DataFrame(
        {
            "point_index": [0, 1],
            "is_coordinate_update": [True, True],
            "timestamp_ms_raw": [0, 5000],
            "latitude_raw": [42.0, 42.0001],
            "longitude_raw": [-83.0, -83.0],
        }
    )
    config = {
        "raw_gps_break_rules": {
            "maximum_coordinate_update_speed_kmh": 200,
            "maximum_coordinate_update_jump_m": 500,
        }
    }
    result = MODULE.classify_trip(frame, "incorrect", config)
    assert result["classification"] == "map_match_error_candidate"
    assert result["delete_original_trip"] is False
