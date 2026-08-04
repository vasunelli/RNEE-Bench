import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "rnee_build" / "06_map_match_trips.py"
SPEC = importlib.util.spec_from_file_location("rnee_map_match", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_profiles_are_frozen_and_quality_only():
    profiles = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "map_match_profiles.yaml"
    )
    assert profiles["profiles"] == {
        "strict": {"gps_accuracy_m": 5, "search_radius_m": 10},
        "primary": {"gps_accuracy_m": 10, "search_radius_m": 30},
        "tolerant": {"gps_accuracy_m": 20, "search_radius_m": 50},
    }
    prohibited = profiles["selection_policy"]["prohibited_selection_inputs"]
    assert "energy_target" in prohibited
    assert "eVED_match_fields" in prohibited


def test_map_match_request_preserves_point_cardinality_and_time():
    config = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "valhalla.yaml"
    )
    trip = pd.DataFrame(
        {
            "Timestamp(ms)": [0, 1000],
            "Latitude[deg]": [42.1, 42.2],
            "Longitude[deg]": [-83.1, -83.2],
        }
    )
    request = MODULE.build_request(
        trip,
        config,
        {"gps_accuracy_m": 10, "search_radius_m": 30},
    )
    assert len(request["shape"]) == len(trip)
    assert request["shape"][1]["time"] == 1.0
    assert request["shape_match"] == "map_snap"
    assert request["trace_options"]["breakage_distance"] == 500


def test_map_match_request_allows_profile_breakage_override():
    config = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "valhalla.yaml"
    )
    trip = pd.DataFrame(
        {
            "Timestamp(ms)": [0, 1000, 2000],
            "Latitude[deg]": [42.0, 42.0001, 42.0002],
            "Longitude[deg]": [-83.0, -83.0001, -83.0002],
        }
    )
    request = MODULE.build_request(
        trip,
        config,
        {
            "gps_accuracy_m": 20,
            "search_radius_m": 75,
            "breakage_distance_m": 1000,
        },
    )
    assert request["trace_options"]["breakage_distance"] == 1000


def test_end_to_end_two_file_contract_is_all_trips_and_does_not_overwrite_map_matching_pointer():
    config = MODULE.load_yaml(ROOT / "configs" / "rnee_build" / "end_to_end_valhalla.yaml")
    assert config["all_trips_in_source_files"] is True
    assert len(config["source_files"]) == 2
    assert config["latest_raw_pointer_name"] == "end_to_end_latest_raw_pointer.json"


def test_silent_fallback_detection_is_measured():
    valid = pd.DataFrame(
        {
            "match_status": ["matched_with_edge", "unmatched", "request_error"],
            "matched_latitude": [42.1, 42.2, None],
            "matched_longitude": [-83.1, -83.2, None],
            "edge_index_valid": [True, False, False],
        }
    )
    assert MODULE.count_silent_fallbacks(valid) == 0
    invalid = valid.iloc[[0]].copy()
    invalid["edge_index_valid"] = False
    assert MODULE.count_silent_fallbacks(invalid) == 1


def test_segmented_trace_breaks_only_at_frozen_raw_gps_rules():
    config = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "valhalla.yaml"
    )
    remediation = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "map_matching_remediation.yaml"
    )
    trace = pd.DataFrame(
        {
            "Timestamp(ms)": [0, 5000, 6000, 11000],
            "Latitude[deg]": [42.0, 42.0001, 42.02, 42.0201],
            "Longitude[deg]": [-83.0, -83.0, -83.0, -83.0],
        }
    )
    chunks, break_before = MODULE.trace_chunk_indices(
        trace, config, remediation
    )
    assert break_before.tolist() == [True, False, True, False]
    assert [chunk.tolist() for chunk in chunks] == [[0, 1], [2, 3]]


def test_month_allocation_conserves_requested_count():
    counts = pd.Series({"2017-11": 10, "2017-12": 20, "2018-01": 70})
    targets = MODULE.allocate_month_targets(counts, 20, minimum_per_month=3)
    assert sum(targets.values()) == 20
    assert min(targets.values()) >= 3


def test_active_snapshot_assignment_is_never_future():
    gps = pd.read_parquet(
        ROOT / "results" / "rnee_build" / "network_freeze_latest_gps_trip_quality.parquet"
    )
    snapshot_dates = {
        "geofabrik_michigan_2017-01-01": date(2017, 1, 1),
        "geofabrik_michigan_2018-01-01": date(2018, 1, 1),
    }
    assert not any(
        snapshot_dates[snapshot_id] > date.fromisoformat(trip_date)
        for snapshot_id, trip_date in zip(
            gps["osm_snapshot_id"], gps["trip_mid_date"]
        )
    )
