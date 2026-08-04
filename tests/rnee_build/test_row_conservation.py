import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rnee_rows", ROOT / "src" / "rnee_build" / "13_assemble_rows.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_row_id_is_deterministic_and_source_specific():
    first = MODULE.row_id("a.csv", 1, 2, 3)
    assert first == MODULE.row_id("a.csv", 1, 2, 3)
    assert first != MODULE.row_id("b.csv", 1, 2, 3)
    assert first != MODULE.row_id("a.csv", 1, 2, 4)


def test_value_hash_preserves_missing_and_value_changes():
    frame = pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]})
    hashes = MODULE.value_hash(frame, ["a", "b"])
    changed = frame.copy(); changed.loc[0, "a"] = 2.0
    changed_hashes = MODULE.value_hash(changed, ["a", "b"])
    assert hashes.iloc[0] != changed_hashes.iloc[0]
    assert hashes.iloc[1] == changed_hashes.iloc[1]


def test_quality_check_abstention_nulls_semantics_without_deleting_rows():
    frame = pd.DataFrame(
        {
            "source_file": ["a.csv", "a.csv", "a.csv"],
            "VehId": [1, 1, 2],
            "Trip": [10, 10, 20],
            "row_id": ["a", "b", "c"],
            "edge_semantic_join_found": [True, True, True],
            "road_class": ["local", "local", "primary"],
            "nearest_graph_node_degree": [2.0, 3.0, 4.0],
        }
    )
    policy = pd.DataFrame(
        {
            "source_file": ["a.csv"],
            "VehId": [1],
            "Trip": [10],
            "road_semantics_available": [False],
            "road_semantics_exclusion_reason": ["off_network_open_lot_quality_1_abstention"],
        }
    )
    result = MODULE.apply_road_semantics_availability(
        frame, ["road_class", "nearest_graph_node_degree"], policy
    )
    assert len(result) == len(frame)
    assert result["row_id"].tolist() == frame["row_id"].tolist()
    assert result.loc[:1, "road_semantics_available"].eq(False).all()
    assert result.loc[:1, ["road_class", "nearest_graph_node_degree"]].isna().all().all()
    assert result.loc[2, "road_semantics_available"]
    assert result.loc[2, "road_class"] == "primary"


def test_unmatched_edge_has_no_road_semantics_even_without_manual_policy():
    frame = pd.DataFrame(
        {
            "source_file": ["a.csv"],
            "VehId": [1],
            "Trip": [10],
            "edge_semantic_join_found": [False],
            "road_class": ["local"],
        }
    )
    result = MODULE.apply_road_semantics_availability(frame, ["road_class"], None)
    assert not result.loc[0, "road_semantics_available"]
    assert pd.isna(result.loc[0, "road_class"])
    assert result.loc[0, "road_semantics_exclusion_reason"] == "map_match_or_edge_semantics_unavailable"
