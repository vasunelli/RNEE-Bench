import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "rnee_build" / "04_build_network.py"
SPEC = importlib.util.spec_from_file_location("rnee_network", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_point_in_bbox_boundaries_are_inclusive():
    bbox = (-84.0, 42.0, -83.0, 43.0)
    assert MODULE.point_in_bbox(-84.0, 42.0, bbox)
    assert MODULE.point_in_bbox(-83.0, 43.0, bbox)
    assert not MODULE.point_in_bbox(-84.1, 42.5, bbox)


def test_segment_intersection_detects_crossing_with_both_endpoints_outside():
    bbox = (-84.0, 42.0, -83.0, 43.0)
    assert MODULE.segment_intersects_bbox((-85.0, 42.5), (-82.0, 42.5), bbox)
    assert not MODULE.segment_intersects_bbox((-85.0, 41.0), (-82.0, 41.0), bbox)


def test_network_config_requires_separate_historical_snapshots():
    config = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "network_build.yaml"
    )
    base = MODULE.load_yaml(Path(config["base_contract"]))
    assert base["osm"]["temporal_policy"] == "historical_snapshots_only"
    assert len(base["osm"]["snapshots"]) == 2
    assert base["osm"]["snapshots"][0]["sha256"] != base["osm"]["snapshots"][1][
        "sha256"
    ]


def test_network_qa_requires_queryable_graph():
    config = MODULE.load_yaml(
        ROOT / "configs" / "rnee_build" / "network_build.yaml"
    )
    assert config["qa"]["minimum_graph_tile_count"] > 0
    assert config["qa"]["minimum_locate_success_rate"] == 1.0
    assert config["qa"]["minimum_locate_edge_reachability"] >= 50
    assert config["valhalla"]["concurrency"] == 1
    assert config["valhalla"]["traffic_extract"] == ""
