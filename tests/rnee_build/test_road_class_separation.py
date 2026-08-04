import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rnee_osm_parser", ROOT / "src" / "rnee_build" / "11_parse_osm_tags.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_functional_class_mapping_is_separate_from_speed_provenance():
    config = yaml.safe_load((ROOT / "configs" / "rnee_build" / "edge_semantics.yaml").read_text(encoding="utf-8"))
    mapping = config["functional_road_class"]
    assert MODULE.functional_class("motorway_link", mapping) == "motorway"
    assert MODULE.functional_class("residential", mapping) == "local"
    assert MODULE.functional_class(None, mapping) is None
    class_names = set(mapping)
    provenance = set(config["speed"]["provenance_values"])
    assert class_names.isdisjoint(provenance)


def test_lane_parser_does_not_silently_collapse_ambiguous_values():
    assert MODULE.parse_lane_count("2")["lane_count"] == 2
    assert MODULE.parse_lane_count("2;3")["lane_count"] is None
    assert MODULE.parse_lane_count("2;3")["status"] == "composite_ambiguous"
