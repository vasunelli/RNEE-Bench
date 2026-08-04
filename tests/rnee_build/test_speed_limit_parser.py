import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rnee_osm_parser", ROOT / "src" / "rnee_build" / "11_parse_osm_tags.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_speed_parser_units_and_special_values():
    assert round(MODULE.parse_speed("35 mph")["speed_kmh"], 3) == 56.327
    assert MODULE.parse_speed("80 km/h")["speed_kmh"] == 80
    assert MODULE.parse_speed("50")["speed_kmh"] == 50
    assert MODULE.parse_speed("signals")["speed_kmh"] is None
    assert MODULE.parse_speed("none")["status"] == "special_none"


def test_speed_parser_preserves_composite_ambiguity():
    result = MODULE.parse_speed("30 mph;50 mph")
    assert result["speed_kmh"] is None
    assert result["status"] == "composite_ambiguous"
    assert len(result["values_kmh"]) == 2


def test_directional_speed_precedes_general_and_valhalla_default():
    config = {"mph_to_kmh": 1.609344, "valhalla_fallback_provenance": "default", "valhalla_fallback_is_inferred": True}
    result = MODULE.choose_speed({"maxspeed": "50", "maxspeed:forward": "35 mph"}, True, 90, config)
    assert result["speed_limit_provenance"] == "directional"
    assert round(result["speed_limit_kmh_used"], 3) == 56.327


def test_unresolved_osm_speed_uses_explicitly_inferred_valhalla_fallback():
    config = {"mph_to_kmh": 1.609344, "valhalla_fallback_provenance": "default", "valhalla_fallback_is_inferred": True}
    result = MODULE.choose_speed({"maxspeed": "signals"}, True, 72, config)
    assert result["speed_limit_kmh_used"] == 72
    assert result["speed_limit_provenance"] == "default"
    assert result["speed_limit_is_inferred"] is True
