#!/usr/bin/env python3
"""Pure parsers for auditable OSM road semantics."""

from __future__ import annotations

import json
import re
from typing import Any


NUMBER = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/ ]*)$")
SPECIAL_SPEEDS = {"signals", "walk", "none", "variable", "national", "unposted"}


def parse_speed(raw: Any, mph_to_kmh: float = 1.609344) -> dict[str, Any]:
    """Parse one OSM maxspeed value without silently resolving ambiguity."""
    if raw is None or str(raw).strip() == "":
        return {"raw": None, "speed_kmh": None, "status": "missing", "values_kmh": []}
    text = str(raw).strip()
    lower = text.lower()
    if lower in SPECIAL_SPEEDS:
        return {"raw": text, "speed_kmh": None, "status": f"special_{lower}", "values_kmh": []}
    if "@" in text:
        return {"raw": text, "speed_kmh": None, "status": "conditional_unresolved", "values_kmh": []}
    tokens = [token.strip() for token in text.split(";") if token.strip()]
    values: list[float] = []
    for token in tokens:
        match = NUMBER.match(token)
        if not match:
            return {"raw": text, "speed_kmh": None, "status": "unparsed", "values_kmh": values}
        value = float(match.group(1))
        unit = match.group(2).strip().lower().replace(" ", "")
        if unit in {"mph", "mi/h", "miles/hour", "mile/hour"}:
            value *= mph_to_kmh
        elif unit not in {"", "km/h", "kmh", "kph"}:
            return {"raw": text, "speed_kmh": None, "status": "unknown_unit", "values_kmh": values}
        values.append(value)
    rounded = [round(value, 6) for value in values]
    if not values:
        return {"raw": text, "speed_kmh": None, "status": "unparsed", "values_kmh": []}
    if len(set(rounded)) > 1:
        return {"raw": text, "speed_kmh": None, "status": "composite_ambiguous", "values_kmh": rounded}
    status = "parsed_composite_equal" if len(values) > 1 else "parsed"
    return {"raw": text, "speed_kmh": rounded[0], "status": status, "values_kmh": rounded}


def parse_lane_count(raw: Any) -> dict[str, Any]:
    if raw is None or str(raw).strip() == "":
        return {"raw": None, "lane_count": None, "status": "missing"}
    text = str(raw).strip()
    tokens = [token.strip() for token in text.split(";") if token.strip()]
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return {"raw": text, "lane_count": None, "status": "unparsed"}
    if len(set(values)) != 1:
        return {"raw": text, "lane_count": None, "status": "composite_ambiguous"}
    value = values[0]
    if not value.is_integer():
        return {"raw": text, "lane_count": value, "status": "parsed_noninteger"}
    return {"raw": text, "lane_count": int(value), "status": "parsed"}


def truthy_osm(raw: Any) -> bool | None:
    if raw is None or str(raw).strip() == "":
        return None
    value = str(raw).strip().lower()
    if value in {"yes", "true", "1"}:
        return True
    if value in {"no", "false", "0"}:
        return False
    return None


def functional_class(highway: Any, mapping: dict[str, list[str]]) -> str | None:
    if highway is None or str(highway).strip() == "":
        return None
    value = str(highway).strip().lower()
    for group, members in mapping.items():
        if value in members:
            return group
    return "other"


def choose_speed(tags: dict[str, Any], forward: bool, valhalla_speed: Any, config: dict[str, Any]) -> dict[str, Any]:
    direction_key = "maxspeed:forward" if forward else "maxspeed:backward"
    candidates = [
        (direction_key, "directional"),
        ("maxspeed", "explicit"),
        ("maxspeed:advisory", "advisory"),
        ("maxspeed:practical", "practical"),
    ]
    attempts = []
    for key, provenance in candidates:
        parsed = parse_speed(tags.get(key), float(config["mph_to_kmh"]))
        attempts.append({"key": key, **parsed})
        if parsed["speed_kmh"] is not None:
            return {
                "speed_limit_kmh_used": parsed["speed_kmh"],
                "speed_limit_provenance": provenance,
                "speed_limit_is_inferred": False,
                "speed_limit_selected_tag": key,
                "speed_limit_parse_status": parsed["status"],
                "speed_limit_raw": parsed["raw"],
                "speed_parse_attempts_json": json.dumps(attempts, sort_keys=True),
            }
    try:
        fallback = float(valhalla_speed)
    except (TypeError, ValueError):
        fallback = float("nan")
    if fallback == fallback and fallback > 0:
        return {
            "speed_limit_kmh_used": fallback,
            "speed_limit_provenance": config["valhalla_fallback_provenance"],
            "speed_limit_is_inferred": bool(config["valhalla_fallback_is_inferred"]),
            "speed_limit_selected_tag": "valhalla_speed_limit",
            "speed_limit_parse_status": "valhalla_fallback",
            "speed_limit_raw": None,
            "speed_parse_attempts_json": json.dumps(attempts, sort_keys=True),
        }
    return {
        "speed_limit_kmh_used": None,
        "speed_limit_provenance": "missing",
        "speed_limit_is_inferred": False,
        "speed_limit_selected_tag": None,
        "speed_limit_parse_status": "missing",
        "speed_limit_raw": None,
        "speed_parse_attempts_json": json.dumps(attempts, sort_keys=True),
    }
