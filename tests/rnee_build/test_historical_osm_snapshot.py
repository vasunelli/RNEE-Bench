from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "src" / "rnee_build" / "03_prepare_osm_snapshot.py"
SPEC = importlib.util.spec_from_file_location("prepare_osm_snapshot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_haversine_known_scale() -> None:
    distance = MODULE.haversine_km(
        np.array([42.0]), np.array([-83.0]), np.array([43.0]), np.array([-83.0])
    )
    assert distance[0] == pytest.approx(111.195, rel=1e-3)


def test_daynum_floor_contract() -> None:
    assert MODULE.daynum_to_date(1.9, date(2017, 10, 31)) == date(2017, 11, 1)


def test_preceding_historical_snapshot_assignment() -> None:
    snapshots = [
        {"snapshot_id": "s2017", "snapshot_date": "2017-01-01"},
        {"snapshot_id": "s2018", "snapshot_date": "2018-01-01"},
        {"snapshot_id": "s2019", "snapshot_date": "2019-01-01"},
    ]
    assert MODULE.latest_snapshot_on_or_before(
        date(2018, 3, 1), snapshots
    ) == ("s2018", 59)
    assert MODULE.latest_snapshot_on_or_before(
        date(2018, 10, 1), snapshots
    )[0] == "s2018"


def test_future_only_snapshot_assignment_is_rejected() -> None:
    snapshots = [{"snapshot_id": "s2018", "snapshot_date": "2018-01-01"}]
    with pytest.raises(ValueError, match="on or before"):
        MODULE.latest_snapshot_on_or_before(date(2017, 12, 1), snapshots)


def test_current_latest_snapshot_is_forbidden() -> None:
    config = {
        "policy": {
            "historical_only": True,
            "current_osm_fallback": "forbidden",
            "maximum_allowed_snapshot_date": "2019-01-01",
        },
        "osm": {
            "snapshots": [
                {
                    "snapshot_id": "bad",
                    "snapshot_date": "2018-01-01",
                    "filename": "michigan-latest.osm.pbf",
                    "url": "https://example.test/michigan-latest.osm.pbf",
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="Current/latest"):
        MODULE.validate_historical_policy(config)


def test_future_snapshot_is_forbidden() -> None:
    config = {
        "policy": {
            "historical_only": True,
            "current_osm_fallback": "forbidden",
            "maximum_allowed_snapshot_date": "2019-01-01",
        },
        "osm": {
            "snapshots": [
                {
                    "snapshot_id": "bad",
                    "snapshot_date": "2020-01-01",
                    "filename": "michigan-200101.osm.pbf",
                    "url": "https://example.test/michigan-200101.osm.pbf",
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="historical cutoff"):
        MODULE.validate_historical_policy(config)


def test_bbox_containment() -> None:
    bbox = {"min_lon": -84.0, "min_lat": 42.0, "max_lon": -83.0, "max_lat": 43.0}
    boxes = [
        {"min_lon": -90.0, "min_lat": 40.0, "max_lon": -80.0, "max_lat": 48.0}
    ]
    assert MODULE.bbox_inside_any(bbox, boxes)
