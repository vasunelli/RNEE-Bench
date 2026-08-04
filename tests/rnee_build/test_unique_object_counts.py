import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rnee_context", ROOT / "src" / "rnee_build" / "12_build_context_semantics.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_poi_taxonomy_is_deterministic_and_multilabel():
    taxonomy = {
        "transport": {"amenity": ["parking"]},
        "parking": {"amenity": ["parking"]},
        "other": {"tag_presence": ["amenity"]},
    }
    labels, primary = MODULE.poi_labels({"amenity": "parking"}, taxonomy, ["transport", "parking", "other"])
    assert labels == ["transport", "parking", "other"]
    assert primary == "transport"


def test_monotonic_context_count_check():
    good = pd.DataFrame({"poi_unique_count_25m": [0, 1], "poi_unique_count_50m": [1, 1], "poi_unique_count_100m": [2, 3], "poi_unique_count_250m": [4, 4]})
    bad = good.copy(); bad.loc[0, "poi_unique_count_100m"] = 0
    assert MODULE.monotonic_violations(good, [25, 50, 100, 250]) == 0
    assert MODULE.monotonic_violations(bad, [25, 50, 100, 250]) == 1


def test_object_uid_is_snapshot_and_layer_specific():
    assert MODULE.stable_uid("s1", "node", 1, "poi") != MODULE.stable_uid("s2", "node", 1, "poi")
    assert MODULE.stable_uid("s1", "node", 1, "poi") != MODULE.stable_uid("s1", "node", 1, "traffic_control")
