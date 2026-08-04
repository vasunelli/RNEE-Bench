from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "47_run_model_contract_m3_add_drop.py"
SPEC = importlib.util.spec_from_file_location("model_contract_m3", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _contract() -> dict:
    groups = {
        "R1_HIERARCHY_REGULATION": ["r1a", "r1b"],
        "R2_PHYSICAL_ACCESS": ["r2"],
        "R3_TOPOLOGY": ["r3"],
        "R4_TRAFFIC_PLACE_OBJECTS": ["r4"],
        "R5_BUILT_ENVIRONMENT": ["r5"],
    }
    return {
        "targets": {"ICE_fuel_L": {"r0_full_obd": ["x0", "x1"]}},
        "feature_groups": {**groups, "R6_FULL_SEMANTICS": ["r1a", "r1b", "r2", "r3", "r4", "r5"]},
    }


def test_variant_grid_is_exact_and_ordered() -> None:
    variants = MODULE.variant_feature_sets(_contract(), "ICE_fuel_L")
    assert list(variants) == [
        "R0",
        "R0_PLUS_R1",
        "R0_PLUS_R2",
        "R0_PLUS_R3",
        "R0_PLUS_R4",
        "R0_PLUS_R5",
        "R6",
        "R6_MINUS_R1",
        "R6_MINUS_R2",
        "R6_MINUS_R3",
        "R6_MINUS_R4",
        "R6_MINUS_R5",
    ]
    assert variants["R0_PLUS_R1"] == ["x0", "x1", "r1a", "r1b"]
    assert variants["R6_MINUS_R1"] == ["x0", "x1", "r2", "r3", "r4", "r5"]
    assert variants["R6"] == ["x0", "x1", "r1a", "r1b", "r2", "r3", "r4", "r5"]


def test_ordered_id_hash_is_order_invariant_and_sensitive() -> None:
    first = MODULE.ordered_id_sha256(pd.Series(["b", "a", "c"]))
    second = MODULE.ordered_id_sha256(["c", "b", "a"])
    assert first == second
    assert first != MODULE.ordered_id_sha256(["a", "b", "d"])


def test_train_active_columns_uses_train_only_variation() -> None:
    frame = pd.DataFrame(
        {
            "varying": [0.0, 1.0, 2.0],
            "constant": [1.0, 1.0, 1.0],
            "missing": [np.nan, np.nan, np.nan],
            "mixed": [np.nan, 2.0, 3.0],
        }
    )
    assert MODULE.train_active_columns(frame) == ["varying", "mixed"]


def test_regression_metrics_match_known_values() -> None:
    metrics = MODULE.regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0]))
    assert np.isclose(metrics["mae"], 2 / 3)
    assert np.isclose(metrics["rmse"], np.sqrt(2 / 3))
    assert np.isclose(metrics["medae"], 1.0)


def test_fit_keys_separate_cells_variants_and_seeds() -> None:
    assert MODULE.fit_key("cold_trip", "ICE_fuel_L", "R6_MINUS_R4", 20260721) == (
        "cold_trip__ICE_fuel_L__R6_MINUS_R4__seed_20260721"
    )
