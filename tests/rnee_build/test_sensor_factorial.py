from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rnee_build"
    / "68_run_sensor_factorial.py"
)
SPEC = importlib.util.spec_from_file_location("sensor_factorial_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_direct_support_requires_pure_direct_and_zero_maf() -> None:
    frame = pd.DataFrame(
        {
            "fuel_direct_duration_share": [1.0, 0.999, 1.0, np.nan],
            "fuel_maf_duration_share": [0.0, 0.001, 0.1, 0.0],
        }
    )
    assert MODULE.direct_support_mask(frame).tolist() == [True, False, False, False]


def test_relative_effects_have_expected_sign_and_interaction() -> None:
    effects = MODULE.relative_effects(
        {"L-R0": 100.0, "L-R1": 90.0, "H-R0": 50.0, "H-R1": 49.0}
    )
    assert np.isclose(effects["low_relative"], 0.10)
    assert np.isclose(effects["high_relative"], 0.02)
    assert np.isclose(effects["interaction_relative"], 0.08)


def test_four_state_classifier_uses_frozen_sesoi() -> None:
    delta = 0.055
    assert MODULE.classify_effect(0.06, 0.09, delta) == "PRACTICAL_POSITIVE"
    assert (
        MODULE.classify_effect(-0.02, 0.03, delta)
        == "EQUIVALENT_NO_MEANINGFUL_EFFECT"
    )
    assert MODULE.classify_effect(-0.10, -0.06, delta) == "PRACTICAL_NEGATIVE"
    assert MODULE.classify_effect(0.01, 0.08, delta) == "UNRESOLVED"


def test_simultaneous_interval_is_not_narrower_than_pointwise_spread() -> None:
    point = np.array([0.1, 0.2])
    bootstrap = np.array(
        [
            [0.08, 0.19],
            [0.12, 0.21],
            [0.09, 0.18],
            [0.11, 0.22],
            [0.10, 0.20],
        ]
    )
    lower, upper, critical = MODULE.simultaneous_max_t_intervals(
        point, bootstrap, 0.05
    )
    assert critical >= 0
    assert np.all(lower <= point)
    assert np.all(upper >= point)


def test_retrospective_route_never_uses_confirmed_language() -> None:
    frame = pd.DataFrame(
        {
            "target_id": ["ICE_fuel_L"] * 4,
            "low_road_gain_relative_state": ["PRACTICAL_POSITIVE"] * 4,
            "high_road_gain_relative_state": [
                "EQUIVALENT_NO_MEANINGFUL_EFFECT"
            ]
            * 4,
            "sensor_interaction_relative_state": ["PRACTICAL_POSITIVE"] * 4,
        }
    )
    route = MODULE.retrospective_route(frame, "ICE_fuel_L")
    assert route["retrospective_pattern"] == "SENSOR_SCARCE_VALUE"
    assert route["route"].startswith("RETROSPECTIVE_PATTERN_")
    assert "CONFIRMED" not in route["route"]
