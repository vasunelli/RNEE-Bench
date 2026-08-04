from __future__ import annotations

from pathlib import Path

import pandas as pd

from artifacts import load_json
from config import EXPECTED_ENVIRONMENTS, EXPECTED_SYSTEMS, load_config
from features import build_feature_contract, shared_real_road_active_masks


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_theory_validation_configuration_contract() -> None:
    config = load_config(
        ROOT / "configs" / "rnee" / "theory_validation" / "theory_validation.yaml"
    )
    assert config["experiment"]["id"] == "THEORY_VALIDATION"
    assert tuple(config["environments"]) == EXPECTED_ENVIRONMENTS
    assert tuple(config["features"]["systems"]) == EXPECTED_SYSTEMS
    assert config["model"]["seeds"] == [0, 1, 2]
    assert (
        config["model"]["active_mask_policy"]
        == "real_R6_train_active_mask_shared_within_sensor_regime"
    )
    assert config["bootstrap"]["repetitions"] == 2000
    assert config["decision"]["relative_sesoi"] == 0.055


def test_actual_upstream_feature_contract_is_exact_and_target_safe() -> None:
    information = load_json(
        ROOT
        / "results"
        / "model_contract"
        / "information_freeze_m0_m1_riva_freeze_20260724_232406"
        / "feature_information_contract.json"
    )
    m1 = load_json(
        ROOT
        / "results"
        / "rnee_build"
        / "model_contract_m1_feature_contract_20260720_034411"
        / "feature_contract.json"
    )
    contract = build_feature_contract(
        information, m1, ["ICE_fuel_L", "HEV_fuel_L"]
    )
    assert contract["low_sensor_count"] == 8
    assert contract["road_semantic_count"] == 142
    for target in contract["target_contracts"].values():
        assert len(target["high_features"]) == 20
        assert target["forbidden_hits"] == []
        assert target["target_source_leakage_hits"] == []
        assert [len(value) for value in target["systems"].values()] == [
            8,
            150,
            150,
            150,
            20,
            162,
            162,
            162,
        ]


def test_real_road_mask_removes_noise_only_effective_dimensions() -> None:
    low = pd.DataFrame({"low": [0.0, 1.0, 2.0, 3.0]})
    high = pd.DataFrame(
        {"low": [0.0, 1.0, 2.0, 3.0], "high": [3.0, 2.0, 1.0, 0.0]}
    )
    real_road = pd.DataFrame(
        {"road_active": [0.0, 1.0, 0.0, 1.0], "road_constant": [1.0] * 4}
    )
    noise_road = pd.DataFrame(
        {
            "road_active": [0.2, 0.4, 0.6, 0.8],
            "road_constant": [-1.0, 0.0, 1.0, 2.0],
        }
    )
    permutation_road = real_road.iloc[[1, 0, 3, 2]].reset_index(drop=True)
    matrices: dict[str, pd.DataFrame] = {}
    for prefix, base in (("L", low), ("H", high)):
        matrices[f"{prefix}-R0"] = base.copy()
        matrices[f"{prefix}-R-real"] = pd.concat(
            [base, real_road], axis=1
        )
        matrices[f"{prefix}-R-noise"] = pd.concat(
            [base, noise_road], axis=1
        )
        matrices[f"{prefix}-R-perm"] = pd.concat(
            [base, permutation_road], axis=1
        )

    masks, audit = shared_real_road_active_masks(matrices)
    for prefix in ("L", "H"):
        assert "road_active" in masks[f"{prefix}-R-real"]
        assert "road_constant" not in masks[f"{prefix}-R-real"]
        assert masks[f"{prefix}-R-real"] == masks[f"{prefix}-R-noise"]
        assert masks[f"{prefix}-R-real"] == masks[f"{prefix}-R-perm"]
    assert audit["sensor_regimes"]["low"][
        "native_only_features_excluded_by_shared_mask"
    ]["L-R-noise"] == ["road_constant"]
