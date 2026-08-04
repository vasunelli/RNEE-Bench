from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


EXPECTED_ENVIRONMENTS = (
    "cold_trip",
    "cold_month",
    "cold_spatial",
    "cold_functional_road_class",
)
EXPECTED_MODELS = ("hgb", "ridge", "catboost")
EXPECTED_SENSORS = ("low", "high")
EXPECTED_NULL_TYPES = ("noise", "permutation")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["experiment"]["id"] != "ROBUSTNESS":
        raise ValueError("The experiment id must be ROBUSTNESS.")
    if not config["experiment"]["protocol_frozen_before_execution"]:
        raise ValueError("ROBUSTNESS must be frozen before execution.")
    if not config["experiment"]["separate_from_theory_validation"]:
        raise ValueError("ROBUSTNESS must remain separate from frozen THEORY_VALIDATION.")
    if config["experiment"]["theory_validation_results_mutable"]:
        raise ValueError("THEORY_VALIDATION artifacts are immutable.")
    if config["data"]["segments"] != "SEGMENT_SPLIT" or config["data"]["splits"] != "SEGMENT_SPLIT":
        raise ValueError("Only frozen SEGMENT_SPLIT segments and splits are allowed.")
    if tuple(config["environments"]) != EXPECTED_ENVIRONMENTS:
        raise ValueError("ROBUSTNESS requires the exact four-environment order.")
    counts = (
        int(config["features"]["low_sensor_count"]),
        int(config["features"]["high_sensor_count"]),
        int(config["features"]["road_semantic_count"]),
    )
    if counts != (8, 20, 142):
        raise ValueError("The frozen X_L/X_H/R6 counts must be 8/20/142.")
    if int(config["null_controls"]["draws_per_type"]) != 20:
        raise ValueError("ROBUSTNESS requires exactly 20 draws per null type.")
    if list(config["models"]["seeds"]) != [0, 1, 2]:
        raise ValueError("All ROBUSTNESS models require seeds [0,1,2].")
    if config["models"]["hgb"]["loss"] != "absolute_error":
        raise ValueError("HGB must retain the frozen L1 loss.")
    if config["models"]["hgb"]["params"]["loss"] != "absolute_error":
        raise ValueError("HGB params must retain the frozen L1 loss.")
    if float(config["models"]["ridge"]["params"]["alpha"]) != 1.0:
        raise ValueError("Ridge alpha is frozen at 1.0 without tuning.")
    if config["models"]["catboost"]["params"]["loss_function"] != "MAE":
        raise ValueError("CatBoost must use the frozen MAE objective.")
    bootstrap = config["bootstrap"]
    if (
        bootstrap["cluster"] != "vehicle"
        or int(bootstrap["repetitions"]) != 2000
        or bootstrap["confidence"] != "max_t"
        or float(bootstrap["alpha"]) != 0.05
    ):
        raise ValueError("ROBUSTNESS requires 2,000 vehicle-cluster max-t replicates.")
    expected_families = {
        "hgb_null_gain": 328,
        "hgb_semantic_specificity": 320,
        "model_real_gain": 24,
    }
    if dict(bootstrap["simultaneous_families_per_target"]) != expected_families:
        raise ValueError("ROBUSTNESS simultaneous family sizes changed.")
    if float(config["decision"]["relative_sesoi"]) != 0.055:
        raise ValueError("The ROBUSTNESS SESOI is frozen at 5.5%.")
    reference = config["decision"]["frozen_positive_reference_cells"]
    expected_reference = [
        {"target_id": "ICE_fuel_L", "environment": "cold_trip", "sensor": "low"},
        {"target_id": "ICE_fuel_L", "environment": "cold_month", "sensor": "low"},
    ]
    if reference != expected_reference:
        raise ValueError("Frozen THEORY_VALIDATION-positive reference cells changed.")
    if not config["integrity"]["preserve_all_negative_environments"]:
        raise ValueError("Negative environments may not be removed.")
    forbidden_true = (
        config["data"]["cross_dataset_inputs_authorized"],
        config["data"]["new_osm_or_map_matching_authorized"],
        config["data"]["new_split_generation_authorized"],
        config["features"]["new_road_features_authorized"],
        config["integrity"]["modify_theory_validation"],
        config["integrity"]["tune_models_after_results"],
        config["integrity"]["select_favorable_model"],
        config["integrity"]["causal_claims_authorized"],
    )
    if any(forbidden_true):
        raise ValueError("An ROBUSTNESS prohibited action was enabled.")


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**32 - 1)
