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
EXPECTED_SYSTEMS = (
    "L-R0",
    "L-R-real",
    "L-R-noise",
    "L-R-perm",
    "H-R0",
    "H-R-real",
    "H-R-noise",
    "H-R-perm",
)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["experiment"]["id"] != "THEORY_VALIDATION":
        raise ValueError("The experiment id must be THEORY_VALIDATION.")
    if not config["experiment"]["protocol_frozen_before_execution"]:
        raise ValueError("THEORY_VALIDATION must be frozen before execution.")
    if config["data"]["segments"] != "SEGMENT_SPLIT" or config["data"]["splits"] != "SEGMENT_SPLIT":
        raise ValueError("Only the frozen SEGMENT_SPLIT segments and splits are allowed.")
    if tuple(config["environments"]) != EXPECTED_ENVIRONMENTS:
        raise ValueError("THEORY_VALIDATION requires the exact frozen four-environment order.")
    if tuple(config["features"]["systems"]) != EXPECTED_SYSTEMS:
        raise ValueError("THEORY_VALIDATION requires the exact frozen eight-system order.")
    feature_counts = (
        int(config["features"]["low_sensor_count"]),
        int(config["features"]["high_sensor_count"]),
        int(config["features"]["road_semantic_count"]),
    )
    if feature_counts != (8, 20, 142):
        raise ValueError("The frozen X_L/X_H/R6 counts must be 8/20/142.")
    model = config["model"]
    if model["name"] != "HistGradientBoostingRegressor":
        raise ValueError("The THEORY_VALIDATION primary model is HistGradientBoostingRegressor.")
    if model["loss"] != "absolute_error" or list(model["seeds"]) != [0, 1, 2]:
        raise ValueError("The frozen model loss/seeds are absolute_error and [0,1,2].")
    if (
        model.get("active_mask_policy")
        != "real_R6_train_active_mask_shared_within_sensor_regime"
    ):
        raise ValueError(
            "THEORY_VALIDATION requires the real-R6 train-active mask to be shared across "
            "real, noise, and permutation systems within each sensor regime."
        )
    bootstrap = config["bootstrap"]
    if (
        bootstrap["cluster"] != "vehicle"
        or int(bootstrap["repetitions"]) != 2000
        or bootstrap["confidence"] != "max_t"
        or float(bootstrap["alpha"]) != 0.05
    ):
        raise ValueError("The frozen bootstrap is vehicle-clustered 2000-replicate max-t.")
    if float(config["decision"]["relative_sesoi"]) != 0.055:
        raise ValueError("The frozen THEORY_VALIDATION SESOI is 5.5%.")
    if not config["integrity"]["preserve_all_negative_environments"]:
        raise ValueError("Negative environments may not be removed.")
    if config["data"]["allow_cross_dataset_inputs"]:
        raise ValueError("THEORY_VALIDATION is RNEE-only; cross-dataset inputs are prohibited.")
    if config["integrity"]["allow_causal_claims"]:
        raise ValueError("THEORY_VALIDATION does not support causal claims.")


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**32 - 1)
