from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import forbidden_hits, train_active_columns


SENSORS = ("low", "high")


def build_feature_contract(
    information_contract: dict[str, Any],
    m1_contract: dict[str, Any],
    target_ids: list[str],
) -> dict[str, Any]:
    low = list(information_contract["components"]["LOW"])
    road = list(information_contract["components"]["ROAD"])
    if len(low) != 8 or len(road) != 142 or len(set(road)) != 142:
        raise RuntimeError("Frozen LOW/R6 counts or uniqueness changed.")
    targets: dict[str, Any] = {}
    for target_id in target_ids:
        target = m1_contract["targets"][target_id]
        high = list(target["r0_full_obd"])
        if len(high) != 20 or not set(low).issubset(high):
            raise RuntimeError(f"{target_id}: frozen target-safe X_H is not 20 columns.")
        all_features = sorted(set(low + high + road))
        hits = forbidden_hits(all_features, m1_contract)
        removed = set(target["target_source_features_removed"])
        leakage = sorted(removed.intersection(all_features))
        if hits or leakage:
            raise RuntimeError(
                f"{target_id}: forbidden={hits}, target_source_leakage={leakage}"
            )
        targets[target_id] = {
            "target_column": target["column"],
            "target_unit": target["unit"],
            "target_channel": target["channel"],
            "low_features": low,
            "high_features": high,
            "road_features": road,
            "forbidden_hits": hits,
            "target_source_features_removed": sorted(removed),
            "target_source_leakage_hits": leakage,
        }
    return {
        "schema_version": 1,
        "feature_contract": "ROBUSTNESS_XL_XH_R6_UNCHANGED_FROM_THEORY_VALIDATION",
        "low_sensor_count": len(low),
        "road_semantic_count": len(road),
        "target_contracts": targets,
    }


def sanitize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def model_matrix(
    base_frame: pd.DataFrame,
    target_contract: dict[str, Any],
    sensor: str,
    road_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor regime: {sensor}")
    base = base_frame[target_contract[f"{sensor}_features"]].reset_index(drop=True)
    if road_frame is None:
        return sanitize_numeric(base)
    road = road_frame[target_contract["road_features"]].reset_index(drop=True)
    return sanitize_numeric(pd.concat([base, road], axis=1))


def active_masks_from_real_train(
    train_base: pd.DataFrame,
    target_contract: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    road = train_base[target_contract["road_features"]].reset_index(drop=True)
    masks: dict[str, list[str]] = {}
    audit: dict[str, Any] = {
        "policy": (
            "real_R6_train_active_mask_shared_across_models_and_all_null_draws"
        ),
        "sensor_regimes": {},
    }
    for sensor in SENSORS:
        r0 = model_matrix(train_base, target_contract, sensor, None)
        real = model_matrix(train_base, target_contract, sensor, road)
        r0_mask = train_active_columns(r0)
        real_mask = train_active_columns(real)
        if not set(r0_mask).issubset(real_mask):
            raise RuntimeError(f"{sensor}: R0 active mask is not contained in real R6.")
        masks[f"{sensor}|baseline"] = r0_mask
        masks[f"{sensor}|road"] = real_mask
        audit["sensor_regimes"][sensor] = {
            "r0_active_features": r0_mask,
            "r0_active_feature_count": len(r0_mask),
            "real_road_active_features": real_mask,
            "real_road_active_feature_count": len(real_mask),
            "shared_with_models": ["hgb", "ridge", "catboost"],
            "shared_with_null_types": ["noise", "permutation"],
            "shared_with_null_draw_count_per_type": 20,
        }
    return masks, audit
