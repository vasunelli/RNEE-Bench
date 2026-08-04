from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from contracts import forbidden_hits, train_active_columns


SENSOR_PREFIX = {"low": "L", "high": "H"}
ROAD_VARIANTS = ("real", "noise", "perm")


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
        systems = {
            "L-R0": low,
            "L-R-real": list(dict.fromkeys(low + road)),
            "L-R-noise": list(dict.fromkeys(low + road)),
            "L-R-perm": list(dict.fromkeys(low + road)),
            "H-R0": high,
            "H-R-real": list(dict.fromkeys(high + road)),
            "H-R-noise": list(dict.fromkeys(high + road)),
            "H-R-perm": list(dict.fromkeys(high + road)),
        }
        expected = [8, 150, 150, 150, 20, 162, 162, 162]
        if [len(value) for value in systems.values()] != expected:
            raise RuntimeError(f"{target_id}: unexpected THEORY_VALIDATION system feature counts.")
        all_features = sorted({item for value in systems.values() for item in value})
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
            "systems": systems,
            "forbidden_hits": hits,
            "target_source_features_removed": sorted(removed),
            "target_source_leakage_hits": leakage,
        }
    return {
        "schema_version": 1,
        "feature_contract": "THEORY_VALIDATION_XL_XH_R6",
        "low_sensor_count": len(low),
        "road_semantic_count": len(road),
        "target_contracts": targets,
    }


def system_matrix(
    base_frame: pd.DataFrame,
    controls: pd.DataFrame,
    target_contract: dict[str, Any],
    system: str,
) -> pd.DataFrame:
    sensor = "low" if system.startswith("L-") else "high"
    base_features = target_contract[f"{sensor}_features"]
    result = base_frame[base_features].copy()
    if system.endswith("R0"):
        return sanitize_numeric(result)
    road_kind = system.rsplit("-", 1)[-1]
    road_features = target_contract["road_features"]
    if road_kind == "real":
        road = base_frame[road_features].copy()
    elif road_kind == "noise":
        road = controls[[f"noise__{column}" for column in road_features]].copy()
        road.columns = road_features
    elif road_kind == "perm":
        road = controls[[f"perm__{column}" for column in road_features]].copy()
        road.columns = road_features
    else:
        raise ValueError(f"Unknown THEORY_VALIDATION system {system}")
    return sanitize_numeric(pd.concat([result, road], axis=1))


def sanitize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def shared_real_road_active_masks(
    matrices: dict[str, pd.DataFrame],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Freeze one effective feature mask per sensor regime.

    R0 keeps its own train-only variance mask. Real, noise, and permutation
    road systems all use the mask learned from the corresponding real-R6
    training matrix. This preserves the frozen preprocessing rule while
    preventing zero-variance real columns from becoming noise-only effective
    dimensions.
    """
    expected = {
        "L-R0",
        "L-R-real",
        "L-R-noise",
        "L-R-perm",
        "H-R0",
        "H-R-real",
        "H-R-noise",
        "H-R-perm",
    }
    if set(matrices) != expected:
        raise RuntimeError(
            "THEORY_VALIDATION active-mask construction requires the exact eight systems."
        )

    masks: dict[str, list[str]] = {}
    audit: dict[str, Any] = {
        "policy": "real_R6_train_active_mask_shared_within_sensor_regime",
        "sensor_regimes": {},
    }
    for prefix, sensor in (("L", "low"), ("H", "high")):
        r0_system = f"{prefix}-R0"
        real_system = f"{prefix}-R-real"
        controlled_systems = (
            real_system,
            f"{prefix}-R-noise",
            f"{prefix}-R-perm",
        )
        r0_mask = train_active_columns(matrices[r0_system])
        shared_mask = train_active_columns(matrices[real_system])
        masks[r0_system] = list(r0_mask)
        for system in controlled_systems:
            missing = sorted(set(shared_mask) - set(matrices[system].columns))
            if missing:
                raise RuntimeError(
                    f"{system}: shared real-R6 active mask has missing columns {missing}."
                )
            masks[system] = list(shared_mask)

        native = {
            system: train_active_columns(matrices[system])
            for system in (r0_system, *controlled_systems)
        }
        effective_counts = {
            system: len(masks[system]) for system in controlled_systems
        }
        audit["sensor_regimes"][sensor] = {
            "r0_system": r0_system,
            "mask_source_system": real_system,
            "shared_active_features": list(shared_mask),
            "shared_active_feature_count": len(shared_mask),
            "effective_active_feature_counts": effective_counts,
            "effective_dimension_equal_across_real_noise_permutation": (
                len(set(effective_counts.values())) == 1
            ),
            "native_active_feature_counts_before_shared_mask": {
                system: len(columns) for system, columns in native.items()
            },
            "native_only_features_excluded_by_shared_mask": {
                system: sorted(set(native[system]) - set(shared_mask))
                for system in controlled_systems
            },
        }
    return masks, audit
