#!/usr/bin/env python3
"""Run SENSOR_FACTORIAL: exploratory RNEE-only sensor x road-semantics factorial.

SENSOR_FACTORIAL is deliberately separate from MODEL_CONTRACT and INFORMATION_FREEZE.  It uses the frozen
MODEL_CONTRACT split/backbone contracts and the frozen INFORMATION_FREEZE L/H/R column definitions,
but it creates only exploratory evidence on the published RNEE
corpus.  The current MODEL_CONTRACT and INFORMATION_FREEZE releases are never modified.

The fuel target in RNEE can use either direct Fuel Rate or a MAF fallback.
Pure-direct support is too small for modeling, so SENSOR_FACTORIAL uses each target's
frozen MODEL_CONTRACT target-safe R0 contract.  Fuel Rate, MAF, and fuel-trim channels
are excluded from every fuel model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


SYSTEMS = ("L-R0", "L-R1", "H-R0", "H-R1")
PREDICTION_COLUMNS = {
    "L-R0": "prediction_l_r0",
    "L-R1": "prediction_l_r1",
    "H-R0": "prediction_h_r0",
    "H-R1": "prediction_h_r1",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha256(values: list[str] | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    unique = sorted(set(paths), key=lambda value: str(value).lower())
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in unique
        ],
    }


def verify_manifest(
    manifest: dict[str, Any], base_directory: Path | None = None
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if not path.is_absolute() and base_directory is not None:
            path = base_directory / path
        if (
            not path.exists()
            or int(path.stat().st_size) != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            mismatches.append(str(path))
    return not mismatches, mismatches


def train_active_columns(frame: pd.DataFrame) -> list[str]:
    active: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.nunique(dropna=True) >= 2:
            active.append(column)
    return active


def forbidden_hits(features: list[str], contract: dict[str, Any]) -> list[str]:
    forbidden = contract["forbidden_feature_contract"]
    exact = {value.lower() for value in forbidden["exact"]}
    tokens = [value.lower() for value in forbidden["tokens"]]
    hits: list[str] = []
    for feature in features:
        lowered = feature.lower()
        if lowered in exact or any(token in lowered for token in tokens):
            hits.append(feature)
    return sorted(set(hits))


def direct_support_mask(
    frame: pd.DataFrame,
    required_direct_share: float = 1.0,
    maximum_maf_share: float = 0.0,
    tolerance: float = 1e-12,
) -> pd.Series:
    direct = pd.to_numeric(frame["fuel_direct_duration_share"], errors="coerce")
    maf = pd.to_numeric(frame["fuel_maf_duration_share"], errors="coerce")
    return direct.ge(required_direct_share - tolerance) & maf.le(
        maximum_maf_share + tolerance
    )


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    residual = target - predicted
    absolute = np.abs(residual)
    denominator = float(np.square(target - np.mean(target)).sum())
    return {
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(np.square(residual).mean())),
        "medae": float(np.median(absolute)),
        "r2": (
            float(1.0 - np.square(residual).sum() / denominator)
            if denominator > 0
            else math.nan
        ),
    }


def relative_effects(mae: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    low_absolute = mae["L-R0"] - mae["L-R1"]
    high_absolute = mae["H-R0"] - mae["H-R1"]
    low_relative = np.divide(
        low_absolute,
        mae["L-R0"],
        out=np.full_like(np.asarray(low_absolute, dtype=float), np.nan),
        where=np.asarray(mae["L-R0"]) != 0,
    )
    high_relative = np.divide(
        high_absolute,
        mae["H-R0"],
        out=np.full_like(np.asarray(high_absolute, dtype=float), np.nan),
        where=np.asarray(mae["H-R0"]) != 0,
    )
    return {
        "low_absolute": low_absolute,
        "high_absolute": high_absolute,
        "low_relative": low_relative,
        "high_relative": high_relative,
        "interaction_relative": low_relative - high_relative,
    }


def classify_effect(lower: float, upper: float, sesoi: float) -> str:
    if lower > sesoi:
        return "PRACTICAL_POSITIVE"
    if lower > -sesoi and upper < sesoi:
        return "EQUIVALENT_NO_MEANINGFUL_EFFECT"
    if upper < -sesoi:
        return "PRACTICAL_NEGATIVE"
    return "UNRESOLVED"


def simultaneous_max_t_intervals(
    point: np.ndarray,
    bootstrap: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-sided simultaneous max-t intervals over environment columns."""
    point = np.asarray(point, dtype=float)
    bootstrap = np.asarray(bootstrap, dtype=float)
    standard_error = np.nanstd(bootstrap, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, 1.0)
    standardized = np.abs((bootstrap - point[None, :]) / safe[None, :])
    standardized[:, standard_error <= 0] = 0.0
    maximum = np.nanmax(standardized, axis=1)
    critical = float(np.nanquantile(maximum, 1.0 - alpha))
    half_width = critical * standard_error
    return point - half_width, point + half_width, critical


def build_backbone(params: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=seed, **params)


def dump_model(path: Path, payload: dict[str, Any]) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def load_model(path: Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def valid_model_record(record: dict[str, Any]) -> bool:
    model_path = Path(record["model_path"])
    metadata_path = Path(record["metadata_path"])
    return (
        model_path.exists()
        and metadata_path.exists()
        and sha256_file(model_path) == record["model_sha256"]
        and sha256_file(metadata_path) == record["metadata_sha256"]
    )


def support_audit(frame: pd.DataFrame, floors: dict[str, Any]) -> dict[str, Any]:
    vehicle_counts = frame["VehId"].astype(str).value_counts()
    vehicle_count = int(vehicle_counts.size)
    trip_count = int(frame["trip_uid"].astype(str).nunique())
    segment_count = int(len(frame))
    maximum_share = (
        float(vehicle_counts.iloc[0] / segment_count) if segment_count else math.nan
    )
    checks = {
        "minimum_vehicles": vehicle_count >= int(floors["minimum_vehicles"]),
        "minimum_trips": trip_count >= int(floors["minimum_trips"]),
        "minimum_segments": segment_count >= int(floors["minimum_segments"]),
        "maximum_single_vehicle_segment_share": maximum_share
        <= float(floors["maximum_single_vehicle_segment_share"]),
    }
    return {
        "vehicle_count": vehicle_count,
        "trip_count": trip_count,
        "segment_count": segment_count,
        "maximum_single_vehicle_segment_share": maximum_share,
        "prospective_floor_checks": checks,
        "prospective_floor_pass": all(checks.values()),
    }


def vehicle_cluster_bootstrap(
    environment_frames: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
    alpha: float,
    sesoi: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Joint vehicle bootstrap preserving dependence across cold environments."""
    environments = list(environment_frames)
    vehicles = sorted(
        {
            str(vehicle)
            for frame in environment_frames.values()
            for vehicle in frame["VehId"].astype(str).unique()
        }
    )
    if len(vehicles) < 2:
        raise RuntimeError("Vehicle bootstrap requires at least two vehicle clusters.")
    vehicle_index = {vehicle: index for index, vehicle in enumerate(vehicles)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(vehicles),
        np.full(len(vehicles), 1.0 / len(vehicles)),
        size=replicates,
    ).astype(float)

    point_mae: dict[str, list[float]] = {system: [] for system in SYSTEMS}
    bootstrap_mae: dict[str, list[np.ndarray]] = {system: [] for system in SYSTEMS}
    support: dict[str, Any] = {}

    for environment in environments:
        frame = environment_frames[environment].copy()
        frame["VehId"] = frame["VehId"].astype(str)
        for system in SYSTEMS:
            frame[f"ae_{system}"] = np.abs(
                pd.to_numeric(frame["target_l"], errors="raise").to_numpy(float)
                - pd.to_numeric(
                    frame[PREDICTION_COLUMNS[system]], errors="raise"
                ).to_numpy(float)
            )
        grouped = frame.groupby("VehId", sort=False).agg(
            count=("segment_id", "size"),
            **{f"sum_{system}": (f"ae_{system}", "sum") for system in SYSTEMS},
        )
        counts = np.zeros(len(vehicles), dtype=float)
        sums = {system: np.zeros(len(vehicles), dtype=float) for system in SYSTEMS}
        for vehicle, row in grouped.iterrows():
            position = vehicle_index[str(vehicle)]
            counts[position] = float(row["count"])
            for system in SYSTEMS:
                sums[system][position] = float(row[f"sum_{system}"])
        denominators = weights @ counts
        if np.any(denominators <= 0):
            raise RuntimeError(f"Bootstrap produced empty support for {environment}.")
        for system in SYSTEMS:
            point_mae[system].append(float(sums[system].sum() / counts.sum()))
            bootstrap_mae[system].append((weights @ sums[system]) / denominators)

    point_arrays = {system: np.asarray(values) for system, values in point_mae.items()}
    bootstrap_arrays = {
        system: np.column_stack(values) for system, values in bootstrap_mae.items()
    }
    point_effects = relative_effects(point_arrays)
    bootstrap_effects = relative_effects(bootstrap_arrays)

    intervals: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name in ("low_relative", "high_relative", "interaction_relative"):
        intervals[name] = simultaneous_max_t_intervals(
            np.asarray(point_effects[name]),
            np.asarray(bootstrap_effects[name]),
            alpha,
        )

    rows: list[dict[str, Any]] = []
    for index, environment in enumerate(environments):
        frame = environment_frames[environment]
        support_record = {
            "vehicle_count": int(frame["VehId"].astype(str).nunique()),
            "trip_count": int(frame["trip_uid"].astype(str).nunique()),
            "segment_count": int(len(frame)),
            "maximum_single_vehicle_segment_share": float(
                frame["VehId"].astype(str).value_counts().iloc[0] / len(frame)
            ),
        }
        support[environment] = support_record
        row: dict[str, Any] = {
            "environment": environment,
            **support_record,
            **{
                f"mae_{system.lower().replace('-', '_')}": float(
                    point_arrays[system][index]
                )
                for system in SYSTEMS
            },
            "low_road_gain_l": float(point_effects["low_absolute"][index]),
            "high_road_gain_l": float(point_effects["high_absolute"][index]),
        }
        for effect_name, output_prefix in (
            ("low_relative", "low_road_gain_relative"),
            ("high_relative", "high_road_gain_relative"),
            ("interaction_relative", "sensor_interaction_relative"),
        ):
            lower, upper, critical = intervals[effect_name]
            estimate = float(np.asarray(point_effects[effect_name])[index])
            row[output_prefix] = estimate
            row[f"{output_prefix}_sim_lcb"] = float(lower[index])
            row[f"{output_prefix}_sim_ucb"] = float(upper[index])
            row[f"{output_prefix}_sim_critical"] = critical
            row[f"{output_prefix}_state"] = classify_effect(
                float(lower[index]), float(upper[index]), sesoi
            )
        rows.append(row)
    diagnostics = {
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "vehicle_union_count": len(vehicles),
        "environment_order": environments,
        "simultaneous_alpha": alpha,
        "relative_sesoi": sesoi,
        "support": support,
    }
    return pd.DataFrame(rows), diagnostics


def retrospective_route(effect_rows: pd.DataFrame, target_id: str) -> dict[str, Any]:
    rows = effect_rows[effect_rows["target_id"].eq(target_id)].copy()
    low = rows["low_road_gain_relative_state"].tolist()
    high = rows["high_road_gain_relative_state"].tolist()
    interaction = rows["sensor_interaction_relative_state"].tolist()

    def count(values: list[str], state: str) -> int:
        return sum(value == state for value in values)

    low_positive = count(low, "PRACTICAL_POSITIVE")
    high_positive = count(high, "PRACTICAL_POSITIVE")
    interaction_positive = count(interaction, "PRACTICAL_POSITIVE")
    low_negative = count(low, "PRACTICAL_NEGATIVE")
    high_negative = count(high, "PRACTICAL_NEGATIVE")

    if (
        low_positive >= 3
        and low_negative == 0
        and interaction_positive >= 3
    ):
        pattern = "SENSOR_SCARCE_VALUE"
    elif low_positive >= 3 and high_positive >= 3 and not (low_negative or high_negative):
        pattern = "GENERAL_COMPLEMENT"
    elif (
        count(low, "EQUIVALENT_NO_MEANINGFUL_EFFECT") == len(low)
        and count(high, "EQUIVALENT_NO_MEANINGFUL_EFFECT") == len(high)
    ):
        pattern = "EQUIVALENT_NO_PRACTICAL_VALUE"
    elif any(
        len(set(values)) > 1
        or "PRACTICAL_NEGATIVE" in values
        for values in (low, high, interaction)
    ):
        pattern = "CONTEXT_DEPENDENT"
    else:
        pattern = "UNRESOLVED"
    return {
        "target_id": target_id,
        "retrospective_pattern": pattern,
        "route": f"RETROSPECTIVE_PATTERN_{pattern}__NOT_CONFIRMATORY",
        "low_positive_count": low_positive,
        "high_positive_count": high_positive,
        "interaction_positive_count": interaction_positive,
        "low_negative_count": low_negative,
        "high_negative_count": high_negative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/sensor_factorial.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m1_pointer_path = Path(config["m1_pointer"])
    m2_pointer_path = Path(config["m2_pointer"])
    segment_pointer_path = Path(config["segment_pointer"])
    information_freeze_pointer_path = Path(config["information_freeze_pointer"])
    m1_pointer = load_json(m1_pointer_path)
    m2_pointer = load_json(m2_pointer_path)
    segment_pointer = load_json(segment_pointer_path)
    information_freeze_pointer = load_json(information_freeze_pointer_path)
    m1_summary = load_json(Path(m1_pointer["summary"]))
    m2_summary = load_json(Path(m2_pointer["summary"]))
    m1_contract = load_json(Path(m1_pointer["feature_contract"]))
    protocol = load_json(Path(m2_pointer["protocol"]))
    information_contract = load_json(Path(information_freeze_pointer["information_contract"]))

    upstream_manifest_paths = [
        Path(m1_pointer["manifest"]),
        Path(m1_pointer["release_manifest"]),
        Path(m2_pointer["manifest"]),
        Path(m2_pointer["release_manifest"]),
        Path(information_freeze_pointer["manifest"]),
    ]
    upstream_verification = [
        verify_manifest(load_json(path), path.parent) for path in upstream_manifest_paths
    ]
    if not all(result[0] for result in upstream_verification):
        raise RuntimeError(f"Upstream manifest failure: {upstream_verification}")
    if m1_summary["status"] != "PASS_MODEL_CONTRACT_M1_B1":
        raise RuntimeError("MODEL_CONTRACT M1 is not authoritative PASS.")
    if (
        m2_summary["status"]
        != "PASS_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE"
    ):
        raise RuntimeError("MODEL_CONTRACT M2 is not authoritative PASS.")
    if information_freeze_pointer["status"] != (
        "PASS_INFORMATION_FREEZE_M0_M1_CONTRACT_FEATURE_POWER_FREEZE__M2_BLOCKED"
    ):
        raise RuntimeError("INFORMATION_FREEZE must remain at its frozen M2-blocked status.")
    information_freeze_release = load_json(Path(information_freeze_pointer["release_manifest"]))
    if (
        information_freeze_release["artifact_manifest_sha256"]
        != sha256_file(Path(information_freeze_pointer["manifest"]))
        or information_freeze_release["status"] != information_freeze_pointer["status"]
        or information_freeze_release["model_fit_count"] != 0
        or information_freeze_release["target_materialization_count"] != 0
    ):
        raise RuntimeError("INFORMATION_FREEZE release controls are not intact.")
    if protocol["selected_backbone"] != "hist_gradient_boosting_l1":
        raise RuntimeError("Unexpected frozen backbone.")
    if tuple(config["scope"]["systems"]) != SYSTEMS:
        raise RuntimeError("SENSOR_FACTORIAL requires the exact frozen four-system order.")
    if list(config["scope"]["seeds"]) != list(protocol["m3_seeds"]):
        raise RuntimeError("SENSOR_FACTORIAL seeds must match the frozen MODEL_CONTRACT seeds.")

    low_features = list(information_contract["components"]["LOW"])
    road_features = list(information_contract["components"]["ROAD"])
    target_ids = sorted({cell["target_id"] for cell in protocol["primary_cells"]})
    systems_by_target: dict[str, dict[str, list[str]]] = {}
    for target_id in target_ids:
        high_road_free = list(m1_contract["targets"][target_id]["r0_full_obd"])
        if not set(low_features).issubset(high_road_free):
            raise RuntimeError(f"LOW is not a subset of target-safe R0 for {target_id}.")
        systems_by_target[target_id] = {
            "L-R0": low_features,
            "L-R1": list(dict.fromkeys(low_features + road_features)),
            "H-R0": high_road_free,
            "H-R1": list(dict.fromkeys(high_road_free + road_features)),
        }
    if any(
        [len(systems_by_target[target_id][name]) for name in SYSTEMS]
        != [8, 150, 20, 162]
        for target_id in target_ids
    ):
        raise RuntimeError("Unexpected RNEE target-safe L/H/R feature counts.")
    all_features = sorted(
        {
            feature
            for target_systems in systems_by_target.values()
            for values in target_systems.values()
            for feature in values
        }
    )
    hits = forbidden_hits(all_features, m1_contract)
    if hits:
        raise RuntimeError(f"Forbidden features in SENSOR_FACTORIAL: {hits}")
    target_source_features = {
        feature
        for target_id in target_ids
        for feature in m1_contract["targets"][target_id][
            "target_source_features_removed"
        ]
    }
    if target_source_features.intersection(all_features):
        raise RuntimeError("A fuel target-source channel leaked into SENSOR_FACTORIAL.")

    m1_root = Path(m1_pointer["output_directory"])
    r2_path = m1_root / "r2_segment_features.parquet"
    supported_cells = list(protocol["primary_cells"])
    seeds = [int(value) for value in config["scope"]["seeds"]]
    membership_paths = {
        f"{cell['split_family']}|{cell['target_id']}": (
            m1_root
            / "target_memberships"
            / cell["split_family"]
            / cell["target_id"]
            / "model_membership.parquet"
        )
        for cell in supported_cells
    }
    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "m1_pointer_sha256": sha256_file(m1_pointer_path),
        "m2_pointer_sha256": sha256_file(m2_pointer_path),
        "segment_pointer_sha256": sha256_file(segment_pointer_path),
        "information_freeze_pointer_sha256": sha256_file(information_freeze_pointer_path),
        "information_contract_sha256": sha256_file(
            Path(information_freeze_pointer["information_contract"])
        ),
        "segments_sha256": sha256_file(Path(segment_pointer["segments_all"])),
        "r2_sha256": sha256_file(r2_path),
        "membership_sha256": {
            key: sha256_file(path) for key, path in membership_paths.items()
        },
    }
    output = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(config["output_root"])
        / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    run_config_path = output / "run_config.yaml"
    if not run_config_path.exists():
        shutil.copy2(args.config, run_config_path)
    state_path = output / "sensor_factorial_state.json"
    summary_path = output / "sensor_factorial_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if state["input_fingerprint"] != fingerprint:
            raise RuntimeError("SENSOR_FACTORIAL resume fingerprint mismatch.")
    else:
        state = {
            "schema_version": 1,
            "status": "FITTING",
            "input_fingerprint": fingerprint,
            "fit_records": {},
            "prediction_records": {},
            "test_gate": {"status": "CLOSED"},
        }
        write_json(state_path, state)

    if (
        state["status"] == "PASS"
        and summary_path.exists()
        and primary_manifest_path.exists()
    ):
        manifest = load_json(primary_manifest_path)
        verified, mismatches = verify_manifest(manifest)
        resume_path = output / "resume_check.json"
        write_json(
            resume_path,
            {
                "status": "PASS" if verified else "FAIL",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_count": len(manifest["artifacts"]),
                "mismatches": mismatches,
                "refit_count": 0,
                "prediction_recompute_count": 0,
                "target_rematerialization_count": 0,
            },
        )
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(resume_path)
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest([primary_manifest_path, summary_path, resume_path])
        release["scope"] = "SENSOR_FACTORIAL post-resume controls"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / (
            f"{config['latest_prefix']}_latest_pointer.json"
        )
        latest = load_json(latest_path) if latest_path.exists() else {}
        latest["release_manifest"] = str(release_manifest_path)
        latest["resume_check"] = str(resume_path)
        latest["resume_noop_verified"] = verified
        write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    segment_schema = set(pq.read_schema(segment_pointer["segments_all"]).names)
    identity_columns = [
        "segment_id",
        "trip_uid",
        "VehId",
        "engine_type",
        "fuel_direct_duration_share",
        "fuel_maf_duration_share",
    ]
    segment_columns = list(
        dict.fromkeys(
            identity_columns + sorted(set(all_features).intersection(segment_schema))
        )
    )
    segments = pd.read_parquet(
        segment_pointer["segments_all"], columns=segment_columns
    )
    r2_features = [feature for feature in all_features if feature not in segments.columns]
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    missing = sorted(set(all_features) - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing SENSOR_FACTORIAL features after R2 join: {missing}")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")
    target_support = config["scope"]["target_source_support"]
    if target_support["mode"] != "frozen_model_contract_target_safe_operational_fuel":
        raise RuntimeError("Unexpected SENSOR_FACTORIAL target-source support mode.")
    backend_params = dict(protocol["selected_params"])
    fit_count_this_run = 0
    support_rows: list[dict[str, Any]] = []

    for cell in supported_cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        key = f"{split_name}|{target_id}"
        target = m1_contract["targets"][target_id]
        if target["unit"] != "L" or target["channel"] != "fuel":
            raise RuntimeError(f"Non-fuel cell entered SENSOR_FACTORIAL: {key}")
        membership = pd.read_parquet(membership_paths[key])
        membership = membership.merge(
            data[
                [
                    "segment_id",
                    "fuel_direct_duration_share",
                    "fuel_maf_duration_share",
                ]
            ],
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        role_counts_before = membership["split_role"].value_counts().to_dict()
        role_counts_after = membership["split_role"].value_counts().to_dict()
        pure_direct = direct_support_mask(membership)
        support_rows.append(
            {
                "split_family": split_name,
                "target_id": target_id,
                **{
                    f"{role}_rows_before_direct_filter": int(
                        role_counts_before.get(role, 0)
                    )
                    for role in ("train", "validation", "calibration", "test")
                },
                **{
                    f"{role}_rows_after_direct_filter": int(
                        role_counts_after.get(role, 0)
                    )
                    for role in ("train", "validation", "calibration", "test")
                },
                **{
                    f"{role}_pure_direct_rows": int(
                        (
                            pure_direct
                            & membership["split_role"].eq(role)
                        ).sum()
                    )
                    for role in ("train", "validation", "calibration", "test")
                },
            }
        )
        train_membership = membership[membership["split_role"].eq("train")].copy()
        train_ids = train_membership["segment_id"].astype(str).tolist()
        target_table = target_dataset.to_table(
            columns=["segment_id", target["column"]],
            filter=ds.field("segment_id").isin(pa.array(train_ids)),
        ).to_pandas()
        train = (
            train_membership[["segment_id", "split_role"]]
            .merge(data, on="segment_id", how="left", validate="one_to_one")
            .merge(target_table, on="segment_id", how="left", validate="one_to_one")
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        if (
            len(train) < int(config["scope"]["minimum_train_rows"])
            or train[target["column"]].isna().any()
        ):
            raise RuntimeError(
                f"Target-safe train support below the frozen minimum for {key}: {len(train)}"
            )
        y_train = pd.to_numeric(train[target["column"]], errors="raise").to_numpy(
            float
        )
        train_hash = ordered_sha256(train["segment_id"])
        cell_systems = systems_by_target[target_id]
        for system in SYSTEMS:
            raw_features = cell_systems[system]
            raw = train[raw_features].apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            active_features = train_active_columns(raw)
            if not active_features:
                raise RuntimeError(f"No active train features for {key}/{system}")
            feature_hash = ordered_sha256(active_features)
            for seed in seeds:
                record_key = f"{key}|{system}|{seed}"
                existing = state["fit_records"].get(record_key)
                if existing and valid_model_record(existing):
                    continue
                model_path = (
                    output
                    / "models"
                    / f"{split_name}__{target_id}"
                    / system.replace("-", "_")
                    / f"seed_{seed}.joblib"
                )
                metadata_path = model_path.with_suffix(".json")
                fit_started = time.perf_counter()
                model = build_backbone(backend_params, seed)
                model.fit(raw[active_features], y_train)
                metadata = {
                    "schema_version": 1,
                    "experiment": "SENSOR_FACTORIAL",
                    "evidence_class": config["evidence_contract"]["evidence_class"],
                    "split_family": split_name,
                    "target_id": target_id,
                    "target_unit": target["unit"],
                    "system": system,
                    "seed": seed,
                    "train_rows": int(len(train)),
                    "train_segment_id_sha256": train_hash,
                    "raw_features": raw_features,
                    "active_features": active_features,
                    "active_feature_sha256": feature_hash,
                    "params": backend_params,
                    "fit_seconds": time.perf_counter() - fit_started,
                    "target_source_support_mode": target_support["mode"],
                    "target_source_features_excluded": sorted(
                        m1_contract["targets"][target_id][
                            "target_source_features_removed"
                        ]
                    ),
                    "validation_target_rows_used": 0,
                    "calibration_target_rows_used": 0,
                    "test_target_rows_used_before_fit_freeze": 0,
                }
                dump_model(model_path, {"model": model, "metadata": metadata})
                write_json(metadata_path, metadata)
                state["fit_records"][record_key] = {
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "model_sha256": sha256_file(model_path),
                    "metadata_sha256": sha256_file(metadata_path),
                }
                fit_count_this_run += 1
                write_json(state_path, state)
                print(
                    f"FIT {len(state['fit_records'])}/"
                    f"{len(supported_cells) * len(SYSTEMS) * len(seeds)} "
                    f"{record_key}",
                    flush=True,
                )

    expected_fits = len(supported_cells) * len(SYSTEMS) * len(seeds)
    if len(state["fit_records"]) != expected_fits:
        raise RuntimeError("Incomplete SENSOR_FACTORIAL fit grid.")
    invalid_models = [
        key
        for key, record in state["fit_records"].items()
        if not valid_model_record(record)
    ]
    if invalid_models:
        raise RuntimeError(f"Invalid SENSOR_FACTORIAL model records: {invalid_models}")
    model_paths = [
        Path(record[field])
        for record in state["fit_records"].values()
        for field in ("model_path", "metadata_path")
    ]
    fit_manifest_path = output / "fit_artifact_manifest.json"
    if state.get("fit_manifest"):
        fit_manifest_path = Path(state["fit_manifest"])
        if sha256_file(fit_manifest_path) != state["fit_manifest_sha256"]:
            raise RuntimeError("SENSOR_FACTORIAL fit manifest hash drift.")
        verified, mismatches = verify_manifest(load_json(fit_manifest_path))
        if not verified:
            raise RuntimeError(f"SENSOR_FACTORIAL fit artifact drift: {mismatches}")
    else:
        fit_manifest = artifact_manifest(model_paths)
        fit_manifest["fit_count"] = expected_fits
        fit_manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(fit_manifest_path, fit_manifest)
        state["fit_manifest"] = str(fit_manifest_path)
        state["fit_manifest_sha256"] = sha256_file(fit_manifest_path)
    state["status"] = "FITS_FROZEN"
    write_json(state_path, state)

    if state["test_gate"]["status"] == "CLOSED":
        verified, mismatches = verify_manifest(load_json(fit_manifest_path))
        if not verified:
            raise RuntimeError(f"Cannot open SENSOR_FACTORIAL test gate: {mismatches}")
        state["test_gate"] = {
            "status": "OPEN",
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "fit_manifest_sha256_at_open": sha256_file(fit_manifest_path),
            "historical_note": "RNEE test labels are fixed by MODEL_CONTRACT; "
            "this is an SENSOR_FACTORIAL one-pass reuse after its own fits were frozen.",
        }
        write_json(state_path, state)

    metric_rows: list[dict[str, Any]] = []
    ensemble_paths: dict[str, Path] = {}
    test_target_materializations_this_run = 0
    prediction_recomputes_this_run = 0
    for cell in supported_cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        key = f"{split_name}|{target_id}"
        target = m1_contract["targets"][target_id]
        membership = pd.read_parquet(membership_paths[key])
        membership = membership.merge(
            data[
                [
                    "segment_id",
                    "fuel_direct_duration_share",
                    "fuel_maf_duration_share",
                ]
            ],
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        test_membership = membership[membership["split_role"].eq("test")].copy()
        test_ids = test_membership["segment_id"].astype(str).tolist()
        input_path = output / "test_gate_inputs" / f"{split_name}__{target_id}.parquet"
        if not input_path.exists():
            target_table = target_dataset.to_table(
                columns=["segment_id", target["column"]],
                filter=ds.field("segment_id").isin(pa.array(test_ids)),
            ).to_pandas()
            test = (
                test_membership[["segment_id", "split_role"]]
                .merge(data, on="segment_id", how="left", validate="one_to_one")
                .merge(target_table, on="segment_id", how="left", validate="one_to_one")
                .sort_values("segment_id")
                .reset_index(drop=True)
            )
            if len(test) == 0 or test[target["column"]].isna().any():
                raise RuntimeError(f"Invalid target-safe SENSOR_FACTORIAL test support: {key}")
            input_columns = list(
                dict.fromkeys(
                    [
                        "segment_id",
                        "trip_uid",
                        "VehId",
                        "engine_type",
                        "fuel_direct_duration_share",
                        "fuel_maf_duration_share",
                        target["column"],
                    ]
                    + all_features
                )
            )
            write_parquet(input_path, test[input_columns])
            test_target_materializations_this_run += 1
        test = pd.read_parquet(input_path)
        y_test = pd.to_numeric(test[target["column"]], errors="raise").to_numpy(float)
        seed_frames: list[pd.DataFrame] = []
        for seed in seeds:
            prediction_key = f"{key}|{seed}"
            prediction_path = (
                output
                / "paired_test_predictions"
                / f"{split_name}__{target_id}"
                / f"seed_{seed}.parquet"
            )
            existing = state["prediction_records"].get(prediction_key)
            if not (
                existing
                and prediction_path.exists()
                and sha256_file(prediction_path) == existing["sha256"]
            ):
                frame = test[["segment_id", "trip_uid", "VehId"]].copy()
                frame["target_l"] = y_test
                for system in SYSTEMS:
                    record = state["fit_records"][f"{key}|{system}|{seed}"]
                    payload = load_model(Path(record["model_path"]))
                    active_features = payload["metadata"]["active_features"]
                    raw = test[active_features].apply(
                        pd.to_numeric, errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan)
                    frame[PREDICTION_COLUMNS[system]] = payload["model"].predict(raw)
                write_parquet(prediction_path, frame)
                state["prediction_records"][prediction_key] = {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                    "rows": int(len(frame)),
                    "segment_id_sha256": ordered_sha256(frame["segment_id"]),
                }
                prediction_recomputes_this_run += 1
                write_json(state_path, state)
            frame = pd.read_parquet(prediction_path)
            seed_frames.append(frame)
            for system in SYSTEMS:
                metrics = regression_metrics(
                    frame["target_l"].to_numpy(float),
                    frame[PREDICTION_COLUMNS[system]].to_numpy(float),
                )
                metric_rows.append(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "seed": seed,
                        "aggregation": "single_seed",
                        "system": system,
                        "rows": int(len(frame)),
                        **metrics,
                    }
                )
        ensemble = seed_frames[0][
            ["segment_id", "trip_uid", "VehId", "target_l"]
        ].copy()
        for system in SYSTEMS:
            values = np.column_stack(
                [frame[PREDICTION_COLUMNS[system]].to_numpy(float) for frame in seed_frames]
            )
            ensemble[PREDICTION_COLUMNS[system]] = values.mean(axis=1)
            ensemble[f"{PREDICTION_COLUMNS[system]}_seed_std"] = values.std(axis=1)
            metrics = regression_metrics(
                ensemble["target_l"].to_numpy(float),
                ensemble[PREDICTION_COLUMNS[system]].to_numpy(float),
            )
            metric_rows.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "seed": "ensemble",
                    "aggregation": "mean_prediction",
                    "system": system,
                    "rows": int(len(ensemble)),
                    **metrics,
                }
            )
        ensemble_path = (
            output / "ensemble_test_predictions" / f"{split_name}__{target_id}.parquet"
        )
        write_parquet(ensemble_path, ensemble)
        ensemble_paths[key] = ensemble_path

    metrics_path = output / "system_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    support_path = output / "target_source_support.csv"
    pd.DataFrame(support_rows).to_csv(support_path, index=False)

    cold_splits = list(config["scope"]["cold_splits"])
    inference = config["inference"]
    effect_frames: list[pd.DataFrame] = []
    bootstrap_diagnostics: dict[str, Any] = {}
    for target_index, target_id in enumerate(
        [config["scope"]["primary_target"], config["scope"]["supporting_target"]]
    ):
        frames = {
            split: pd.read_parquet(ensemble_paths[f"{split}|{target_id}"])
            for split in cold_splits
        }
        effect, diagnostics = vehicle_cluster_bootstrap(
            frames,
            replicates=int(inference["bootstrap_replicates"]),
            seed=int(inference["bootstrap_seed"]) + target_index,
            alpha=float(inference["simultaneous_family_alpha"]),
            sesoi=float(inference["relative_sesoi"]),
        )
        effect.insert(0, "target_id", target_id)
        effect_frames.append(effect)
        bootstrap_diagnostics[target_id] = diagnostics
    effects = pd.concat(effect_frames, ignore_index=True)
    floors = config["prospective_support_audit"]
    floor_passes: list[bool] = []
    for index, row in effects.iterrows():
        checks = {
            "minimum_vehicles": int(row["vehicle_count"])
            >= int(floors["minimum_vehicles"]),
            "minimum_trips": int(row["trip_count"]) >= int(floors["minimum_trips"]),
            "minimum_segments": int(row["segment_count"])
            >= int(floors["minimum_segments"]),
            "maximum_single_vehicle_segment_share": float(
                row["maximum_single_vehicle_segment_share"]
            )
            <= float(floors["maximum_single_vehicle_segment_share"]),
        }
        passed = all(checks.values())
        floor_passes.append(passed)
        effects.loc[index, "prospective_support_floor_pass"] = passed
        effects.loc[index, "prospective_support_failed_checks"] = "|".join(
            name for name, value in checks.items() if not value
        )
    effects_path = output / "cold_ood_factorial_effects.csv"
    effects.to_csv(effects_path, index=False)
    bootstrap_path = output / "bootstrap_diagnostics.json"
    write_json(bootstrap_path, bootstrap_diagnostics)

    reference_rows: list[dict[str, Any]] = []
    reference_split = config["scope"]["reference_split"]
    for target_id in (
        config["scope"]["primary_target"],
        config["scope"]["supporting_target"],
    ):
        frame = pd.read_parquet(ensemble_paths[f"{reference_split}|{target_id}"])
        mae = {
            system: float(
                np.abs(
                    frame["target_l"].to_numpy(float)
                    - frame[PREDICTION_COLUMNS[system]].to_numpy(float)
                ).mean()
            )
            for system in SYSTEMS
        }
        effect = relative_effects(mae)
        reference_rows.append(
            {
                "target_id": target_id,
                "split_family": reference_split,
                **{f"mae_{system.lower().replace('-', '_')}": mae[system] for system in SYSTEMS},
                **{name: float(np.asarray(value)) for name, value in effect.items()},
                **support_audit(frame, floors),
            }
        )
    reference_path = output / "random_reference_effects.json"
    write_json(reference_path, reference_rows)

    routes = [
        retrospective_route(effects, config["scope"]["primary_target"]),
        retrospective_route(effects, config["scope"]["supporting_target"]),
    ]
    routes_path = output / "retrospective_routes.json"
    write_json(routes_path, routes)

    gate_rows = [
        ("upstream_manifests_verified", all(result[0] for result in upstream_verification)),
        (
            "exact_four_systems",
            all(tuple(systems_by_target[target_id]) == SYSTEMS for target_id in target_ids),
        ),
        (
            "exact_system_feature_counts",
            all(
                [len(systems_by_target[target_id][name]) for name in SYSTEMS]
                == [8, 150, 20, 162]
                for target_id in target_ids
            ),
        ),
        ("zero_forbidden_feature_hits", not hits),
        (
            "all_fuel_target_source_features_excluded",
            not target_source_features.intersection(all_features),
        ),
        (
            "target_source_support_mode_frozen",
            target_support["mode"] == "frozen_model_contract_target_safe_operational_fuel",
        ),
        ("exact_ten_cells", len(supported_cells) == 10),
        ("exact_three_seeds", seeds == [20260720, 20260721, 20260722]),
        ("complete_fit_grid", len(state["fit_records"]) == expected_fits),
        ("fit_manifest_verified_before_test_gate", state["test_gate"]["status"] == "OPEN"),
        ("complete_prediction_grid", len(state["prediction_records"]) == len(supported_cells) * len(seeds)),
        ("cross_dataset_evidence_zero", config["evidence_contract"]["cross_dataset_evidence_used"] is False),
        ("information_freeze_status_unchanged", information_freeze_pointer["status"].endswith("M2_BLOCKED")),
        ("confirmation_claim_prohibited", config["evidence_contract"]["independent_confirmation_included"] is False),
    ]
    gate_path = output / "gate_checks.csv"
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in gate_rows
        ]
    ).to_csv(gate_path, index=False)
    if not all(passed for _, passed in gate_rows):
        raise RuntimeError(f"SENSOR_FACTORIAL gate failure: {gate_rows}")

    primary_route = next(
        route
        for route in routes
        if route["target_id"] == config["scope"]["primary_target"]
    )
    primary_effects = effects[
        effects["target_id"].eq(config["scope"]["primary_target"])
    ]
    prospective_primary_support_pass = bool(
        primary_effects["prospective_support_floor_pass"].all()
    )
    report_path = output / "SENSOR_FACTORIAL_RNEE_SENSOR_FACTORIAL_REPORT.md"
    table_lines = [
        "| Environment | Low gain | Low simultaneous CI | Low state | High gain | High state | Interaction | Interaction state |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for _, row in primary_effects.iterrows():
        table_lines.append(
            "| {environment} | {low:.2%} | [{lcb:.2%}, {ucb:.2%}] | {low_state} | "
            "{high:.2%} | {high_state} | {interaction:.2%} | {interaction_state} |".format(
                environment=row["environment"],
                low=row["low_road_gain_relative"],
                lcb=row["low_road_gain_relative_sim_lcb"],
                ucb=row["low_road_gain_relative_sim_ucb"],
                low_state=row["low_road_gain_relative_state"],
                high=row["high_road_gain_relative"],
                high_state=row["high_road_gain_relative_state"],
                interaction=row["sensor_interaction_relative"],
                interaction_state=row["sensor_interaction_relative_state"],
            )
        )
    report = "\n".join(
        [
            "# SENSOR_FACTORIAL RNEE-only sensor x road-semantics factorial",
            "",
            f"- Status: `PASS_SENSOR_FACTORIAL_RNEE_RETROSPECTIVE_FACTORIAL`",
            f"- Evidence class: `{config['evidence_contract']['evidence_class']}`",
            f"- Primary route: `{primary_route['route']}`",
            f"- Prospective INFORMATION_FREEZE support floors passed in all primary cold cells: `{prospective_primary_support_pass}`",
            f"- New fits: `{expected_fits}`; target: RNEE operational 60 s fuel L with all target-source channels excluded from predictors.",
            f"- Cross-dataset inputs: `0`.",
            "",
            "## Primary ICE result",
            "",
            *table_lines,
            "",
            "## Interpretation boundary",
            "",
            "This is an exploratory RNEE discovery branch. It does not modify "
            "MODEL_CONTRACT, does not advance INFORMATION_FREEZE, and cannot be called independent confirmation. "
            "The next RNEE-only step is authorized only if the pattern warrants fixed "
            "negative controls, support-domain and calibration analysis.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    immutable_paths = (
        model_paths
        + [Path(record["path"]) for record in state["prediction_records"].values()]
        + list(ensemble_paths.values())
        + [
            run_config_path,
            fit_manifest_path,
            metrics_path,
            support_path,
            effects_path,
            bootstrap_path,
            reference_path,
            routes_path,
            gate_path,
            report_path,
        ]
    )
    primary_manifest = artifact_manifest(immutable_paths)
    primary_manifest["experiment"] = "SENSOR_FACTORIAL"
    primary_manifest["evidence_class"] = config["evidence_contract"]["evidence_class"]
    write_json(primary_manifest_path, primary_manifest)
    summary = {
        "schema_version": 1,
        "experiment": "SENSOR_FACTORIAL",
        "stage": config["stage"],
        "status": "PASS_SENSOR_FACTORIAL_RNEE_RETROSPECTIVE_FACTORIAL",
        "evidence_class": config["evidence_contract"]["evidence_class"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "execution_backend": "local_cpu",
        "gpu_used": False,
        "fit_count": expected_fits,
        "fit_count_this_run": fit_count_this_run,
        "test_target_cell_materialization_count_this_run": test_target_materializations_this_run,
        "prediction_recompute_count_this_run": prediction_recomputes_this_run,
        "cell_count": len(supported_cells),
        "system_count": len(SYSTEMS),
        "seed_count": len(seeds),
        "bootstrap_replicates": int(inference["bootstrap_replicates"]),
        "primary_route": primary_route,
        "supporting_route": routes[1],
        "prospective_primary_support_pass": prospective_primary_support_pass,
        "cross_dataset_inputs": 0,
        "model_contract_modified": False,
        "information_freeze_status": information_freeze_pointer["status"],
        "confirmation_authorized": False,
        "report": str(report_path),
        "effects": str(effects_path),
        "metrics": str(metrics_path),
        "support": str(support_path),
        "routes": str(routes_path),
        "manifest": str(primary_manifest_path),
        "output_directory": str(output),
        "resume_noop_verified": False,
    }
    write_json(summary_path, summary)
    state["status"] = "PASS"
    state["completed_at_utc"] = summary["completed_at_utc"]
    write_json(state_path, state)
    latest_path = Path(config["output_root"]) / (
        f"{config['latest_prefix']}_latest_pointer.json"
    )
    latest = {
        "status": summary["status"],
        "evidence_class": summary["evidence_class"],
        "summary": str(summary_path),
        "report": str(report_path),
        "effects": str(effects_path),
        "metrics": str(metrics_path),
        "support": str(support_path),
        "routes": str(routes_path),
        "manifest": str(primary_manifest_path),
        "output_directory": str(output),
        "resume_noop_verified": False,
    }
    write_json(latest_path, latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
