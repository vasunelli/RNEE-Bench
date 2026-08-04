#!/usr/bin/env python3
"""Run MODEL_CONTRACT M2 validation-only backbone selection and a bounded smoke test.

The script fits candidate models on train rows and evaluates validation rows
only.  Calibration and test targets are never selected into a modeling frame.
The selected protocol is frozen for the later R0--R6 add/drop stage.
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(frame: pd.DataFrame, limit: int, seed: int, role: str) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.sort_values("segment_id").reset_index(drop=True)
    key = frame["segment_id"].astype(str) + f"|{seed}|{role}"
    hashes = key.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    return (
        frame.assign(_sample_hash=hashes)
        .sort_values("_sample_hash", kind="stable")
        .head(limit)
        .drop(columns="_sample_hash")
        .reset_index(drop=True)
    )


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float)
    absolute = np.abs(residual)
    denominator = float(np.square(y_true - np.mean(y_true)).sum())
    return {
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(np.square(residual).mean())),
        "medae": float(np.median(absolute)),
        "r2": float(1.0 - np.square(residual).sum() / denominator) if denominator > 0 else math.nan,
    }


def train_active_columns(frame: pd.DataFrame) -> list[str]:
    """Keep columns with at least two distinct observed train values."""
    active = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.nunique(dropna=True) >= 2:
            active.append(column)
    return active


def build_candidate(name: str, spec: dict[str, Any], seed: int) -> Any:
    params = dict(spec["params"])
    kind = spec["kind"]
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, **params)
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, **params)
    if kind == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unsupported candidate kind: {kind}")


def build_transparent_reference(alpha: float) -> Any:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def forbidden_hits(features: list[str], contract: dict[str, Any]) -> list[str]:
    forbidden = contract["forbidden_feature_contract"]
    exact = {value.lower() for value in forbidden["exact"]}
    tokens = [value.lower() for value in forbidden["tokens"]]
    hits = []
    for feature in features:
        lowered = feature.lower()
        if lowered in exact or any(token in lowered for token in tokens):
            hits.append(feature)
    return hits


def current_rss_gb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss / 1024**3)


def artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in sorted(set(paths), key=lambda value: str(value).lower())
        ],
    }


def verify_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append(str(path))
    return not mismatches, mismatches


def exclude_globally_protected_ids(
    development: pd.DataFrame, protected_ids: set[str]
) -> tuple[pd.DataFrame, int]:
    protected_mask = development["segment_id"].astype(str).isin(protected_ids)
    selected = development.loc[~protected_mask].copy()
    residual_overlap = int(selected["segment_id"].astype(str).isin(protected_ids).sum())
    return selected, residual_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/model_contract_m2_backbone_smoke.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m1_pointer_path = Path(config["m1_pointer"])
    segment_pointer_path = Path(config["segment_pointer"])
    m1_pointer = load_json(m1_pointer_path)
    segment_pointer = load_json(segment_pointer_path)
    m1_root = Path(m1_pointer["output_directory"])
    m1_summary = load_json(Path(m1_pointer["summary"]))
    m1_audit = load_json(Path(m1_pointer["experiment_audit_json"]))
    m1_primary_manifest = load_json(Path(m1_pointer["manifest"]))
    m1_release_manifest = load_json(Path(m1_pointer["release_manifest"]))
    feature_contract_path = Path(m1_pointer["feature_contract"])
    feature_contract = load_json(feature_contract_path)
    r2_path = m1_root / "r2_segment_features.parquet"
    if m1_summary["status"] != "PASS_MODEL_CONTRACT_M1_B1" or not m1_summary["resume_noop_verified"]:
        raise RuntimeError("MODEL_CONTRACT M1 is not a reproducible PASS release.")
    if m1_audit["overall_verdict"] != "pass":
        raise RuntimeError("MODEL_CONTRACT M1 integrity audit is not PASS.")
    m1_primary_verified, m1_primary_mismatches = verify_manifest(m1_primary_manifest)
    m1_release_verified, m1_release_mismatches = verify_manifest(m1_release_manifest)
    if not m1_primary_verified or not m1_release_verified:
        raise RuntimeError(
            "MODEL_CONTRACT M1 recursive manifest verification failed: "
            f"primary={m1_primary_mismatches}, release={m1_release_mismatches}"
        )
    if segment_pointer["status"] != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("Frozen primary SEGMENT_SPLIT segment release is not PASS.")

    supported_cells = config["primary_supported_cells"]
    if len(supported_cells) != 10:
        raise RuntimeError("M2 must use exactly the ten M1 primary-supported cells.")
    membership_paths = {
        f"{cell['split_family']}|{cell['target_id']}": m1_root
        / "target_memberships"
        / cell["split_family"]
        / cell["target_id"]
        / "model_membership.parquet"
        for cell in supported_cells
    }
    memberships = {
        key: pd.read_parquet(path)
        for key, path in membership_paths.items()
    }
    globally_protected_ids: dict[str, set[str]] = {}
    for target_id in sorted({cell["target_id"] for cell in supported_cells}):
        relevant = [
            memberships[f"{cell['split_family']}|{cell['target_id']}"]
            for cell in supported_cells
            if cell["target_id"] == target_id
        ]
        globally_protected_ids[target_id] = set().union(
            *[
                set(frame.loc[frame["split_role"].isin(["calibration", "test"]), "segment_id"].astype(str))
                for frame in relevant
            ]
        )
    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "m1_pointer_sha256": sha256_file(m1_pointer_path),
        "m1_release_manifest_sha256": sha256_file(Path(m1_pointer["release_manifest"])),
        "m1_feature_contract_sha256": sha256_file(feature_contract_path),
        "segment_pointer_sha256": sha256_file(segment_pointer_path),
        "segments_sha256": sha256_file(Path(segment_pointer["segments_all"])),
        "r2_sha256": sha256_file(r2_path),
        "membership_sha256": {key: sha256_file(path) for key, path in membership_paths.items()},
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "run_config.yaml").exists():
        shutil.copy2(args.config, output / "run_config.yaml")
    state_path = output / "m2_state.json"
    summary_path = output / "model_contract_m2_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if state["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        state = {"schema_version": 1, "input_fingerprint": fingerprint, "status": "PENDING"}
        write_json(state_path, state)

    if state["status"] == "PASS" and summary_path.exists() and primary_manifest_path.exists():
        manifest = load_json(primary_manifest_path)
        verified, mismatches = verify_manifest(manifest)
        resume_path = output / "resume_check.json"
        write_json(
            resume_path,
            {
                "status": "PASS" if verified else "FAIL",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "artifact_count": len(manifest["artifacts"]),
                "mismatches": mismatches,
                "refit_count": 0,
            },
        )
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(resume_path)
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest([primary_manifest_path, summary_path, resume_path])
        release["scope"] = "post-resume control files; immutable smoke outputs are enumerated by artifact_manifest.json"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
        if latest_path.exists():
            latest = load_json(latest_path)
            latest["release_manifest"] = str(release_manifest_path)
            write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    candidate_names = list(config["candidate_models"])
    if not 1 <= len(candidate_names) <= 3:
        raise RuntimeError("M2 candidate count must be between one and three.")
    target_contracts = feature_contract["targets"]
    road_features = feature_contract["feature_groups"]["R6_FULL_SEMANTICS"]
    target_features: dict[str, list[str]] = {}
    for target_id in sorted({cell["target_id"] for cell in supported_cells}):
        target_features[target_id] = list(
            dict.fromkeys(target_contracts[target_id]["r0_full_obd"] + road_features)
        )
    all_features = sorted(set(feature for values in target_features.values() for feature in values))
    reference_features = feature_contract["feature_groups"]["R0_LOW_SENSOR"]
    hits = forbidden_hits(all_features + reference_features, feature_contract)
    if hits:
        raise RuntimeError(f"Forbidden feature hits in M2: {hits}")
    for target_id, features in target_features.items():
        removed = set(target_contracts[target_id]["target_source_features_removed"])
        if removed.intersection(features):
            raise RuntimeError(f"Target-source leakage restored for {target_id}")

    segment_schema = set(pq.read_schema(segment_pointer["segments_all"]).names)
    segment_columns = [
        "segment_id",
        "engine_type",
        *sorted(set(all_features + reference_features).intersection(segment_schema)),
    ]
    segment_columns = list(dict.fromkeys(segment_columns))
    segments = pd.read_parquet(segment_pointer["segments_all"], columns=segment_columns)
    r2_features = [feature for feature in all_features if feature not in segments.columns]
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")
    missing_features = sorted(set(all_features + reference_features) - set(data.columns))
    if missing_features:
        raise RuntimeError(f"M2 features absent after R2 join: {missing_features}")

    metric_records: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    peak_rss = current_rss_gb()
    smoke_seed = int(config["smoke_seed"])
    for cell in supported_cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        target = target_contracts[target_id]
        if target["unit"] != "L" or target["channel"] != "fuel":
            raise RuntimeError(f"Unsupported primary M2 target: {target_id}")
        membership = memberships[f"{split_name}|{target_id}"]
        role_counts = membership["split_role"].value_counts().to_dict()
        development = membership[membership["split_role"].isin(["train", "validation"])].copy()
        protected = globally_protected_ids[target_id]
        selected_membership, protected_id_overlap_count = exclude_globally_protected_ids(
            development, protected
        )
        if set(selected_membership["split_role"].unique()) != {"train", "validation"}:
            raise RuntimeError(f"Missing train/validation role: {split_name}/{target_id}")
        selected_ids = selected_membership["segment_id"].astype(str).tolist()
        target_table = target_dataset.to_table(
            columns=["segment_id", target["column"]],
            filter=ds.field("segment_id").isin(pa.array(selected_ids)),
        ).to_pandas()
        if set(target_table["segment_id"].astype(str)) != set(selected_ids) or target_table["segment_id"].duplicated().any():
            raise RuntimeError(f"Target materialization escaped safe development IDs: {split_name}/{target_id}")
        modeling = (
            selected_membership
            .merge(data, on="segment_id", how="left", validate="one_to_one")
            .merge(target_table, on="segment_id", how="left", validate="one_to_one")
        )
        if modeling[target["column"]].isna().any():
            raise RuntimeError(f"Missing target after membership join: {split_name}/{target_id}")
        train = deterministic_sample(
            modeling[modeling["split_role"].eq("train")],
            int(config["smoke_max_train_rows"]),
            smoke_seed,
            "train",
        )
        validation = deterministic_sample(
            modeling[modeling["split_role"].eq("validation")],
            int(config["smoke_max_validation_rows"]),
            smoke_seed,
            "validation",
        )
        if len(train) < int(config["minimum_smoke_train_rows"]) or len(validation) < int(config["minimum_smoke_validation_rows"]):
            raise RuntimeError(f"Insufficient smoke support: {split_name}/{target_id}")
        train_iqr = float(train[target["column"]].quantile(0.75) - train[target["column"]].quantile(0.25))
        if not np.isfinite(train_iqr) or train_iqr <= 0:
            raise RuntimeError(f"Non-positive train target IQR: {split_name}/{target_id}")
        features = target_features[target_id]
        x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_validation = validation[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        active_features = train_active_columns(x_train)
        if not active_features:
            raise RuntimeError(f"No active train-derived features: {split_name}/{target_id}")
        x_train = x_train[active_features]
        x_validation = x_validation[active_features]
        y_train = pd.to_numeric(train[target["column"]], errors="raise").to_numpy(float)
        y_validation = pd.to_numeric(validation[target["column"]], errors="raise").to_numpy(float)
        for candidate_name, spec in config["candidate_models"].items():
            fit_started = time.perf_counter()
            model = build_candidate(candidate_name, spec, smoke_seed)
            model.fit(x_train, y_train)
            prediction = np.asarray(model.predict(x_validation), dtype=float)
            metrics = regression_metrics(y_validation, prediction)
            metric_records.append(
                {
                    "system_role": "candidate",
                    "model": candidate_name,
                    "kind": spec["kind"],
                    "backend": spec["backend"],
                    "split_family": split_name,
                    "target_id": target_id,
                    "unit": target["unit"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "raw_feature_count": int(len(features)),
                    "feature_count": int(len(active_features)),
                    "train_target_iqr": train_iqr,
                    **metrics,
                    "normalized_mae_by_train_iqr": metrics["mae"] / train_iqr,
                    "fit_predict_seconds": time.perf_counter() - fit_started,
                    "seed": smoke_seed,
                }
            )
            peak_rss = max(peak_rss, current_rss_gb())
        reference_started = time.perf_counter()
        reference = build_transparent_reference(float(config["transparent_reference"]["alpha"]))
        ref_x_train = train[reference_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        ref_x_validation = validation[reference_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        active_reference_features = train_active_columns(ref_x_train)
        if not active_reference_features:
            raise RuntimeError(f"No active transparent-reference features: {split_name}/{target_id}")
        ref_x_train = ref_x_train[active_reference_features]
        ref_x_validation = ref_x_validation[active_reference_features]
        reference.fit(ref_x_train, y_train)
        reference_prediction = np.asarray(reference.predict(ref_x_validation), dtype=float)
        reference_metrics = regression_metrics(y_validation, reference_prediction)
        metric_records.append(
            {
                "system_role": "transparent_reference",
                "model": config["transparent_reference"]["name"],
                "kind": "ridge_standardized_train_median",
                "backend": "sklearn_cpu",
                "split_family": split_name,
                "target_id": target_id,
                "unit": target["unit"],
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "raw_feature_count": int(len(reference_features)),
                "feature_count": int(len(active_reference_features)),
                "train_target_iqr": train_iqr,
                **reference_metrics,
                "normalized_mae_by_train_iqr": reference_metrics["mae"] / train_iqr,
                "fit_predict_seconds": time.perf_counter() - reference_started,
                "seed": smoke_seed,
            }
        )
        role_records.append(
            {
                "split_family": split_name,
                "target_id": target_id,
                "membership_train_rows": int(role_counts.get("train", 0)),
                "membership_validation_rows": int(role_counts.get("validation", 0)),
                "membership_calibration_rows": int(role_counts.get("calibration", 0)),
                "membership_test_rows": int(role_counts.get("test", 0)),
                "development_rows_before_global_protection": int(len(development)),
                "globally_protected_rows_excluded": int(len(development) - len(selected_membership)),
                "protected_id_overlap_count": protected_id_overlap_count,
                "safe_development_rows_materialized": int(len(selected_membership)),
                "smoke_train_rows_used": int(len(train)),
                "smoke_validation_rows_used": int(len(validation)),
                "calibration_target_rows_used": 0,
                "test_target_rows_used": 0,
            }
        )
        peak_rss = max(peak_rss, current_rss_gb())

    metrics = pd.DataFrame(metric_records)
    role_audit = pd.DataFrame(role_records)
    candidate_metrics = metrics[metrics["system_role"].eq("candidate")]
    candidate_summary = (
        candidate_metrics.groupby(["model", "kind", "backend"], as_index=False)
        .agg(
            completed_cells=("normalized_mae_by_train_iqr", "size"),
            mean_normalized_mae=("normalized_mae_by_train_iqr", "mean"),
            median_normalized_mae=("normalized_mae_by_train_iqr", "median"),
            mean_validation_mae_L=("mae", "mean"),
            total_fit_predict_seconds=("fit_predict_seconds", "sum"),
        )
        .sort_values(["mean_normalized_mae", "median_normalized_mae", "total_fit_predict_seconds", "model"])
        .reset_index(drop=True)
    )
    candidate_summary["selection_rank"] = np.arange(1, len(candidate_summary) + 1)
    selected_name = str(candidate_summary.iloc[0]["model"])
    selected_spec = config["candidate_models"][selected_name]
    expected_cells = len(supported_cells)

    metrics_path = output / "validation_metrics.csv"
    candidate_summary_path = output / "candidate_summary.csv"
    role_audit_path = output / "role_access_audit.csv"
    metrics.to_csv(metrics_path, index=False)
    candidate_summary.to_csv(candidate_summary_path, index=False)
    role_audit.to_csv(role_audit_path, index=False)
    protocol = {
        "schema_version": 1,
        "experiment": "MODEL_CONTRACT",
        "stage": "M2_frozen_backbone_protocol",
        "selection_data": "train fitting plus validation metrics only",
        "cross_split_protection": "for each target, exclude from M2 every segment assigned calibration or test in any of the five primary split families before materializing target values",
        "calibration_used_for_selection": False,
        "test_used_for_selection": False,
        "selected_backbone": selected_name,
        "selected_kind": selected_spec["kind"],
        "selected_backend_for_smoke": selected_spec["backend"],
        "selected_params": selected_spec["params"],
        "selection_feature_set": "target-specific R0_FULL_OBD plus R6_FULL_SEMANTICS",
        "train_only_variance_rule": "within each target/split/feature variant, retain only columns with at least two distinct observed train values; validation/calibration/test never affect the filter",
        "selection_criterion": config["selection_criterion"],
        "primary_cells": supported_cells,
        "m3_seeds": config["m3_seeds"],
        "m3_train_cap": None,
        "m3_validation_selection": "none; protocol already frozen",
        "m3_calibration_role": "held out from fitting and selection",
        "m3_test_role": "evaluate once only after all add/drop fits are complete",
        "metrics": ["MAE", "RMSE", "MedAE", "R2"],
        "primary_metric": "MAE",
        "transparent_reference": config["transparent_reference"],
        "feature_contract_path": str(feature_contract_path),
        "feature_contract_sha256": sha256_file(feature_contract_path),
        "target_feature_counts": {key: len(value) for key, value in target_features.items()},
        "forbidden_feature_hits": hits,
        "target_source_leakage_hits": [],
    }
    protocol_path = output / "frozen_backbone_protocol.json"
    write_json(protocol_path, protocol)

    checks = {
        "m1_reproducible_pass": "PASS",
        "m1_primary_manifest_recursive_verification": "PASS" if m1_primary_verified else "FAIL",
        "m1_release_manifest_recursive_verification": "PASS" if m1_release_verified else "FAIL",
        "exact_ten_primary_cells": "PASS" if expected_cells == 10 else "FAIL",
        "candidate_count_at_most_three": "PASS" if len(candidate_names) <= 3 else "FAIL",
        "all_candidate_cells_completed": "PASS"
        if candidate_summary["completed_cells"].eq(expected_cells).all()
        else "FAIL",
        "validation_metrics_finite": "PASS"
        if np.isfinite(candidate_metrics[["mae", "rmse", "medae", "r2", "normalized_mae_by_train_iqr"]].to_numpy(float)).all()
        else "FAIL",
        "calibration_target_rows_used_zero": "PASS"
        if role_audit["calibration_target_rows_used"].sum() == 0
        else "FAIL",
        "test_target_rows_used_zero": "PASS" if role_audit["test_target_rows_used"].sum() == 0 else "FAIL",
        "cross_split_calibration_test_protection": "PASS"
        if role_audit["globally_protected_rows_excluded"].gt(0).all()
        and role_audit["protected_id_overlap_count"].eq(0).all()
        else "FAIL",
        "safe_target_materialization_only": "PASS"
        if role_audit["safe_development_rows_materialized"].ge(
            role_audit["smoke_train_rows_used"] + role_audit["smoke_validation_rows_used"]
        ).all()
        else "FAIL",
        "zero_forbidden_feature_hits": "PASS" if not hits else "FAIL",
        "zero_target_source_leakage_hits": "PASS",
        "target_units_separate": "PASS" if set(metrics["unit"]) == {"L"} else "FAIL",
        "transparent_reference_not_mechanistic": "PASS"
        if config["transparent_reference"]["claim_status"] == "STATISTICAL_REFERENCE_NOT_MECHANISTIC"
        else "FAIL",
        "cpu_backend_frozen": "PASS"
        if set(candidate_summary["backend"]) == {"cpu"}
        else "FAIL",
        "memory_ceiling": "PASS" if peak_rss <= float(config["memory_ceiling_gb"]) else "FAIL",
    }
    checks_path = output / "m2_gate_checks.csv"
    pd.DataFrame([{"check": key, "status": value} for key, value in checks.items()]).to_csv(checks_path, index=False)
    passed = "FAIL" not in checks.values()
    summary = {
        "experiment": "MODEL_CONTRACT",
        "stage": "M2_validation_only_backbone_selection_smoke",
        "status": "PASS_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE"
        if passed
        else "FAIL_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE",
        "checks": checks,
        "candidate_count": len(candidate_names),
        "primary_cell_count": expected_cells,
        "candidate_fit_count": int(len(candidate_metrics)),
        "transparent_reference_fit_count": int((metrics["system_role"] == "transparent_reference").sum()),
        "selected_backbone": selected_name,
        "selected_mean_normalized_validation_mae": float(candidate_summary.iloc[0]["mean_normalized_mae"]),
        "test_target_rows_used": int(role_audit["test_target_rows_used"].sum()),
        "calibration_target_rows_used": int(role_audit["calibration_target_rows_used"].sum()),
        "protected_id_overlap_count": int(role_audit["protected_id_overlap_count"].sum()),
        "test_evaluation_performed": False,
        "gpu_used": False,
        "execution_backend": "local_cpu",
        "peak_process_rss_gb": peak_rss,
        "elapsed_seconds": time.perf_counter() - started,
        "resume_noop_verified": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output),
    }
    write_json(summary_path, summary)
    report_path = output / "MODEL_CONTRACT_M2_GATE.md"
    table = candidate_summary[["selection_rank", "model", "mean_normalized_mae", "median_normalized_mae", "mean_validation_mae_L", "total_fit_predict_seconds"]]
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    report_path.write_text(
        "\n".join(
            [
                "# MODEL_CONTRACT M2 validation-only backbone-selection smoke gate",
                "",
                f"- Status: `{summary['status']}`",
                f"- Selected backbone: `{selected_name}`",
                f"- Primary cells: {expected_cells}",
                f"- Candidate/reference fits: {len(candidate_metrics)} / {summary['transparent_reference_fit_count']}",
                "- Calibration target rows used: 0",
                "- Test target rows used: 0",
                "- Test evaluation performed: no",
                f"- Peak process RSS: {peak_rss:.3f} GB",
                "- Execution backend: local CPU; the local GPU was not assigned for this bounded smoke stage.",
                "",
                "## Validation-only candidate ranking",
                "",
                header,
                separator,
                *rows,
                "",
                "## Decision",
                "",
                "The selected protocol is frozen for M3. These validation metrics are pipeline-selection evidence only and must not be reported as final test performance or road-semantics gain.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state["status"] = "PASS" if passed else "FAIL"
    state["selected_backbone"] = selected_name
    write_json(state_path, state)
    primary_paths = [
        output / "run_config.yaml",
        state_path,
        metrics_path,
        candidate_summary_path,
        role_audit_path,
        protocol_path,
        checks_path,
        report_path,
    ]
    write_json(primary_manifest_path, artifact_manifest(primary_paths))
    latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
    write_json(
        latest_path,
        {
            "status": summary["status"],
            "summary": str(summary_path),
            "report": str(report_path),
            "protocol": str(protocol_path),
            "metrics": str(metrics_path),
            "candidate_summary": str(candidate_summary_path),
            "role_access_audit": str(role_audit_path),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
