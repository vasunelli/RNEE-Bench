#!/usr/bin/env python3
"""Run the frozen MODEL_CONTRACT M3 primary R0--R6 add/drop grid.

The runner has two irreversible phases.  It first fits and serializes every
prespecified train-only model.  Only after the complete fit manifest verifies
does it materialize each cell's test target once and write wide, segment-level
paired predictions.  Validation and calibration targets are never read.
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


def ordered_id_sha256(values: pd.Series | list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": [
            {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
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
    active = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.nunique(dropna=True) >= 2:
            active.append(column)
    return active


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


def variant_feature_sets(contract: dict[str, Any], target_id: str) -> dict[str, list[str]]:
    target = contract["targets"][target_id]
    r0 = list(target["r0_full_obd"])
    group_names = [
        "R1_HIERARCHY_REGULATION",
        "R2_PHYSICAL_ACCESS",
        "R3_TOPOLOGY",
        "R4_TRAFFIC_PLACE_OBJECTS",
        "R5_BUILT_ENVIRONMENT",
    ]
    groups = {name: list(contract["feature_groups"][name]) for name in group_names}
    variants: dict[str, list[str]] = {"R0": r0}
    for index, name in enumerate(group_names, start=1):
        variants[f"R0_PLUS_R{index}"] = list(dict.fromkeys(r0 + groups[name]))
    r6 = list(dict.fromkeys(r0 + contract["feature_groups"]["R6_FULL_SEMANTICS"]))
    variants["R6"] = r6
    for index, name in enumerate(group_names, start=1):
        removed = set(groups[name])
        variants[f"R6_MINUS_R{index}"] = [feature for feature in r6 if feature not in removed]
    return variants


def build_backbone(params: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=seed, **params)


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


def current_rss_gb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**3)
    except ImportError:
        return None


def cell_key(split_family: str, target_id: str) -> str:
    return f"{split_family}__{target_id}"


def fit_key(split_family: str, target_id: str, variant: str, seed: int) -> str:
    return f"{cell_key(split_family, target_id)}__{variant}__seed_{seed}"


def reference_key(split_family: str, target_id: str) -> str:
    return f"{cell_key(split_family, target_id)}__TRANSPARENT_REFERENCE"


def dump_model(path: Path, payload: dict[str, Any]) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def load_model(path: Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def validate_model_record(record: dict[str, Any]) -> bool:
    model_path = Path(record["model_path"])
    metadata_path = Path(record["metadata_path"])
    return (
        model_path.exists()
        and metadata_path.exists()
        and sha256_file(model_path) == record["model_sha256"]
        and sha256_file(metadata_path) == record["metadata_sha256"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/model_contract_m3_add_drop.yaml"),
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
    m1_pointer = load_json(m1_pointer_path)
    m2_pointer = load_json(m2_pointer_path)
    segment_pointer = load_json(segment_pointer_path)
    m1_summary = load_json(Path(m1_pointer["summary"]))
    m2_summary = load_json(Path(m2_pointer["summary"]))
    m1_audit = load_json(Path(m1_pointer["experiment_audit_json"]))
    m2_audit = load_json(Path(m2_pointer["experiment_audit_json"]))
    m1_primary = load_json(Path(m1_pointer["manifest"]))
    m1_release = load_json(Path(m1_pointer["release_manifest"]))
    m2_primary = load_json(Path(m2_pointer["manifest"]))
    m2_release = load_json(Path(m2_pointer["release_manifest"]))
    feature_contract_path = Path(m1_pointer["feature_contract"])
    feature_contract = load_json(feature_contract_path)
    protocol_path = Path(m2_pointer["protocol"])
    protocol = load_json(protocol_path)
    m1_root = Path(m1_pointer["output_directory"])
    r2_path = m1_root / "r2_segment_features.parquet"

    upstream_manifests = [m1_primary, m1_release, m2_primary, m2_release]
    upstream_results = [verify_manifest(manifest) for manifest in upstream_manifests]
    if not all(result[0] for result in upstream_results):
        raise RuntimeError(f"Recursive upstream manifest verification failed: {upstream_results}")
    if m1_summary["status"] != "PASS_MODEL_CONTRACT_M1_B1" or not m1_summary["resume_noop_verified"]:
        raise RuntimeError("MODEL_CONTRACT M1 is not a reproducible PASS release.")
    if m2_summary["status"] != "PASS_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE" or not m2_summary["resume_noop_verified"]:
        raise RuntimeError("MODEL_CONTRACT M2 is not a reproducible PASS release.")
    if m1_audit["overall_verdict"] != "pass" or m2_audit["overall_verdict"] != "pass":
        raise RuntimeError("An upstream independent integrity audit is not PASS.")
    if segment_pointer["status"] != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("Frozen primary SEGMENT_SPLIT segment release is not PASS.")
    if protocol["selected_backbone"] != "hist_gradient_boosting_l1":
        raise RuntimeError("Frozen M2 backbone is not hist_gradient_boosting_l1.")

    supported_cells = protocol["primary_cells"]
    seeds = [int(value) for value in protocol["m3_seeds"]]
    if len(supported_cells) != 10 or seeds != [20260720, 20260721, 20260722]:
        raise RuntimeError("M3 scope must remain the frozen ten cells and three seeds.")
    membership_paths = {
        f"{cell['split_family']}|{cell['target_id']}": m1_root
        / "target_memberships"
        / cell["split_family"]
        / cell["target_id"]
        / "model_membership.parquet"
        for cell in supported_cells
    }
    memberships = {key: pd.read_parquet(path) for key, path in membership_paths.items()}

    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "m1_pointer_sha256": sha256_file(m1_pointer_path),
        "m1_primary_manifest_sha256": sha256_file(Path(m1_pointer["manifest"])),
        "m1_release_manifest_sha256": sha256_file(Path(m1_pointer["release_manifest"])),
        "m1_audit_sha256": sha256_file(Path(m1_pointer["experiment_audit_json"])),
        "m2_pointer_sha256": sha256_file(m2_pointer_path),
        "m2_primary_manifest_sha256": sha256_file(Path(m2_pointer["manifest"])),
        "m2_release_manifest_sha256": sha256_file(Path(m2_pointer["release_manifest"])),
        "m2_audit_sha256": sha256_file(Path(m2_pointer["experiment_audit_json"])),
        "protocol_sha256": sha256_file(protocol_path),
        "feature_contract_sha256": sha256_file(feature_contract_path),
        "segment_pointer_sha256": sha256_file(segment_pointer_path),
        "segments_sha256": sha256_file(Path(segment_pointer["segments_all"])),
        "r2_sha256": sha256_file(r2_path),
        "membership_sha256": {key: sha256_file(path) for key, path in membership_paths.items()},
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
    state_path = output / "m3_state.json"
    summary_path = output / "model_contract_m3_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if state["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        state = {
            "schema_version": 1,
            "status": "FITTING",
            "input_fingerprint": fingerprint,
            "fit_records": {},
            "reference_records": {},
            "test_gate": {"status": "CLOSED", "inputs": {}, "predictions": {}, "references": {}},
            "peak_process_rss_gb": current_rss_gb(),
        }
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
                "test_target_rematerialization_count": 0,
                "test_prediction_recompute_count": 0,
            },
        )
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(resume_path)
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest([primary_manifest_path, summary_path, resume_path])
        release["scope"] = "post-resume controls; immutable M3 artifacts are enumerated by artifact_manifest.json"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
        latest = load_json(latest_path) if latest_path.exists() else {}
        latest["release_manifest"] = str(release_manifest_path)
        write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    target_ids = sorted({cell["target_id"] for cell in supported_cells})
    variants_by_target = {target_id: variant_feature_sets(feature_contract, target_id) for target_id in target_ids}
    expected_variants = [
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
    if any(list(variants) != expected_variants for variants in variants_by_target.values()):
        raise RuntimeError("Unexpected or reordered M3 variant grid.")
    all_features = sorted(
        set(
            feature
            for target_variants in variants_by_target.values()
            for features in target_variants.values()
            for feature in features
        )
    )
    reference_features = list(feature_contract["feature_groups"]["R0_LOW_SENSOR"])
    hits = forbidden_hits(all_features + reference_features, feature_contract)
    if hits:
        raise RuntimeError(f"Forbidden feature hits in M3: {hits}")
    for target_id, target_variants in variants_by_target.items():
        removed = set(feature_contract["targets"][target_id]["target_source_features_removed"])
        restored = sorted(removed.intersection(feature for values in target_variants.values() for feature in values))
        if restored:
            raise RuntimeError(f"Target-source leakage restored for {target_id}: {restored}")

    segment_schema = set(pq.read_schema(segment_pointer["segments_all"]).names)
    identity_columns = ["segment_id", "trip_uid", "VehId", "engine_type"]
    segment_columns = list(
        dict.fromkeys(identity_columns + sorted(set(all_features + reference_features).intersection(segment_schema)))
    )
    segments = pd.read_parquet(segment_pointer["segments_all"], columns=segment_columns)
    r2_features = [feature for feature in all_features if feature not in segments.columns]
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    missing_features = sorted(set(all_features + reference_features) - set(data.columns))
    if missing_features:
        raise RuntimeError(f"M3 features absent after R2 join: {missing_features}")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")
    backend_params = dict(protocol["selected_params"])
    fit_count_this_run = 0
    reference_fit_count_this_run = 0
    train_role_records: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        key = f"{split_name}|{target_id}"
        target = feature_contract["targets"][target_id]
        if target["unit"] != "L" or target["channel"] != "fuel":
            raise RuntimeError(f"Unsupported primary M3 target: {target_id}")
        membership = memberships[key]
        role_counts = membership["split_role"].value_counts().to_dict()
        train_membership = membership[membership["split_role"].eq("train")].copy()
        train_ids = train_membership["segment_id"].astype(str).tolist()
        target_table = target_dataset.to_table(
            columns=["segment_id", target["column"]],
            filter=ds.field("segment_id").isin(pa.array(train_ids)),
        ).to_pandas()
        if set(target_table["segment_id"].astype(str)) != set(train_ids) or target_table["segment_id"].duplicated().any():
            raise RuntimeError(f"Train target materialization escaped its role: {key}")
        train = (
            train_membership.merge(data, on="segment_id", how="left", validate="one_to_one")
            .merge(target_table, on="segment_id", how="left", validate="one_to_one")
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        if len(train) < int(config["minimum_train_rows"]) or train[target["column"]].isna().any():
            raise RuntimeError(f"Invalid train support for {key}")
        y_train = pd.to_numeric(train[target["column"]], errors="raise").to_numpy(float)
        train_hash = ordered_id_sha256(train["segment_id"])
        train_iqr = float(np.quantile(y_train, 0.75) - np.quantile(y_train, 0.25))
        if not np.isfinite(train_iqr) or train_iqr <= 0:
            raise RuntimeError(f"Non-positive train target IQR for {key}")

        for variant, features in variants_by_target[target_id].items():
            x_train_raw = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            active_features = train_active_columns(x_train_raw)
            if not active_features:
                raise RuntimeError(f"No active train-derived features for {key}/{variant}")
            x_train = x_train_raw[active_features]
            feature_hash = ordered_id_sha256(active_features)
            for seed in seeds:
                record_key = fit_key(split_name, target_id, variant, seed)
                existing = state["fit_records"].get(record_key)
                if existing and validate_model_record(existing):
                    continue
                model_path = output / "models" / cell_key(split_name, target_id) / variant / f"seed_{seed}.joblib"
                metadata_path = model_path.with_suffix(".json")
                fit_started = time.perf_counter()
                model = build_backbone(backend_params, seed)
                model.fit(x_train, y_train)
                metadata = {
                    "schema_version": 1,
                    "system_role": "frozen_backbone",
                    "backbone": protocol["selected_backbone"],
                    "split_family": split_name,
                    "target_id": target_id,
                    "unit": target["unit"],
                    "variant": variant,
                    "seed": seed,
                    "train_rows": int(len(train)),
                    "train_segment_id_sha256": train_hash,
                    "train_target_iqr": train_iqr,
                    "raw_features": features,
                    "active_features": active_features,
                    "active_feature_sha256": feature_hash,
                    "params": backend_params,
                    "fit_seconds": time.perf_counter() - fit_started,
                    "validation_target_rows_used": 0,
                    "calibration_target_rows_used": 0,
                    "test_target_rows_used": 0,
                }
                dump_model(model_path, {"model": model, "metadata": metadata})
                write_json(metadata_path, metadata)
                state["fit_records"][record_key] = {
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "model_sha256": sha256_file(model_path),
                    "metadata_sha256": sha256_file(metadata_path),
                    "train_rows": int(len(train)),
                    "train_segment_id_sha256": train_hash,
                    "active_feature_sha256": feature_hash,
                }
                fit_count_this_run += 1
                rss = current_rss_gb()
                if rss is not None:
                    previous = state.get("peak_process_rss_gb") or 0.0
                    state["peak_process_rss_gb"] = max(float(previous), rss)
                write_json(state_path, state)
                print(f"FIT {len(state['fit_records'])}/360 {record_key}", flush=True)

        ref_key = reference_key(split_name, target_id)
        ref_existing = state["reference_records"].get(ref_key)
        if not (ref_existing and validate_model_record(ref_existing)):
            ref_raw = train[reference_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            ref_active = train_active_columns(ref_raw)
            reference = build_transparent_reference(float(protocol["transparent_reference"]["alpha"]))
            reference_started = time.perf_counter()
            reference.fit(ref_raw[ref_active], y_train)
            ref_model_path = output / "reference_models" / cell_key(split_name, target_id) / "transparent_reference.joblib"
            ref_metadata_path = ref_model_path.with_suffix(".json")
            ref_metadata = {
                "schema_version": 1,
                "system_role": "transparent_reference",
                "claim_status": protocol["transparent_reference"]["claim_status"],
                "split_family": split_name,
                "target_id": target_id,
                "unit": target["unit"],
                "train_rows": int(len(train)),
                "train_segment_id_sha256": train_hash,
                "raw_features": reference_features,
                "active_features": ref_active,
                "fit_seconds": time.perf_counter() - reference_started,
                "validation_target_rows_used": 0,
                "calibration_target_rows_used": 0,
                "test_target_rows_used": 0,
            }
            dump_model(ref_model_path, {"model": reference, "metadata": ref_metadata})
            write_json(ref_metadata_path, ref_metadata)
            state["reference_records"][ref_key] = {
                "model_path": str(ref_model_path),
                "metadata_path": str(ref_metadata_path),
                "model_sha256": sha256_file(ref_model_path),
                "metadata_sha256": sha256_file(ref_metadata_path),
                "train_rows": int(len(train)),
                "train_segment_id_sha256": train_hash,
            }
            reference_fit_count_this_run += 1
            write_json(state_path, state)
            print(f"REFERENCE {len(state['reference_records'])}/10 {ref_key}", flush=True)

        train_role_records.append(
            {
                "split_family": split_name,
                "target_id": target_id,
                "membership_train_rows": int(role_counts.get("train", 0)),
                "membership_validation_rows": int(role_counts.get("validation", 0)),
                "membership_calibration_rows": int(role_counts.get("calibration", 0)),
                "membership_test_rows": int(role_counts.get("test", 0)),
                "train_target_rows_used": int(len(train)),
                "train_segment_id_sha256": train_hash,
                "validation_target_rows_used": 0,
                "calibration_target_rows_used": 0,
                "test_target_rows_used_before_gate": 0,
            }
        )
        print(f"CELL TRAIN COMPLETE {cell_index}/10 {key}", flush=True)

    expected_fit_count = len(supported_cells) * len(expected_variants) * len(seeds)
    expected_reference_count = len(supported_cells)
    if len(state["fit_records"]) != expected_fit_count or len(state["reference_records"]) != expected_reference_count:
        raise RuntimeError("The complete M3 fit grid is not frozen.")
    invalid_fit_records = [key for key, value in state["fit_records"].items() if not validate_model_record(value)]
    invalid_reference_records = [key for key, value in state["reference_records"].items() if not validate_model_record(value)]
    if invalid_fit_records or invalid_reference_records:
        raise RuntimeError(f"Frozen fit artifact mismatch: {invalid_fit_records + invalid_reference_records}")
    fit_paths = [
        Path(record[path_key])
        for records in [state["fit_records"], state["reference_records"]]
        for record in records.values()
        for path_key in ["model_path", "metadata_path"]
    ]
    fit_manifest_path = output / "fit_artifact_manifest.json"
    recorded_fit_manifest = state.get("fit_manifest")
    recorded_fit_manifest_sha256 = state.get("fit_manifest_sha256")
    if recorded_fit_manifest and Path(recorded_fit_manifest).exists():
        if sha256_file(Path(recorded_fit_manifest)) != recorded_fit_manifest_sha256:
            raise RuntimeError("Recorded fit-manifest hash mismatch.")
        fit_manifest = load_json(Path(recorded_fit_manifest))
        verified, mismatches = verify_manifest(fit_manifest)
        if not verified:
            raise RuntimeError(f"Recorded fit artifacts do not verify: {mismatches}")
    else:
        if state["test_gate"]["status"] != "CLOSED":
            raise RuntimeError("Test gate was opened without an immutable fit manifest.")
        fit_manifest = artifact_manifest(fit_paths)
        fit_manifest["strong_fit_count"] = expected_fit_count
        fit_manifest["transparent_reference_fit_count"] = expected_reference_count
        fit_manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(fit_manifest_path, fit_manifest)
        state["fit_manifest"] = str(fit_manifest_path)
        state["fit_manifest_sha256"] = sha256_file(fit_manifest_path)
        state["fits_frozen_at_utc"] = fit_manifest["frozen_at_utc"]
    state["status"] = "FITS_FROZEN"
    write_json(state_path, state)

    if not bool(config["allow_one_time_test_evaluation"]):
        print(json.dumps({"status": "FITS_FROZEN_TEST_GATE_CLOSED", "output_directory": str(output)}, indent=2))
        return 0
    if state["test_gate"]["status"] == "CLOSED":
        verified, mismatches = verify_manifest(load_json(fit_manifest_path))
        if not verified:
            raise RuntimeError(f"Cannot open test gate; fit manifest mismatch: {mismatches}")
        state["test_gate"].update(
            {
                "status": "OPEN",
                "opened_at_utc": datetime.now(timezone.utc).isoformat(),
                "fit_manifest_sha256_at_open": sha256_file(fit_manifest_path),
            }
        )
        write_json(state_path, state)

    test_target_materializations_this_run = 0
    for cell in supported_cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        key = f"{split_name}|{target_id}"
        storage_key = cell_key(split_name, target_id)
        target = feature_contract["targets"][target_id]
        membership = memberships[key]
        test_membership = membership[membership["split_role"].eq("test")].copy()
        test_ids = test_membership["segment_id"].astype(str).tolist()
        test_input_path = output / "test_gate_inputs" / f"{storage_key}.parquet"
        existing = state["test_gate"]["inputs"].get(storage_key)
        if existing and test_input_path.exists() and sha256_file(test_input_path) == existing["sha256"]:
            continue
        if test_input_path.exists():
            candidate = pd.read_parquet(test_input_path)
            if (
                set(candidate.columns) == {"segment_id", "trip_uid", "VehId", "engine_type", "y_true"}
                and set(candidate["segment_id"].astype(str)) == set(test_ids)
                and not candidate["segment_id"].duplicated().any()
            ):
                state["test_gate"]["inputs"][storage_key] = {
                    "path": str(test_input_path),
                    "sha256": sha256_file(test_input_path),
                    "rows": int(len(candidate)),
                    "segment_id_sha256": ordered_id_sha256(candidate["segment_id"]),
                    "source_materialization_count": 1,
                }
                write_json(state_path, state)
                continue
        target_table = target_dataset.to_table(
            columns=["segment_id", target["column"]],
            filter=ds.field("segment_id").isin(pa.array(test_ids)),
        ).to_pandas()
        if set(target_table["segment_id"].astype(str)) != set(test_ids) or target_table["segment_id"].duplicated().any():
            raise RuntimeError(f"Test target materialization escaped its role: {key}")
        test_input = (
            test_membership.merge(data[identity_columns], on="segment_id", how="left", validate="one_to_one")
            .merge(target_table, on="segment_id", how="left", validate="one_to_one")
            .rename(columns={target["column"]: "y_true"})
            [["segment_id", "trip_uid", "VehId", "engine_type", "y_true"]]
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        if test_input["y_true"].isna().any():
            raise RuntimeError(f"Missing test targets at the one-time gate: {key}")
        write_parquet(test_input_path, test_input)
        state["test_gate"]["inputs"][storage_key] = {
            "path": str(test_input_path),
            "sha256": sha256_file(test_input_path),
            "rows": int(len(test_input)),
            "segment_id_sha256": ordered_id_sha256(test_input["segment_id"]),
            "source_materialization_count": 1,
        }
        test_target_materializations_this_run += 1
        write_json(state_path, state)
        print(f"TEST TARGET MATERIALIZED {len(state['test_gate']['inputs'])}/10 {key}", flush=True)

    if len(state["test_gate"]["inputs"]) != len(supported_cells):
        raise RuntimeError("Not all test cells were materialized at the one-time gate.")
    test_input_paths = [Path(record["path"]) for record in state["test_gate"]["inputs"].values()]
    test_manifest_path = output / "test_target_manifest.json"
    test_manifest = artifact_manifest(test_input_paths)
    test_manifest["cell_count"] = len(supported_cells)
    test_manifest["total_target_rows"] = sum(record["rows"] for record in state["test_gate"]["inputs"].values())
    test_manifest["source_materialization_count_per_cell"] = 1
    write_json(test_manifest_path, test_manifest)
    state["test_gate"]["status"] = "MATERIALIZED"
    state["test_gate"]["target_manifest"] = str(test_manifest_path)
    state["test_gate"]["target_manifest_sha256"] = sha256_file(test_manifest_path)
    write_json(state_path, state)

    prediction_recomputes_this_run = 0
    metric_records: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        storage_key = cell_key(split_name, target_id)
        target = feature_contract["targets"][target_id]
        test_input = pd.read_parquet(state["test_gate"]["inputs"][storage_key]["path"])
        test = test_input.merge(data, on=identity_columns, how="left", validate="one_to_one")
        y_true = pd.to_numeric(test["y_true"], errors="raise").to_numpy(float)
        test_hash = ordered_id_sha256(test["segment_id"])
        for seed in seeds:
            prediction_key = f"{storage_key}__seed_{seed}"
            prediction_path = output / "paired_test_predictions" / storage_key / f"seed_{seed}.parquet"
            existing = state["test_gate"]["predictions"].get(prediction_key)
            if existing and prediction_path.exists() and sha256_file(prediction_path) == existing["sha256"]:
                paired = pd.read_parquet(prediction_path)
            else:
                paired = test_input.copy()
                for variant, features in variants_by_target[target_id].items():
                    record = state["fit_records"][fit_key(split_name, target_id, variant, seed)]
                    payload = load_model(Path(record["model_path"]))
                    active = payload["metadata"]["active_features"]
                    x_test = test[active].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                    paired[f"prediction__{variant}"] = np.asarray(payload["model"].predict(x_test), dtype=float)
                write_parquet(prediction_path, paired)
                state["test_gate"]["predictions"][prediction_key] = {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                    "rows": int(len(paired)),
                    "segment_id_sha256": test_hash,
                    "variant_count": len(expected_variants),
                }
                prediction_recomputes_this_run += 1
                write_json(state_path, state)
                print(f"PAIRED PREDICTION {len(state['test_gate']['predictions'])}/30 {prediction_key}", flush=True)
            for variant in expected_variants:
                metrics = regression_metrics(y_true, paired[f"prediction__{variant}"].to_numpy(float))
                metric_records.append(
                    {
                        "system_role": "frozen_backbone",
                        "backbone": protocol["selected_backbone"],
                        "split_family": split_name,
                        "target_id": target_id,
                        "unit": target["unit"],
                        "variant": variant,
                        "seed": seed,
                        "test_rows": int(len(test)),
                        "test_segment_id_sha256": test_hash,
                        **metrics,
                    }
                )

        reference_prediction_path = output / "reference_test_predictions" / f"{storage_key}.parquet"
        ref_existing = state["test_gate"]["references"].get(storage_key)
        if ref_existing and reference_prediction_path.exists() and sha256_file(reference_prediction_path) == ref_existing["sha256"]:
            reference_prediction = pd.read_parquet(reference_prediction_path)
        else:
            ref_record = state["reference_records"][reference_key(split_name, target_id)]
            ref_payload = load_model(Path(ref_record["model_path"]))
            ref_active = ref_payload["metadata"]["active_features"]
            ref_x_test = test[ref_active].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            reference_prediction = test_input.copy()
            reference_prediction["prediction__TRANSPARENT_REFERENCE"] = np.asarray(
                ref_payload["model"].predict(ref_x_test), dtype=float
            )
            write_parquet(reference_prediction_path, reference_prediction)
            state["test_gate"]["references"][storage_key] = {
                "path": str(reference_prediction_path),
                "sha256": sha256_file(reference_prediction_path),
                "rows": int(len(reference_prediction)),
                "segment_id_sha256": test_hash,
            }
            write_json(state_path, state)
        ref_metrics = regression_metrics(
            y_true, reference_prediction["prediction__TRANSPARENT_REFERENCE"].to_numpy(float)
        )
        metric_records.append(
            {
                "system_role": "transparent_reference",
                "backbone": protocol["transparent_reference"]["name"],
                "split_family": split_name,
                "target_id": target_id,
                "unit": target["unit"],
                "variant": "TRANSPARENT_REFERENCE",
                "seed": seeds[0],
                "test_rows": int(len(test)),
                "test_segment_id_sha256": test_hash,
                **ref_metrics,
            }
        )
        print(f"CELL TEST COMPLETE {cell_index}/10 {split_name}|{target_id}", flush=True)

    metrics = pd.DataFrame(metric_records).sort_values(
        ["system_role", "split_family", "target_id", "variant", "seed"]
    )
    metrics_path = output / "m3_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    fit_audit_path = output / "fit_audit.csv"
    fit_rows = []
    for record_key, record in sorted(state["fit_records"].items()):
        metadata = load_json(Path(record["metadata_path"]))
        fit_rows.append({"fit_key": record_key, **{key: value for key, value in metadata.items() if not isinstance(value, (list, dict))}})
    pd.DataFrame(fit_rows).to_csv(fit_audit_path, index=False)
    role_audit = pd.DataFrame(train_role_records)
    input_map = state["test_gate"]["inputs"]
    role_audit["test_target_rows_used_at_gate"] = [
        input_map[cell_key(row.split_family, row.target_id)]["rows"] for row in role_audit.itertuples()
    ]
    role_audit["test_source_materialization_count"] = 1
    role_audit_path = output / "role_access_audit.csv"
    role_audit.to_csv(role_audit_path, index=False)

    prediction_paths = [Path(record["path"]) for record in state["test_gate"]["predictions"].values()]
    reference_prediction_paths = [Path(record["path"]) for record in state["test_gate"]["references"].values()]
    finite_columns = ["mae", "rmse", "medae", "r2"]
    strong_metrics = metrics[metrics["system_role"].eq("frozen_backbone")]
    checks = {
        "upstream_recursive_manifests": "PASS",
        "upstream_independent_audits": "PASS",
        "exact_ten_primary_cells": "PASS" if len(supported_cells) == 10 else "FAIL",
        "exact_three_frozen_seeds": "PASS" if seeds == [20260720, 20260721, 20260722] else "FAIL",
        "exact_twelve_prespecified_variants": "PASS" if len(expected_variants) == 12 else "FAIL",
        "complete_frozen_backbone_grid": "PASS" if len(state["fit_records"]) == 360 else "FAIL",
        "complete_transparent_reference_grid": "PASS" if len(state["reference_records"]) == 10 else "FAIL",
        "fit_manifest_verified_before_test_gate": "PASS"
        if verify_manifest(load_json(fit_manifest_path))[0]
        and state["test_gate"]["fit_manifest_sha256_at_open"] == sha256_file(fit_manifest_path)
        else "FAIL",
        "validation_target_rows_used_zero": "PASS" if role_audit["validation_target_rows_used"].sum() == 0 else "FAIL",
        "calibration_target_rows_used_zero": "PASS" if role_audit["calibration_target_rows_used"].sum() == 0 else "FAIL",
        "test_materialized_once_per_cell": "PASS"
        if role_audit["test_source_materialization_count"].eq(1).all()
        and all(record["source_materialization_count"] == 1 for record in input_map.values())
        else "FAIL",
        "paired_prediction_files_complete": "PASS" if len(prediction_paths) == 30 else "FAIL",
        "paired_prediction_support_identical": "PASS"
        if all(
            record["segment_id_sha256"] == input_map[key.rsplit("__seed_", 1)[0]]["segment_id_sha256"]
            for key, record in state["test_gate"]["predictions"].items()
        )
        else "FAIL",
        "raw_metrics_complete": "PASS" if len(strong_metrics) == 360 and len(metrics) == 370 else "FAIL",
        "raw_metrics_finite": "PASS" if np.isfinite(metrics[finite_columns].to_numpy(float)).all() else "FAIL",
        "zero_forbidden_feature_hits": "PASS" if not hits else "FAIL",
        "zero_target_source_leakage_hits": "PASS",
        "target_units_separate": "PASS" if set(metrics["unit"]) == {"L"} else "FAIL",
        "transparent_reference_not_mechanistic": "PASS"
        if protocol["transparent_reference"]["claim_status"] == "STATISTICAL_REFERENCE_NOT_MECHANISTIC"
        else "FAIL",
        "memory_ceiling": "PASS"
        if state.get("peak_process_rss_gb") is None
        or float(state["peak_process_rss_gb"]) <= float(config["memory_ceiling_gb"])
        else "FAIL",
    }
    checks_path = output / "m3_gate_checks.csv"
    pd.DataFrame([{"check": key, "status": value} for key, value in checks.items()]).to_csv(checks_path, index=False)
    passed = "FAIL" not in checks.values()
    state["test_gate"]["status"] = "COMPLETE"
    state["status"] = "PASS" if passed else "FAIL"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)

    summary = {
        "schema_version": 1,
        "experiment": "MODEL_CONTRACT",
        "stage": "M3_primary_add_drop",
        "status": "PASS_MODEL_CONTRACT_M3_PRIMARY_ADD_DROP" if passed else "FAIL_MODEL_CONTRACT_M3_PRIMARY_ADD_DROP",
        "checks": checks,
        "primary_cell_count": len(supported_cells),
        "variant_count": len(expected_variants),
        "seed_count": len(seeds),
        "frozen_backbone_fit_count": len(state["fit_records"]),
        "transparent_reference_fit_count": len(state["reference_records"]),
        "fit_count_this_run": fit_count_this_run,
        "transparent_reference_fit_count_this_run": reference_fit_count_this_run,
        "test_target_cell_materialization_count_this_run": test_target_materializations_this_run,
        "paired_prediction_file_count": len(prediction_paths),
        "paired_prediction_recompute_count_this_run": prediction_recomputes_this_run,
        "raw_metric_rows": int(len(metrics)),
        "validation_target_rows_used": 0,
        "calibration_target_rows_used": 0,
        "test_target_rows_used": int(test_manifest["total_target_rows"]),
        "test_gate_opened_at_utc": state["test_gate"]["opened_at_utc"],
        "test_evaluation_performed": True,
        "selected_backbone": protocol["selected_backbone"],
        "execution_backend": "local_cpu",
        "gpu_used": False,
        "peak_process_rss_gb": state.get("peak_process_rss_gb"),
        "elapsed_seconds": time.perf_counter() - started,
        "resume_noop_verified": False,
        "output_directory": str(output),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_path, summary)
    report_path = output / "MODEL_CONTRACT_M3_GATE.md"
    report_path.write_text(
        "\n".join(
            [
                "# MODEL_CONTRACT M3 primary add/drop gate",
                "",
                f"- Status: `{summary['status']}`",
                f"- Frozen backbone: `{protocol['selected_backbone']}`",
                f"- Primary cells / variants / seeds: {len(supported_cells)} / {len(expected_variants)} / {len(seeds)}",
                f"- Frozen backbone fits: {len(state['fit_records'])}",
                f"- Transparent statistical-reference fits: {len(state['reference_records'])}",
                f"- Paired segment-prediction files: {len(prediction_paths)}",
                f"- Raw metric rows: {len(metrics)}",
                "- Validation target rows used: 0",
                "- Calibration target rows used: 0",
                f"- Test target rows materialized at the one-time gate: {test_manifest['total_target_rows']}",
                f"- Peak process RSS: {state.get('peak_process_rss_gb')}",
                "- Execution backend: frozen local CPU HistGradientBoosting L1 protocol.",
                "",
                "## Decision",
                "",
                "M3 freezes raw segment-level predictions and metrics only. Road-semantics claims remain prohibited until M4 completes 2,000 trip-clustered paired inference, target/family-specific FDR, calibration, and the joint support/error/reliability decision.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    primary_paths = [
        run_config_path,
        state_path,
        fit_manifest_path,
        test_manifest_path,
        metrics_path,
        fit_audit_path,
        role_audit_path,
        checks_path,
        report_path,
        *fit_paths,
        *test_input_paths,
        *prediction_paths,
        *reference_prediction_paths,
    ]
    write_json(primary_manifest_path, artifact_manifest(primary_paths))
    latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
    write_json(
        latest_path,
        {
            "status": summary["status"],
            "summary": str(summary_path),
            "report": str(report_path),
            "metrics": str(metrics_path),
            "paired_predictions_root": str(output / "paired_test_predictions"),
            "reference_predictions_root": str(output / "reference_test_predictions"),
            "fit_manifest": str(fit_manifest_path),
            "test_target_manifest": str(test_manifest_path),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
