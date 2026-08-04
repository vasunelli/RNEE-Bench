#!/usr/bin/env python3
"""Run MODEL_CONTRACT M4 clustered inference, FDR, and calibration reliability gates.

This stage consumes the frozen M3 models and paired test predictions.  It does
not refit models or rematerialize test targets.  Calibration targets are read
once after recursive M3 verification and seed-deduplication checks.
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


VARIANTS = [
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


def load_model(path: Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


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


def deterministic_seed(master_seed: int, label: str) -> int:
    value = hashlib.sha256(f"{master_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(value[:4], "little", signed=False)


def fdr_adjust(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("FDR input must be a non-empty finite vector.")
    if method not in {"bh", "by"}:
        raise ValueError(f"Unsupported FDR method: {method}")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    m = len(p)
    factor = float(sum(1.0 / index for index in range(1, m + 1))) if method == "by" else 1.0
    adjusted_ranked = ranked * m * factor / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def conformal_qhat(residuals: np.ndarray, coverage: float) -> float:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not 0.0 < coverage < 1.0:
        raise ValueError("Conformal calibration requires finite residuals and 0 < coverage < 1.")
    rank = min(len(values), int(math.ceil((len(values) + 1) * coverage)))
    return float(np.partition(values, rank - 1)[rank - 1])


def bootstrap_weights(cluster_ids: pd.Series, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusters, inverse = np.unique(cluster_ids.astype(str).to_numpy(), return_inverse=True)
    if len(clusters) < 2:
        raise ValueError("At least two trip clusters are required.")
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=replicates
    ).astype(np.uint16)
    return weights, inverse, counts


def cluster_bootstrap_means(
    values: np.ndarray, weights: np.ndarray, inverse: np.ndarray, cluster_counts: np.ndarray
) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if len(numeric) != len(inverse) or not np.isfinite(numeric).all():
        raise ValueError("Bootstrap values must be finite and aligned with cluster IDs.")
    cluster_sums = np.bincount(inverse, weights=numeric, minlength=len(cluster_counts))
    denominator = weights @ cluster_counts
    return (weights @ cluster_sums) / denominator


def bootstrap_summary(replicates: np.ndarray, point_estimate: float) -> dict[str, float]:
    values = np.asarray(replicates, dtype=float)
    lower, upper = np.quantile(values, [0.025, 0.975])
    lower_tail = (1.0 + float(np.sum(values <= 0.0))) / (len(values) + 1.0)
    upper_tail = (1.0 + float(np.sum(values >= 0.0))) / (len(values) + 1.0)
    return {
        "effect": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
    }


def evidence_grade(
    positive_add: int,
    positive_drop: int,
    negative_add: int,
    negative_drop: int,
) -> str:
    positive_total = positive_add + positive_drop
    negative_total = negative_add + negative_drop
    if negative_total > 0 and positive_total == 0:
        return "HARMFUL_UNDER_OOD"
    if positive_add >= 3 and positive_drop >= 1 and negative_total == 0:
        return "ROBUST_GAIN"
    if positive_add > 0 and positive_drop == 0:
        return "REDUNDANT_SIGNAL"
    if positive_add == 0 and positive_drop > 0:
        return "SYNERGISTIC_SIGNAL"
    if positive_total > 0:
        return "CONDITIONAL_GAIN"
    return "NO_GAIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/model_contract_m4_inference.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m1_pointer_path = Path(config["m1_pointer"])
    m3_pointer_path = Path(config["m3_pointer"])
    segment_pointer_path = Path(config["segment_pointer"])
    m1_pointer = load_json(m1_pointer_path)
    m3_pointer = load_json(m3_pointer_path)
    segment_pointer = load_json(segment_pointer_path)
    m1_summary = load_json(Path(m1_pointer["summary"]))
    m1_audit = load_json(Path(m1_pointer["experiment_audit_json"]))
    m3_summary = load_json(Path(m3_pointer["summary"]))
    m3_audit = load_json(Path(m3_pointer["experiment_audit_json"]))
    m3_state_path = Path(m3_pointer["output_directory"]) / "m3_state.json"
    m3_state = load_json(m3_state_path)
    m1_root = Path(m1_pointer["output_directory"])
    m3_root = Path(m3_pointer["output_directory"])
    feature_contract_path = Path(m1_pointer["feature_contract"])
    feature_contract = load_json(feature_contract_path)
    r2_path = m1_root / "r2_segment_features.parquet"

    upstream_manifest_paths = [
        Path(m1_pointer["manifest"]),
        Path(m1_pointer["release_manifest"]),
        Path(m3_pointer["manifest"]),
        Path(m3_pointer["release_manifest"]),
        Path(m3_pointer["audit_manifest"]),
    ]
    upstream_results = [verify_manifest(load_json(path)) for path in upstream_manifest_paths]
    if not all(result[0] for result in upstream_results):
        raise RuntimeError(f"Recursive upstream manifest verification failed: {upstream_results}")
    if m1_summary["status"] != "PASS_MODEL_CONTRACT_M1_B1" or not m1_summary["resume_noop_verified"]:
        raise RuntimeError("MODEL_CONTRACT M1 is not a reproducible PASS release.")
    if m1_audit["overall_verdict"] != "pass":
        raise RuntimeError("MODEL_CONTRACT M1 audit is not PASS.")
    if m3_summary["status"] != "PASS_MODEL_CONTRACT_M3_PRIMARY_ADD_DROP" or not m3_summary["resume_noop_verified"]:
        raise RuntimeError("MODEL_CONTRACT M3 is not a reproducible PASS release.")
    if m3_audit["overall_verdict"] != "pass" or m3_audit["m4_disposition"] != "ELIGIBLE_TO_PROCEED_WITH_FROZEN_PREDICTIONS":
        raise RuntimeError("MODEL_CONTRACT M3 independent contract checks do not permit M4.")
    if m3_state["status"] != "PASS" or m3_state["test_gate"]["status"] != "COMPLETE":
        raise RuntimeError("MODEL_CONTRACT M3 state/test gate is not frozen PASS.")
    if segment_pointer["status"] != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("Frozen SEGMENT_SPLIT primary segment release is not PASS.")

    cells = config["primary_cells"]
    if len(cells) != 10:
        raise RuntimeError("M4 must retain exactly ten primary cells.")
    canonical_seed = int(config["canonical_seed"])
    audit_seeds = [int(value) for value in config["audit_seeds"]]
    if canonical_seed not in audit_seeds or audit_seeds != [20260720, 20260721, 20260722]:
        raise RuntimeError("M4 seed-deduplication contract drifted.")
    membership_paths = {
        f"{cell['split_family']}|{cell['target_id']}": m1_root
        / "target_memberships"
        / cell["split_family"]
        / cell["target_id"]
        / "model_membership.parquet"
        for cell in cells
    }
    memberships = {key: pd.read_parquet(path) for key, path in membership_paths.items()}

    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "m1_pointer_sha256": sha256_file(m1_pointer_path),
        "m1_primary_manifest_sha256": sha256_file(Path(m1_pointer["manifest"])),
        "m1_release_manifest_sha256": sha256_file(Path(m1_pointer["release_manifest"])),
        "m1_audit_sha256": sha256_file(Path(m1_pointer["experiment_audit_json"])),
        "m3_pointer_sha256": sha256_file(m3_pointer_path),
        "m3_primary_manifest_sha256": sha256_file(Path(m3_pointer["manifest"])),
        "m3_release_manifest_sha256": sha256_file(Path(m3_pointer["release_manifest"])),
        "m3_audit_manifest_sha256": sha256_file(Path(m3_pointer["audit_manifest"])),
        "m3_audit_sha256": sha256_file(Path(m3_pointer["experiment_audit_json"])),
        "m3_state_sha256": sha256_file(m3_state_path),
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
    state_path = output / "m4_state.json"
    summary_path = output / "model_contract_m4_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if state["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        state = {
            "schema_version": 1,
            "status": "PENDING",
            "input_fingerprint": fingerprint,
            "seed_equivalence_verified": False,
            "calibration_gate": {"status": "CLOSED", "cells": {}},
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
                "model_refit_count": 0,
                "test_target_rematerialization_count": 0,
                "calibration_target_rematerialization_count": 0,
                "bootstrap_recompute_count": 0,
            },
        )
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(resume_path)
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest([primary_manifest_path, summary_path, resume_path])
        release["scope"] = "post-resume controls; immutable M4 outputs are enumerated by artifact_manifest.json"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
        latest = load_json(latest_path) if latest_path.exists() else {}
        latest["release_manifest"] = str(release_manifest_path)
        write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    seed_audit_records = []
    for cell in cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        storage_key = cell_key(split_name, target_id)
        frames = {
            seed: pd.read_parquet(m3_root / "paired_test_predictions" / storage_key / f"seed_{seed}.parquet")
            for seed in audit_seeds
        }
        canonical = frames[canonical_seed]
        prediction_columns = [f"prediction__{variant}" for variant in VARIANTS]
        for seed, frame in frames.items():
            ids_equal = frame["segment_id"].astype(str).tolist() == canonical["segment_id"].astype(str).tolist()
            target_difference = float(np.max(np.abs(frame["y_true"].to_numpy(float) - canonical["y_true"].to_numpy(float))))
            prediction_difference = float(
                np.max(np.abs(frame[prediction_columns].to_numpy(float) - canonical[prediction_columns].to_numpy(float)))
            )
            seed_audit_records.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "seed": seed,
                    "rows": int(len(frame)),
                    "segment_order_equal_to_canonical": ids_equal,
                    "max_abs_target_difference": target_difference,
                    "max_abs_prediction_difference": prediction_difference,
                }
            )
    seed_audit = pd.DataFrame(seed_audit_records)
    tolerance = float(config["seed_equivalence_tolerance"])
    if (
        not seed_audit["segment_order_equal_to_canonical"].all()
        or seed_audit["max_abs_target_difference"].max() > tolerance
        or seed_audit["max_abs_prediction_difference"].max() > tolerance
    ):
        raise RuntimeError("Frozen M3 seeds are not equivalent under the M4 deduplication contract.")
    seed_audit_path = output / "seed_equivalence_audit.csv"
    seed_audit.to_csv(seed_audit_path, index=False)
    state["seed_equivalence_verified"] = True
    state["canonical_seed"] = canonical_seed
    state["seed_audit_sha256"] = sha256_file(seed_audit_path)
    write_json(state_path, state)

    all_active_features: set[str] = set()
    for cell in cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        for variant in VARIANTS:
            record = m3_state["fit_records"][fit_key(split_name, target_id, variant, canonical_seed)]
            metadata = load_json(Path(record["metadata_path"]))
            all_active_features.update(metadata["active_features"])
    segment_schema = set(pq.read_schema(segment_pointer["segments_all"]).names)
    identity_columns = ["segment_id", "trip_uid", "VehId", "engine_type"]
    segment_columns = list(
        dict.fromkeys(identity_columns + sorted(all_active_features.intersection(segment_schema)))
    )
    segments = pd.read_parquet(segment_pointer["segments_all"], columns=segment_columns)
    r2_features = sorted(all_active_features - set(segments.columns))
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    missing_features = sorted(all_active_features - set(data.columns))
    if missing_features:
        raise RuntimeError(f"M4 calibration features are missing: {missing_features}")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")

    if state["calibration_gate"]["status"] == "CLOSED":
        if not state["seed_equivalence_verified"]:
            raise RuntimeError("Seed equivalence must verify before calibration access.")
        state["calibration_gate"].update(
            {
                "status": "OPEN",
                "opened_at_utc": datetime.now(timezone.utc).isoformat(),
                "m3_primary_manifest_sha256_at_open": sha256_file(Path(m3_pointer["manifest"])),
                "m3_release_manifest_sha256_at_open": sha256_file(Path(m3_pointer["release_manifest"])),
                "m3_audit_manifest_sha256_at_open": sha256_file(Path(m3_pointer["audit_manifest"])),
            }
        )
        write_json(state_path, state)

    calibration_materializations_this_run = 0
    calibration_paths: list[Path] = []
    for cell_index, cell in enumerate(cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        key = f"{split_name}|{target_id}"
        storage_key = cell_key(split_name, target_id)
        target = feature_contract["targets"][target_id]
        membership = memberships[key]
        calibration_membership = membership[membership["split_role"].eq("calibration")].copy()
        calibration_ids = calibration_membership["segment_id"].astype(str).tolist()
        calibration_path = output / "calibration_predictions" / f"{storage_key}.parquet"
        existing = state["calibration_gate"]["cells"].get(storage_key)
        if existing:
            if not calibration_path.exists() or sha256_file(calibration_path) != existing["sha256"]:
                raise RuntimeError(f"Recorded calibration artifact mismatch: {storage_key}")
            calibration_paths.append(calibration_path)
            continue
        if calibration_path.exists():
            raise RuntimeError(f"Orphan calibration artifact requires fail-closed quality_check: {calibration_path}")
        target_table = target_dataset.to_table(
            columns=["segment_id", target["column"]],
            filter=ds.field("segment_id").isin(pa.array(calibration_ids)),
        ).to_pandas()
        if set(target_table["segment_id"].astype(str)) != set(calibration_ids) or target_table["segment_id"].duplicated().any():
            raise RuntimeError(f"Calibration target materialization escaped its role: {key}")
        calibration = (
            calibration_membership.merge(data, on="segment_id", how="left", validate="one_to_one")
            .merge(target_table, on="segment_id", how="left", validate="one_to_one")
            .rename(columns={target["column"]: "y_true"})
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        if calibration["y_true"].isna().any():
            raise RuntimeError(f"Missing calibration targets: {key}")
        output_frame = calibration[identity_columns + ["y_true"]].copy()
        for variant in VARIANTS:
            record = m3_state["fit_records"][fit_key(split_name, target_id, variant, canonical_seed)]
            payload = load_model(Path(record["model_path"]))
            active = payload["metadata"]["active_features"]
            x_calibration = calibration[active].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            output_frame[f"prediction__{variant}"] = np.asarray(payload["model"].predict(x_calibration), dtype=float)
        write_parquet(calibration_path, output_frame)
        state["calibration_gate"]["cells"][storage_key] = {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
            "rows": int(len(output_frame)),
            "segment_id_sha256": ordered_id_sha256(output_frame["segment_id"]),
            "source_materialization_count": 1,
        }
        calibration_materializations_this_run += 1
        calibration_paths.append(calibration_path)
        write_json(state_path, state)
        print(f"CALIBRATION {cell_index}/10 {key}", flush=True)
    state["calibration_gate"]["status"] = "MATERIALIZED"
    write_json(state_path, state)

    bootstrap_replicates = int(config["bootstrap_replicates"])
    master_seed = int(config["bootstrap_master_seed"])
    coverage_levels = [float(value) for value in config["coverage_levels"]]
    primary_coverage = float(config["primary_coverage"])
    effect_records: list[dict[str, Any]] = []
    effect_bootstrap_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    coverage_bootstrap_records: list[dict[str, Any]] = []
    local_coverage_records: list[dict[str, Any]] = []
    bootstrap_seed_records: list[dict[str, Any]] = []

    fit_audit = pd.read_csv(m3_root / "fit_audit.csv")
    role_audit = pd.read_csv(m3_root / "role_access_audit.csv")
    m3_metrics = pd.read_csv(m3_root / "m3_test_metrics.csv")
    canonical_metrics = m3_metrics[
        (m3_metrics["system_role"] == "frozen_backbone") & (m3_metrics["seed"] == canonical_seed)
    ].copy()

    for cell_index, cell in enumerate(cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        storage_key = cell_key(split_name, target_id)
        test = pd.read_parquet(m3_root / "paired_test_predictions" / storage_key / f"seed_{canonical_seed}.parquet")
        calibration = pd.read_parquet(output / "calibration_predictions" / f"{storage_key}.parquet")
        if test["trip_uid"].isna().any() or calibration["trip_uid"].isna().any():
            raise RuntimeError(f"Missing trip cluster ID: {storage_key}")
        cell_seed = deterministic_seed(master_seed, storage_key)
        weights, inverse, cluster_counts = bootstrap_weights(
            test["trip_uid"], bootstrap_replicates, cell_seed
        )
        bootstrap_seed_records.append(
            {
                "split_family": split_name,
                "target_id": target_id,
                "bootstrap_seed": cell_seed,
                "replicates": bootstrap_replicates,
                "test_trips": int(test["trip_uid"].nunique()),
            }
        )
        y_true = test["y_true"].to_numpy(float)
        comparisons = []
        for index in range(1, 6):
            comparisons.append(("add", f"R{index}", "R0", f"R0_PLUS_R{index}"))
            comparisons.append(("drop", f"R{index}", "R6", f"R6_MINUS_R{index}"))
        for family, semantic_group, baseline, comparison in comparisons:
            baseline_error = np.abs(y_true - test[f"prediction__{baseline}"].to_numpy(float))
            comparison_error = np.abs(y_true - test[f"prediction__{comparison}"].to_numpy(float))
            row_effect = (
                baseline_error - comparison_error if family == "add" else comparison_error - baseline_error
            )
            point = float(row_effect.mean())
            replicates = cluster_bootstrap_means(row_effect, weights, inverse, cluster_counts)
            summary = bootstrap_summary(replicates, point)
            baseline_metric = canonical_metrics[
                (canonical_metrics["split_family"] == split_name)
                & (canonical_metrics["target_id"] == target_id)
                & (canonical_metrics["variant"] == baseline)
            ].iloc[0]
            comparison_metric = canonical_metrics[
                (canonical_metrics["split_family"] == split_name)
                & (canonical_metrics["target_id"] == target_id)
                & (canonical_metrics["variant"] == comparison)
            ].iloc[0]
            train_iqr = float(
                fit_audit[
                    (fit_audit["split_family"] == split_name)
                    & (fit_audit["target_id"] == target_id)
                    & (fit_audit["variant"] == baseline)
                    & (fit_audit["seed"] == canonical_seed)
                ]["train_target_iqr"].iloc[0]
            )
            effect_records.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "unit": "L",
                    "comparison_family": family,
                    "semantic_group": semantic_group,
                    "baseline_variant": baseline,
                    "comparison_variant": comparison,
                    "canonical_seed": canonical_seed,
                    "test_segments": int(len(test)),
                    "test_trips": int(test["trip_uid"].nunique()),
                    "test_vehicles": int(test["VehId"].nunique()),
                    "test_segment_id_sha256": ordered_id_sha256(test["segment_id"]),
                    "baseline_mae": float(baseline_metric["mae"]),
                    "comparison_mae": float(comparison_metric["mae"]),
                    "baseline_rmse": float(baseline_metric["rmse"]),
                    "comparison_rmse": float(comparison_metric["rmse"]),
                    "baseline_medae": float(baseline_metric["medae"]),
                    "comparison_medae": float(comparison_metric["medae"]),
                    "baseline_r2": float(baseline_metric["r2"]),
                    "comparison_r2": float(comparison_metric["r2"]),
                    "train_target_iqr": train_iqr,
                    **summary,
                    "effect_relative_to_baseline_mae_pct": 100.0 * point / float(baseline_metric["mae"]),
                    "effect_relative_to_train_iqr": point / train_iqr,
                }
            )
            effect_bootstrap_records.extend(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "comparison_family": family,
                    "semantic_group": semantic_group,
                    "replicate": index_value,
                    "effect": float(value),
                }
                for index_value, value in enumerate(replicates)
            )

        calibration_abs_target = np.abs(calibration["y_true"].to_numpy(float))
        raw_edges = np.quantile(calibration_abs_target, [0.0, 0.25, 0.5, 0.75, 1.0])
        local_edges = np.unique(raw_edges)
        if len(local_edges) < 2:
            local_edges = np.array([-np.inf, np.inf])
        else:
            local_edges[0] = -np.inf
            local_edges[-1] = np.inf
        test_abs_target = np.abs(y_true)
        for variant in VARIANTS:
            calibration_residual = np.abs(
                calibration["y_true"].to_numpy(float) - calibration[f"prediction__{variant}"].to_numpy(float)
            )
            test_residual = np.abs(y_true - test[f"prediction__{variant}"].to_numpy(float))
            for coverage in coverage_levels:
                qhat = conformal_qhat(calibration_residual, coverage)
                covered = (test_residual <= qhat).astype(float)
                coverage_point = float(covered.mean())
                coverage_replicates = cluster_bootstrap_means(covered, weights, inverse, cluster_counts)
                coverage_ci = np.quantile(coverage_replicates, [0.025, 0.975])
                calibration_records.append(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "variant": variant,
                        "coverage_nominal": coverage,
                        "calibration_rows": int(len(calibration)),
                        "calibration_trips": int(calibration["trip_uid"].nunique()),
                        "calibration_segment_id_sha256": ordered_id_sha256(calibration["segment_id"]),
                        "qhat_abs_error_L": qhat,
                        "interval_width_L": 2.0 * qhat,
                        "test_coverage": coverage_point,
                        "test_coverage_ci_lower": float(coverage_ci[0]),
                        "test_coverage_ci_upper": float(coverage_ci[1]),
                        "marginal_undercoverage": coverage - coverage_point,
                    }
                )
                coverage_bootstrap_records.extend(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "variant": variant,
                        "coverage_nominal": coverage,
                        "replicate": index_value,
                        "coverage": float(value),
                    }
                    for index_value, value in enumerate(coverage_replicates)
                )
                bin_index = np.digitize(test_abs_target, local_edges[1:-1], right=True)
                for local_index in range(len(local_edges) - 1):
                    mask = bin_index == local_index
                    if not mask.any():
                        continue
                    local_coverage = float(covered[mask].mean())
                    local_coverage_records.append(
                        {
                            "split_family": split_name,
                            "target_id": target_id,
                            "variant": variant,
                            "coverage_nominal": coverage,
                            "local_bin": local_index + 1,
                            "abs_target_lower_L": float(local_edges[local_index]),
                            "abs_target_upper_L": float(local_edges[local_index + 1]),
                            "test_segments": int(mask.sum()),
                            "test_coverage": local_coverage,
                            "local_undercoverage": coverage - local_coverage,
                            "eligible_for_local_gate": int(mask.sum()) >= int(config["minimum_local_test_rows"]),
                        }
                    )
        print(f"INFERENCE {cell_index}/10 {split_name}|{target_id}", flush=True)
        rss = current_rss_gb()
        if rss is not None:
            state["peak_process_rss_gb"] = max(float(state.get("peak_process_rss_gb") or 0.0), rss)
            write_json(state_path, state)

    effects = pd.DataFrame(effect_records)
    for (target_id, family), indexes in effects.groupby(["target_id", "comparison_family"]).groups.items():
        selected = list(indexes)
        effects.loc[selected, "q_value_bh"] = fdr_adjust(
            effects.loc[selected, "p_value_two_sided"].to_numpy(float), "bh"
        )
        effects.loc[selected, "q_value_by_dependency_sensitivity"] = fdr_adjust(
            effects.loc[selected, "p_value_two_sided"].to_numpy(float), "by"
        )
        effects.loc[selected, "fdr_family_size"] = len(selected)
    effects["bh_direction"] = np.where(
        (effects["q_value_bh"] <= float(config["fdr_alpha"])) & (effects["ci_lower"] > 0),
        "positive",
        np.where(
            (effects["q_value_bh"] <= float(config["fdr_alpha"])) & (effects["ci_upper"] < 0),
            "negative",
            "not_supported",
        ),
    )
    effects["by_dependency_direction"] = np.where(
        (effects["q_value_by_dependency_sensitivity"] <= float(config["fdr_alpha"]))
        & (effects["ci_lower"] > 0),
        "positive",
        np.where(
            (effects["q_value_by_dependency_sensitivity"] <= float(config["fdr_alpha"]))
            & (effects["ci_upper"] < 0),
            "negative",
            "not_supported",
        ),
    )

    calibration_summary = pd.DataFrame(calibration_records)
    local_coverage = pd.DataFrame(local_coverage_records)
    local_gate = (
        local_coverage[
            (local_coverage["coverage_nominal"] == primary_coverage)
            & local_coverage["eligible_for_local_gate"]
        ]
        .groupby(["split_family", "target_id", "variant"], as_index=False)
        .agg(
            eligible_local_bins=("local_bin", "size"),
            worst_local_undercoverage=("local_undercoverage", "max"),
            worst_local_coverage=("test_coverage", "min"),
        )
    )
    primary_calibration = calibration_summary[calibration_summary["coverage_nominal"] == primary_coverage].merge(
        local_gate, on=["split_family", "target_id", "variant"], how="left", validate="one_to_one"
    )
    primary_calibration["reliability_pass"] = (
        (primary_calibration["test_coverage"] >= primary_coverage - float(config["marginal_undercoverage_tolerance"]))
        & (primary_calibration["eligible_local_bins"].fillna(0) > 0)
        & (
            primary_calibration["worst_local_undercoverage"]
            <= float(config["worst_local_undercoverage_tolerance"])
        )
    )

    joint_records = []
    role_lookup = role_audit.set_index(["split_family", "target_id"])
    calibration_lookup = primary_calibration.set_index(["split_family", "target_id", "variant"])
    for row in effects.itertuples(index=False):
        role = role_lookup.loc[(row.split_family, row.target_id)]
        baseline_cal = calibration_lookup.loc[(row.split_family, row.target_id, row.baseline_variant)]
        comparison_cal = calibration_lookup.loc[(row.split_family, row.target_id, row.comparison_variant)]
        reliability_joint = bool(baseline_cal["reliability_pass"] and comparison_cal["reliability_pass"])
        direction = "not_supported"
        if reliability_joint and row.by_dependency_direction in {"positive", "negative"}:
            direction = row.by_dependency_direction
        joint_records.append(
            {
                **row._asdict(),
                "train_segments": int(role["train_target_rows_used"]),
                "calibration_segments": int(baseline_cal["calibration_rows"]),
                "primary_coverage_nominal": primary_coverage,
                "baseline_test_coverage": float(baseline_cal["test_coverage"]),
                "comparison_test_coverage": float(comparison_cal["test_coverage"]),
                "baseline_worst_local_undercoverage": float(baseline_cal["worst_local_undercoverage"]),
                "comparison_worst_local_undercoverage": float(comparison_cal["worst_local_undercoverage"]),
                "baseline_reliability_pass": bool(baseline_cal["reliability_pass"]),
                "comparison_reliability_pass": bool(comparison_cal["reliability_pass"]),
                "reliability_joint_status": "PASS" if reliability_joint else "BOUNDARY",
                "joint_dependency_robust_direction": direction,
                "split_family_dependence_note": "correlated_family_results_not_independent_replications",
                "seed_inference_note": "canonical_seed_only_three_frozen_seeds_numerically_identical",
            }
        )
    joint = pd.DataFrame(joint_records)

    grade_records = []
    cold_splits = set(config["primary_cold_splits"])
    for (target_id, semantic_group), group in joint.groupby(["target_id", "semantic_group"]):
        cold = group[group["split_family"].isin(cold_splits)]
        add = cold[cold["comparison_family"] == "add"]
        drop = cold[cold["comparison_family"] == "drop"]
        positive_add = int((add["joint_dependency_robust_direction"] == "positive").sum())
        positive_drop = int((drop["joint_dependency_robust_direction"] == "positive").sum())
        negative_add = int((add["joint_dependency_robust_direction"] == "negative").sum())
        negative_drop = int((drop["joint_dependency_robust_direction"] == "negative").sum())
        grade_records.append(
            {
                "target_id": target_id,
                "semantic_group": semantic_group,
                "primary_cold_split_count": int(len(cold_splits)),
                "dependency_robust_positive_add_splits": positive_add,
                "dependency_robust_positive_drop_splits": positive_drop,
                "dependency_robust_negative_add_splits": negative_add,
                "dependency_robust_negative_drop_splits": negative_drop,
                "reliability_boundary_comparisons": int((cold["reliability_joint_status"] != "PASS").sum()),
                "evidence_grade": evidence_grade(
                    positive_add, positive_drop, negative_add, negative_drop
                ),
            }
        )
    grades = pd.DataFrame(grade_records).sort_values(["target_id", "semantic_group"])

    effects_path = output / "paired_effect_inference.csv"
    effect_bootstrap_path = output / "paired_effect_bootstrap.parquet"
    calibration_summary_path = output / "calibration_summary.csv"
    coverage_bootstrap_path = output / "coverage_bootstrap.parquet"
    local_coverage_path = output / "local_coverage.csv"
    joint_path = output / "joint_gain_reliability.csv"
    grades_path = output / "evidence_grades.csv"
    bootstrap_seeds_path = output / "bootstrap_seed_audit.csv"
    effects.to_csv(effects_path, index=False)
    write_parquet(effect_bootstrap_path, pd.DataFrame(effect_bootstrap_records))
    calibration_summary.to_csv(calibration_summary_path, index=False)
    write_parquet(coverage_bootstrap_path, pd.DataFrame(coverage_bootstrap_records))
    local_coverage.to_csv(local_coverage_path, index=False)
    joint.to_csv(joint_path, index=False)
    grades.to_csv(grades_path, index=False)
    pd.DataFrame(bootstrap_seed_records).to_csv(bootstrap_seeds_path, index=False)

    finite_effect_columns = [
        "effect",
        "ci_lower",
        "ci_upper",
        "p_value_two_sided",
        "q_value_bh",
        "q_value_by_dependency_sensitivity",
    ]
    checks = {
        "recursive_m1_m3_audit_manifests": "PASS",
        "m3_reproducible_pass": "PASS",
        "exact_ten_primary_cells": "PASS" if len(cells) == 10 else "FAIL",
        "canonical_seed_deduplicated": "PASS"
        if seed_audit["max_abs_prediction_difference"].max() <= tolerance
        else "FAIL",
        "no_model_refitting": "PASS",
        "no_test_target_rematerialization": "PASS",
        "calibration_materialized_once_per_cell": "PASS"
        if len(state["calibration_gate"]["cells"]) == 10
        and all(record["source_materialization_count"] == 1 for record in state["calibration_gate"]["cells"].values())
        else "FAIL",
        "complete_paired_comparisons": "PASS" if len(effects) == 100 else "FAIL",
        "exact_2000_trip_clustered_replicates": "PASS"
        if bootstrap_replicates == 2000 and len(effect_bootstrap_records) == 200000
        else "FAIL",
        "complete_target_family_bh_fdr": "PASS"
        if effects.groupby(["target_id", "comparison_family"]).size().eq(25).all()
        else "FAIL",
        "dependency_bounded_with_by_sensitivity": "PASS"
        if effects["q_value_by_dependency_sensitivity"].notna().all()
        else "FAIL",
        "complete_calibration_levels": "PASS" if len(calibration_summary) == 240 else "FAIL",
        "complete_joint_reliability_table": "PASS" if len(joint) == 100 else "FAIL",
        "complete_evidence_grades": "PASS" if len(grades) == 10 else "FAIL",
        "finite_inference_outputs": "PASS"
        if np.isfinite(effects[finite_effect_columns].to_numpy(float)).all()
        else "FAIL",
        "split_family_dependence_disclosed": "PASS",
        "deterministic_seed_variance_not_inflated": "PASS",
        "target_units_separate": "PASS" if set(effects["unit"]) == {"L"} else "FAIL",
        "memory_ceiling": "PASS"
        if state.get("peak_process_rss_gb") is None
        or float(state["peak_process_rss_gb"]) <= float(config["memory_ceiling_gb"])
        else "FAIL",
    }
    checks_path = output / "m4_gate_checks.csv"
    pd.DataFrame([{"check": key, "status": value} for key, value in checks.items()]).to_csv(checks_path, index=False)
    passed = "FAIL" not in checks.values()
    state["calibration_gate"]["status"] = "COMPLETE"
    state["status"] = "PASS" if passed else "FAIL"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)

    grade_counts = grades["evidence_grade"].value_counts().sort_index().to_dict()
    summary = {
        "schema_version": 1,
        "experiment": "MODEL_CONTRACT",
        "stage": "M4_clustered_inference_fdr_calibration_reliability",
        "status": "PASS_MODEL_CONTRACT_M4_INFERENCE_AND_RELIABILITY" if passed else "FAIL_MODEL_CONTRACT_M4_INFERENCE_AND_RELIABILITY",
        "inference_gate": "PASS_MODEL_CONTRACT_INFERENCE_GATE" if passed else "FAIL_MODEL_CONTRACT_INFERENCE_GATE",
        "joint_interpretation_gate": "PASS_GAIN_RELIABILITY_JOINT_INTERPRETATION" if passed else "FAIL_GAIN_RELIABILITY_JOINT_INTERPRETATION",
        "checks": checks,
        "primary_cell_count": len(cells),
        "paired_comparison_count": int(len(effects)),
        "bootstrap_replicates_per_comparison": bootstrap_replicates,
        "paired_effect_bootstrap_rows": int(len(effect_bootstrap_records)),
        "coverage_bootstrap_rows": int(len(coverage_bootstrap_records)),
        "calibration_summary_rows": int(len(calibration_summary)),
        "joint_reliability_rows": int(len(joint)),
        "evidence_grade_rows": int(len(grades)),
        "evidence_grade_counts": grade_counts,
        "bh_supported_positive_comparisons": int((effects["bh_direction"] == "positive").sum()),
        "bh_supported_negative_comparisons": int((effects["bh_direction"] == "negative").sum()),
        "by_supported_positive_comparisons": int((effects["by_dependency_direction"] == "positive").sum()),
        "by_supported_negative_comparisons": int((effects["by_dependency_direction"] == "negative").sum()),
        "joint_dependency_robust_positive_comparisons": int((joint["joint_dependency_robust_direction"] == "positive").sum()),
        "joint_dependency_robust_negative_comparisons": int((joint["joint_dependency_robust_direction"] == "negative").sum()),
        "reliability_boundary_comparisons": int((joint["reliability_joint_status"] != "PASS").sum()),
        "canonical_seed": canonical_seed,
        "max_abs_prediction_difference_across_frozen_seeds": float(seed_audit["max_abs_prediction_difference"].max()),
        "calibration_target_rows_used": int(sum(record["rows"] for record in state["calibration_gate"]["cells"].values())),
        "calibration_target_cell_materialization_count_this_run": calibration_materializations_this_run,
        "test_target_rows_rematerialized": 0,
        "model_refit_count": 0,
        "execution_backend": "local_cpu",
        "gpu_used": False,
        "peak_process_rss_gb": state.get("peak_process_rss_gb"),
        "elapsed_seconds": time.perf_counter() - started,
        "resume_noop_verified": False,
        "claim_authorization": "M4_EVIDENCE_GRADES_ONLY_PENDING_INDEPENDENT_AUDIT",
        "output_directory": str(output),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_path, summary)
    report_path = output / "MODEL_CONTRACT_M4_GATE.md"
    report_path.write_text(
        "\n".join(
            [
                "# MODEL_CONTRACT M4 clustered inference, FDR and reliability gate",
                "",
                f"- Status: `{summary['status']}`",
                f"- Inference gate: `{summary['inference_gate']}`",
                f"- Joint reliability gate: `{summary['joint_interpretation_gate']}`",
                f"- Paired comparisons: {len(effects)}",
                f"- Trip-clustered replicates per comparison: {bootstrap_replicates}",
                f"- BH positive / negative comparisons: {summary['bh_supported_positive_comparisons']} / {summary['bh_supported_negative_comparisons']}",
                f"- BY positive / negative comparisons: {summary['by_supported_positive_comparisons']} / {summary['by_supported_negative_comparisons']}",
                f"- Reliability-joint positive / negative comparisons: {summary['joint_dependency_robust_positive_comparisons']} / {summary['joint_dependency_robust_negative_comparisons']}",
                f"- Reliability-boundary comparisons: {summary['reliability_boundary_comparisons']}",
                f"- Evidence grades: {json.dumps(grade_counts, sort_keys=True)}",
                f"- Calibration target rows used once: {summary['calibration_target_rows_used']}",
                "- Test target rows rematerialized: 0",
                "- Model refits: 0",
                "- Frozen seeds: numerically identical; canonical seed only used for inference.",
                "- Split families: correlated; BH is primary and BY is the dependency sensitivity.",
                "",
                "## Decision",
                "",
                "M4 machine-readable evidence is complete. Interpretation remains bounded by the independent integrity audit and the joint reliability columns. No causal or external-transfer claim is authorized.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    primary_paths = [
        run_config_path,
        state_path,
        seed_audit_path,
        *calibration_paths,
        effects_path,
        effect_bootstrap_path,
        calibration_summary_path,
        coverage_bootstrap_path,
        local_coverage_path,
        joint_path,
        grades_path,
        bootstrap_seeds_path,
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
            "paired_effect_inference": str(effects_path),
            "paired_effect_bootstrap": str(effect_bootstrap_path),
            "calibration_summary": str(calibration_summary_path),
            "local_coverage": str(local_coverage_path),
            "joint_gain_reliability": str(joint_path),
            "evidence_grades": str(grades_path),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
