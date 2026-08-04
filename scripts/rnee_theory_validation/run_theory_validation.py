#!/usr/bin/env python3
"""Execute THEORY_VALIDATION under the frozen RNEE theory-validation protocol.

The runner creates equal-dimensional noise and whole-R6 role-restricted
permutation controls, fits the frozen HistGradientBoosting-L1 matrix, opens
the already-historical RNEE test target only after all THEORY_VALIDATION fits are frozen,
and performs vehicle-cluster simultaneous max-t inference. It never changes
the SEGMENT_SPLIT data/splits, MODEL_CONTRACT/SENSOR_FACTORIAL artifacts, SESOI, or bootstrap design.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from artifacts import (
    artifact_manifest,
    load_json,
    ordered_sha256,
    sha256_file,
    verify_manifest,
    write_csv,
    write_json,
    write_parquet,
)
from bootstrap import final_decision, vehicle_cluster_bootstrap
from config import EXPECTED_ENVIRONMENTS, EXPECTED_SYSTEMS, load_config, stable_seed
from contracts import (
    null_control_contract_pass,
    validate_segment_split_release_manifests,
    validate_membership,
    validate_numeric_frame,
    validate_split_checks,
    validate_upstream_artifact_manifests,
)
from evaluation import SYSTEM_TO_PREDICTION, evaluate_system
from features import (
    build_feature_contract,
    shared_real_road_active_masks,
    system_matrix,
)
from models import dump_model, load_model, train_model
from null_controls import generate_noise_road, generate_permuted_road


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()


def _runtime_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "scikit-learn",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rnee/theory_validation/theory_validation.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _valid_record(record: dict[str, Any], path_key: str = "path") -> bool:
    path = Path(record[path_key])
    return path.exists() and sha256_file(path) == record["sha256"]


def _valid_model_record(record: dict[str, Any]) -> bool:
    model_path = Path(record["model_path"])
    metadata_path = Path(record["metadata_path"])
    return (
        model_path.exists()
        and metadata_path.exists()
        and sha256_file(model_path) == record["model_sha256"]
        and sha256_file(metadata_path) == record["metadata_sha256"]
    )


def _materialize_target(
    target_dataset: ds.Dataset,
    membership: pd.DataFrame,
    role: str,
    target_column: str,
) -> pd.DataFrame:
    selected = membership[membership["split_role"].eq(role)][
        ["segment_id", "split_role"]
    ].copy()
    identifiers = selected["segment_id"].astype(str).tolist()
    table = target_dataset.to_table(
        columns=["segment_id", target_column],
        filter=ds.field("segment_id").isin(pa.array(identifiers)),
    ).to_pandas()
    result = selected.merge(table, on="segment_id", how="left", validate="one_to_one")
    if len(result) != len(selected) or result[target_column].isna().any():
        raise RuntimeError(
            f"Invalid {role} target materialization for {target_column}: "
            f"{len(result)}/{len(selected)}."
        )
    return result


def _road_representation(
    base: pd.DataFrame,
    controls: pd.DataFrame,
    road_features: list[str],
    representation: str,
) -> pd.DataFrame:
    if representation == "real":
        return base[road_features].reset_index(drop=True)
    prefix = "noise" if representation == "noise" else "perm"
    frame = controls[[f"{prefix}__{column}" for column in road_features]].copy()
    frame.columns = road_features
    return frame.reset_index(drop=True)


def _support_rows(
    cell_key: str,
    split_name: str,
    target_id: str,
    base: pd.DataFrame,
    controls: pd.DataFrame,
    road_features: list[str],
    floors: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ("train", "validation", "calibration", "test"):
        mask = base["split_role"].eq(role).to_numpy()
        role_base = base.loc[mask].reset_index(drop=True)
        vehicle_counts = role_base["VehId"].astype(str).value_counts()
        for representation in ("real", "noise", "permutation"):
            road = _road_representation(
                base, controls, road_features, representation
            ).loc[mask].reset_index(drop=True)
            missing_rate = float(road.isna().to_numpy().mean())
            all_missing_rows = int(road.isna().all(axis=1).sum())
            segment_count = int(len(role_base))
            record = {
                "cell_key": cell_key,
                "split_family": split_name,
                "target_id": target_id,
                "split_role": role,
                "road_representation": representation,
                "segment_count": segment_count,
                "trip_count": int(role_base["trip_uid"].astype(str).nunique()),
                "vehicle_count": int(vehicle_counts.size),
                "maximum_single_vehicle_segment_share": (
                    float(vehicle_counts.iloc[0] / segment_count)
                    if segment_count
                    else math.nan
                ),
                "road_missing_cell_rate": missing_rate,
                "all_road_features_missing_rows": all_missing_rows,
                "fuel_direct_duration_share_mean": float(
                    pd.to_numeric(
                        role_base["fuel_direct_duration_share"], errors="coerce"
                    ).mean()
                ),
                "fuel_maf_duration_share_mean": float(
                    pd.to_numeric(
                        role_base["fuel_maf_duration_share"], errors="coerce"
                    ).mean()
                ),
            }
            checks = {
                "minimum_vehicles": record["vehicle_count"]
                >= int(floors["minimum_vehicles"]),
                "minimum_trips": record["trip_count"]
                >= int(floors["minimum_trips"]),
                "minimum_segments": record["segment_count"]
                >= int(floors["minimum_segments"]),
                "maximum_single_vehicle_segment_share": record[
                    "maximum_single_vehicle_segment_share"
                ]
                <= float(floors["maximum_single_vehicle_segment_share"]),
            }
            record["prospective_support_floor_pass"] = bool(
                role == "test" and all(checks.values())
            )
            record["prospective_support_failed_checks"] = (
                "|".join(name for name, value in checks.items() if not value)
                if role == "test"
                else "NOT_APPLICABLE_NON_TEST_ROLE"
            )
            rows.append(record)
    return rows


def _shift_rows(
    cell_key: str,
    split_name: str,
    target_id: str,
    base: pd.DataFrame,
    controls: pd.DataFrame,
    road_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_mask = base["split_role"].eq("train").to_numpy()
    test_mask = base["split_role"].eq("test").to_numpy()
    for representation in ("real", "noise", "permutation"):
        road = _road_representation(base, controls, road_features, representation)
        train = road.loc[train_mask].reset_index(drop=True)
        test = road.loc[test_mask].reset_index(drop=True)
        for column in road_features:
            train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(
                float
            )
            test_values = pd.to_numeric(test[column], errors="coerce").to_numpy(
                float
            )
            train_finite = train_values[np.isfinite(train_values)]
            test_finite = test_values[np.isfinite(test_values)]
            train_mean = (
                float(train_finite.mean()) if len(train_finite) else math.nan
            )
            test_mean = float(test_finite.mean()) if len(test_finite) else math.nan
            train_std = (
                float(train_finite.std(ddof=0)) if len(train_finite) else math.nan
            )
            standardized_shift = (
                float((test_mean - train_mean) / train_std)
                if np.isfinite(train_std) and train_std > 0
                else math.nan
            )
            if len(train_finite) and len(test_finite):
                train_min = float(train_finite.min())
                train_max = float(train_finite.max())
                outside = float(
                    np.mean((test_finite < train_min) | (test_finite > train_max))
                )
            else:
                train_min = train_max = outside = math.nan
            rows.append(
                {
                    "cell_key": cell_key,
                    "split_family": split_name,
                    "target_id": target_id,
                    "road_representation": representation,
                    "feature": column,
                    "train_observed_count": int(len(train_finite)),
                    "test_observed_count": int(len(test_finite)),
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "train_std": train_std,
                    "standardized_mean_shift": standardized_shift,
                    "absolute_standardized_mean_shift": (
                        abs(standardized_shift)
                        if np.isfinite(standardized_shift)
                        else math.nan
                    ),
                    "train_missing_rate": float(train[column].isna().mean()),
                    "test_missing_rate": float(test[column].isna().mean()),
                    "missing_rate_shift": float(
                        test[column].isna().mean() - train[column].isna().mean()
                    ),
                    "train_min": train_min,
                    "train_max": train_max,
                    "test_outside_train_range_fraction": outside,
                }
            )
    return rows


def _plot_results(effects: pd.DataFrame, output: Path, target_id: str) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = effects[effects["target_id"].eq(target_id)].copy()
    environment_labels = {
        "cold_trip": "Cold trip",
        "cold_month": "Cold month",
        "cold_spatial": "Cold spatial",
        "cold_functional_road_class": "Cold functional class",
    }
    paths: list[Path] = []

    gain = primary[primary["effect_type"].isin(
        ["gain_real", "gain_noise", "gain_permutation"]
    )]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    colors = {
        "gain_real": "#1f77b4",
        "gain_noise": "#7f7f7f",
        "gain_permutation": "#d95f02",
    }
    labels = {
        "gain_real": "Real road",
        "gain_noise": "Noise road",
        "gain_permutation": "Permutation road",
    }
    x = np.arange(len(EXPECTED_ENVIRONMENTS))
    offsets = {"gain_real": -0.18, "gain_noise": 0.0, "gain_permutation": 0.18}
    for axis, sensor in zip(axes, ("low", "high"), strict=True):
        for effect_type in ("gain_real", "gain_noise", "gain_permutation"):
            subset = (
                gain[gain["sensor"].eq(sensor) & gain["effect_type"].eq(effect_type)]
                .set_index("environment")
                .loc[list(EXPECTED_ENVIRONMENTS)]
            )
            estimate = subset["estimate"].to_numpy(float) * 100
            lower = subset["simultaneous_lcb"].to_numpy(float) * 100
            upper = subset["simultaneous_ucb"].to_numpy(float) * 100
            axis.errorbar(
                x + offsets[effect_type],
                estimate,
                yerr=np.vstack([estimate - lower, upper - estimate]),
                fmt="o",
                capsize=3,
                color=colors[effect_type],
                label=labels[effect_type],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(5.5, color="#2ca02c", linestyle="--", linewidth=0.9)
        axis.axhline(-5.5, color="#b2182b", linestyle="--", linewidth=0.9)
        axis.set_ylabel("Relative MAE gain (%)")
        axis.set_title(f"{sensor.capitalize()} sensor")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, loc="best")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [environment_labels[value] for value in EXPECTED_ENVIRONMENTS],
        rotation=15,
        ha="right",
    )
    fig.suptitle("THEORY_VALIDATION road-block gains with 95% simultaneous max-t intervals")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures / f"theory_validation_primary_road_gains.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    specificity = primary[primary["effect_type"].isin(
        ["real_minus_noise", "real_minus_permutation"]
    )]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    colors = {
        "real_minus_noise": "#4daf4a",
        "real_minus_permutation": "#984ea3",
    }
    labels = {
        "real_minus_noise": "Real − noise",
        "real_minus_permutation": "Real − permutation",
    }
    offsets = {"real_minus_noise": -0.10, "real_minus_permutation": 0.10}
    for axis, sensor in zip(axes, ("low", "high"), strict=True):
        for effect_type in ("real_minus_noise", "real_minus_permutation"):
            subset = (
                specificity[
                    specificity["sensor"].eq(sensor)
                    & specificity["effect_type"].eq(effect_type)
                ]
                .set_index("environment")
                .loc[list(EXPECTED_ENVIRONMENTS)]
            )
            estimate = subset["estimate"].to_numpy(float) * 100
            lower = subset["simultaneous_lcb"].to_numpy(float) * 100
            upper = subset["simultaneous_ucb"].to_numpy(float) * 100
            axis.errorbar(
                x + offsets[effect_type],
                estimate,
                yerr=np.vstack([estimate - lower, upper - estimate]),
                fmt="o",
                capsize=3,
                color=colors[effect_type],
                label=labels[effect_type],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(5.5, color="#2ca02c", linestyle="--", linewidth=0.9)
        axis.axhline(-5.5, color="#b2182b", linestyle="--", linewidth=0.9)
        axis.set_ylabel("Gain difference (percentage points)")
        axis.set_title(f"{sensor.capitalize()} sensor")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, loc="best")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [environment_labels[value] for value in EXPECTED_ENVIRONMENTS],
        rotation=15,
        ha="right",
    )
    fig.suptitle("THEORY_VALIDATION semantic-specificity contrasts with simultaneous intervals")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures / f"theory_validation_primary_semantic_specificity.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _format_effect_table(effects: pd.DataFrame, target_id: str) -> list[str]:
    target = effects[effects["target_id"].eq(target_id)]
    lines = [
        "| Environment | Sensor | Real gain (sim. CI) | Real−noise (sim. CI) | Real−permutation (sim. CI) | Cell pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for environment in EXPECTED_ENVIRONMENTS:
        for sensor in ("low", "high"):
            cell = target[
                target["environment"].eq(environment)
                & target["sensor"].eq(sensor)
            ].set_index("effect_type")
            real = cell.loc["gain_real"]
            noise = cell.loc["real_minus_noise"]
            permutation = cell.loc["real_minus_permutation"]
            passed = (
                float(real["simultaneous_lcb"]) > 0.055
                and float(noise["simultaneous_lcb"]) > 0
                and float(permutation["simultaneous_lcb"]) > 0
            )

            def value(row: pd.Series) -> str:
                return (
                    f"{float(row['estimate']):.2%} "
                    f"[{float(row['simultaneous_lcb']):.2%}, "
                    f"{float(row['simultaneous_ucb']):.2%}]"
                )

            lines.append(
                f"| {environment} | {sensor} | {value(real)} | {value(noise)} | "
                f"{value(permutation)} | {passed} |"
            )
    return lines


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    output = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(config["output"]["root"])
        / (
            f"{config['output']['run_name_prefix']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    state_path = output / "theory_validation_state.json"
    summary_path = output / "theory_validation_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"

    # A completed run is immutable. The only allowed resume action is a hash
    # verification that writes separate release-control artifacts.
    if state_path.exists():
        state = load_json(state_path)
        if (
            state.get("status") == "PASS"
            and primary_manifest_path.exists()
            and summary_path.exists()
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
                    "control_recompute_count": 0,
                    "prediction_recompute_count": 0,
                    "target_rematerialization_count": 0,
                    "bootstrap_recompute_count": 0,
                },
            )
            primary_summary = load_json(summary_path)
            release_summary_path = output / "release_summary.json"
            release_summary = {
                **primary_summary,
                "primary_summary": str(summary_path),
                "resume_noop_status": "PASS" if verified else "FAIL",
                "resume_noop_verified": verified,
                "resume_check": str(resume_path),
                "release_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(release_summary_path, release_summary)
            release_path = output / "release_manifest.json"
            write_json(
                release_path,
                artifact_manifest(
                    [
                        primary_manifest_path,
                        summary_path,
                        resume_path,
                        release_summary_path,
                    ],
                    experiment="THEORY_VALIDATION",
                    scope="post_resume_release_controls",
                ),
            )
            latest_path = Path(config["output"]["latest_pointer"])
            latest = load_json(latest_path) if latest_path.exists() else {}
            latest.update(
                {
                    "resume_noop_verified": verified,
                    "resume_check": str(resume_path),
                    "primary_summary": str(summary_path),
                    "summary": str(release_summary_path),
                    "release_manifest": str(release_path),
                }
            )
            write_json(latest_path, latest)
            print(
                json.dumps(
                    {
                        "status": "PASS_THEORY_VALIDATION_NOOP_REPRODUCTION"
                        if verified
                        else "FAIL_THEORY_VALIDATION_NOOP_REPRODUCTION",
                        "run_directory": str(output),
                        "mismatches": mismatches,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if verified else 1
        raise RuntimeError("THEORY_VALIDATION run directory contains an incomplete or invalid state.")

    if output.exists() and any(output.iterdir()):
        raise RuntimeError("A new THEORY_VALIDATION run requires an empty or absent run directory.")
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "logs" / "execution.log")
    logger.log("START THEORY_VALIDATION frozen theory-validation run")

    inputs = config["inputs"]
    pointer_paths = {
        name: Path(inputs[name])
        for name in (
            "m1_pointer",
            "m2_pointer",
            "segment_pointer",
            "split_pointer",
            "information_freeze_pointer",
            "sensor_factorial_pointer",
        )
    }
    pointers = {name: load_json(path) for name, path in pointer_paths.items()}
    m1_pointer = pointers["m1_pointer"]
    m2_pointer = pointers["m2_pointer"]
    segment_pointer = pointers["segment_pointer"]
    split_pointer = pointers["split_pointer"]
    information_freeze_pointer = pointers["information_freeze_pointer"]
    sensor_factorial_pointer = pointers["sensor_factorial_pointer"]

    if m1_pointer["status"] != "PASS_MODEL_CONTRACT_M1_B1":
        raise RuntimeError("MODEL_CONTRACT M1 upstream contract is not authoritative PASS.")
    if m2_pointer["status"] != "PASS_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE":
        raise RuntimeError("MODEL_CONTRACT M2 upstream contract is not authoritative PASS.")
    if segment_pointer["status"] != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("SEGMENT_SPLIT segment release is not authoritative PASS.")
    if split_pointer["status"] != "PASS_BENCHMARK_SPLITS":
        raise RuntimeError("SEGMENT_SPLIT split release is not authoritative PASS.")
    if sensor_factorial_pointer["status"] != "PASS_SENSOR_FACTORIAL_RNEE_RETROSPECTIVE_FACTORIAL":
        raise RuntimeError("SENSOR_FACTORIAL upstream evidence is not authoritative PASS.")
    if not information_freeze_pointer["status"].endswith("M2_BLOCKED"):
        raise RuntimeError("INFORMATION_FREEZE must remain frozen at M2_BLOCKED.")

    upstream_manifest_paths = [
        Path(m1_pointer["manifest"]),
        Path(m1_pointer["release_manifest"]),
        Path(m2_pointer["manifest"]),
        Path(m2_pointer["release_manifest"]),
        Path(information_freeze_pointer["manifest"]),
        Path(information_freeze_pointer["release_manifest"]),
        Path(sensor_factorial_pointer["manifest"]),
        Path(sensor_factorial_pointer["release_manifest"]),
    ]
    upstream_verification = validate_upstream_artifact_manifests(
        upstream_manifest_paths
    )
    split_verification = validate_split_checks(Path(split_pointer["checks"]))
    segment_split_release_verification = validate_segment_split_release_manifests(
        Path(segment_pointer["manifest"]),
        Path(split_pointer["manifest"]),
    )
    upstream_release_verification_path = (
        output / "upstream_release_verification.json"
    )
    write_json(
        upstream_release_verification_path,
        {
            "upstream_artifact_manifests": upstream_verification,
            "segment_split_release_manifests": segment_split_release_verification,
            "segment_split_leakage_checks": split_verification,
        },
    )
    logger.log(
        f"VERIFIED upstream manifests={len(upstream_manifest_paths)} "
        f"split_checks={split_verification['check_count']} "
        f"segment_split_hash_records="
        f"{segment_split_release_verification['segment_hash_record_count'] + segment_split_release_verification['split_hash_record_count']}"
    )

    m1_contract = load_json(Path(m1_pointer["feature_contract"]))
    information_contract = load_json(Path(information_freeze_pointer["information_contract"]))
    protocol = load_json(Path(m2_pointer["protocol"]))
    if protocol["selected_backbone"] != "hist_gradient_boosting_l1":
        raise RuntimeError("The frozen MODEL_CONTRACT backbone changed.")
    if dict(config["model"]["params"]) != dict(protocol["selected_params"]):
        raise RuntimeError("THEORY_VALIDATION model parameters differ from the frozen MODEL_CONTRACT backbone.")
    target_ids = [
        config["target"]["primary_upstream_id"],
        config["target"]["secondary_upstream_id"],
    ]
    feature_contract = build_feature_contract(
        information_contract, m1_contract, target_ids
    )
    road_features = list(
        feature_contract["target_contracts"][target_ids[0]]["road_features"]
    )
    all_features = sorted(
        {
            feature
            for target in feature_contract["target_contracts"].values()
            for values in target["systems"].values()
            for feature in values
        }
    )

    m1_root = Path(m1_pointer["output_directory"])
    r2_path = m1_root / "r2_segment_features.parquet"
    supported_cells = [
        cell
        for cell in protocol["primary_cells"]
        if cell["split_family"] in EXPECTED_ENVIRONMENTS
        and cell["target_id"] in target_ids
    ]
    expected_cell_keys = {
        f"{environment}|{target_id}"
        for environment in EXPECTED_ENVIRONMENTS
        for target_id in target_ids
    }
    observed_cell_keys = {
        f"{cell['split_family']}|{cell['target_id']}" for cell in supported_cells
    }
    if observed_cell_keys != expected_cell_keys or len(supported_cells) != 8:
        raise RuntimeError("THEORY_VALIDATION did not resolve the exact frozen eight cells.")
    membership_paths = {
        key: (
            m1_root
            / "target_memberships"
            / key.split("|")[0]
            / key.split("|")[1]
            / "model_membership.parquet"
        )
        for key in sorted(expected_cell_keys)
    }

    # Public reproducibility fingerprints use only files shipped in this repository.
    protocol_paths = [args.config, REPO_ROOT / "docs" / "full_reproduction.md"]
    module_paths = sorted(Path(__file__).parent.glob("*.py"))
    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": {
            str(path): sha256_file(path) for path in protocol_paths
        },
        "module_sha256": {str(path): sha256_file(path) for path in module_paths},
        "pointer_sha256": {
            name: sha256_file(path) for name, path in pointer_paths.items()
        },
        "segment_split_segment_manifest_sha256": segment_split_release_verification[
            "segment_manifest_sha256"
        ],
        "segment_split_split_manifest_sha256": segment_split_release_verification[
            "split_manifest_sha256"
        ],
        "segments_sha256": sha256_file(Path(segment_pointer["segments_all"])),
        "r2_sha256": sha256_file(r2_path),
        "membership_sha256": {
            key: sha256_file(path) for key, path in membership_paths.items()
        },
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "PREPARING_CONTROLS",
        "input_fingerprint": fingerprint,
        "control_records": {},
        "fit_records": {},
        "prediction_records": {},
        "test_gate": {"status": "CLOSED"},
    }
    write_json(state_path, state)
    config_snapshot_path = output / "config_snapshot.json"
    runtime_versions = _runtime_versions()
    write_json(
        config_snapshot_path,
        {
            "source_config": str(args.config),
            "source_config_sha256": fingerprint["config_sha256"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "input_fingerprint": fingerprint,
            "runtime_versions": runtime_versions,
        },
    )
    shutil.copy2(args.config, output / "run_config.yaml")

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
    logger.log(f"LOAD segment features columns={len(segment_columns)}")
    segments = pd.read_parquet(
        segment_pointer["segments_all"], columns=segment_columns
    )
    r2_features = [feature for feature in all_features if feature not in segments]
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    missing_features = sorted(set(all_features) - set(data.columns))
    if missing_features:
        raise RuntimeError(f"Missing THEORY_VALIDATION features: {missing_features}")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")

    control_paths: dict[str, Path] = {}
    control_metadata_paths: list[Path] = []
    mapping_paths: list[Path] = []
    support_records: list[dict[str, Any]] = []
    shift_records: list[dict[str, Any]] = []
    membership_audits: list[dict[str, Any]] = []
    active_mask_audits: dict[str, Any] = {}
    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        membership = pd.read_parquet(membership_paths[cell_key])
        membership_audits.append(validate_membership(membership, cell_key))
        cell_base = (
            membership[["segment_id", "split_role"]]
            .merge(data, on="segment_id", how="left", validate="one_to_one")
            .sort_values(["split_role", "segment_id"])
            .reset_index(drop=True)
        )
        road = cell_base[road_features].copy()
        validate_numeric_frame(road, road_features, f"{cell_key}/real_R6")
        train_reference = road[cell_base["split_role"].eq("train")].reset_index(
            drop=True
        )
        noise_seed = stable_seed(
            int(config["null_controls"]["noise"]["seed"]), split_name, target_id
        )
        permutation_seed = stable_seed(
            int(config["null_controls"]["permutation"]["seed"]),
            split_name,
            target_id,
        )
        noise, noise_metadata = generate_noise_road(
            road,
            train_reference=train_reference,
            seed=noise_seed,
            zero_variance_scale=float(
                config["null_controls"]["noise"]["zero_variance_scale"]
            ),
        )
        permutation, mapping, permutation_metadata = generate_permuted_road(
            road,
            cell_base["split_role"],
            seed=permutation_seed,
        )
        mapping["recipient_segment_id"] = cell_base.loc[
            mapping["recipient_position"], "segment_id"
        ].to_numpy()
        mapping["source_segment_id"] = cell_base.loc[
            mapping["source_position"], "segment_id"
        ].to_numpy()
        mapping["recipient_role"] = cell_base.loc[
            mapping["recipient_position"], "split_role"
        ].to_numpy()
        mapping["source_role"] = cell_base.loc[
            mapping["source_position"], "split_role"
        ].to_numpy()
        if not mapping["recipient_role"].eq(mapping["source_role"]).all():
            raise RuntimeError(f"{cell_key}: permutation crossed split roles.")
        for role in ("train", "validation", "calibration", "test"):
            mask = cell_base["split_role"].eq(role).to_numpy()
            real_hashes = np.sort(
                pd.util.hash_pandas_object(
                    road.loc[mask].reset_index(drop=True), index=False
                ).to_numpy()
            )
            perm_hashes = np.sort(
                pd.util.hash_pandas_object(
                    permutation.loc[mask].reset_index(drop=True), index=False
                ).to_numpy()
            )
            if not np.array_equal(real_hashes, perm_hashes):
                raise RuntimeError(
                    f"{cell_key}: whole-block multiset not preserved in {role}."
                )
        control_columns: dict[str, Any] = {
            "segment_id": cell_base["segment_id"].to_numpy(),
            "split_role": cell_base["split_role"].to_numpy(),
        }
        for column in road_features:
            control_columns[f"noise__{column}"] = noise[column].to_numpy()
            control_columns[f"perm__{column}"] = permutation[column].to_numpy()
        controls = pd.DataFrame(control_columns)
        control_path = output / "null_controls" / f"{split_name}__{target_id}.parquet"
        mapping_path = (
            output / "null_controls" / "mappings" / f"{split_name}__{target_id}.parquet"
        )
        metadata_path = (
            output / "null_controls" / "metadata" / f"{split_name}__{target_id}.json"
        )
        write_parquet(control_path, controls)
        write_parquet(mapping_path, mapping)
        metadata = {
            "schema_version": 1,
            "cell_key": cell_key,
            "target_values_read": 0,
            "noise": noise_metadata,
            "permutation": permutation_metadata,
            "contracts": {
                "same_rows": len(road) == len(noise) == len(permutation),
                "same_columns": list(road) == list(noise) == list(permutation),
                "same_dtypes": [str(value) for value in road.dtypes]
                == [str(value) for value in noise.dtypes]
                == [str(value) for value in permutation.dtypes],
                "noise_recipient_missing_mask_exact": noise.isna().equals(
                    road.isna()
                ),
                "permutation_role_block_multiset_exact": True,
                "permutation_cross_role_moves": 0,
                "permutation_fixed_points": 0,
            },
        }
        write_json(metadata_path, metadata)
        state["control_records"][cell_key] = {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
            "mapping_path": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
        write_json(state_path, state)
        control_paths[cell_key] = control_path
        control_metadata_paths.append(metadata_path)
        mapping_paths.append(mapping_path)
        support_records.extend(
            _support_rows(
                cell_key,
                split_name,
                target_id,
                cell_base,
                controls,
                road_features,
                config["support"],
            )
        )
        shift_records.extend(
            _shift_rows(
                cell_key,
                split_name,
                target_id,
                cell_base,
                controls,
                road_features,
            )
        )
        logger.log(f"CONTROL {cell_index}/8 {cell_key}")

    support_path = output / "support_results.csv"
    shift_path = output / "shift_results.csv"
    write_csv(support_path, pd.DataFrame(support_records))
    write_csv(shift_path, pd.DataFrame(shift_records))
    feature_manifest_path = output / "feature_manifest.json"
    write_json(
        feature_manifest_path,
        {
            **feature_contract,
            "experiment": "THEORY_VALIDATION",
            "systems": list(EXPECTED_SYSTEMS),
            "environments": list(EXPECTED_ENVIRONMENTS),
            "membership_audits": membership_audits,
            "null_control_records": state["control_records"],
            "all_control_contracts_pass": True,
            "active_mask_policy": config["model"]["active_mask_policy"],
            "active_mask_audits": "PENDING_MODEL_FIT_PREPARATION",
        },
    )
    state["status"] = "FITTING"
    write_json(state_path, state)

    model_paths: list[Path] = []
    fit_count_this_run = 0
    seeds = [int(value) for value in config["model"]["seeds"]]
    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        target_contract = feature_contract["target_contracts"][target_id]
        if (
            target_contract["target_unit"] != "L"
            or target_contract["target_channel"] != "fuel"
        ):
            raise RuntimeError(f"Non-fuel target entered THEORY_VALIDATION: {cell_key}")
        membership = pd.read_parquet(membership_paths[cell_key])
        train_target = _materialize_target(
            target_dataset,
            membership,
            "train",
            target_contract["target_column"],
        )
        train_base = (
            train_target.merge(data, on="segment_id", how="left", validate="one_to_one")
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        train_control_source = (
            pd.read_parquet(control_paths[cell_key])
            .query("split_role == 'train'")
            .drop(columns=["split_role"])
        )
        train_controls = train_base[["segment_id"]].merge(
            train_control_source,
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        if len(train_base) < int(config["model"]["minimum_train_rows"]):
            raise RuntimeError(f"{cell_key}: insufficient frozen train support.")
        y_train = pd.to_numeric(
            train_base[target_contract["target_column"]], errors="raise"
        ).to_numpy(float)
        train_hash = ordered_sha256(train_base["segment_id"])
        train_matrices = {
            system: system_matrix(
                train_base, train_controls, target_contract, system
            )
            for system in EXPECTED_SYSTEMS
        }
        active_masks, active_mask_audit = shared_real_road_active_masks(
            train_matrices
        )
        active_mask_audit.update(
            {
                "cell_key": cell_key,
                "train_rows": int(len(train_base)),
                "train_segment_id_sha256": train_hash,
            }
        )
        active_mask_audits[cell_key] = active_mask_audit
        for system in EXPECTED_SYSTEMS:
            matrix = train_matrices[system]
            active_features = active_masks[system]
            if not active_features:
                raise RuntimeError(f"{cell_key}/{system}: no active train features.")
            for seed in seeds:
                record_key = f"{cell_key}|{system}|{seed}"
                model_path = (
                    output
                    / "models"
                    / f"{split_name}__{target_id}"
                    / system.replace("-", "_")
                    / f"seed_{seed}.joblib"
                )
                metadata_path = model_path.with_suffix(".json")
                fit_started = time.perf_counter()
                model = train_model(
                    matrix[active_features], y_train, config["model"], seed
                )
                metadata = {
                    "schema_version": 1,
                    "experiment": "THEORY_VALIDATION",
                    "evidence_tier": config["experiment"]["evidence_tier"],
                    "split_family": split_name,
                    "target_id": target_id,
                    "target_alias": (
                        config["target"]["primary"]
                        if target_id == config["target"]["primary_upstream_id"]
                        else config["target"]["secondary"]
                    ),
                    "target_unit": "L",
                    "system": system,
                    "seed": seed,
                    "train_rows": int(len(train_base)),
                    "train_segment_id_sha256": train_hash,
                    "raw_features": target_contract["systems"][system],
                    "active_features": active_features,
                    "active_feature_count": len(active_features),
                    "active_mask_policy": config["model"]["active_mask_policy"],
                    "active_mask_source_system": (
                        system
                        if system.endswith("R0")
                        else f"{system[0]}-R-real"
                    ),
                    "params": config["model"]["params"],
                    "fit_seconds": time.perf_counter() - fit_started,
                    "validation_target_rows_used": 0,
                    "calibration_target_rows_used": 0,
                    "test_target_rows_used_before_fit_freeze": 0,
                    "post_test_tuning": False,
                }
                dump_model(model_path, model, metadata)
                write_json(metadata_path, metadata)
                state["fit_records"][record_key] = {
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "model_sha256": sha256_file(model_path),
                    "metadata_sha256": sha256_file(metadata_path),
                }
                model_paths.extend([model_path, metadata_path])
                fit_count_this_run += 1
                write_json(state_path, state)
                logger.log(
                    f"FIT {fit_count_this_run}/"
                    f"{len(supported_cells) * len(EXPECTED_SYSTEMS) * len(seeds)} "
                    f"{record_key}"
                )

    expected_fits = len(supported_cells) * len(EXPECTED_SYSTEMS) * len(seeds)
    if len(state["fit_records"]) != expected_fits or any(
        not _valid_model_record(record)
        for record in state["fit_records"].values()
    ):
        raise RuntimeError("THEORY_VALIDATION fit grid or model hashes are incomplete.")
    equal_effective_dimension_controls = all(
        sensor_audit[
            "effective_dimension_equal_across_real_noise_permutation"
        ]
        for cell_audit in active_mask_audits.values()
        for sensor_audit in cell_audit["sensor_regimes"].values()
    )
    if not equal_effective_dimension_controls:
        raise RuntimeError(
            "THEORY_VALIDATION real/noise/permutation effective dimensions are not equal."
        )
    active_masks_path = output / "active_feature_masks.json"
    write_json(
        active_masks_path,
        {
            "schema_version": 1,
            "policy": config["model"]["active_mask_policy"],
            "all_effective_dimensions_equal": equal_effective_dimension_controls,
            "cells": active_mask_audits,
        },
    )
    write_json(
        feature_manifest_path,
        {
            **feature_contract,
            "experiment": "THEORY_VALIDATION",
            "systems": list(EXPECTED_SYSTEMS),
            "environments": list(EXPECTED_ENVIRONMENTS),
            "membership_audits": membership_audits,
            "null_control_records": state["control_records"],
            "all_control_contracts_pass": True,
            "active_mask_policy": config["model"]["active_mask_policy"],
            "all_effective_dimensions_equal": equal_effective_dimension_controls,
            "active_feature_masks": str(active_masks_path),
            "active_mask_audits": active_mask_audits,
        },
    )
    model_manifest_path = output / "model_manifest.json"
    model_manifest = artifact_manifest(
        model_paths,
        experiment="THEORY_VALIDATION",
        model="HistGradientBoostingRegressor",
        loss="absolute_error",
        seeds=seeds,
        fit_count=expected_fits,
        records=state["fit_records"],
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(model_manifest_path, model_manifest)
    state["model_manifest"] = str(model_manifest_path)
    state["model_manifest_sha256"] = sha256_file(model_manifest_path)
    state["status"] = "FITS_FROZEN"
    write_json(state_path, state)
    verified_models, model_mismatches = verify_manifest(model_manifest)
    if not verified_models:
        raise RuntimeError(f"Cannot open THEORY_VALIDATION test gate: {model_mismatches}")
    state["test_gate"] = {
        "status": "OPEN",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256_at_open": sha256_file(model_manifest_path),
        "historical_note": "RNEE test targets were previously opened by MODEL_CONTRACT; "
        "THEORY_VALIDATION reuses them only after its complete frozen fit grid.",
    }
    write_json(state_path, state)
    logger.log("TEST_GATE OPEN after complete model-manifest verification")

    metric_rows: list[dict[str, Any]] = []
    ensemble_paths: dict[str, Path] = {}
    prediction_paths: list[Path] = []
    test_target_materializations = 0
    prediction_recomputes = 0
    maximum_seed_prediction_std = 0.0
    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        target_contract = feature_contract["target_contracts"][target_id]
        membership = pd.read_parquet(membership_paths[cell_key])
        test_target = _materialize_target(
            target_dataset,
            membership,
            "test",
            target_contract["target_column"],
        )
        test_target_materializations += 1
        test_base = (
            test_target.merge(data, on="segment_id", how="left", validate="one_to_one")
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        test_control_source = (
            pd.read_parquet(control_paths[cell_key])
            .query("split_role == 'test'")
            .drop(columns=["split_role"])
        )
        test_controls = test_base[["segment_id"]].merge(
            test_control_source,
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        input_path = (
            output / "test_gate_inputs" / f"{split_name}__{target_id}.parquet"
        )
        write_parquet(
            input_path,
            test_base[
                [
                    "segment_id",
                    "trip_uid",
                    "VehId",
                    "engine_type",
                    target_contract["target_column"],
                    *all_features,
                ]
            ],
        )
        prediction_paths.append(input_path)
        seed_frames: list[pd.DataFrame] = []
        for seed in seeds:
            prediction_path = (
                output
                / "paired_test_predictions"
                / f"{split_name}__{target_id}"
                / f"seed_{seed}.parquet"
            )
            frame = test_base[["segment_id", "trip_uid", "VehId"]].copy()
            frame["target_l"] = pd.to_numeric(
                test_base[target_contract["target_column"]], errors="raise"
            ).to_numpy(float)
            for system in EXPECTED_SYSTEMS:
                record = state["fit_records"][f"{cell_key}|{system}|{seed}"]
                payload = load_model(Path(record["model_path"]))
                matrix = system_matrix(
                    test_base, test_controls, target_contract, system
                )
                active = payload["metadata"]["active_features"]
                frame[SYSTEM_TO_PREDICTION[system]] = payload["model"].predict(
                    matrix[active]
                )
                metrics = evaluate_system(
                    frame,
                    SYSTEM_TO_PREDICTION[system],
                )
                metric_rows.append(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "seed": seed,
                        "aggregation": "single_seed",
                        "system": system,
                        **metrics,
                    }
                )
            write_parquet(prediction_path, frame)
            state["prediction_records"][f"{cell_key}|{seed}"] = {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "rows": int(len(frame)),
                "segment_id_sha256": ordered_sha256(frame["segment_id"]),
            }
            write_json(state_path, state)
            seed_frames.append(frame)
            prediction_paths.append(prediction_path)
            prediction_recomputes += 1
        ensemble = seed_frames[0][
            ["segment_id", "trip_uid", "VehId", "target_l"]
        ].copy()
        for system in EXPECTED_SYSTEMS:
            column = SYSTEM_TO_PREDICTION[system]
            values = np.column_stack(
                [frame[column].to_numpy(float) for frame in seed_frames]
            )
            ensemble[column] = values.mean(axis=1)
            ensemble[f"{column}_seed_std"] = values.std(axis=1)
            maximum_seed_prediction_std = max(
                maximum_seed_prediction_std,
                float(np.nanmax(ensemble[f"{column}_seed_std"].to_numpy(float))),
            )
            metrics = evaluate_system(ensemble, column)
            metric_rows.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "seed": "ensemble",
                    "aggregation": "mean_prediction",
                    "system": system,
                    **metrics,
                }
            )
        ensemble_path = (
            output / "ensemble_test_predictions" / f"{split_name}__{target_id}.parquet"
        )
        write_parquet(ensemble_path, ensemble)
        ensemble_paths[cell_key] = ensemble_path
        prediction_paths.append(ensemble_path)
        logger.log(f"PREDICT {cell_index}/8 {cell_key}")

    expected_predictions = len(supported_cells) * len(seeds)
    if len(state["prediction_records"]) != expected_predictions or any(
        not _valid_record(record) for record in state["prediction_records"].values()
    ):
        raise RuntimeError("THEORY_VALIDATION prediction grid or hashes are incomplete.")
    metrics_path = output / "system_metrics.csv"
    write_csv(metrics_path, pd.DataFrame(metric_rows))

    effects_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    bootstrap_diagnostics: dict[str, Any] = {}
    for target_index, target_id in enumerate(target_ids):
        frames = {
            environment: pd.read_parquet(
                ensemble_paths[f"{environment}|{target_id}"]
            )
            for environment in EXPECTED_ENVIRONMENTS
        }
        effects, samples, diagnostics = vehicle_cluster_bootstrap(
            frames,
            repetitions=int(config["bootstrap"]["repetitions"]),
            seed=int(config["bootstrap"]["seed"]) + target_index,
            alpha=float(config["bootstrap"]["alpha"]),
            sesoi=float(config["decision"]["relative_sesoi"]),
        )
        effects.insert(0, "target_id", target_id)
        samples.insert(0, "target_id", target_id)
        effects_frames.append(effects)
        bootstrap_frames.append(samples)
        bootstrap_diagnostics[target_id] = diagnostics
        logger.log(
            f"BOOTSTRAP target={target_id} repetitions="
            f"{config['bootstrap']['repetitions']}"
        )
    effects_frame = pd.concat(effects_frames, ignore_index=True)
    bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
    effects_path = output / "effects.csv"
    bootstrap_path = output / "bootstrap_results.csv"
    bootstrap_diagnostics_path = output / "bootstrap_diagnostics.json"
    write_csv(effects_path, effects_frame)
    write_csv(bootstrap_path, bootstrap_frame)
    write_json(bootstrap_diagnostics_path, bootstrap_diagnostics)

    primary_target_id = config["target"]["primary_upstream_id"]
    primary_effects = effects_frame[
        effects_frame["target_id"].eq(primary_target_id)
    ]
    decision = final_decision(
        primary_effects,
        sesoi=float(config["decision"]["relative_sesoi"]),
        minimum_environment_passes=3,
    )
    supporting_decision = final_decision(
        effects_frame[
            effects_frame["target_id"].eq(
                config["target"]["secondary_upstream_id"]
            )
        ],
        sesoi=float(config["decision"]["relative_sesoi"]),
        minimum_environment_passes=3,
    )
    decision_path = output / "final_decision.json"
    write_json(
        decision_path,
        {
            "primary_target": config["target"]["primary"],
            "primary_upstream_id": primary_target_id,
            "primary": decision,
            "secondary_target": config["target"]["secondary"],
            "secondary_upstream_id": config["target"]["secondary_upstream_id"],
            "secondary_supporting_only": supporting_decision,
            "causal_interpretation": False,
            "independent_confirmation": False,
        },
    )
    figure_paths = _plot_results(effects_frame, output, primary_target_id)

    control_metadata = [load_json(path) for path in control_metadata_paths]
    control_contract_pass = all(
        null_control_contract_pass(metadata) for metadata in control_metadata
    )
    whole_block_permutation_only = all(
        metadata["permutation"]["method"]
        == "whole_R6_block_sattolo_derangement_within_split_role"
        and not metadata["permutation"]["independent_column_shuffle"]
        for metadata in control_metadata
    )
    no_cross_role_permutation = all(
        int(metadata["permutation"]["cross_group_moves"]) == 0
        and int(metadata["contracts"]["permutation_cross_role_moves"]) == 0
        for metadata in control_metadata
    )
    exact_feature_counts = (
        feature_contract["low_sensor_count"] == 8
        and feature_contract["road_semantic_count"] == 142
        and all(
            len(target_contract["high_features"]) == 20
            for target_contract in feature_contract["target_contracts"].values()
        )
    )
    zero_forbidden_or_target_source_features = all(
        not target_contract["forbidden_hits"]
        and not target_contract["target_source_leakage_hits"]
        for target_contract in feature_contract["target_contracts"].values()
    )
    simultaneous_families_frozen = all(
        int(diagnostics["bootstrap_repetitions"]) == 2000
        and int(diagnostics["families"]["road_gain"]["member_count"]) == 24
        and int(
            diagnostics["families"]["semantic_specificity"]["member_count"]
        )
        == 16
        for diagnostics in bootstrap_diagnostics.values()
    )
    preserved_environments = (
        set(effects_frame["environment"].astype(str).unique())
        == set(EXPECTED_ENVIRONMENTS)
    )
    support_frame = pd.DataFrame(support_records)
    test_real_support = support_frame[
        support_frame["split_role"].eq("test")
        & support_frame["road_representation"].eq("real")
    ].copy()
    failed_test_support_cells = int(
        (~test_real_support["prospective_support_floor_pass"]).sum()
    )
    direct_duration_share_range = [
        float(test_real_support["fuel_direct_duration_share_mean"].min()),
        float(test_real_support["fuel_direct_duration_share_mean"].max()),
    ]
    maf_duration_share_range = [
        float(test_real_support["fuel_maf_duration_share_mean"].min()),
        float(test_real_support["fuel_maf_duration_share_mean"].max()),
    ]
    gate_rows = [
        (
            "protocol_frozen_before_execution",
            bool(config["experiment"]["protocol_frozen_before_execution"]),
        ),
        (
            "upstream_artifact_manifests_verified",
            upstream_verification["all_verified"],
        ),
        (
            "segment_split_release_manifest_hashes_verified",
            segment_split_release_verification["all_hashes_verified"],
        ),
        (
            "segment_split_segment_partition_contracts_pass",
            segment_split_release_verification["segment_contracts_pass"],
        ),
        (
            "segment_split_split_manifest_contract_pass",
            segment_split_release_verification["split_contract_pass"],
        ),
        ("segment_split_split_checks_all_pass", split_verification["fail_count"] == 0),
        ("exact_eight_cells", len(supported_cells) == 8),
        ("exact_four_environments", tuple(config["environments"]) == EXPECTED_ENVIRONMENTS),
        ("exact_eight_systems", tuple(config["features"]["systems"]) == EXPECTED_SYSTEMS),
        ("exact_feature_counts_8_20_142", exact_feature_counts),
        (
            "zero_forbidden_or_target_source_features",
            zero_forbidden_or_target_source_features,
        ),
        ("null_control_contracts_pass", control_contract_pass),
        ("whole_block_permutation_only", whole_block_permutation_only),
        ("no_cross_role_permutation", no_cross_role_permutation),
        (
            "equal_effective_dimension_real_noise_permutation",
            equal_effective_dimension_controls,
        ),
        (
            "complete_model_grid_192",
            expected_fits == 192 and len(state["fit_records"]) == 192,
        ),
        ("model_manifest_verified_before_test", verified_models),
        (
            "complete_prediction_grid_24",
            expected_predictions == 24
            and len(state["prediction_records"]) == 24,
        ),
        ("bootstrap_repetitions_2000", int(config["bootstrap"]["repetitions"]) == 2000),
        ("simultaneous_max_t_families_frozen", simultaneous_families_frozen),
        ("all_negative_environments_preserved", preserved_environments),
        ("cross_dataset_inputs_zero", not config["data"]["allow_cross_dataset_inputs"]),
        ("causal_claims_prohibited", not config["integrity"]["allow_causal_claims"]),
        ("information_freeze_status_unchanged_blocked", information_freeze_pointer["status"].endswith("M2_BLOCKED")),
    ]
    gate_path = output / "contract_checks.csv"
    write_csv(
        gate_path,
        pd.DataFrame(
            [
                {"check": name, "status": "PASS" if passed else "FAIL"}
                for name, passed in gate_rows
            ]
        ),
    )
    if not all(passed for _, passed in gate_rows):
        raise RuntimeError(f"THEORY_VALIDATION contract gate failure: {gate_rows}")

    report_path = output / "THEORY_VALIDATION_REPORT.md"
    report = "\n".join(
        [
            "# THEORY_VALIDATION RNEE theory validation",
            "",
            f"- Completion status: `PASS_THEORY_VALIDATION_EXECUTION_AND_INFERENCE`",
            f"- Final decision: `{decision['route']}`",
            f"- Decision basis: {decision['reason']}",
            f"- Evidence tier: `{config['experiment']['evidence_tier']}`",
            "- Target: ICE operational fuel-L; HEV operational fuel-L is supporting evidence.",
            "- Model: frozen HistGradientBoostingRegressor with absolute-error loss and seeds 0/1/2.",
            "- Inference: 2,000 vehicle-cluster replicates with per-target simultaneous max-t families.",
            "- SESOI: 5.5% relative MAE.",
            "- Protocol note: real, noise, and permutation road systems share "
            "the real-R6 train-active mask within each sensor regime; the target, "
            "systems, controls, model, SESOI, bootstrap, and decision rule are unchanged.",
            f"- Runtime: Python {runtime_versions['python']}; scikit-learn "
            f"{runtime_versions['packages']['scikit-learn']}; pandas "
            f"{runtime_versions['packages']['pandas']}; pyarrow "
            f"{runtime_versions['packages']['pyarrow']}.",
            "",
            "## Primary ICE effect table",
            "",
            *_format_effect_table(effects_frame, primary_target_id),
            "",
            "## Supporting HEV route",
            "",
            f"`{supporting_decision['route']}` — {supporting_decision['reason']}",
            "",
            "## Diagnostics that limit interpretation",
            "",
            f"- All {failed_test_support_cells}/8 target-environment test cells fail "
            "the frozen prospective trip and segment support floors. These floors are "
            "diagnostic, not exclusionary.",
            f"- Mean direct-Fuel-Rate duration share across test cells ranges from "
            f"{direct_duration_share_range[0]:.4%} to "
            f"{direct_duration_share_range[1]:.4%}; the corresponding MAF-derived "
            f"share ranges from {maf_duration_share_range[0]:.4%} to "
            f"{maf_duration_share_range[1]:.4%}. The target is therefore an "
            "assumption-bound operational fuel-volume target.",
            f"- The three configured HGB seeds are deterministic for this frozen "
            f"configuration (maximum saved row-wise seed standard deviation "
            f"{maximum_seed_prediction_std:.3e} L); they do not supply empirical "
            "model-instability robustness.",
            "- Each cell uses one frozen Gaussian-noise draw and one frozen whole-block "
            "permutation mapping; null-draw sensitivity was not added after test quality_check.",
            "- Every hash-bearing SEGMENT_SPLIT segment-partition and split-release record was "
            "verified before fitting.",
            "",
            "## Scientific interpretation boundary",
            "",
            "This is a technical theory-validation experiment on the published "
            "RNEE test corpus. It tests whether aligned road semantics outperform "
            "equal-dimensional noise and a role-restricted whole-block permutation. "
            "It does not identify causal road effects or support external-fleet or deployment claims. All four frozen "
            "cold environments are retained regardless of direction.",
            "",
            "## Required artifacts",
            "",
            "- `effects.csv`: point effects, pointwise intervals, simultaneous intervals, and states.",
            "- `bootstrap_results.csv`: all saved effect replicates.",
            "- `support_results.csv`: role/representation support and missingness diagnostics.",
            "- `shift_results.csv`: train-test feature shift diagnostics.",
            "- `feature_manifest.json` and `model_manifest.json`: frozen information and fit contracts.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "schema_version": 1,
        "experiment": "THEORY_VALIDATION",
        "status": "PASS_THEORY_VALIDATION_EXECUTION_AND_INFERENCE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "execution_backend": "local_cpu",
        "gpu_used": False,
        "runtime_versions": runtime_versions,
        "evidence_tier": config["experiment"]["evidence_tier"],
        "protocol_update": config["experiment"]["protocol_update"],
        "primary_decision": decision,
        "secondary_supporting_decision": supporting_decision,
        "cell_count": len(supported_cells),
        "system_count": len(EXPECTED_SYSTEMS),
        "seed_count": len(seeds),
        "fit_count": expected_fits,
        "fit_count_this_run": fit_count_this_run,
        "test_target_materialization_count_this_run": test_target_materializations,
        "prediction_recompute_count_this_run": prediction_recomputes,
        "bootstrap_replicates": int(config["bootstrap"]["repetitions"]),
        "active_mask_policy": config["model"]["active_mask_policy"],
        "equal_effective_dimension_controls": equal_effective_dimension_controls,
        "failed_prospective_test_support_cells": failed_test_support_cells,
        "test_cell_count": int(len(test_real_support)),
        "fuel_direct_duration_share_mean_range": direct_duration_share_range,
        "fuel_maf_duration_share_mean_range": maf_duration_share_range,
        "maximum_seed_prediction_std": maximum_seed_prediction_std,
        "null_control_draws_per_cell": {
            "gaussian_noise": 1,
            "whole_block_permutation": 1,
        },
        "cross_dataset_inputs": 0,
        "model_contract_modified": False,
        "sensor_factorial_modified": False,
        "information_freeze_status": information_freeze_pointer["status"],
        "independent_confirmation_included": False,
        "report": str(report_path),
        "config_snapshot": str(config_snapshot_path),
        "feature_manifest": str(feature_manifest_path),
        "model_manifest": str(model_manifest_path),
        "effects": str(effects_path),
        "bootstrap_results": str(bootstrap_path),
        "support_results": str(support_path),
        "shift_results": str(shift_path),
        "decision": str(decision_path),
        "contract_checks": str(gate_path),
        "upstream_release_verification": str(
            upstream_release_verification_path
        ),
        "active_feature_masks": str(active_masks_path),
        "output_directory": str(output),
        "resume_noop_status": "PENDING_REQUIRED_POST_PRIMARY_RELEASE",
    }
    write_json(summary_path, summary)
    logger.log(f"DECISION {decision['route']}")
    logger.log("COMPLETE THEORY_VALIDATION primary run")
    logger.close()

    primary_paths = (
        model_paths
        + list(control_paths.values())
        + control_metadata_paths
        + mapping_paths
        + prediction_paths
        + figure_paths
        + [
            config_snapshot_path,
            output / "run_config.yaml",
            upstream_release_verification_path,
            feature_manifest_path,
            active_masks_path,
            model_manifest_path,
            metrics_path,
            effects_path,
            bootstrap_path,
            bootstrap_diagnostics_path,
            support_path,
            shift_path,
            decision_path,
            gate_path,
            report_path,
            summary_path,
            output / "logs" / "execution.log",
        ]
    )
    write_json(
        primary_manifest_path,
        artifact_manifest(
            primary_paths,
            experiment="THEORY_VALIDATION",
            evidence_tier=config["experiment"]["evidence_tier"],
        ),
    )
    state["status"] = "PASS"
    state["completed_at_utc"] = summary["completed_at_utc"]
    state["artifact_manifest"] = str(primary_manifest_path)
    state["artifact_manifest_sha256"] = sha256_file(primary_manifest_path)
    write_json(state_path, state)
    latest_path = Path(config["output"]["latest_pointer"])
    write_json(
        latest_path,
        {
            "experiment": "THEORY_VALIDATION",
            "status": summary["status"],
            "decision": decision["route"],
            "evidence_tier": summary["evidence_tier"],
            "summary": str(summary_path),
            "report": str(report_path),
            "effects": str(effects_path),
            "bootstrap_results": str(bootstrap_path),
            "support_results": str(support_path),
            "shift_results": str(shift_path),
            "feature_manifest": str(feature_manifest_path),
            "active_feature_masks": str(active_masks_path),
            "model_manifest": str(model_manifest_path),
            "upstream_release_verification": str(
                upstream_release_verification_path
            ),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
            "resume_noop_verified": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
