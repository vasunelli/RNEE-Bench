#!/usr/bin/env python3
"""Execute the frozen ROBUSTNESS RNEE robustness extension.

ROBUSTNESS is a separate technical robustness branch. It hash-verifies and reuses
only the frozen THEORY_VALIDATION HGB baseline/real predictions, generates 20 new
train-moment Gaussian and 20 new role-restricted whole-block permutation
controls, fits the complete HGB null grid plus untuned Ridge/CatBoost
baseline/real grids, and opens test targets only after every model artifact is
frozen and verified.
"""
from __future__ import annotations

import argparse
import gc
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

from .artifacts import (
    artifact_manifest,
    load_json,
    ordered_sha256,
    sha256_file,
    verify_manifest,
    write_csv,
    write_json,
    write_parquet,
)
from .config import (
    EXPECTED_ENVIRONMENTS,
    EXPECTED_MODELS,
    EXPECTED_NULL_TYPES,
    EXPECTED_SENSORS,
    load_config,
    stable_seed,
)
from .contracts import (
    validate_segment_split_release_manifests,
    validate_membership,
    validate_numeric_frame,
    validate_split_checks,
    validate_upstream_artifact_manifests,
)
from .evaluation import regression_metrics
from .features import (
    active_masks_from_real_train,
    build_feature_contract,
    model_matrix,
)
from .inference import (
    bootstrap_target,
    final_decision,
    model_comparison,
    summarize_null_distributions,
)
from .models import dump_model, load_model, predict_model, train_model
from .null_controls import generate_noise_road, generate_permuted_road


class RunLogger:
    def __init__(self, path: Path, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open(
            "a" if append else "w", encoding="utf-8", buffering=1
        )

    def log(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()


def _runtime_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "catboost",
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
        default=Path(
            "configs/rnee/robustness/robustness_robustness_extension.yaml"
        ),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _valid_record(record: dict[str, Any]) -> bool:
    path = Path(record["path"])
    return path.exists() and sha256_file(path) == record["sha256"]


def _valid_pair_record(record: dict[str, Any]) -> bool:
    pairs = (
        ("model_path", "model_sha256"),
        ("metadata_path", "metadata_sha256"),
    )
    return all(
        Path(record[path_key]).exists()
        and sha256_file(Path(record[path_key])) == record[hash_key]
        for path_key, hash_key in pairs
    )


def _valid_control_record(record: dict[str, Any]) -> bool:
    pairs = [
        ("control_path", "control_sha256"),
        ("metadata_path", "metadata_sha256"),
    ]
    if record["null_type"] == "permutation":
        pairs.append(("mapping_path", "mapping_sha256"))
    return all(
        Path(record[path_key]).exists()
        and sha256_file(Path(record[path_key])) == record[hash_key]
        for path_key, hash_key in pairs
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


def _cell_directory(cell_key: str) -> str:
    return cell_key.replace("|", "__")


def _model_key(
    cell_key: str,
    model: str,
    sensor: str,
    representation: str,
    seed: int,
    null_type: str = "",
    draw: int | None = None,
) -> str:
    parts = [cell_key, model, sensor, representation]
    if null_type:
        parts.append(null_type)
    if draw is not None:
        parts.append(f"{draw:02d}")
    parts.append(str(seed))
    return "|".join(parts)


def _model_path(
    output: Path,
    cell_key: str,
    model: str,
    sensor: str,
    representation: str,
    seed: int,
    null_type: str = "",
    draw: int | None = None,
) -> Path:
    variant = representation
    if null_type:
        variant = f"{null_type}_draw_{draw:02d}"
    return (
        output
        / "models"
        / _cell_directory(cell_key)
        / model
        / sensor
        / variant
        / f"seed_{seed}.joblib"
    )


def _prediction_column(key: str) -> str:
    return "p__" + key.replace("|", "__")


def _control_paths(
    output: Path, cell_key: str, null_type: str, draw: int
) -> tuple[Path, Path, Path | None]:
    root = output / "null_controls" / _cell_directory(cell_key)
    control = root / null_type / f"draw_{draw:02d}.parquet"
    metadata = root / "metadata" / null_type / f"draw_{draw:02d}.json"
    mapping = (
        root / "permutation_mappings" / f"draw_{draw:02d}.parquet"
        if null_type == "permutation"
        else None
    )
    return control, metadata, mapping


def _reference_theory_validation_models(
    state: dict[str, Any],
    theory_validation_manifest: dict[str, Any],
    cells: list[dict[str, Any]],
    seeds: list[int],
) -> None:
    records = theory_validation_manifest["records"]
    for cell in cells:
        cell_key = f"{cell['split_family']}|{cell['target_id']}"
        for sensor, prefix in (("low", "L"), ("high", "H")):
            for representation, suffix in (
                ("baseline", "R0"),
                ("real", "R-real"),
            ):
                system = f"{prefix}-{suffix}"
                for seed in seeds:
                    source_key = f"{cell_key}|{system}|{seed}"
                    if source_key not in records:
                        raise RuntimeError(f"Missing THEORY_VALIDATION model record: {source_key}")
                    source = records[source_key]
                    if not _valid_pair_record(source):
                        raise RuntimeError(
                            f"Invalid frozen THEORY_VALIDATION model record: {source_key}"
                        )
                    key = _model_key(
                        cell_key,
                        "hgb",
                        sensor,
                        representation,
                        seed,
                    )
                    state["model_records"][key] = {
                        **source,
                        "record_source": "verified_THEORY_VALIDATION_reference",
                        "source_record_key": source_key,
                    }


def _create_control(
    output: Path,
    state: dict[str, Any],
    cell_key: str,
    road: pd.DataFrame,
    split_roles: pd.Series,
    segment_ids: pd.Series,
    train_reference: pd.DataFrame,
    null_type: str,
    draw: int,
    seed: int,
    zero_variance_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    record_key = f"{cell_key}|{null_type}|{draw:02d}"
    existing = state["control_records"].get(record_key)
    if existing and _valid_control_record(existing):
        return pd.read_parquet(existing["control_path"]), existing, False
    control_path, metadata_path, mapping_path = _control_paths(
        output, cell_key, null_type, draw
    )
    if null_type == "noise":
        generated, generator_metadata = generate_noise_road(
            road,
            train_reference=train_reference,
            seed=seed,
            zero_variance_scale=zero_variance_scale,
        )
        mapping = None
    else:
        generated, mapping, generator_metadata = generate_permuted_road(
            road,
            split_roles,
            seed=seed,
        )
    validate_numeric_frame(generated, list(road), f"{record_key}/generated")
    contracts = {
        "same_rows": len(generated) == len(road),
        "same_columns": list(generated) == list(road),
        "same_dtypes": [str(value) for value in generated.dtypes]
        == [str(value) for value in road.dtypes],
        "noise_recipient_missing_mask_exact": (
            generated.isna().equals(road.isna()) if null_type == "noise" else None
        ),
        "permutation_role_block_multiset_exact": (
            True if null_type == "permutation" else None
        ),
        "permutation_cross_role_moves": 0 if null_type == "permutation" else None,
        "permutation_fixed_points": 0 if null_type == "permutation" else None,
    }
    if null_type == "permutation":
        assert mapping is not None and mapping_path is not None
        mapping["recipient_segment_id"] = segment_ids.iloc[
            mapping["recipient_position"].to_numpy(int)
        ].to_numpy()
        mapping["source_segment_id"] = segment_ids.iloc[
            mapping["source_position"].to_numpy(int)
        ].to_numpy()
        mapping["recipient_role"] = split_roles.iloc[
            mapping["recipient_position"].to_numpy(int)
        ].to_numpy()
        mapping["source_role"] = split_roles.iloc[
            mapping["source_position"].to_numpy(int)
        ].to_numpy()
        if not mapping["recipient_role"].eq(mapping["source_role"]).all():
            raise RuntimeError(f"{record_key}: permutation crossed split roles.")
        for role in ("train", "validation", "calibration", "test"):
            mask = split_roles.eq(role).to_numpy()
            real_hashes = np.sort(
                pd.util.hash_pandas_object(
                    road.loc[mask].reset_index(drop=True), index=False
                ).to_numpy()
            )
            perm_hashes = np.sort(
                pd.util.hash_pandas_object(
                    generated.loc[mask].reset_index(drop=True), index=False
                ).to_numpy()
            )
            if not np.array_equal(real_hashes, perm_hashes):
                raise RuntimeError(
                    f"{record_key}: whole-block multiset changed in {role}."
                )
        write_parquet(mapping_path, mapping)
    artifact = pd.concat(
        [
            pd.DataFrame(
                {
                    "segment_id": segment_ids.to_numpy(),
                    "split_role": split_roles.to_numpy(),
                }
            ),
            generated.reset_index(drop=True),
        ],
        axis=1,
    )
    write_parquet(control_path, artifact)
    metadata = {
        "schema_version": 1,
        "experiment": "ROBUSTNESS",
        "cell_key": cell_key,
        "null_type": null_type,
        "draw": draw,
        "seed": seed,
        "generator": generator_metadata,
        "contracts": contracts,
        "target_values_read": 0,
        "full_142_dimensional_artifact_persisted": True,
    }
    write_json(metadata_path, metadata)
    record = {
        "cell_key": cell_key,
        "null_type": null_type,
        "draw": draw,
        "seed": seed,
        "control_path": str(control_path),
        "control_sha256": sha256_file(control_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "mapping_path": str(mapping_path) if mapping_path else "",
        "mapping_sha256": sha256_file(mapping_path) if mapping_path else "",
        "rows": len(artifact),
        "columns": len(road.columns),
        "contract_pass": bool(
            contracts["same_rows"]
            and contracts["same_columns"]
            and contracts["same_dtypes"]
            and (
                contracts["noise_recipient_missing_mask_exact"]
                if null_type == "noise"
                else contracts["permutation_role_block_multiset_exact"]
                and contracts["permutation_cross_role_moves"] == 0
                and contracts["permutation_fixed_points"] == 0
            )
        ),
    }
    state["control_records"][record_key] = record
    return artifact, record, True


def _fit_and_record(
    output: Path,
    state: dict[str, Any],
    cell_key: str,
    model_name: str,
    sensor: str,
    representation: str,
    seed: int,
    matrix: pd.DataFrame,
    active_features: list[str],
    target: np.ndarray,
    models_config: dict[str, Any],
    train_segment_hash: str,
    null_type: str = "",
    draw: int | None = None,
) -> bool:
    key = _model_key(
        cell_key,
        model_name,
        sensor,
        representation,
        seed,
        null_type,
        draw,
    )
    existing = state["model_records"].get(key)
    if existing and _valid_pair_record(existing):
        return False
    model_path = _model_path(
        output,
        cell_key,
        model_name,
        sensor,
        representation,
        seed,
        null_type,
        draw,
    )
    metadata_path = model_path.with_suffix(".json")
    started = time.perf_counter()
    model = train_model(
        model_name,
        matrix[active_features],
        target,
        models_config,
        seed,
    )
    fit_seconds = time.perf_counter() - started
    config_key = model_name
    metadata = {
        "schema_version": 1,
        "experiment": "ROBUSTNESS",
        "cell_key": cell_key,
        "model": model_name,
        "sensor": sensor,
        "representation": representation,
        "null_type": null_type,
        "draw": draw,
        "seed": seed,
        "train_rows": len(matrix),
        "train_segment_id_sha256": train_segment_hash,
        "raw_feature_count": matrix.shape[1],
        "active_features": active_features,
        "active_feature_count": len(active_features),
        "active_mask_policy": (
            "frozen_THEORY_VALIDATION_real_R6_train_active_mask_shared_across_"
            "models_and_all_null_draws"
        ),
        "params": models_config[config_key]["params"],
        "fit_seconds": fit_seconds,
        "validation_target_rows_used": 0,
        "calibration_target_rows_used": 0,
        "test_target_rows_used_before_fit_freeze": 0,
        "post_test_tuning": False,
    }
    dump_model(model_path, model, metadata)
    write_json(metadata_path, metadata)
    state["model_records"][key] = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_sha256": sha256_file(model_path),
        "metadata_sha256": sha256_file(metadata_path),
        "record_source": "ROBUSTNESS_new_fit",
    }
    del model
    return True


def _support_and_shift(
    cell_key: str,
    split_name: str,
    target_id: str,
    base: pd.DataFrame,
    road_features: list[str],
    active_road_features: list[str],
    floors: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_mask = base["split_role"].eq("train").to_numpy()
    test_mask = base["split_role"].eq("test").to_numpy()
    road = base[road_features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    train_active = road.loc[train_mask, active_road_features].reset_index(drop=True)
    train_mean = train_active.mean(axis=0)
    train_std = train_active.std(axis=0, ddof=0)
    valid_distance_features = train_std[
        np.isfinite(train_std.to_numpy(float)) & train_std.gt(0)
    ].index.tolist()
    distance_by_role: dict[str, np.ndarray] = {}
    for role in ("train", "validation", "calibration", "test"):
        role_mask = base["split_role"].eq(role).to_numpy()
        values = road.loc[role_mask, valid_distance_features].reset_index(drop=True)
        z = values.subtract(train_mean[valid_distance_features]).divide(
            train_std[valid_distance_features]
        )
        z = z.fillna(0.0).to_numpy(float)
        distance_by_role[role] = np.sqrt(np.mean(np.square(z), axis=1))
    train_p95 = float(np.quantile(distance_by_role["train"], 0.95))

    support_rows: list[dict[str, Any]] = []
    for role in ("train", "validation", "calibration", "test"):
        role_base = base[base["split_role"].eq(role)].reset_index(drop=True)
        role_road = road.loc[base["split_role"].eq(role)].reset_index(drop=True)
        vehicle_counts = role_base["VehId"].astype(str).value_counts()
        segment_count = len(role_base)
        checks = {
            "minimum_vehicles": int(vehicle_counts.size)
            >= int(floors["minimum_vehicles"]),
            "minimum_trips": int(role_base["trip_uid"].astype(str).nunique())
            >= int(floors["minimum_trips"]),
            "minimum_segments": segment_count >= int(floors["minimum_segments"]),
            "maximum_single_vehicle_segment_share": (
                float(vehicle_counts.iloc[0] / segment_count)
                <= float(floors["maximum_single_vehicle_segment_share"])
            ),
        }
        distances = distance_by_role[role]
        support_rows.append(
            {
                "cell_key": cell_key,
                "split_family": split_name,
                "target_id": target_id,
                "split_role": role,
                "segment_count": segment_count,
                "trip_count": int(role_base["trip_uid"].astype(str).nunique()),
                "vehicle_count": int(vehicle_counts.size),
                "maximum_single_vehicle_segment_share": float(
                    vehicle_counts.iloc[0] / segment_count
                ),
                "road_missing_cell_rate": float(role_road.isna().to_numpy().mean()),
                "all_road_features_missing_rows": int(
                    role_road.isna().all(axis=1).sum()
                ),
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
                "support_distance_definition": "diagonal_train_centroid_rms_z",
                "support_distance_feature_count": len(valid_distance_features),
                "support_distance_mean": float(distances.mean()),
                "support_distance_p50": float(np.quantile(distances, 0.50)),
                "support_distance_p95": float(np.quantile(distances, 0.95)),
                "train_support_distance_p95": train_p95,
                "fraction_beyond_train_support_p95": float(
                    np.mean(distances > train_p95)
                ),
                "prospective_support_floor_pass": bool(
                    role == "test" and all(checks.values())
                ),
                "prospective_support_failed_checks": (
                    "|".join(name for name, passed in checks.items() if not passed)
                    if role == "test"
                    else "NOT_APPLICABLE_NON_TEST_ROLE"
                ),
                "functional_failure_analysis_focus": (
                    split_name == "cold_functional_road_class"
                ),
            }
        )

    composition_tokens = [
        str(value).lower()
        for value in floors["functional_failure_analysis"][
            "composition_feature_tokens"
        ]
    ]
    shift_rows: list[dict[str, Any]] = []
    train = road.loc[train_mask].reset_index(drop=True)
    test = road.loc[test_mask].reset_index(drop=True)
    for column in road_features:
        train_values = train[column].to_numpy(float)
        test_values = test[column].to_numpy(float)
        train_finite = train_values[np.isfinite(train_values)]
        test_finite = test_values[np.isfinite(test_values)]
        train_column_mean = (
            float(train_finite.mean()) if train_finite.size else math.nan
        )
        test_column_mean = (
            float(test_finite.mean()) if test_finite.size else math.nan
        )
        train_column_std = (
            float(train_finite.std(ddof=0)) if train_finite.size else math.nan
        )
        standardized_shift = (
            float((test_column_mean - train_column_mean) / train_column_std)
            if np.isfinite(train_column_std) and train_column_std > 0
            else math.nan
        )
        if train_finite.size and test_finite.size:
            train_min = float(train_finite.min())
            train_max = float(train_finite.max())
            outside = float(
                np.mean((test_finite < train_min) | (test_finite > train_max))
            )
        else:
            train_min = train_max = outside = math.nan
        shift_rows.append(
            {
                "cell_key": cell_key,
                "split_family": split_name,
                "target_id": target_id,
                "feature": column,
                "train_observed_count": int(train_finite.size),
                "test_observed_count": int(test_finite.size),
                "train_mean": train_column_mean,
                "test_mean": test_column_mean,
                "train_std": train_column_std,
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
                "functional_composition_feature": any(
                    token in column.lower() for token in composition_tokens
                ),
                "functional_failure_analysis_focus": (
                    split_name == "cold_functional_road_class"
                ),
            }
        )
    return support_rows, shift_rows


def _write_bootstrap_long(
    path: Path,
    target_id: str,
    effects: pd.DataFrame,
    values: dict[str, np.ndarray],
    append: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    header = not append
    batch: list[pd.DataFrame] = []
    batch_rows = 0
    for row in effects.sort_values("effect_id").itertuples(index=False):
        array = values[row.effect_id]
        frame = pd.DataFrame(
            {
                "target_id": target_id,
                "replicate": np.arange(len(array), dtype=int),
                "effect_id": row.effect_id,
                "family": row.family,
                "environment": row.environment,
                "model": row.model,
                "sensor": row.sensor,
                "effect_type": row.effect_type,
                "null_type": row.null_type,
                "draw": row.draw,
                "value": array,
            }
        )
        batch.append(frame)
        batch_rows += len(frame)
        if batch_rows >= 100_000:
            pd.concat(batch, ignore_index=True).to_csv(
                path, mode=mode, header=header, index=False
            )
            mode = "a"
            header = False
            batch = []
            batch_rows = 0
    if batch:
        pd.concat(batch, ignore_index=True).to_csv(
            path, mode=mode, header=header, index=False
        )


def _plot_results(
    effects: pd.DataFrame,
    null_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output: Path,
    primary_target: str,
) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = {
        "cold_trip": "Trip",
        "cold_month": "Month",
        "cold_spatial": "Spatial",
        "cold_functional_road_class": "Functional class",
    }
    primary = effects[effects["target_id"].eq(primary_target)]
    for null_type, title, filename, color in (
        ("noise", "Noise-road gain distributions", "noise_gain_distribution", "#777777"),
        (
            "permutation",
            "Whole-block permutation gain distributions",
            "permutation_gain_distribution",
            "#d95f02",
        ),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
        for axis, sensor in zip(axes, EXPECTED_SENSORS, strict=True):
            groups: list[np.ndarray] = []
            real_values: list[float] = []
            for environment in EXPECTED_ENVIRONMENTS:
                cell = primary[
                    primary["environment"].eq(environment)
                    & primary["sensor"].eq(sensor)
                ]
                groups.append(
                    cell[
                        cell["family"].eq("hgb_null_gain")
                        & cell["null_type"].eq(null_type)
                    ]
                    .sort_values("draw")["estimate"]
                    .to_numpy(float)
                    * 100
                )
                real_values.append(
                    float(
                        cell[
                            cell["family"].eq("hgb_null_gain")
                            & cell["effect_type"].eq("gain_real")
                        ]["estimate"].iloc[0]
                    )
                    * 100
                )
            positions = np.arange(1, len(groups) + 1)
            box = axis.boxplot(groups, positions=positions, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
            axis.scatter(
                positions,
                real_values,
                color="#1f77b4",
                marker="D",
                s=35,
                label="Real road",
                zorder=3,
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axhline(5.5, color="#2ca02c", linestyle="--", linewidth=0.9)
            axis.set_ylabel("Relative MAE gain (%)")
            axis.set_title(f"{sensor.capitalize()} sensor")
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(loc="best")
        axes[-1].set_xticks(np.arange(1, 5))
        axes[-1].set_xticklabels(
            [labels[value] for value in EXPECTED_ENVIRONMENTS],
            rotation=12,
            ha="right",
        )
        fig.suptitle(f"ROBUSTNESS ICE: {title} (20 draws per cell)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"{filename}.{suffix}"
            fig.savefig(
                path, dpi=300 if suffix == "png" else None, bbox_inches="tight"
            )
            paths.append(path)
        plt.close(fig)

    model_rows = comparison[
        comparison["record_type"].eq("cell_agreement")
        & comparison["target_id"].eq(primary_target)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    colors = {"hgb": "#1f77b4", "ridge": "#4daf4a", "catboost": "#984ea3"}
    offsets = {"hgb": -0.18, "ridge": 0.0, "catboost": 0.18}
    for axis, sensor in zip(axes, EXPECTED_SENSORS, strict=True):
        subset = model_rows[model_rows["sensor"].eq(sensor)].set_index("environment")
        for model in EXPECTED_MODELS:
            axis.scatter(
                np.arange(4) + offsets[model],
                subset.loc[list(EXPECTED_ENVIRONMENTS), f"{model}_gain"].to_numpy(float)
                * 100,
                color=colors[model],
                label=model.upper() if model == "hgb" else model.capitalize(),
                s=35,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(5.5, color="#2ca02c", linestyle="--", linewidth=0.9)
        axis.axhline(-5.5, color="#b2182b", linestyle="--", linewidth=0.9)
        axis.set_ylabel("Real-road MAE gain (%)")
        axis.set_title(f"{sensor.capitalize()} sensor")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, loc="best")
    axes[-1].set_xticks(np.arange(4))
    axes[-1].set_xticklabels(
        [labels[value] for value in EXPECTED_ENVIRONMENTS],
        rotation=12,
        ha="right",
    )
    fig.suptitle("ROBUSTNESS ICE model-family comparison")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures / f"model_family_comparison.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    summary = null_summary[null_summary["target_id"].eq(primary_target)].copy()
    summary["cell"] = summary["sensor"] + " / " + summary["environment"].map(labels)
    pivot = summary.pivot(
        index="cell",
        columns="null_type",
        values="probability_real_exceeds_null",
    )
    ordered = [
        f"{sensor} / {labels[environment]}"
        for sensor in EXPECTED_SENSORS
        for environment in EXPECTED_ENVIRONMENTS
    ]
    pivot = pivot.loc[ordered, list(EXPECTED_NULL_TYPES)]
    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    image = axis.imshow(pivot.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = float(pivot.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.65 else "black",
            )
    axis.set_xticks(range(2))
    axis.set_xticklabels(["Noise", "Permutation"])
    axis.set_yticks(range(len(pivot)))
    axis.set_yticklabels(pivot.index)
    axis.set_title("ROBUSTNESS ICE probability: real-road gain exceeds null draw")
    fig.colorbar(image, ax=axis, label="Empirical probability over 20 draws")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures / f"environment_robustness.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _build_report(
    path: Path,
    decision: dict[str, Any],
    null_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    support: pd.DataFrame,
    shift: pd.DataFrame,
    runtime: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    reference_lines = [
        "| Environment | Real>Noise | Real>Permutation | Strict null | All-model strict |",
        "|---|---:|---:|---|---|",
    ]
    for item in decision["reference_cell_results"]:
        reference_lines.append(
            f"| {item['environment']} / {item['sensor']} | "
            f"{item['noise_probability_real_exceeds']:.2f} | "
            f"{item['permutation_probability_real_exceeds']:.2f} | "
            f"{item['strict_null_pass']} | {item['strict_model_pass']} |"
        )
    cell_comparison = comparison[
        comparison["record_type"].eq("cell_agreement")
        & comparison["target_id"].eq("ICE_fuel_L")
    ].copy()
    model_lines = [
        "| Environment | Sensor | HGB | Ridge | CatBoost | Direction consistent |",
        "|---|---|---:|---:|---:|---|",
    ]
    for environment in EXPECTED_ENVIRONMENTS:
        for sensor in EXPECTED_SENSORS:
            row = cell_comparison[
                cell_comparison["environment"].eq(environment)
                & cell_comparison["sensor"].eq(sensor)
            ].iloc[0]
            model_lines.append(
                f"| {environment} | {sensor} | "
                f"{_format_percent(float(row['hgb_gain']))} | "
                f"{_format_percent(float(row['ridge_gain']))} | "
                f"{_format_percent(float(row['catboost_gain']))} | "
                f"{bool(row['direction_consistent'])} |"
            )
    functional_support = support[
        support["split_family"].eq("cold_functional_road_class")
        & support["split_role"].eq("test")
    ]
    functional_shift = shift[
        shift["split_family"].eq("cold_functional_road_class")
    ].copy()
    top_shift = (
        functional_shift.groupby("feature", as_index=False)[
            "absolute_standardized_mean_shift"
        ]
        .mean()
        .sort_values("absolute_standardized_mean_shift", ascending=False)
        .head(5)
    )
    top_shift_text = ", ".join(
        f"{row.feature} ({row.absolute_standardized_mean_shift:.2f} SD)"
        for row in top_shift.itertuples(index=False)
    )
    report = "\n".join(
        [
            "# ROBUSTNESS RNEE robustness extension",
            "",
            f"- Completion status: `PASS_ROBUSTNESS_EXECUTION_AND_INFERENCE`",
            f"- Robustness route: `{decision['route']}`",
            f"- THEORY_VALIDATION disposition: `{decision['theory_validation_disposition']}`",
            f"- Decision basis: {decision['reason']}",
            "- Evidence role: technical robustness analysis on the published "
            "RNEE corpus; it is not independent confirmation.",
            "- Target: assumption-bound ICE operational fuel-L; HEV operational "
            "fuel-L is supporting only.",
            "- Frozen features: X_L=8, X_H=20, R6=142; no added road features.",
            "- Nulls: 20 independent train-moment Gaussian draws and 20 "
            "role-restricted whole-block R6 permutations per target-environment cell.",
            "- Models: frozen HGB-L1 plus untuned Ridge and CatBoost-MAE; all "
            "use seeds 0/1/2 and the same real-R6 train-active masks.",
            "- Inference: 2,000 vehicle-cluster bootstrap replicates, separate "
            "per-target simultaneous max-t families, SESOI=5.5%.",
            f"- Runtime: {elapsed_seconds / 60:.1f} min; Python "
            f"{runtime['python']}; scikit-learn "
            f"{runtime['packages']['scikit-learn']}; CatBoost "
            f"{runtime['packages']['catboost']}.",
            "",
            "## 1. Does real road stably exceed noise?",
            "",
            "The two frozen THEORY_VALIDATION-positive ICE low-sensor cells are summarized "
            "below. Empirical probabilities are over the 20 pre-frozen draws; "
            "strict null status additionally requires every draw-specific "
            "simultaneous lower bound to exceed zero and the real gain lower "
            "bound to exceed 5.5%.",
            "",
            *reference_lines,
            "",
            "## 2. Does real road stably exceed permutation?",
            "",
            "The same table reports whole-block permutation robustness. "
            "Permutations preserve the complete 142-dimensional row block and "
            "remain within train/validation/calibration/test roles; no "
            "column-wise shuffling is used.",
            "",
            "## 3. Are model families consistent?",
            "",
            *model_lines,
            "",
            "Pairwise effect correlations and every disagreement case are saved "
            "in `model_comparison.csv`; no best-model selection is performed.",
            "",
            "## 4. Functional-class failure analysis",
            "",
            f"- Functional-class test support-distance exceedance (fraction above "
            f"the corresponding train 95th percentile) ranges from "
            f"{functional_support['fraction_beyond_train_support_p95'].min():.2%} "
            f"to {functional_support['fraction_beyond_train_support_p95'].max():.2%}.",
            f"- Largest mean absolute standardized R6 shifts: {top_shift_text}.",
            "- `support_results.csv` and `shift_results.csv` retain vehicle/trip/"
            "segment support, road composition, train-centroid distance, "
            "out-of-range exposure and missingness diagnostics for both targets.",
            "",
            "## 5. THEORY_VALIDATION conclusion",
            "",
            f"`{decision['theory_validation_disposition']}`. ROBUSTNESS does not rewrite THEORY_VALIDATION "
            "or convert retrospective technical evidence into confirmation. "
            "Cold spatial and functional-road-class results remain in the matrix, "
            "and any adverse direction is retained.",
            "",
            "## Interpretation boundary",
            "",
            "The analysis tests predictive semantic specificity under frozen "
            "models, controls and splits. It does not identify causal effects of "
            "roads, does not establish external-fleet transfer, and does not "
            "repair the failed prospective trip/segment support floors or the "
            "predominantly MAF-derived target.",
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def _completed_noop(
    output: Path,
    config: dict[str, Any],
    state: dict[str, Any],
) -> int:
    manifest_path = output / "artifact_manifest.json"
    summary_path = output / "robustness_summary.json"
    manifest = load_json(manifest_path)
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
            "theory_validation_write_count": 0,
        },
    )
    primary = load_json(summary_path)
    release_summary_path = output / "release_summary.json"
    release = {
        **primary,
        "primary_summary": str(summary_path),
        "resume_noop_verified": verified,
        "resume_noop_status": "PASS" if verified else "FAIL",
        "resume_check": str(resume_path),
        "release_completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(release_summary_path, release)
    release_manifest_path = output / "release_manifest.json"
    write_json(
        release_manifest_path,
        artifact_manifest(
            [manifest_path, summary_path, resume_path, release_summary_path],
            experiment="ROBUSTNESS",
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
            "release_manifest": str(release_manifest_path),
        }
    )
    write_json(latest_path, latest)
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ROBUSTNESS_NOOP_REPRODUCTION"
                    if verified
                    else "FAIL_ROBUSTNESS_NOOP_REPRODUCTION"
                ),
                "run_directory": str(output),
                "mismatches": mismatches,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verified else 1


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
    state_path = output / "robustness_state.json"
    summary_path = output / "robustness_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if (
            state.get("status") == "PASS"
            and primary_manifest_path.exists()
            and summary_path.exists()
        ):
            return _completed_noop(output, config, state)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("A new ROBUSTNESS run requires an empty run directory.")
        output.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 1,
            "status": "INITIALIZING",
            "control_records": {},
            "model_records": {},
            "prediction_records": {},
            "test_gate": {"status": "CLOSED"},
        }
        write_json(state_path, state)
    logger = RunLogger(
        output / "logs" / "execution.log",
        append=(output / "logs" / "execution.log").exists(),
    )
    logger.log("START/RESUME ROBUSTNESS frozen robustness extension")

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
            "theory_validation_pointer",
        )
    }
    pointers = {name: load_json(path) for name, path in pointer_paths.items()}
    m1_pointer = pointers["m1_pointer"]
    m2_pointer = pointers["m2_pointer"]
    segment_pointer = pointers["segment_pointer"]
    split_pointer = pointers["split_pointer"]
    information_freeze_pointer = pointers["information_freeze_pointer"]
    sensor_factorial_pointer = pointers["sensor_factorial_pointer"]
    theory_validation_pointer = pointers["theory_validation_pointer"]
    expected_statuses = {
        "m1": (m1_pointer["status"], "PASS_MODEL_CONTRACT_M1_B1"),
        "m2": (
            m2_pointer["status"],
            "PASS_MODEL_CONTRACT_M2_VALIDATION_ONLY_BACKBONE_SELECTION_SMOKE",
        ),
        "segment": (
            segment_pointer["status"],
            "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE",
        ),
        "split": (split_pointer["status"], "PASS_BENCHMARK_SPLITS"),
        "sensor_factorial": (
            sensor_factorial_pointer["status"],
            "PASS_SENSOR_FACTORIAL_RNEE_RETROSPECTIVE_FACTORIAL",
        ),
        "theory_validation": (
            theory_validation_pointer["status"],
            "PASS_THEORY_VALIDATION_EXECUTION_AND_INFERENCE",
        ),
    }
    mismatched_statuses = {
        key: value for key, value in expected_statuses.items() if value[0] != value[1]
    }
    if mismatched_statuses:
        raise RuntimeError(f"Upstream status mismatch: {mismatched_statuses}")
    if not information_freeze_pointer["status"].endswith("M2_BLOCKED"):
        raise RuntimeError("INFORMATION_FREEZE must remain M2_BLOCKED.")
    if theory_validation_pointer["decision"] != "CONTEXT_DEPENDENT_ONLY":
        raise RuntimeError("The frozen THEORY_VALIDATION decision changed.")
    if not theory_validation_pointer.get("resume_noop_verified", False):
        raise RuntimeError("THEORY_VALIDATION no-op release verification is required.")

    upstream_manifest_paths = [
        Path(m1_pointer["manifest"]),
        Path(m1_pointer["release_manifest"]),
        Path(m2_pointer["manifest"]),
        Path(m2_pointer["release_manifest"]),
        Path(information_freeze_pointer["manifest"]),
        Path(information_freeze_pointer["release_manifest"]),
        Path(sensor_factorial_pointer["manifest"]),
        Path(sensor_factorial_pointer["release_manifest"]),
        Path(theory_validation_pointer["manifest"]),
        Path(theory_validation_pointer["release_manifest"]),
    ]
    upstream_verification = validate_upstream_artifact_manifests(
        upstream_manifest_paths
    )
    split_verification = validate_split_checks(Path(split_pointer["checks"]))
    segment_split_verification = validate_segment_split_release_manifests(
        Path(segment_pointer["manifest"]),
        Path(split_pointer["manifest"]),
    )
    theory_validation_guard_hashes = {
        "artifact_manifest": sha256_file(Path(theory_validation_pointer["manifest"])),
        "release_manifest": sha256_file(Path(theory_validation_pointer["release_manifest"])),
        "effects": sha256_file(Path(theory_validation_pointer["effects"])),
        "report": sha256_file(Path(theory_validation_pointer["report"])),
        "model_manifest": sha256_file(Path(theory_validation_pointer["model_manifest"])),
    }
    upstream_release_path = output / "upstream_release_verification.json"
    write_json(
        upstream_release_path,
        {
            "upstream_artifact_manifests": upstream_verification,
            "segment_split_release_manifests": segment_split_verification,
            "segment_split_leakage_checks": split_verification,
            "theory_validation_guard_hashes_before": theory_validation_guard_hashes,
        },
    )
    logger.log(
        f"VERIFIED upstream manifests={len(upstream_manifest_paths)} "
        f"SEGMENT_SPLIT checks={split_verification['check_count']}"
    )

    m1_contract = load_json(Path(m1_pointer["feature_contract"]))
    information_contract = load_json(Path(information_freeze_pointer["information_contract"]))
    protocol = load_json(Path(m2_pointer["protocol"]))
    if protocol["selected_backbone"] != "hist_gradient_boosting_l1":
        raise RuntimeError("Frozen MODEL_CONTRACT backbone changed.")
    if dict(config["models"]["hgb"]["params"]) != dict(
        protocol["selected_params"]
    ):
        raise RuntimeError("ROBUSTNESS HGB params differ from frozen MODEL_CONTRACT/THEORY_VALIDATION.")
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
            for key in ("low_features", "high_features", "road_features")
            for feature in target[key]
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
    environment_order = {value: index for index, value in enumerate(EXPECTED_ENVIRONMENTS)}
    target_order = {value: index for index, value in enumerate(target_ids)}
    supported_cells.sort(
        key=lambda cell: (
            environment_order[cell["split_family"]],
            target_order[cell["target_id"]],
        )
    )
    expected_cell_keys = {
        f"{environment}|{target_id}"
        for environment in EXPECTED_ENVIRONMENTS
        for target_id in target_ids
    }
    observed_cell_keys = {
        f"{cell['split_family']}|{cell['target_id']}" for cell in supported_cells
    }
    if observed_cell_keys != expected_cell_keys or len(supported_cells) != 8:
        raise RuntimeError("ROBUSTNESS requires the exact frozen eight cells.")
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
        "segments_sha256": sha256_file(Path(segment_pointer["segments_all"])),
        "r2_sha256": sha256_file(r2_path),
        "membership_sha256": {
            key: sha256_file(path) for key, path in membership_paths.items()
        },
        "theory_validation_guard_hashes": theory_validation_guard_hashes,
    }
    if "input_fingerprint" in state and state["input_fingerprint"] != fingerprint:
        raise RuntimeError("Incomplete ROBUSTNESS run fingerprint changed; use a new run dir.")
    state["input_fingerprint"] = fingerprint
    write_json(state_path, state)
    runtime_versions = _runtime_versions()
    config_snapshot_path = output / "config_snapshot.json"
    if not config_snapshot_path.exists():
        write_json(
            config_snapshot_path,
            {
                "source_config": str(args.config),
                "source_config_sha256": fingerprint["config_sha256"],
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "input_fingerprint": fingerprint,
                "runtime_versions": runtime_versions,
                "decision_rules_frozen_before_test_access": True,
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
    logger.log(f"LOAD segment feature columns={len(segment_columns)}")
    segments = pd.read_parquet(
        segment_pointer["segments_all"], columns=segment_columns
    )
    r2_features = [feature for feature in all_features if feature not in segments]
    r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_features])
    data = segments.merge(r2, on="segment_id", how="left", validate="one_to_one")
    missing_features = sorted(set(all_features) - set(data.columns))
    if missing_features:
        raise RuntimeError(f"Missing ROBUSTNESS features: {missing_features}")
    target_dataset = ds.dataset(str(segment_pointer["segments_all"]), format="parquet")

    theory_validation_root = Path(theory_validation_pointer["output_directory"])
    theory_validation_model_manifest = load_json(Path(theory_validation_pointer["model_manifest"]))
    theory_validation_active_masks = load_json(Path(theory_validation_pointer["active_feature_masks"]))
    seeds = [int(value) for value in config["models"]["seeds"]]
    _reference_theory_validation_models(
        state, theory_validation_model_manifest, supported_cells, seeds
    )
    write_json(state_path, state)

    support_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    membership_audits: list[dict[str, Any]] = []
    active_mask_audits: dict[str, Any] = {}
    control_recompute_count = 0
    fit_count_this_run = 0
    draws = int(config["null_controls"]["draws_per_type"])
    state["status"] = "PRETEST_FITTING"
    write_json(state_path, state)

    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        target_contract = feature_contract["target_contracts"][target_id]
        if (
            target_contract["target_unit"] != "L"
            or target_contract["target_channel"] != "fuel"
        ):
            raise RuntimeError(f"Non-fuel target entered ROBUSTNESS: {cell_key}")
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
        train_role_mask = cell_base["split_role"].eq("train").to_numpy()
        train_base_without_target = cell_base.loc[train_role_mask].reset_index(
            drop=True
        )
        masks, mask_audit = active_masks_from_real_train(
            train_base_without_target, target_contract
        )
        theory_validation_cell_masks = theory_validation_active_masks["cells"][cell_key]["sensor_regimes"]
        for sensor in EXPECTED_SENSORS:
            if (
                masks[f"{sensor}|road"]
                != theory_validation_cell_masks[sensor]["shared_active_features"]
            ):
                raise RuntimeError(f"{cell_key}/{sensor}: THEORY_VALIDATION road mask changed.")
            prefix = "L" if sensor == "low" else "H"
            reference_metadata = load_json(
                Path(
                    state["model_records"][
                        _model_key(
                            cell_key, "hgb", sensor, "baseline", seeds[0]
                        )
                    ]["metadata_path"]
                )
            )
            if masks[f"{sensor}|baseline"] != reference_metadata["active_features"]:
                raise RuntimeError(f"{cell_key}/{sensor}: THEORY_VALIDATION R0 mask changed.")
        mask_audit.update(
            {
                "cell_key": cell_key,
                "train_rows": len(train_base_without_target),
                "train_segment_id_sha256": ordered_sha256(
                    train_base_without_target["segment_id"]
                ),
                "exact_THEORY_VALIDATION_mask_match": True,
            }
        )
        active_mask_audits[cell_key] = mask_audit
        active_road_features = [
            feature
            for feature in masks["low|road"]
            if feature in set(road_features)
        ]
        cell_support, cell_shift = _support_and_shift(
            cell_key,
            split_name,
            target_id,
            cell_base,
            road_features,
            active_road_features,
            config["support"],
        )
        support_rows.extend(cell_support)
        shift_rows.extend(cell_shift)

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
        if len(train_base) < int(config["models"]["minimum_train_rows"]):
            raise RuntimeError(f"{cell_key}: insufficient train support.")
        target = pd.to_numeric(
            train_base[target_contract["target_column"]], errors="raise"
        ).to_numpy(float)
        train_hash = ordered_sha256(train_base["segment_id"])
        train_real_road = train_base[road_features].reset_index(drop=True)

        for model_name in ("ridge", "catboost"):
            for sensor in EXPECTED_SENSORS:
                for representation, road_frame in (
                    ("baseline", None),
                    ("real", train_real_road),
                ):
                    matrix = model_matrix(
                        train_base, target_contract, sensor, road_frame
                    )
                    active = masks[
                        f"{sensor}|{'baseline' if road_frame is None else 'road'}"
                    ]
                    for seed in seeds:
                        fitted = _fit_and_record(
                            output,
                            state,
                            cell_key,
                            model_name,
                            sensor,
                            representation,
                            seed,
                            matrix,
                            active,
                            target,
                            config["models"],
                            train_hash,
                        )
                        fit_count_this_run += int(fitted)
                        if fitted:
                            write_json(state_path, state)
                    del matrix
                    gc.collect()
        logger.log(f"MODEL_ROBUSTNESS_FITS {cell_index}/8 {cell_key}")

        train_reference = road.loc[train_role_mask].reset_index(drop=True)
        for null_type in EXPECTED_NULL_TYPES:
            base_seed = int(
                config["null_controls"][null_type]["base_seed"]
            )
            for draw in range(1, draws + 1):
                control_seed = stable_seed(
                    base_seed, split_name, target_id, f"draw_{draw:02d}"
                )
                control, control_record, generated = _create_control(
                    output,
                    state,
                    cell_key,
                    road.reset_index(drop=True),
                    cell_base["split_role"].reset_index(drop=True),
                    cell_base["segment_id"].reset_index(drop=True),
                    train_reference,
                    null_type,
                    draw,
                    control_seed,
                    float(
                        config["null_controls"]["noise"]["zero_variance_scale"]
                    ),
                )
                control_recompute_count += int(generated)
                if generated:
                    write_json(state_path, state)
                train_control = (
                    control[control["split_role"].eq("train")]
                    .sort_values("segment_id")
                    .reset_index(drop=True)
                )
                if not train_control["segment_id"].astype(str).equals(
                    train_base["segment_id"].astype(str)
                ):
                    raise RuntimeError(
                        f"{cell_key}/{null_type}/{draw}: train alignment changed."
                    )
                control_road = train_control[road_features]
                for sensor in EXPECTED_SENSORS:
                    matrix = model_matrix(
                        train_base, target_contract, sensor, control_road
                    )
                    active = masks[f"{sensor}|road"]
                    for seed in seeds:
                        fitted = _fit_and_record(
                            output,
                            state,
                            cell_key,
                            "hgb",
                            sensor,
                            "null",
                            seed,
                            matrix,
                            active,
                            target,
                            config["models"],
                            train_hash,
                            null_type,
                            draw,
                        )
                        fit_count_this_run += int(fitted)
                        if fitted:
                            write_json(state_path, state)
                    del matrix
                del control, train_control, control_road
                gc.collect()
                logger.log(
                    f"NULL_FIT {cell_index}/8 {cell_key} "
                    f"{null_type} {draw:02d}/{draws}"
                )
        del cell_base, road, train_base, train_real_road, target
        gc.collect()

    support_path = output / "support_results.csv"
    shift_path = output / "shift_results.csv"
    write_csv(support_path, pd.DataFrame(support_rows))
    write_csv(shift_path, pd.DataFrame(shift_rows))
    active_masks_path = output / "active_feature_masks.json"
    write_json(
        active_masks_path,
        {
            "schema_version": 1,
            "policy": config["features"]["active_mask_policy"],
            "exact_THEORY_VALIDATION_match_all_cells": True,
            "cells": active_mask_audits,
        },
    )
    feature_manifest_path = output / "feature_manifest.json"
    write_json(
        feature_manifest_path,
        {
            **feature_contract,
            "experiment": "ROBUSTNESS",
            "environments": list(EXPECTED_ENVIRONMENTS),
            "models": list(EXPECTED_MODELS),
            "null_types": list(EXPECTED_NULL_TYPES),
            "null_draws_per_type": draws,
            "membership_audits": membership_audits,
            "active_mask_policy": config["features"]["active_mask_policy"],
            "active_feature_masks": str(active_masks_path),
            "active_mask_audits": active_mask_audits,
            "new_road_features_added": 0,
        },
    )

    noise_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    null_paths: list[Path] = []
    for record in state["control_records"].values():
        if not _valid_control_record(record):
            raise RuntimeError(f"Invalid null record: {record}")
        row = {
            "cell_key": record["cell_key"],
            "draw": record["draw"],
            "seed": record["seed"],
            "rows": record["rows"],
            "columns": record["columns"],
            "control_path": record["control_path"],
            "control_sha256": record["control_sha256"],
            "metadata_path": record["metadata_path"],
            "metadata_sha256": record["metadata_sha256"],
            "mapping_path": record["mapping_path"],
            "mapping_sha256": record["mapping_sha256"],
            "contract_pass": record["contract_pass"],
        }
        (noise_rows if record["null_type"] == "noise" else permutation_rows).append(
            row
        )
        null_paths.extend(
            [Path(record["control_path"]), Path(record["metadata_path"])]
        )
        if record["mapping_path"]:
            null_paths.append(Path(record["mapping_path"]))
    noise_frame = pd.DataFrame(noise_rows).sort_values(["cell_key", "draw"])
    permutation_frame = pd.DataFrame(permutation_rows).sort_values(
        ["cell_key", "draw"]
    )
    noise_manifest_path = output / "noise_draw_manifest.csv"
    permutation_manifest_path = output / "permutation_manifest.csv"
    write_csv(noise_manifest_path, noise_frame)
    write_csv(permutation_manifest_path, permutation_frame)
    if len(noise_frame) != 160 or len(permutation_frame) != 160:
        raise RuntimeError("ROBUSTNESS requires 160 cell-specific draws per null type.")
    null_manifest_path = output / "null_manifest.json"
    write_json(
        null_manifest_path,
        artifact_manifest(
            null_paths,
            experiment="ROBUSTNESS",
            noise_draw_records=len(noise_frame),
            permutation_draw_records=len(permutation_frame),
            draws_per_cell_per_type=draws,
            full_control_artifacts_persisted=True,
            all_contracts_pass=bool(
                noise_frame["contract_pass"].all()
                and permutation_frame["contract_pass"].all()
            ),
        ),
    )
    null_verified, null_mismatches = verify_manifest(load_json(null_manifest_path))
    if not null_verified:
        raise RuntimeError(f"Null manifest verification failed: {null_mismatches}")

    expected_reference_models = 8 * 2 * 2 * 3
    expected_new_null_models = 8 * 2 * 2 * 20 * 3
    expected_new_robustness_models = 8 * 2 * 2 * 2 * 3
    expected_total_models = (
        expected_reference_models
        + expected_new_null_models
        + expected_new_robustness_models
    )
    if len(state["model_records"]) != expected_total_models:
        raise RuntimeError(
            f"Model grid {len(state['model_records'])} != {expected_total_models}."
        )
    model_paths: list[Path] = []
    for record in state["model_records"].values():
        if not _valid_pair_record(record):
            raise RuntimeError(f"Invalid model record: {record}")
        model_paths.extend([Path(record["model_path"]), Path(record["metadata_path"])])
    model_manifest_path = output / "model_manifest.json"
    write_json(
        model_manifest_path,
        artifact_manifest(
            model_paths,
            experiment="ROBUSTNESS",
            records=state["model_records"],
            total_model_record_count=expected_total_models,
            verified_THEORY_VALIDATION_reference_count=expected_reference_models,
            ROBUSTNESS_new_null_fit_count=expected_new_null_models,
            ROBUSTNESS_new_model_robustness_fit_count=expected_new_robustness_models,
            seeds=seeds,
            no_tuning=True,
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    model_verified, model_mismatches = verify_manifest(
        load_json(model_manifest_path)
    )
    if not model_verified:
        raise RuntimeError(f"Model manifest verification failed: {model_mismatches}")
    state["status"] = "FITS_FROZEN"
    state["model_manifest"] = str(model_manifest_path)
    state["model_manifest_sha256"] = sha256_file(model_manifest_path)
    state["null_manifest"] = str(null_manifest_path)
    state["null_manifest_sha256"] = sha256_file(null_manifest_path)
    state["test_gate"] = {
        "status": "OPEN",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256_at_open": sha256_file(model_manifest_path),
        "null_manifest_sha256_at_open": sha256_file(null_manifest_path),
        "historical_note": (
            "RNEE test targets are fixed by MODEL_CONTRACT/SENSOR_FACTORIAL/THEORY_VALIDATION; "
            "ROBUSTNESS nevertheless froze and verified its complete fit/null grid "
            "before ROBUSTNESS test materialization."
        ),
    }
    write_json(state_path, state)
    logger.log(
        f"TEST_GATE OPEN models={expected_total_models} "
        f"null_artifacts={len(null_paths)}"
    )

    prediction_frames: dict[str, pd.DataFrame] = {}
    prediction_columns_by_cell: dict[str, dict[str, str]] = {}
    prediction_paths: list[Path] = []
    prediction_metadata_paths: list[Path] = []
    test_target_materializations = 0
    prediction_recomputes = 0
    maximum_seed_std_by_model = {model: 0.0 for model in EXPECTED_MODELS}
    for cell_index, cell in enumerate(supported_cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        prediction_path = (
            output
            / "ensemble_test_predictions"
            / f"{_cell_directory(cell_key)}.parquet"
        )
        metadata_path = prediction_path.with_suffix(".json")
        existing = state["prediction_records"].get(cell_key)
        if (
            existing
            and _valid_record(
                {"path": existing["path"], "sha256": existing["sha256"]}
            )
            and _valid_record(
                {
                    "path": existing["metadata_path"],
                    "sha256": existing["metadata_sha256"],
                }
            )
        ):
            frame = pd.read_parquet(prediction_path)
            metadata = load_json(metadata_path)
            prediction_columns = metadata["prediction_columns"]
            for model, value in metadata["maximum_seed_std_by_model"].items():
                maximum_seed_std_by_model[model] = max(
                    maximum_seed_std_by_model[model], float(value)
                )
        else:
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
            frame = test_base[["segment_id", "trip_uid", "VehId"]].copy()
            frame["target_l"] = pd.to_numeric(
                test_base[target_contract["target_column"]], errors="raise"
            ).to_numpy(float)
            prediction_columns: dict[str, str] = {}
            prediction_seed_std: dict[str, float] = {}
            theory_validation_ensemble = pd.read_parquet(
                theory_validation_root
                / "ensemble_test_predictions"
                / f"{_cell_directory(cell_key)}.parquet"
            ).sort_values("segment_id").reset_index(drop=True)
            if not frame["segment_id"].astype(str).equals(
                theory_validation_ensemble["segment_id"].astype(str)
            ):
                raise RuntimeError(f"{cell_key}: THEORY_VALIDATION prediction support changed.")
            if not np.array_equal(
                frame["target_l"].to_numpy(float),
                theory_validation_ensemble["target_l"].to_numpy(float),
            ):
                raise RuntimeError(f"{cell_key}: THEORY_VALIDATION target values changed.")
            theory_validation_columns = {
                "hgb|low|baseline": "prediction_l_r0",
                "hgb|low|real": "prediction_l_r_real",
                "hgb|high|baseline": "prediction_h_r0",
                "hgb|high|real": "prediction_h_r_real",
            }
            for key, source_column in theory_validation_columns.items():
                column = _prediction_column(key)
                frame[column] = theory_validation_ensemble[source_column].to_numpy(float)
                prediction_columns[key] = column
                seed_std_column = f"{source_column}_seed_std"
                prediction_seed_std[key] = float(
                    theory_validation_ensemble[seed_std_column].max()
                )
                maximum_seed_std_by_model["hgb"] = max(
                    maximum_seed_std_by_model["hgb"],
                    prediction_seed_std[key],
                )
            masks = {
                f"{sensor}|baseline": active_mask_audits[cell_key][
                    "sensor_regimes"
                ][sensor]["r0_active_features"]
                for sensor in EXPECTED_SENSORS
            }
            masks.update(
                {
                    f"{sensor}|road": active_mask_audits[cell_key][
                        "sensor_regimes"
                    ][sensor]["real_road_active_features"]
                    for sensor in EXPECTED_SENSORS
                }
            )
            test_real_road = test_base[road_features].reset_index(drop=True)
            for model_name in ("ridge", "catboost"):
                for sensor in EXPECTED_SENSORS:
                    for representation, road_frame in (
                        ("baseline", None),
                        ("real", test_real_road),
                    ):
                        matrix = model_matrix(
                            test_base, target_contract, sensor, road_frame
                        )
                        active = masks[
                            f"{sensor}|{'baseline' if road_frame is None else 'road'}"
                        ]
                        seed_predictions: list[np.ndarray] = []
                        for seed in seeds:
                            record = state["model_records"][
                                _model_key(
                                    cell_key,
                                    model_name,
                                    sensor,
                                    representation,
                                    seed,
                                )
                            ]
                            payload = load_model(Path(record["model_path"]))
                            seed_predictions.append(
                                predict_model(payload["model"], matrix[active])
                            )
                        stacked = np.column_stack(seed_predictions)
                        key = f"{model_name}|{sensor}|{representation}"
                        column = _prediction_column(key)
                        frame[column] = stacked.mean(axis=1)
                        prediction_columns[key] = column
                        prediction_seed_std[key] = float(
                            np.max(stacked.std(axis=1))
                        )
                        maximum_seed_std_by_model[model_name] = max(
                            maximum_seed_std_by_model[model_name],
                            prediction_seed_std[key],
                        )
                        del matrix, stacked, seed_predictions
                        gc.collect()
            for null_type in EXPECTED_NULL_TYPES:
                for draw in range(1, draws + 1):
                    control_path, _, _ = _control_paths(
                        output, cell_key, null_type, draw
                    )
                    test_control = pd.read_parquet(control_path)
                    test_control = (
                        test_control[test_control["split_role"].eq("test")]
                        .sort_values("segment_id")
                        .reset_index(drop=True)
                    )
                    if not test_control["segment_id"].astype(str).equals(
                        test_base["segment_id"].astype(str)
                    ):
                        raise RuntimeError(
                            f"{cell_key}/{null_type}/{draw}: test alignment changed."
                        )
                    control_road = test_control[road_features]
                    for sensor in EXPECTED_SENSORS:
                        matrix = model_matrix(
                            test_base, target_contract, sensor, control_road
                        )
                        active = masks[f"{sensor}|road"]
                        seed_predictions = []
                        for seed in seeds:
                            record = state["model_records"][
                                _model_key(
                                    cell_key,
                                    "hgb",
                                    sensor,
                                    "null",
                                    seed,
                                    null_type,
                                    draw,
                                )
                            ]
                            payload = load_model(Path(record["model_path"]))
                            seed_predictions.append(
                                predict_model(payload["model"], matrix[active])
                            )
                        stacked = np.column_stack(seed_predictions)
                        key = f"hgb|{sensor}|{null_type}|{draw:02d}"
                        column = _prediction_column(key)
                        frame[column] = stacked.mean(axis=1)
                        prediction_columns[key] = column
                        prediction_seed_std[key] = float(
                            np.max(stacked.std(axis=1))
                        )
                        maximum_seed_std_by_model["hgb"] = max(
                            maximum_seed_std_by_model["hgb"],
                            prediction_seed_std[key],
                        )
                        del matrix, stacked, seed_predictions
                    del test_control, control_road
                    gc.collect()
            write_parquet(prediction_path, frame)
            metadata = {
                "schema_version": 1,
                "cell_key": cell_key,
                "rows": len(frame),
                "segment_id_sha256": ordered_sha256(frame["segment_id"]),
                "prediction_columns": prediction_columns,
                "prediction_seed_std": prediction_seed_std,
                "maximum_seed_std_by_model": {
                    model: max(
                        [
                            value
                            for key, value in prediction_seed_std.items()
                            if key.startswith(f"{model}|")
                        ],
                        default=0.0,
                    )
                    for model in EXPECTED_MODELS
                },
                "THEORY_VALIDATION_hgb_baseline_real_reused": True,
                "test_target_materialization_count": 1,
            }
            write_json(metadata_path, metadata)
            state["prediction_records"][cell_key] = {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "rows": len(frame),
                "segment_id_sha256": ordered_sha256(frame["segment_id"]),
            }
            write_json(state_path, state)
            prediction_recomputes += 1
        prediction_frames[cell_key] = frame
        prediction_columns_by_cell[cell_key] = prediction_columns
        prediction_paths.append(prediction_path)
        prediction_metadata_paths.append(metadata_path)
        logger.log(f"PREDICT {cell_index}/8 {cell_key}")

    if len(state["prediction_records"]) != 8:
        raise RuntimeError("ROBUSTNESS prediction grid is incomplete.")
    metric_rows: list[dict[str, Any]] = []
    for cell in supported_cells:
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        cell_key = f"{split_name}|{target_id}"
        frame = prediction_frames[cell_key]
        for key, column in prediction_columns_by_cell[cell_key].items():
            parts = key.split("|")
            model_name, sensor, representation = parts[:3]
            draw = int(parts[3]) if len(parts) == 4 else None
            metrics = regression_metrics(
                frame["target_l"].to_numpy(float), frame[column].to_numpy(float)
            )
            metric_rows.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "model": model_name,
                    "sensor": sensor,
                    "representation": representation,
                    "draw": draw,
                    "aggregation": (
                        "verified_THEORY_VALIDATION_seed_ensemble_reuse"
                        if model_name == "hgb"
                        and representation in ("baseline", "real")
                        else "mean_prediction_across_seeds_0_1_2"
                    ),
                    "sample_count": len(frame),
                    "vehicle_count": int(frame["VehId"].astype(str).nunique()),
                    "trip_count": int(frame["trip_uid"].astype(str).nunique()),
                    **metrics,
                }
            )
    metrics_path = output / "system_metrics.csv"
    write_csv(metrics_path, pd.DataFrame(metric_rows))

    effects_frames: list[pd.DataFrame] = []
    bootstrap_path = output / "bootstrap_results.csv"
    bootstrap_diagnostics: dict[str, Any] = {}
    append_bootstrap = False
    for target_index, target_id in enumerate(target_ids):
        environment_frames = {
            environment: prediction_frames[f"{environment}|{target_id}"]
            for environment in EXPECTED_ENVIRONMENTS
        }
        prediction_columns = {
            environment: prediction_columns_by_cell[f"{environment}|{target_id}"]
            for environment in EXPECTED_ENVIRONMENTS
        }
        effects, bootstrap_values, diagnostics = bootstrap_target(
            target_id,
            environment_frames,
            prediction_columns,
            draws,
            int(config["bootstrap"]["repetitions"]),
            int(config["bootstrap"]["seed"]) + target_index,
            float(config["bootstrap"]["alpha"]),
            float(config["decision"]["relative_sesoi"]),
        )
        effects_frames.append(effects)
        _write_bootstrap_long(
            bootstrap_path,
            target_id,
            effects,
            bootstrap_values,
            append_bootstrap,
        )
        append_bootstrap = True
        bootstrap_diagnostics[target_id] = diagnostics
        logger.log(f"BOOTSTRAP {target_id} effects={len(effects)}")
        del bootstrap_values
        gc.collect()
    effects_frame = pd.concat(effects_frames, ignore_index=True)
    effects_path = output / "effects.csv"
    write_csv(effects_path, effects_frame)
    bootstrap_diagnostics_path = output / "bootstrap_diagnostics.json"
    write_json(bootstrap_diagnostics_path, bootstrap_diagnostics)
    null_summary = summarize_null_distributions(effects_frame)
    null_summary_path = output / "null_distribution_summary.csv"
    write_csv(null_summary_path, null_summary)
    comparison = model_comparison(effects_frame)
    model_comparison_path = output / "model_comparison.csv"
    write_csv(model_comparison_path, comparison)
    decision = final_decision(effects_frame, null_summary, config)
    decision_path = output / "final_decision.json"
    write_json(decision_path, decision)
    figure_paths = _plot_results(
        effects_frame,
        null_summary,
        comparison,
        output,
        config["target"]["primary_upstream_id"],
    )

    theory_validation_guard_hashes_after = {
        "artifact_manifest": sha256_file(Path(theory_validation_pointer["manifest"])),
        "release_manifest": sha256_file(Path(theory_validation_pointer["release_manifest"])),
        "effects": sha256_file(Path(theory_validation_pointer["effects"])),
        "report": sha256_file(Path(theory_validation_pointer["report"])),
        "model_manifest": sha256_file(Path(theory_validation_pointer["model_manifest"])),
    }
    theory_validation_unchanged = theory_validation_guard_hashes_after == theory_validation_guard_hashes
    expected_family_sizes = config["bootstrap"]["simultaneous_families_per_target"]
    family_sizes_pass = all(
        int(bootstrap_diagnostics[target_id]["families"][family]["member_count"])
        == int(expected)
        for target_id in target_ids
        for family, expected in expected_family_sizes.items()
    )
    all_environments_preserved = set(
        effects_frame["environment"].astype(str).unique()
    ) == set(EXPECTED_ENVIRONMENTS)
    support_frame = pd.DataFrame(support_rows)
    failed_test_support = int(
        (
            ~support_frame[
                support_frame["split_role"].eq("test")
            ]["prospective_support_floor_pass"].astype(bool)
        ).sum()
    )
    test_support = support_frame[support_frame["split_role"].eq("test")]
    direct_share_range = [
        float(test_support["fuel_direct_duration_share_mean"].min()),
        float(test_support["fuel_direct_duration_share_mean"].max()),
    ]
    maf_share_range = [
        float(test_support["fuel_maf_duration_share_mean"].min()),
        float(test_support["fuel_maf_duration_share_mean"].max()),
    ]
    gate_rows = [
        (
            "protocol_frozen_before_execution",
            bool(config["experiment"]["protocol_frozen_before_execution"]),
        ),
        ("ROBUSTNESS_separate_from_THEORY_VALIDATION", config["experiment"]["separate_from_theory_validation"]),
        ("THEORY_VALIDATION_guard_hashes_unchanged", theory_validation_unchanged),
        (
            "upstream_artifact_manifests_verified",
            upstream_verification["all_verified"],
        ),
        (
            "SEGMENT_SPLIT_release_hashes_verified",
            segment_split_verification["all_hashes_verified"],
        ),
        ("SEGMENT_SPLIT_leakage_checks_all_pass", split_verification["fail_count"] == 0),
        ("exact_eight_cells", len(supported_cells) == 8),
        ("exact_four_environments", all_environments_preserved),
        (
            "exact_feature_counts_8_20_142",
            feature_contract["low_sensor_count"] == 8
            and feature_contract["road_semantic_count"] == 142
            and all(
                len(contract["high_features"]) == 20
                for contract in feature_contract["target_contracts"].values()
            ),
        ),
        (
            "active_masks_exact_THEORY_VALIDATION_match",
            all(
                audit["exact_THEORY_VALIDATION_mask_match"]
                for audit in active_mask_audits.values()
            ),
        ),
        ("noise_draw_records_160", len(noise_frame) == 160),
        ("permutation_draw_records_160", len(permutation_frame) == 160),
        ("all_null_contracts_pass", bool(
            noise_frame["contract_pass"].all()
            and permutation_frame["contract_pass"].all()
        )),
        ("null_manifest_verified", null_verified),
        ("whole_block_permutation_only", bool(
            all(
                load_json(Path(path))["generator"]["method"]
                == "whole_R6_block_sattolo_derangement_within_split_role"
                and not load_json(Path(path))["generator"][
                    "independent_column_shuffle"
                ]
                for path in permutation_frame["metadata_path"]
            )
        )),
        ("complete_model_grid_2208", len(state["model_records"]) == 2208),
        ("verified_THEORY_VALIDATION_model_references_96", expected_reference_models == 96),
        ("new_ROBUSTNESS_fits_2112", expected_new_null_models + expected_new_robustness_models == 2112),
        ("model_manifest_verified_before_test", model_verified),
        ("test_gate_opened_after_fit_and_null_freeze", state["test_gate"]["status"] == "OPEN"),
        ("complete_prediction_grid_8", len(state["prediction_records"]) == 8),
        ("bootstrap_repetitions_2000", int(config["bootstrap"]["repetitions"]) == 2000),
        ("simultaneous_family_sizes_frozen", family_sizes_pass),
        ("negative_environments_preserved", all_environments_preserved),
        ("no_model_tuning", not config["models"]["post_test_tuning_authorized"]),
        ("no_new_road_features", not config["features"]["new_road_features_authorized"]),
        ("no_cross_dataset_inputs", not config["data"]["cross_dataset_inputs_authorized"]),
        ("causal_claims_prohibited", not config["integrity"]["causal_claims_authorized"]),
        ("INFORMATION_FREEZE_remains_M2_blocked", information_freeze_pointer["status"].endswith("M2_BLOCKED")),
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
        raise RuntimeError(f"ROBUSTNESS contract gate failed: {gate_rows}")

    report_path = output / "ROBUSTNESS_REPORT.md"
    elapsed_seconds = time.perf_counter() - started
    _build_report(
        report_path,
        decision,
        null_summary,
        comparison,
        support_frame,
        pd.DataFrame(shift_rows),
        runtime_versions,
        elapsed_seconds,
    )
    summary = {
        "schema_version": 1,
        "experiment": "ROBUSTNESS",
        "status": "PASS_ROBUSTNESS_EXECUTION_AND_INFERENCE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "execution_backend": "local_CPU_HGB_Ridge__GPU_CatBoost",
        "gpu_used": True,
        "runtime_versions": runtime_versions,
        "evidence_tier": config["experiment"]["evidence_tier"],
        "decision": decision,
        "cell_count": 8,
        "null_draws_per_cell_per_type": 20,
        "null_control_record_count": len(state["control_records"]),
        "model_record_count": len(state["model_records"]),
        "verified_THEORY_VALIDATION_model_reference_count": expected_reference_models,
        "new_fit_count_expected": 2112,
        "new_fit_count_this_invocation": fit_count_this_run,
        "control_recompute_count_this_invocation": control_recompute_count,
        "test_target_materialization_count_this_invocation": test_target_materializations,
        "prediction_recompute_count_this_invocation": prediction_recomputes,
        "bootstrap_replicates": int(config["bootstrap"]["repetitions"]),
        "maximum_seed_prediction_std_by_model": maximum_seed_std_by_model,
        "failed_prospective_test_support_cells": failed_test_support,
        "test_cell_count": int(len(test_support)),
        "fuel_direct_duration_share_mean_range": direct_share_range,
        "fuel_maf_duration_share_mean_range": maf_share_range,
        "THEORY_VALIDATION_modified": False,
        "THEORY_VALIDATION_guard_hashes_before": theory_validation_guard_hashes,
        "THEORY_VALIDATION_guard_hashes_after": theory_validation_guard_hashes_after,
        "cross_dataset_inputs": 0,
        "confirmation_authorized": False,
        "output_directory": str(output),
        "report": str(report_path),
        "config_snapshot": str(config_snapshot_path),
        "feature_manifest": str(feature_manifest_path),
        "model_manifest": str(model_manifest_path),
        "null_manifest": str(null_manifest_path),
        "noise_draw_manifest": str(noise_manifest_path),
        "permutation_manifest": str(permutation_manifest_path),
        "effects": str(effects_path),
        "bootstrap_results": str(bootstrap_path),
        "model_comparison": str(model_comparison_path),
        "null_distribution_summary": str(null_summary_path),
        "support_results": str(support_path),
        "shift_results": str(shift_path),
        "decision_path": str(decision_path),
        "contract_checks": str(gate_path),
        "upstream_release_verification": str(upstream_release_path),
        "resume_noop_status": "PENDING_REQUIRED_POST_PRIMARY_RELEASE",
    }
    write_json(summary_path, summary)
    logger.log(f"DECISION {decision['route']}")
    logger.log("COMPLETE ROBUSTNESS primary run")
    logger.close()

    primary_paths = (
        model_paths
        + null_paths
        + prediction_paths
        + prediction_metadata_paths
        + figure_paths
        + [
            config_snapshot_path,
            output / "run_config.yaml",
            upstream_release_path,
            feature_manifest_path,
            active_masks_path,
            model_manifest_path,
            null_manifest_path,
            noise_manifest_path,
            permutation_manifest_path,
            metrics_path,
            effects_path,
            bootstrap_path,
            bootstrap_diagnostics_path,
            model_comparison_path,
            null_summary_path,
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
            experiment="ROBUSTNESS",
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
            "experiment": "ROBUSTNESS",
            "status": summary["status"],
            "decision": decision["route"],
            "theory_validation_disposition": decision["theory_validation_disposition"],
            "evidence_tier": summary["evidence_tier"],
            "summary": str(summary_path),
            "report": str(report_path),
            "effects": str(effects_path),
            "bootstrap_results": str(bootstrap_path),
            "model_comparison": str(model_comparison_path),
            "null_distribution_summary": str(null_summary_path),
            "support_results": str(support_path),
            "shift_results": str(shift_path),
            "feature_manifest": str(feature_manifest_path),
            "model_manifest": str(model_manifest_path),
            "null_manifest": str(null_manifest_path),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
            "resume_noop_verified": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
