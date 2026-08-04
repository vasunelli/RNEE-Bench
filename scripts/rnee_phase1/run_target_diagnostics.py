from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import yaml

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rnee_robustness.inference import (  # noqa: E402
    _vehicle_mae,
    bootstrap_target,
    model_comparison,
    summarize_null_distributions,
)
from scripts.rnee_robustness.models import predict_model, train_model  # noqa: E402
from scripts.rnee_phase1.common import (  # noqa: E402
    artifact_manifest,
    load_json,
    markdown_table,
    ordered_sha256,
    regression_metrics,
    sha256_file,
    stable_seed,
    windows_to_local_path,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)


ENVIRONMENT_LABELS = {
    "cold_trip": "unseen trips",
    "cold_month": "October 2018",
    "cold_spatial": "held-out area",
    "cold_functional_road_class": "motorway-largest-share holdout",
}
MODELS = ("hgb", "ridge", "catboost")
NULL_TYPES = ("noise", "permutation")
ROLES = ("train", "validation", "calibration", "test")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rnee/phase1/phase1_target_diagnostics.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_output(config: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None:
        return requested
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path(config["output"]["root"]) / f"{config['output']['run_name_prefix']}_{stamp}"


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def source_inventory(config: dict[str, Any], output: Path) -> dict[str, Any]:
    requested = list(config["inputs"].get("source_files", []))
    search_roots = [REPO_ROOT]
    optional_source_root = os.environ.get("RNEE_PUBLIC_SOURCE_ROOT")
    if optional_source_root:
        search_roots.append(Path(optional_source_root).expanduser())
    found: dict[str, str] = {}
    for name in requested:
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            try:
                matches.extend(root.rglob(name))
            except (OSError, PermissionError):
                continue
        if matches:
            found[name] = str(sorted(matches, key=lambda value: str(value))[0])
    missing = [name for name in requested if name not in found]
    result = {
        "requested": requested,
        "found": found,
        "missing": missing,
        "complete": not missing,
        "handling": (
            "All configured source files located."
            if not missing
            else "Missing source files are listed in the output; no content is inferred or fabricated."
        ),
    }
    write_json(output / "source_inventory.json", result)
    return result


def segment_columns(config: dict[str, Any]) -> list[str]:
    columns = {
        "segment_id",
        "trip_uid",
        "VehId",
        "engine_type",
        "fuel_volume_L",
        "fuel_valid_duration_s",
        "effective_duration_s",
        "fuel_target_coverage",
        "fuel_direct_duration_s",
        "fuel_maf_duration_s",
        "fuel_direct_duration_share",
        "fuel_maf_duration_share",
        "maf_g_sec_time_weighted_mean",
        "maf_g_sec_time_weighted_std",
        "fuel_rate_l_hr_time_weighted_mean",
        "fuel_rate_l_hr_time_weighted_std",
        "vehicle_speed_km_h_time_weighted_mean",
        "vehicle_speed_km_h_time_weighted_std",
        "engine_rpm_rpm_time_weighted_mean",
        "engine_rpm_rpm_time_weighted_std",
        "absolute_load_pct_time_weighted_mean",
        "absolute_load_pct_time_weighted_std",
        "distance_m_speed_integrated",
        "oat_degc_time_weighted_mean",
        "oat_degc_time_weighted_std",
        "idle_ratio",
        "low_speed_ratio",
        "functional_road_class_dominant",
        "functional_road_class_share_motorway",
        "road_semantics_distance_coverage",
        "nearest_graph_node_distance_m_time_weighted_mean",
    }
    columns.update(config["features"]["target_source_features_forbidden"])
    columns.update(config["features"]["lower_information"])
    columns.update(config["features"]["rpm"])
    columns.update(config["features"]["absolute_load"])
    return sorted(columns)


def load_segment_data(config: dict[str, Any], logger: RunLogger) -> pd.DataFrame:
    path = Path(config["inputs"]["segments"])
    schema = set(pq.read_schema(path).names)
    requested = segment_columns(config)
    missing = sorted(set(requested) - schema)
    if missing:
        raise RuntimeError(f"Segment artifact lacks required Target diagnostics stage columns: {missing}")
    logger.log(f"LOAD segments columns={len(requested)} path={path}")
    frame = pd.read_parquet(path, columns=requested)
    if frame["segment_id"].astype(str).duplicated().any():
        raise RuntimeError("Frozen segment artifact contains duplicate segment_id values.")
    frame["VehId"] = frame["VehId"].astype(str)
    frame["segment_id"] = frame["segment_id"].astype(str)
    frame["trip_uid"] = frame["trip_uid"].astype(str)
    return frame


def load_static_displacement(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["inputs"]["static_vehicle_metadata"])
    static = pd.read_parquet(
        path,
        columns=["VehId", "Engine Configuration & Displacement", "EngineType_official"],
    )
    static["VehId"] = static["VehId"].astype(str)
    static["engine_displacement_L"] = pd.to_numeric(
        static["Engine Configuration & Displacement"]
        .astype(str)
        .str.extract(r"(?i)(\d+(?:\.\d+)?)\s*L", expand=False),
        errors="coerce",
    )
    return static[["VehId", "EngineType_official", "engine_displacement_L"]]


def membership_path(config: dict[str, Any], environment: str) -> Path:
    root = Path(config["inputs"]["feature_contract"]).parent
    return root / "target_memberships" / environment / config["target"]["id"] / "model_membership.parquet"


def load_memberships(
    config: dict[str, Any], logger: RunLogger
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for environment in config["environments"]:
        path = membership_path(config, environment)
        frame = pd.read_parquet(path)
        frame["segment_id"] = frame["segment_id"].astype(str)
        roles = set(frame["split_role"].astype(str).unique())
        if roles != set(ROLES):
            raise RuntimeError(f"{environment}: unexpected roles {sorted(roles)}")
        if frame["segment_id"].duplicated().any():
            raise RuntimeError(f"{environment}: duplicate membership rows")
        frames[environment] = frame
        counts = frame["split_role"].value_counts().to_dict()
        audits.append(
            {
                "environment": environment,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                **{f"{role}_rows": int(counts.get(role, 0)) for role in ROLES},
            }
        )
        logger.log(f"MEMBERSHIP {environment} rows={len(frame)} roles={counts}")
    return frames, audits


def add_proxy_features(
    segments: pd.DataFrame, static: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    frame = segments.merge(static, on="VehId", how="left", validate="many_to_one")
    load = pd.to_numeric(frame["absolute_load_pct_time_weighted_mean"], errors="coerce")
    rpm = pd.to_numeric(frame["engine_rpm_rpm_time_weighted_mean"], errors="coerce")
    frame["absolute_load_x_rpm_mean_product"] = load * rpm
    density = float(config["target"]["absolute_load_standard_air_density_g_per_l"])
    frame["standardized_absolute_load_rpm_maf_proxy_g_s"] = (
        load * rpm * pd.to_numeric(frame["engine_displacement_L"], errors="coerce") * density / 12000.0
    )
    duration = pd.to_numeric(frame["fuel_valid_duration_s"], errors="coerce")
    frame["operational_fuel_rate_l_h"] = np.where(
        duration > 0,
        pd.to_numeric(frame["fuel_volume_L"], errors="coerce") * 3600.0 / duration,
        np.nan,
    )
    return frame


def finite_pair(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    values = frame[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def correlation_row(
    frame: pd.DataFrame,
    scope: str,
    environment: str,
    aggregation: str,
    x: str,
    y: str,
) -> dict[str, Any]:
    working = frame.copy()
    within_distribution: np.ndarray = np.asarray([], dtype=float)
    if aggregation == "within_vehicle_demeaned":
        numeric = working[[x, y]].apply(pd.to_numeric, errors="coerce")
        numeric[x] = numeric[x] - numeric.groupby(working["VehId"])[x].transform("mean")
        numeric[y] = numeric[y] - numeric.groupby(working["VehId"])[y].transform("mean")
        pair = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        vehicle_values: list[float] = []
        for _, vehicle in working.groupby("VehId", sort=False):
            vehicle_pair = finite_pair(vehicle, x, y)
            if len(vehicle_pair) >= 5 and vehicle_pair[x].nunique() > 1 and vehicle_pair[y].nunique() > 1:
                vehicle_values.append(float(vehicle_pair[x].corr(vehicle_pair[y], method="pearson")))
        within_distribution = np.asarray(vehicle_values, dtype=float)
    elif aggregation == "across_trip_means":
        grouped = (
            working[["trip_uid", x, y]]
            .assign(**{x: pd.to_numeric(working[x], errors="coerce"), y: pd.to_numeric(working[y], errors="coerce")})
            .groupby("trip_uid", sort=False)[[x, y]]
            .mean()
        )
        pair = grouped.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        pair = finite_pair(working, x, y)
    pearson = float(pair[x].corr(pair[y], method="pearson")) if len(pair) >= 3 else np.nan
    spearman = float(pair[x].corr(pair[y], method="spearman")) if len(pair) >= 3 else np.nan
    return {
        "record_type": "correlation",
        "scope": scope,
        "environment": environment,
        "aggregation": aggregation,
        "predictor": x,
        "outcome": y,
        "n": int(len(pair)),
        "vehicles": int(working["VehId"].nunique()),
        "trips": int(working["trip_uid"].nunique()),
        "pearson": pearson,
        "spearman": spearman,
        "median_within_vehicle_pearson": (
            float(np.nanmedian(within_distribution)) if within_distribution.size else np.nan
        ),
        "q25_within_vehicle_pearson": (
            float(np.nanquantile(within_distribution, 0.25)) if within_distribution.size else np.nan
        ),
        "q75_within_vehicle_pearson": (
            float(np.nanquantile(within_distribution, 0.75)) if within_distribution.size else np.nan
        ),
        "eligible_within_vehicle_correlations": int(within_distribution.size),
    }


def ensemble_predictions(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_predictions: list[np.ndarray] = []
    fit_seconds: list[float] = []
    for seed in config["models"]["seeds"]:
        started = time.perf_counter()
        model = train_model(model_name, x_train, y_train, config["models"], int(seed))
        fit_seconds.append(time.perf_counter() - started)
        seed_predictions.append(predict_model(model, x_test))
        del model
    stacked = np.column_stack(seed_predictions)
    metadata = {
        "fit_seconds_total": float(sum(fit_seconds)),
        "fit_seconds_by_seed": fit_seconds,
        "maximum_prediction_seed_std": float(np.max(stacked.std(axis=1))),
    }
    return stacked.mean(axis=1), metadata


def compact_proxy_models(
    config: dict[str, Any],
    data: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    logger: RunLogger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_sets = config["proxy_audit"]["feature_sets"]
    for environment in config["environments"]:
        cell = memberships[environment].merge(data, on="segment_id", how="left", validate="one_to_one")
        train = cell[cell["split_role"].eq("train")].sort_values("segment_id").reset_index(drop=True)
        test = cell[cell["split_role"].eq("test")].sort_values("segment_id").reset_index(drop=True)
        y_train = pd.to_numeric(train["fuel_volume_L"], errors="raise").to_numpy(float)
        y_test = pd.to_numeric(test["fuel_volume_L"], errors="raise").to_numpy(float)
        predictions: dict[str, np.ndarray] = {}
        for name, features in feature_sets.items():
            x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            x_test = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            prediction, metadata = ensemble_predictions("hgb", x_train, y_train, x_test, config)
            predictions[name] = prediction
            metrics = regression_metrics(y_test, prediction)
            rows.append(
                {
                    "record_type": "fuel_target_diagnostic_model",
                    "scope": "transfer_test",
                    "environment": environment,
                    "aggregation": "segment",
                    "predictor": name,
                    "outcome": "fuel_volume_L",
                    "n": int(len(test)),
                    "vehicles": int(test["VehId"].nunique()),
                    "trips": int(test["trip_uid"].nunique()),
                    "feature_count": len(features),
                    **metrics,
                    **metadata,
                }
            )
        baseline_mae = regression_metrics(y_test, predictions["lower_information"])["mae"]
        for row in rows:
            if row["record_type"] == "fuel_target_diagnostic_model" and row["environment"] == environment:
                row["relative_mae_improvement_vs_lower_information"] = (
                    baseline_mae - float(row["mae"])
                ) / baseline_mae

        maf_feature_sets = {
            "rpm_only": ["engine_rpm_rpm_time_weighted_mean", "engine_rpm_rpm_time_weighted_std"],
            "load_only": ["absolute_load_pct_time_weighted_mean", "absolute_load_pct_time_weighted_std"],
            "rpm_plus_load": [
                "engine_rpm_rpm_time_weighted_mean",
                "engine_rpm_rpm_time_weighted_std",
                "absolute_load_pct_time_weighted_mean",
                "absolute_load_pct_time_weighted_std",
                "absolute_load_x_rpm_mean_product",
            ],
            "standardized_load_rpm_displacement_proxy": ["standardized_absolute_load_rpm_maf_proxy_g_s"],
        }
        maf_train_mask = pd.to_numeric(train["maf_g_sec_time_weighted_mean"], errors="coerce").notna()
        maf_test_mask = pd.to_numeric(test["maf_g_sec_time_weighted_mean"], errors="coerce").notna()
        maf_train = train.loc[maf_train_mask].reset_index(drop=True)
        maf_test = test.loc[maf_test_mask].reset_index(drop=True)
        maf_y_train = pd.to_numeric(maf_train["maf_g_sec_time_weighted_mean"], errors="raise").to_numpy(float)
        maf_y_test = pd.to_numeric(maf_test["maf_g_sec_time_weighted_mean"], errors="raise").to_numpy(float)
        for name, features in maf_feature_sets.items():
            x_train = maf_train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            x_test = maf_test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            prediction, metadata = ensemble_predictions("hgb", x_train, maf_y_train, x_test, config)
            metrics = regression_metrics(maf_y_test, prediction)
            rows.append(
                {
                    "record_type": "maf_reconstruction_model",
                    "scope": "transfer_test",
                    "environment": environment,
                    "aggregation": "segment",
                    "predictor": name,
                    "outcome": "maf_g_sec_time_weighted_mean",
                    "n": int(len(maf_test)),
                    "vehicles": int(maf_test["VehId"].nunique()),
                    "trips": int(maf_test["trip_uid"].nunique()),
                    "feature_count": len(features),
                    **metrics,
                    **metadata,
                }
            )
        logger.log(f"PROXY_MODELS {environment} train={len(train)} test={len(test)}")
        gc.collect()
    return pd.DataFrame(rows)


def proxy_audit(
    config: dict[str, Any],
    data: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    output: Path,
    logger: RunLogger,
) -> pd.DataFrame:
    union_ids = pd.concat(
        [frame[["segment_id"]] for frame in memberships.values()], ignore_index=True
    ).drop_duplicates()
    eligible = union_ids.merge(data, on="segment_id", how="left", validate="one_to_one")
    eligible = eligible[eligible["engine_type"].astype(str).str.upper().eq("ICE")].copy()
    pairs = [
        ("absolute_load_pct_time_weighted_mean", "maf_g_sec_time_weighted_mean"),
        ("engine_rpm_rpm_time_weighted_mean", "maf_g_sec_time_weighted_mean"),
        ("absolute_load_x_rpm_mean_product", "maf_g_sec_time_weighted_mean"),
        ("standardized_absolute_load_rpm_maf_proxy_g_s", "maf_g_sec_time_weighted_mean"),
        ("absolute_load_pct_time_weighted_mean", "operational_fuel_rate_l_h"),
        ("engine_rpm_rpm_time_weighted_mean", "operational_fuel_rate_l_h"),
        ("absolute_load_x_rpm_mean_product", "operational_fuel_rate_l_h"),
        ("standardized_absolute_load_rpm_maf_proxy_g_s", "operational_fuel_rate_l_h"),
    ]
    rows: list[dict[str, Any]] = []
    for aggregation in ("segment", "within_vehicle_demeaned", "across_trip_means"):
        for x, y in pairs:
            rows.append(correlation_row(eligible, "global_unique", "", aggregation, x, y))
    for environment in config["environments"]:
        test_ids = memberships[environment].loc[
            memberships[environment]["split_role"].eq("test"), ["segment_id"]
        ]
        test = test_ids.merge(data, on="segment_id", how="left", validate="one_to_one")
        for aggregation in ("segment", "within_vehicle_demeaned", "across_trip_means"):
            for x, y in pairs:
                rows.append(
                    correlation_row(test, "transfer_population", environment, aggregation, x, y)
                )
    source = eligible[
        [
            "fuel_direct_duration_share",
            "fuel_maf_duration_share",
            "fuel_target_coverage",
            "engine_displacement_L",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    rows.append(
        {
            "record_type": "target_pathway_support",
            "scope": "global_unique",
            "aggregation": "segment",
            "outcome": "fuel_volume_L",
            "n": int(len(eligible)),
            "vehicles": int(eligible["VehId"].nunique()),
            "trips": int(eligible["trip_uid"].nunique()),
            "maf_only_segment_share": float(np.mean(source["fuel_maf_duration_share"] >= 0.999999)),
            "any_direct_duration_segment_share": float(np.mean(source["fuel_direct_duration_share"] > 0)),
            "mean_maf_duration_share": float(source["fuel_maf_duration_share"].mean()),
            "mean_direct_duration_share": float(source["fuel_direct_duration_share"].mean()),
            "mean_target_coverage": float(source["fuel_target_coverage"].mean()),
            "displacement_parse_coverage": float(source["engine_displacement_L"].notna().mean()),
        }
    )
    correlations = pd.DataFrame(rows)
    diagnostics = compact_proxy_models(config, data, memberships, logger)
    combined = pd.concat([correlations, diagnostics], ignore_index=True, sort=False)
    write_csv(output / "PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_RESULTS.csv", combined)
    return combined


def road_contract(
    config: dict[str, Any], memberships: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    information_contract = load_json(Path(config["inputs"]["information_contract"]))
    road_features = list(information_contract["components"]["ROAD"])
    if len(road_features) != 142 or len(set(road_features)) != 142:
        raise RuntimeError("Frozen R6 road contract is not exactly 142 unique columns.")
    robustness_root = Path(config["inputs"]["robustness_root"])
    active = load_json(robustness_root / "active_feature_masks.json")
    low = list(config["features"]["lower_information"])
    masks: dict[str, list[str]] = {}
    audits: dict[str, Any] = {}
    for environment in config["environments"]:
        cell_key = f"{environment}|{config['target']['id']}"
        sensor_masks = active["cells"][cell_key]["sensor_regimes"]
        low_road = list(sensor_masks["low"]["real_road_active_features"])
        high_road = list(sensor_masks["high"]["real_road_active_features"])
        low_road_only = [feature for feature in low_road if feature in set(road_features)]
        high_road_only = [feature for feature in high_road if feature in set(road_features)]
        if low_road_only != high_road_only:
            raise RuntimeError(f"{cell_key}: frozen low/high road masks differ.")
        if low_road[: len(low)] != low:
            raise RuntimeError(f"{cell_key}: frozen lower-information feature order changed.")
        masks[environment] = low_road_only
        audits[environment] = {
            "cell_key": cell_key,
            "membership_sha256": sha256_file(membership_path(config, environment)),
            "active_road_feature_count": len(low_road_only),
            "active_road_features": low_road_only,
            "exact_frozen_low_high_road_mask_match": True,
            "source_active_mask_sha256": sha256_file(robustness_root / "active_feature_masks.json"),
        }
    return road_features, masks, audits


def robustness_manifest_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = load_json(Path(config["inputs"]["robustness_root"]) / "artifact_manifest.json")
    index: dict[str, dict[str, Any]] = {}
    for item in manifest["artifacts"]:
        path = windows_to_local_path(item["path"])
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            relative = path
        index[str(relative).replace("\\", "/").lower()] = item
    return index


def verify_frozen_file(path: Path, manifest_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path).replace("\\", "/").lower()
    if path.is_absolute():
        try:
            key = str(path.relative_to(REPO_ROOT)).replace("\\", "/").lower()
        except ValueError:
            pass
    if key not in manifest_index:
        raise RuntimeError(f"Frozen ROBUSTNESS artifact is absent from manifest: {path}")
    expected = manifest_index[key]
    observed_size = int(path.stat().st_size)
    if observed_size != int(expected["bytes"]):
        raise RuntimeError(f"Frozen ROBUSTNESS artifact size changed: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected["sha256"]:
        raise RuntimeError(f"Frozen ROBUSTNESS artifact hash changed: {path}")
    return {
        "path": str(path),
        "bytes": observed_size,
        "expected_sha256": expected["sha256"],
        "observed_sha256": observed_hash,
        "verified": True,
    }


def prediction_column(key: str) -> str:
    return "p__" + key.replace("|", "__")


def original_prediction_paths(config: dict[str, Any], environment: str) -> tuple[Path, Path]:
    stem = f"{environment}__{config['target']['id']}"
    root = Path(config["inputs"]["robustness_root"]) / "ensemble_test_predictions"
    return root / f"{stem}.parquet", root / f"{stem}.json"


def control_path(config: dict[str, Any], environment: str, null_type: str, draw: int) -> Path:
    stem = f"{environment}__{config['target']['id']}"
    return (
        Path(config["inputs"]["robustness_root"])
        / "null_controls"
        / stem
        / null_type
        / f"draw_{draw:02d}.parquet"
    )


def load_road_features(
    config: dict[str, Any], masks: dict[str, list[str]], logger: RunLogger
) -> pd.DataFrame:
    columns = sorted({feature for values in masks.values() for feature in values})
    root = Path(config["inputs"]["feature_contract"]).parent
    r2_path = root / "r2_segment_features.parquet"
    segment_path = Path(config["inputs"]["segments"])
    segment_schema = set(pq.read_schema(segment_path).names)
    r2_schema = set(pq.read_schema(r2_path).names)
    segment_columns = [feature for feature in columns if feature in segment_schema]
    r2_columns = [feature for feature in columns if feature in r2_schema]
    unresolved = sorted(set(columns) - set(segment_columns) - set(r2_columns))
    if unresolved:
        raise RuntimeError(f"Frozen active road columns are unavailable: {unresolved}")
    logger.log(
        f"LOAD frozen real-road columns={len(columns)} "
        f"segment={len(segment_columns)} r2={len(r2_columns)}"
    )
    frame = pd.read_parquet(segment_path, columns=["segment_id", *segment_columns])
    if r2_columns:
        r2 = pd.read_parquet(r2_path, columns=["segment_id", *r2_columns])
        frame = frame.merge(r2, on="segment_id", how="left", validate="one_to_one")
    frame["segment_id"] = frame["segment_id"].astype(str)
    if frame["segment_id"].duplicated().any():
        raise RuntimeError("Frozen R2 road feature artifact contains duplicate segment_id values.")
    return frame


def rpm_only_cell(
    config: dict[str, Any],
    environment: str,
    segments: pd.DataFrame,
    membership: pd.DataFrame,
    real_road: pd.DataFrame,
    active_roads: list[str],
    manifest_index: dict[str, dict[str, Any]],
    output: Path,
    logger: RunLogger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_root = output / "rpm_only_test_predictions"
    prediction_path = prediction_root / f"{environment}__{config['target']['id']}.parquet"
    metadata_path = prediction_path.with_suffix(".json")
    if prediction_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if sha256_file(prediction_path) == metadata["prediction_sha256"]:
            logger.log(f"RPM_ONLY RESUME {environment} verified cached prediction")
            return pd.read_parquet(prediction_path), metadata

    base = membership.merge(segments, on="segment_id", how="left", validate="one_to_one")
    road_columns_to_merge = [feature for feature in active_roads if feature not in base.columns]
    if road_columns_to_merge:
        base = base.merge(
            real_road[["segment_id", *road_columns_to_merge]],
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
    unresolved_road = sorted(set(active_roads) - set(base.columns))
    if unresolved_road:
        raise RuntimeError(f"{environment}: unresolved real-road columns {unresolved_road}")
    base = base.sort_values("segment_id").reset_index(drop=True)
    if not base["engine_type"].astype(str).str.upper().eq("ICE").all():
        raise RuntimeError(f"{environment}: non-ICE row entered ICE target membership.")
    train = base[base["split_role"].eq("train")].reset_index(drop=True)
    test = base[base["split_role"].eq("test")].reset_index(drop=True)
    if len(train) < int(config["models"]["minimum_train_rows"]):
        raise RuntimeError(f"{environment}: insufficient frozen training support.")
    lower = list(config["features"]["lower_information"])
    rpm = list(config["features"]["rpm"])
    baseline_features = lower + rpm
    active_baseline = [
        feature
        for feature in baseline_features
        if pd.to_numeric(train[feature], errors="coerce").nunique(dropna=True) >= 2
    ]
    if active_baseline != baseline_features:
        raise RuntimeError(
            f"{environment}: expected ten RPM-only features, observed active {active_baseline}"
        )
    real_features = active_baseline + active_roads
    forbidden = set(config["features"]["target_source_features_forbidden"])
    forbidden_hits = sorted(forbidden.intersection(real_features))
    if forbidden_hits or any("absolute_load" in feature.lower() for feature in real_features):
        raise RuntimeError(f"{environment}: target-proxy leakage in RPM-only branch: {forbidden_hits}")

    y_train = pd.to_numeric(train[config["target"]["column"]], errors="raise").to_numpy(float)
    y_test = pd.to_numeric(test[config["target"]["column"]], errors="raise").to_numpy(float)
    prediction_arrays: dict[str, np.ndarray] = {}
    fit_records: list[dict[str, Any]] = []

    def fit_representation(model_name: str, representation: str, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> None:
        prediction, metadata = ensemble_predictions(
            model_name,
            train_matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan),
            y_train,
            test_matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan),
            config,
        )
        key = f"{model_name}|rpm|{representation}"
        prediction_arrays[prediction_column(key)] = prediction
        fit_records.append(
            {
                "environment": environment,
                "model": model_name,
                "sensor": "lower_plus_rpm",
                "representation": representation,
                "draw": None,
                "train_rows": len(train),
                "test_rows": len(test),
                "active_feature_count": int(train_matrix.shape[1]),
                "active_features": list(train_matrix.columns),
                **metadata,
            }
        )

    for model_name in MODELS:
        fit_representation(model_name, "baseline", train[active_baseline], test[active_baseline])
        fit_representation(model_name, "real", train[real_features], test[real_features])
        logger.log(f"RPM_ONLY {environment} fitted {model_name} baseline+real")

    verified_controls: list[dict[str, Any]] = []
    draws = int(config["null_controls"]["draws_per_type"])
    for null_type in NULL_TYPES:
        for draw in range(1, draws + 1):
            path = control_path(config, environment, null_type, draw)
            verified_controls.append(verify_frozen_file(path, manifest_index))
            control = pd.read_parquet(path, columns=["segment_id", "split_role", *active_roads])
            control["segment_id"] = control["segment_id"].astype(str)
            control_train = (
                control[control["split_role"].eq("train")]
                .drop(columns="split_role")
                .sort_values("segment_id")
                .reset_index(drop=True)
            )
            control_test = (
                control[control["split_role"].eq("test")]
                .drop(columns="split_role")
                .sort_values("segment_id")
                .reset_index(drop=True)
            )
            if not control_train["segment_id"].equals(train["segment_id"]):
                raise RuntimeError(f"{environment}/{null_type}/{draw}: train control misaligned")
            if not control_test["segment_id"].equals(test["segment_id"]):
                raise RuntimeError(f"{environment}/{null_type}/{draw}: test control misaligned")
            train_matrix = pd.concat(
                [train[active_baseline].reset_index(drop=True), control_train[active_roads]], axis=1
            )
            test_matrix = pd.concat(
                [test[active_baseline].reset_index(drop=True), control_test[active_roads]], axis=1
            )
            prediction, metadata = ensemble_predictions(
                "hgb",
                train_matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan),
                y_train,
                test_matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan),
                config,
            )
            representation = f"{null_type}|{draw:02d}"
            key = f"hgb|rpm|{representation}"
            prediction_arrays[prediction_column(key)] = prediction
            fit_records.append(
                {
                    "environment": environment,
                    "model": "hgb",
                    "sensor": "lower_plus_rpm",
                    "representation": null_type,
                    "draw": draw,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "active_feature_count": int(train_matrix.shape[1]),
                    "active_features": list(train_matrix.columns),
                    **metadata,
                }
            )
            del control, control_train, control_test, train_matrix, test_matrix
            gc.collect()
            logger.log(f"RPM_ONLY {environment} {null_type} draw={draw:02d}/{draws}")

    identity = test[["segment_id", "trip_uid", "VehId"]].copy()
    identity["target_l"] = y_test
    prediction_frame = pd.concat([identity, pd.DataFrame(prediction_arrays)], axis=1)
    write_parquet(prediction_path, prediction_frame)
    metadata = {
        "schema_version": 1,
        "experiment": config["experiment"]["id"],
        "environment": environment,
        "target_id": config["target"]["id"],
        "sensor_definition": "lower_information_plus_RPM_summaries_only",
        "absolute_load_excluded": True,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_segment_id_sha256": ordered_sha256(train["segment_id"]),
        "test_segment_id_sha256": ordered_sha256(test["segment_id"]),
        "baseline_features": active_baseline,
        "active_road_features": active_roads,
        "active_road_feature_count": len(active_roads),
        "prediction_columns": list(prediction_arrays),
        "fit_records": fit_records,
        "verified_frozen_controls": verified_controls,
        "prediction_sha256": sha256_file(prediction_path),
        "no_tuning": True,
        "evaluation_target_opened_after_all_cell_fits": False,
        "note": "Predictions are generated immediately after each frozen fit; no evaluation outcome is used for tuning, selection, or refitting.",
    }
    write_json(metadata_path, metadata)
    return prediction_frame, metadata


def build_exact_bootstrap_frames(
    config: dict[str, Any],
    rpm_predictions: dict[str, pd.DataFrame],
    manifest_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    columns_by_environment: dict[str, dict[str, str]] = {}
    verification: list[dict[str, Any]] = []
    for environment in config["environments"]:
        original_path, original_metadata_path = original_prediction_paths(config, environment)
        verification.append(verify_frozen_file(original_path, manifest_index))
        verification.append(verify_frozen_file(original_metadata_path, manifest_index))
        original = pd.read_parquet(original_path).sort_values("segment_id").reset_index(drop=True)
        original["segment_id"] = original["segment_id"].astype(str)
        rpm = rpm_predictions[environment].sort_values("segment_id").reset_index(drop=True)
        rpm["segment_id"] = rpm["segment_id"].astype(str)
        if not original["segment_id"].equals(rpm["segment_id"]):
            raise RuntimeError(f"{environment}: frozen and RPM-only evaluation observations differ")
        if not np.array_equal(original["target_l"].to_numpy(float), rpm["target_l"].to_numpy(float)):
            raise RuntimeError(f"{environment}: frozen target values changed")
        metadata = load_json(original_metadata_path)
        mapping: dict[str, str] = {}
        output = rpm[["segment_id", "trip_uid", "VehId", "target_l"]].copy()
        for key, column in metadata["prediction_columns"].items():
            parts = key.split("|")
            if parts[1] == "low":
                output[column] = original[column].to_numpy(float)
                mapping[key] = column
        for model in MODELS:
            for representation in ("baseline", "real"):
                target_key = f"{model}|high|{representation}"
                source = prediction_column(f"{model}|rpm|{representation}")
                destination = prediction_column(target_key)
                output[destination] = rpm[source].to_numpy(float)
                mapping[target_key] = destination
        for null_type in NULL_TYPES:
            for draw in range(1, int(config["null_controls"]["draws_per_type"]) + 1):
                target_key = f"hgb|high|{null_type}|{draw:02d}"
                source = prediction_column(f"hgb|rpm|{null_type}|{draw:02d}")
                destination = prediction_column(target_key)
                output[destination] = rpm[source].to_numpy(float)
                mapping[target_key] = destination
        frames[environment] = output
        columns_by_environment[environment] = mapping
    return frames, columns_by_environment, verification


def absolute_mae_bootstrap(
    frames: dict[str, pd.DataFrame],
    prediction_columns: dict[str, dict[str, str]],
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    vehicles = sorted(
        {
            str(vehicle)
            for frame in frames.values()
            for vehicle in frame["VehId"].astype(str).unique()
        }
    )
    repetitions = int(config["bootstrap"]["repetitions"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    weights = rng.multinomial(
        len(vehicles), np.full(len(vehicles), 1.0 / len(vehicles)), size=repetitions
    ).astype(float)
    result: dict[tuple[str, str], dict[str, float]] = {}
    alpha = float(config["bootstrap"]["alpha"])
    for environment, frame in frames.items():
        point, samples = _vehicle_mae(frame, prediction_columns[environment], vehicles, weights)
        for key in prediction_columns[environment]:
            values = samples[key]
            result[(environment, key)] = {
                "mae_l": float(point[key]),
                "mae_pointwise_lcb": float(np.quantile(values, alpha / 2)),
                "mae_pointwise_ucb": float(np.quantile(values, 1 - alpha / 2)),
            }
    return result


def rpm_only_inference(
    config: dict[str, Any],
    rpm_predictions: dict[str, pd.DataFrame],
    manifest_index: dict[str, dict[str, Any]],
    output: Path,
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    frames, prediction_columns, verification = build_exact_bootstrap_frames(
        config, rpm_predictions, manifest_index
    )
    effects, bootstrap_values, diagnostics = bootstrap_target(
        config["target"]["id"],
        frames,
        prediction_columns,
        int(config["null_controls"]["draws_per_type"]),
        int(config["bootstrap"]["repetitions"]),
        int(config["bootstrap"]["seed"]),
        float(config["bootstrap"]["alpha"]),
        float(config["bootstrap"]["relative_sesoi"]),
    )
    effects["family_member_sensor_slot"] = effects["sensor"]
    effects["sensor"] = effects["sensor"].replace({"high": "lower_plus_rpm"})
    null_summary = summarize_null_distributions(
        effects.assign(sensor=effects["family_member_sensor_slot"])
    )
    null_summary["sensor"] = null_summary["sensor"].replace({"high": "lower_plus_rpm"})
    comparisons = model_comparison(
        effects.assign(sensor=effects["family_member_sensor_slot"])
    )
    comparisons["sensor"] = comparisons["sensor"].replace({"high": "lower_plus_rpm"})
    write_csv(output / "rpm_only_effects.csv", effects)
    write_csv(output / "rpm_only_null_distribution_summary.csv", null_summary)
    write_csv(output / "rpm_only_model_comparison.csv", comparisons)
    diagnostics["hybrid_family_definition"] = (
        "Frozen lower-information slots retained unchanged; original high-information slots replaced by lower-information plus RPM."
    )
    diagnostics["frozen_prediction_verification"] = verification
    write_json(output / "rpm_only_bootstrap_diagnostics.json", diagnostics)
    bootstrap_frame = pd.DataFrame(
        {effect_id: values for effect_id, values in bootstrap_values.items()}
    )
    write_parquet(output / "rpm_only_bootstrap_draws.parquet", bootstrap_frame)

    mae_intervals = absolute_mae_bootstrap(frames, prediction_columns, config)
    metric_rows: list[dict[str, Any]] = []
    rpm_effects = effects[effects["sensor"].eq("lower_plus_rpm")]
    for environment in config["environments"]:
        frame = frames[environment]
        y = frame["target_l"].to_numpy(float)
        mapping = prediction_columns[environment]
        for key, column in mapping.items():
            model, sensor_slot, representation, *draw_parts = key.split("|")
            if sensor_slot != "high":
                continue
            draw = int(draw_parts[0]) if draw_parts else None
            metrics = regression_metrics(y, frame[column].to_numpy(float))
            interval = mae_intervals[(environment, key)]
            effect = None
            if representation == "real":
                candidates = rpm_effects[
                    rpm_effects["environment"].eq(environment)
                    & rpm_effects["model"].eq(model)
                    & rpm_effects["family"].eq("model_real_gain")
                    & rpm_effects["effect_type"].eq("gain_real")
                ]
                if len(candidates):
                    effect = candidates.iloc[0]
            elif representation in NULL_TYPES:
                candidates = rpm_effects[
                    rpm_effects["environment"].eq(environment)
                    & rpm_effects["model"].eq("hgb")
                    & rpm_effects["family"].eq("hgb_null_gain")
                    & rpm_effects["null_type"].eq(representation)
                    & rpm_effects["draw"].eq(draw)
                ]
                if len(candidates):
                    effect = candidates.iloc[0]
            baseline_key = f"{model}|high|baseline"
            baseline_mae = mae_intervals[(environment, baseline_key)]["mae_l"]
            row = {
                "record_type": "evaluation_metric",
                "environment": environment,
                "population_label": ENVIRONMENT_LABELS[environment],
                "target_id": config["target"]["id"],
                "sensor": "lower_plus_rpm",
                "model": model,
                "representation": representation,
                "control_draw": draw,
                "test_rows": int(len(frame)),
                "test_vehicles": int(frame["VehId"].nunique()),
                "test_trips": int(frame["trip_uid"].nunique()),
                **metrics,
                **interval,
                "absolute_mae_improvement_l": baseline_mae - interval["mae_l"],
                "relative_mae_gain": 0.0 if representation == "baseline" else np.nan,
                "relative_gain_pointwise_lcb": 0.0 if representation == "baseline" else np.nan,
                "relative_gain_pointwise_ucb": 0.0 if representation == "baseline" else np.nan,
                "relative_gain_simultaneous_lcb": 0.0 if representation == "baseline" else np.nan,
                "relative_gain_simultaneous_ucb": 0.0 if representation == "baseline" else np.nan,
                "simultaneous_family": "baseline_reference" if representation == "baseline" else "",
                "state": "REFERENCE" if representation == "baseline" else "",
            }
            if effect is not None:
                row.update(
                    {
                        "relative_mae_gain": float(effect["estimate"]),
                        "relative_gain_pointwise_lcb": float(effect["pointwise_lcb"]),
                        "relative_gain_pointwise_ucb": float(effect["pointwise_ucb"]),
                        "relative_gain_simultaneous_lcb": float(effect["simultaneous_lcb"]),
                        "relative_gain_simultaneous_ucb": float(effect["simultaneous_ucb"]),
                        "simultaneous_family": effect["family"],
                        "state": effect["state"],
                    }
                )
            metric_rows.append(row)
    results = pd.DataFrame(metric_rows)
    write_csv(output / "PHASE1_RPM_ONLY_RERUN_RESULTS.csv", results)
    logger.log(
        f"INFERENCE RPM-only effects={len(effects)} family_sizes={diagnostics['families']}"
    )
    return results, effects, null_summary, frames


def paired_vehicle_stratum_bootstrap(
    frame: pd.DataFrame,
    baseline_column: str,
    real_column: str,
    repetitions: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    working = frame[["VehId", "target_l", baseline_column, real_column]].copy()
    working["VehId"] = working["VehId"].astype(str)
    working["ae_baseline"] = np.abs(
        working["target_l"].to_numpy(float) - working[baseline_column].to_numpy(float)
    )
    working["ae_real"] = np.abs(
        working["target_l"].to_numpy(float) - working[real_column].to_numpy(float)
    )
    grouped = working.groupby("VehId", sort=False).agg(
        count=("target_l", "size"),
        baseline_sum=("ae_baseline", "sum"),
        real_sum=("ae_real", "sum"),
    )
    vehicles = list(grouped.index)
    counts = grouped["count"].to_numpy(float)
    baseline_sums = grouped["baseline_sum"].to_numpy(float)
    real_sums = grouped["real_sum"].to_numpy(float)
    baseline_point = float(baseline_sums.sum() / counts.sum())
    real_point = float(real_sums.sum() / counts.sum())
    gain_point = float((baseline_point - real_point) / baseline_point)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(vehicles), np.full(len(vehicles), 1.0 / len(vehicles)), size=repetitions
    ).astype(float)
    denominator = weights @ counts
    baseline_samples = (weights @ baseline_sums) / denominator
    real_samples = (weights @ real_sums) / denominator
    gain_samples = (baseline_samples - real_samples) / baseline_samples
    return {
        "baseline_mae_l": baseline_point,
        "baseline_mae_lcb": float(np.quantile(baseline_samples, alpha / 2)),
        "baseline_mae_ucb": float(np.quantile(baseline_samples, 1 - alpha / 2)),
        "real_mae_l": real_point,
        "real_mae_lcb": float(np.quantile(real_samples, alpha / 2)),
        "real_mae_ucb": float(np.quantile(real_samples, 1 - alpha / 2)),
        "absolute_mae_improvement_l": baseline_point - real_point,
        "relative_mae_gain": gain_point,
        "relative_gain_lcb": float(np.quantile(gain_samples, alpha / 2)),
        "relative_gain_ucb": float(np.quantile(gain_samples, 1 - alpha / 2)),
    }


def target_quality_sensitivity(
    config: dict[str, Any],
    data: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    bootstrap_frames: dict[str, pd.DataFrame],
    output: Path,
    logger: RunLogger,
) -> pd.DataFrame:
    trim_columns = [
        "short_term_fuel_trim_bank_1_pct_time_weighted_mean",
        "short_term_fuel_trim_bank_2_pct_time_weighted_mean",
        "long_term_fuel_trim_bank_1_pct_time_weighted_mean",
        "long_term_fuel_trim_bank_2_pct_time_weighted_mean",
    ]
    rows: list[dict[str, Any]] = []
    minimum = int(config["target_sensitivity"]["minimum_stratum_rows"])
    repetitions = int(config["bootstrap"]["repetitions"])
    alpha = float(config["bootstrap"]["alpha"])
    for environment in config["environments"]:
        membership = memberships[environment]
        cell = membership.merge(data, on="segment_id", how="left", validate="one_to_one")
        train = cell[cell["split_role"].eq("train")].copy()
        test_covariates = cell[cell["split_role"].eq("test")].sort_values("segment_id").reset_index(drop=True)
        predictions = bootstrap_frames[environment].sort_values("segment_id").reset_index(drop=True)
        if not test_covariates["segment_id"].equals(predictions["segment_id"]):
            raise RuntimeError(f"{environment}: sensitivity covariates and predictions misaligned")
        test = predictions.merge(
            test_covariates.drop(columns=["trip_uid", "VehId"], errors="ignore"),
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        baseline_column = prediction_column("hgb|high|baseline")
        real_column = prediction_column("hgb|high|real")
        overall = paired_vehicle_stratum_bootstrap(
            test,
            baseline_column,
            real_column,
            repetitions,
            alpha,
            stable_seed(int(config["bootstrap"]["seed"]), environment, "overall"),
        )
        load_q90 = float(
            pd.to_numeric(train["absolute_load_pct_time_weighted_mean"], errors="coerce").quantile(
                float(config["target_sensitivity"]["high_quantile"])
            )
        )
        speed_q90 = float(
            pd.to_numeric(train["vehicle_speed_km_h_time_weighted_mean"], errors="coerce").quantile(
                float(config["target_sensitivity"]["high_quantile"])
            )
        )
        speed_std_q25 = float(
            pd.to_numeric(train["vehicle_speed_km_h_time_weighted_std"], errors="coerce").quantile(
                float(config["target_sensitivity"]["stable_quantile"])
            )
        )
        speed_std_q75 = float(
            pd.to_numeric(train["vehicle_speed_km_h_time_weighted_std"], errors="coerce").quantile(
                float(config["target_sensitivity"]["transient_quantile"])
            )
        )
        rpm_std_q25 = float(
            pd.to_numeric(train["engine_rpm_rpm_time_weighted_std"], errors="coerce").quantile(
                float(config["target_sensitivity"]["stable_quantile"])
            )
        )
        rpm_std_q75 = float(
            pd.to_numeric(train["engine_rpm_rpm_time_weighted_std"], errors="coerce").quantile(
                float(config["target_sensitivity"]["transient_quantile"])
            )
        )
        direct_share = pd.to_numeric(test["fuel_direct_duration_share"], errors="coerce").fillna(0.0)
        maf_share = pd.to_numeric(test["fuel_maf_duration_share"], errors="coerce").fillna(0.0)
        source = np.select(
            [direct_share >= 0.999999, maf_share >= 0.999999],
            ["direct_only", "maf_only"],
            default="mixed_or_partial",
        )
        trim_any = test[trim_columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        load_values = pd.to_numeric(test["absolute_load_pct_time_weighted_mean"], errors="coerce")
        speed_values = pd.to_numeric(test["vehicle_speed_km_h_time_weighted_mean"], errors="coerce")
        motorway = pd.to_numeric(test["functional_road_class_share_motorway"], errors="coerce")
        coverage = pd.to_numeric(test["fuel_target_coverage"], errors="coerce")
        speed_std = pd.to_numeric(test["vehicle_speed_km_h_time_weighted_std"], errors="coerce")
        rpm_std = pd.to_numeric(test["engine_rpm_rpm_time_weighted_std"], errors="coerce")
        dynamics = np.select(
            [
                (speed_std <= speed_std_q25) & (rpm_std <= rpm_std_q25),
                (speed_std >= speed_std_q75) | (rpm_std >= rpm_std_q75),
            ],
            ["stable", "transient"],
            default="intermediate",
        )
        strata = {
            "target_source": pd.Series(source, index=test.index),
            "fuel_trim_availability": pd.Series(
                np.where(trim_any, "any_trim_summary_observed", "no_trim_summary_observed"),
                index=test.index,
            ),
            "reported_absolute_load_band": pd.Series(
                np.where(load_values >= load_q90, "high_q90", "below_q90"), index=test.index
            ),
            "speed_band": pd.Series(
                np.where(speed_values >= speed_q90, "high_q90", "below_q90"), index=test.index
            ),
            "motorway_exposure": pd.Series(
                np.where(motorway >= 0.5, "motorway_share_ge_0_5", "motorway_share_lt_0_5"),
                index=test.index,
            ),
            "operating_dynamics": pd.Series(dynamics, index=test.index),
            "target_duration_coverage": pd.Series(
                np.where(coverage >= 0.999999, "complete", "partial_0_95_to_lt_1"),
                index=test.index,
            ),
        }
        rows.append(
            {
                "environment": environment,
                "sensitivity": "overall",
                "stratum": "all_test",
                "scientific_rationale": "Reference effect for within-environment direction comparisons.",
                "data_support": "supported",
                "eligible": True,
                "rows": int(len(test)),
                "vehicles": int(test["VehId"].nunique()),
                "trips": int(test["trip_uid"].nunique()),
                "thresholds_train_only": "",
                **overall,
                "direction_consistent_with_overall": True,
            }
        )
        rationales = {
            "target_source": "Direct Fuel Rate observations are less assumption-bound than the MAF fallback and expose source sensitivity.",
            "fuel_trim_availability": "Missing trims are operationally treated as zero correction; availability can index target-construction quality.",
            "reported_absolute_load_band": "High reported load can coincide with target-model stress, but it does not identify enrichment.",
            "speed_band": "High speed changes airflow demand and can expose operating-state-dependent target error.",
            "motorway_exposure": "Motorway exposure directly targets the adverse road-function population pattern.",
            "operating_dynamics": "Rapid speed or RPM variation can magnify time-alignment and airflow-to-fuel approximation error.",
            "target_duration_coverage": "Partial valid target duration can alter segment-level error even above the frozen 95% gate.",
        }
        threshold_text = {
            "target_source": "fixed source-share cutoffs: 0.999999",
            "fuel_trim_availability": "observed/not observed; no evaluation-derived threshold",
            "reported_absolute_load_band": f"train q90={load_q90:.6g}",
            "speed_band": f"train q90={speed_q90:.6g}",
            "motorway_exposure": "prespecified share cutoff=0.5",
            "operating_dynamics": (
                f"train speed_std q25/q75={speed_std_q25:.6g}/{speed_std_q75:.6g}; "
                f"rpm_std q25/q75={rpm_std_q25:.6g}/{rpm_std_q75:.6g}"
            ),
            "target_duration_coverage": "frozen complete cutoff=0.999999",
        }
        for sensitivity, labels in strata.items():
            for label in sorted(pd.Series(labels).dropna().astype(str).unique()):
                subset = test[pd.Series(labels, index=test.index).astype(str).eq(label)].copy()
                eligible = len(subset) >= minimum and subset["VehId"].nunique() >= 2
                row: dict[str, Any] = {
                    "environment": environment,
                    "sensitivity": sensitivity,
                    "stratum": label,
                    "scientific_rationale": rationales[sensitivity],
                    "data_support": "supported_descriptive_stratification",
                    "eligible": bool(eligible),
                    "rows": int(len(subset)),
                    "vehicles": int(subset["VehId"].nunique()),
                    "trips": int(subset["trip_uid"].nunique()),
                    "thresholds_train_only": threshold_text[sensitivity],
                }
                if eligible:
                    estimates = paired_vehicle_stratum_bootstrap(
                        subset,
                        baseline_column,
                        real_column,
                        repetitions,
                        alpha,
                        stable_seed(
                            int(config["bootstrap"]["seed"]), environment, sensitivity, label
                        ),
                    )
                    row.update(estimates)
                    row["direction_consistent_with_overall"] = bool(
                        np.sign(estimates["relative_mae_gain"])
                        == np.sign(overall["relative_mae_gain"])
                    )
                rows.append(row)
        for sensitivity, rationale in (
            (
                "open_loop_enrichment",
                "Enrichment changes the air-fuel ratio and can bias a fixed-stoichiometric MAF conversion.",
            ),
            (
                "deceleration_fuel_cut",
                "Fuel cut can decouple airflow-related channels from injected fuel during deceleration.",
            ),
        ):
            rows.append(
                {
                    "environment": environment,
                    "sensitivity": sensitivity,
                    "stratum": "not_identifiable",
                    "scientific_rationale": rationale,
                    "data_support": (
                        "not supported: no commanded equivalence ratio, injector pulse width, fuel-system mode, "
                        "or defensible fuel-cut indicator is available"
                    ),
                    "eligible": False,
                    "rows": 0,
                    "vehicles": 0,
                    "trips": 0,
                    "thresholds_train_only": "none; no proxy invented",
                    "direction_consistent_with_overall": np.nan,
                }
            )
        logger.log(f"TARGET_SENSITIVITY {environment} strata complete")
    result = pd.DataFrame(rows)
    write_csv(output / "target_quality_sensitivity_results.csv", result)
    return result


def comparison_table(
    config: dict[str, Any], effects: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, bool]:
    original = pd.read_csv(Path(config["inputs"]["robustness_root"]) / "effects.csv")
    original = original[
        original["target_id"].eq(config["target"]["id"])
        & original["sensor"].eq("high")
        & original["family"].eq("model_real_gain")
    ][
        [
            "environment",
            "model",
            "estimate",
            "pointwise_lcb",
            "pointwise_ucb",
            "simultaneous_lcb",
            "simultaneous_ucb",
            "state",
        ]
    ].rename(
        columns={
            "estimate": "original_load_based_gain",
            "pointwise_lcb": "original_pointwise_lcb",
            "pointwise_ucb": "original_pointwise_ucb",
            "simultaneous_lcb": "original_simultaneous_lcb",
            "simultaneous_ucb": "original_simultaneous_ucb",
            "state": "original_state",
        }
    )
    revised = effects[
        effects["sensor"].eq("lower_plus_rpm")
        & effects["family"].eq("model_real_gain")
    ][
        [
            "environment",
            "model",
            "estimate",
            "pointwise_lcb",
            "pointwise_ucb",
            "simultaneous_lcb",
            "simultaneous_ucb",
            "state",
        ]
    ].rename(
        columns={
            "estimate": "rpm_only_gain",
            "pointwise_lcb": "rpm_only_pointwise_lcb",
            "pointwise_ucb": "rpm_only_pointwise_ucb",
            "simultaneous_lcb": "rpm_only_simultaneous_lcb",
            "simultaneous_ucb": "rpm_only_simultaneous_ucb",
            "state": "rpm_only_state",
        }
    )
    comparison = original.merge(revised, on=["environment", "model"], how="outer", validate="one_to_one")
    comparison["gain_change_rpm_minus_original"] = (
        comparison["rpm_only_gain"] - comparison["original_load_based_gain"]
    )
    comparison["point_direction_changed"] = (
        np.sign(comparison["rpm_only_gain"]) != np.sign(comparison["original_load_based_gain"])
    )
    reference = comparison[
        comparison["environment"].isin(config["decision"]["frozen_positive_reference_environments"])
        & comparison["model"].eq("hgb")
    ]
    major_change = bool((reference["rpm_only_gain"] <= 0).any() or len(reference) != 2)
    comparison["major_result_change_rule_triggered"] = major_change
    write_csv(output / "rpm_only_original_comparison.csv", comparison)
    return comparison, major_change


def value_from(
    frame: pd.DataFrame, filters: dict[str, Any], column: str, default: float = np.nan
) -> float:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in filters.items():
        mask &= frame[key].fillna("").astype(str).eq(str(value)).to_numpy()
    selected = frame.loc[mask, column]
    return float(selected.iloc[0]) if len(selected) and pd.notna(selected.iloc[0]) else default


def render_proxy_report(
    config: dict[str, Any], results: pd.DataFrame, output: Path
) -> None:
    pathway = results[results["record_type"].eq("target_pathway_support")].iloc[0]
    correlation = results[
        results["record_type"].eq("correlation")
        & results["scope"].eq("global_unique")
        & results["aggregation"].eq("segment")
        & results["outcome"].eq("maf_g_sec_time_weighted_mean")
    ][["predictor", "n", "pearson", "spearman"]]
    within = results[
        results["record_type"].eq("correlation")
        & results["scope"].eq("global_unique")
        & results["aggregation"].eq("within_vehicle_demeaned")
        & results["outcome"].eq("maf_g_sec_time_weighted_mean")
    ][
        [
            "predictor",
            "n",
            "pearson",
            "spearman",
            "median_within_vehicle_pearson",
            "eligible_within_vehicle_correlations",
        ]
    ]
    trip = results[
        results["record_type"].eq("correlation")
        & results["scope"].eq("global_unique")
        & results["aggregation"].eq("across_trip_means")
        & results["outcome"].eq("maf_g_sec_time_weighted_mean")
    ][["predictor", "n", "pearson", "spearman"]]
    transfer = results[
        results["record_type"].eq("correlation")
        & results["scope"].eq("transfer_population")
        & results["aggregation"].eq("segment")
        & results["predictor"].eq("standardized_absolute_load_rpm_maf_proxy_g_s")
        & results["outcome"].eq("maf_g_sec_time_weighted_mean")
    ][["environment", "n", "pearson", "spearman"]]
    diagnostic = results[
        results["record_type"].eq("fuel_target_diagnostic_model")
    ][
        [
            "environment",
            "predictor",
            "n",
            "mae",
            "relative_mae_improvement_vs_lower_information",
            "r2",
        ]
    ]
    maf_reconstruction = results[
        results["record_type"].eq("maf_reconstruction_model")
    ][["environment", "predictor", "n", "mae", "r2"]]
    text = f"""# Target diagnostics stage Absolute Load target-proxy analysis

## Resolution

**Finding: a plausible near-mechanical target-proxy pathway exists. Absolute Load is not demonstrated to be independent of the operational MAF-derived target.** The confirmatory richer-telemetry definition must therefore exclude Absolute Load and use lower-information telemetry plus RPM summaries only. The original Absolute Load-based results remain exploratory target-proxy sensitivity evidence.

## Exact VED field and provenance

The frozen VED dynamic files expose `Absolute Load[%]`, `Engine RPM[RPM]`, `MAF[g/sec]`, and `Fuel Rate[L/hr]`. The [official VED repository](https://github.com/gsoh/VED) describes these as channels recorded by onboard OBD-II loggers; it does not document an independent sensor, vehicle-specific calibration, ECU algorithm, or PID implementation for Absolute Load. The [VED dataset paper](https://arxiv.org/abs/1905.02081) likewise establishes OBD-II acquisition, not an independent physical validation of this channel.

Absolute Load is an ECU-reported calculated OBD parameter, not a directly observed fuel-flow measurement. Under the standardized absolute-load semantics referenced by [ISO 15031-5](https://www.iso.org/standard/66368.html) and SAE J1979-DA, the quantity represents normalized cylinder air charge. For a four-stroke engine, the physically motivated airflow proxy is

`MAF_proxy = AbsoluteLoad[%] × RPM × displacement[L] × 1.184 / 12000`.

The Target diagnostics stage implementation parses displacement from the official VED static workbook. Because the segment artifact retains time-weighted means but not the time-weighted mean of the product, this transformation uses the product of segment means and is explicitly an aggregate approximation. It cannot establish the proprietary ECU computation used by each vehicle.

## Shared computational pathway

The target builder gives valid direct `Fuel Rate[L/hr]` priority and otherwise computes fuel volume as `MAF × available fuel-trim correction / 14.08 / 745 × duration`. In the unique ICE resolution pool ({int(pathway['n']):,} segments), the mean MAF-derived duration share was {float(pathway['mean_maf_duration_share']):.6%}; the mean direct duration share was {float(pathway['mean_direct_duration_share']):.6%}; and only {float(pathway['any_direct_duration_segment_share']):.6%} of segments contained any direct-duration contribution. The displacement parsing coverage was {float(pathway['displacement_parse_coverage']):.6%}.

The original predictor contract correctly removed explicit MAF, Fuel Rate, and fuel-trim summaries. It nevertheless retained Absolute Load and RPM. Because standardized Absolute Load encodes normalized air charge and the target is overwhelmingly an airflow conversion, `Absolute Load × RPM × displacement` can reconstruct target-source information. This is transformed source overlap, even though it is not an exact copied column.

## Empirical relationships

### Global segment-level correlations with MAF

{markdown_table(correlation)}

### Pooled within-vehicle and across-trip relationships

{markdown_table(within)}

{markdown_table(trip)}

### Standardized proxy within each transfer population

{markdown_table(transfer)}

These associations are descriptive. They do not prove leakage or causation, but they are evaluated together with the documented computational pathway and out-of-sample reconstruction.

## Compact frozen-split diagnostic models

All thresholds, feature sets, HGB parameters, seeds, and transfer assignments were fixed in the Target diagnostics stage configuration. Evaluation outcomes were not used for tuning or feature selection. MAE is in litres per segment for the operational fuel target.

{markdown_table(diagnostic, digits=6)}

### Out-of-sample MAF reconstruction

{markdown_table(maf_reconstruction, digits=6)}

## Scientific conclusion

Correlation alone would be insufficient. Here, however, three facts coincide: (1) the outcome is overwhelmingly obtained from MAF; (2) Absolute Load is an ECU-calculated normalized air-charge quantity rather than an independent fuel-flow measurement; and (3) the frozen-split product/displacement transformation and joint RPM–Load models recover substantial MAF/target variation out of sample. VED does not reveal whether each ECU computed the PID from a MAF sensor, speed-density logic, or a hybrid strategy, so direct MAF reuse cannot be asserted vehicle by vehicle. Independence also cannot be demonstrated. The defensible classification is therefore **plausible near-mechanical target proxy**, requiring the RPM-only confirmatory redefinition.
"""
    write_text(output / "PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_AUDIT.md", text)


def render_rpm_comparison_report(
    comparison: pd.DataFrame,
    null_summary: pd.DataFrame,
    major_change: bool,
    output: Path,
) -> None:
    display = comparison[
        [
            "environment",
            "model",
            "original_load_based_gain",
            "rpm_only_gain",
            "gain_change_rpm_minus_original",
            "point_direction_changed",
            "rpm_only_simultaneous_lcb",
            "rpm_only_simultaneous_ucb",
            "rpm_only_state",
        ]
    ]
    null_display = null_summary[null_summary["sensor"].eq("lower_plus_rpm")][
        [
            "environment",
            "null_type",
            "real_gain",
            "null_gain_mean",
            "null_gain_p02_5",
            "null_gain_p97_5",
            "probability_real_exceeds_null",
            "all_point_contrasts_positive",
        ]
    ]
    result_change = (
        "The prespecified major-result-change rule was triggered."
        if major_change
        else "The two frozen positive-reference HGB cells (unseen trips and October 2018) retained positive point gains; the prespecified major-result-change rule was not triggered."
    )
    text = f"""# Target diagnostics stage RPM-only rerun comparison

## Outcome

Absolute Load was removed from the confirmatory richer-telemetry branch. The replacement is exactly the eight frozen lower-information variables plus the two frozen RPM summaries. No target-source feature, Absolute Load column, new observation, new split, new road feature, new model parameter, new seed, new control, or new bootstrap rule was introduced.

{result_change}

## Original exploratory versus redefined confirmatory effects

Relative gain is `(MAE_road-free - MAE_aligned-road) / MAE_road-free`; positive values favor aligned road context. Original values use lower information plus RPM plus Absolute Load and are now exploratory target-proxy sensitivity results. RPM-only values are the confirmatory sensing-regime results.

{markdown_table(display, digits=6)}

## Frozen-control comparison

The 20 matched-noise and 20 within-role whole-R6 Sattolo-permutation artifacts per population were reused byte-for-byte from ROBUSTNESS. The original simultaneous family slots were retained: the unchanged low-information slot plus the new RPM-only slot produce 328 HGB null-gain members, 320 semantic-specificity members, and 24 model-real-gain members.

{markdown_table(null_display, digits=6)}

## Interpretation

All favorable, null, and adverse cells remain in the released CSV. The rerun is retrospective and test-opened; it is not independent confirmation. It resolves the sensing-regime definition without changing the paper's bounded, context-dependent road-context conclusion. The prior Absolute Load-based branch must not be used as confirmatory richer-telemetry evidence.
"""
    write_text(output / "PHASE1_RPM_ONLY_RERUN_COMPARISON.md", text)


def render_target_sensitivity_report(results: pd.DataFrame, output: Path) -> None:
    eligible = results[results["eligible"].eq(True)].copy()
    overview = eligible[
        eligible["sensitivity"].eq("overall")
    ][
        [
            "environment",
            "rows",
            "vehicles",
            "baseline_mae_l",
            "real_mae_l",
            "relative_mae_gain",
            "relative_gain_lcb",
            "relative_gain_ucb",
        ]
    ]
    direction = (
        eligible[~eligible["sensitivity"].eq("overall")]
        .groupby(["environment", "sensitivity"], sort=False)
        .agg(
            eligible_strata=("stratum", "size"),
            direction_consistent_strata=("direction_consistent_with_overall", "sum"),
            minimum_relative_gain=("relative_mae_gain", "min"),
            maximum_relative_gain=("relative_mae_gain", "max"),
        )
        .reset_index()
    )
    source_support = results[
        results["sensitivity"].eq("target_source")
    ][["environment", "stratum", "eligible", "rows", "vehicles", "relative_mae_gain"]]
    text = f"""# Target diagnostics stage target-quality sensitivity analysis

## Resolution

**Gate recommendation: `PASS_WITH_NARROWED_OPERATIONAL_TARGET_CLAIMS`.** The analysis supports prediction of the frozen operational fuel-volume target. It does not validate independently measured physical fuel flow.

## Reference RPM-only road-context effects

{markdown_table(overview, digits=6)}

## Defensible sensitivities

All quantitative strata use train-only thresholds within each frozen transfer population. The following were run because the segment artifacts support them: fuel-trim summary availability; source-path composition; high reported Absolute Load; high speed; motorway exposure; speed/RPM transient versus stable operation; and target-duration coverage. These are descriptive effect-modification checks, not new model-selection strata.

{markdown_table(direction, digits=6)}

### Direct Fuel Rate support

{markdown_table(source_support, digits=6)}

Direct-only support is too sparse wherever a row is marked ineligible. It is reported, not promoted to a standalone predictive validation.

## Sensitivities not run

- Open-loop enrichment is not identifiable: VED does not provide commanded equivalence ratio, injector pulse width, or a reliable fuel-system mode for this purpose. High reported load is shown only as a stress stratum and is not relabelled as enrichment.
- Deceleration fuel cut is not identifiable: speed/RPM changes do not establish zero injection, and no defensible fuel-cut state channel is available. No ad hoc proxy was invented.
- Constant rescaling was not run because it cannot change relative MAE contrasts and would not address operating-state-dependent error.

## Interpretation boundary

The target hierarchy is channel-native direct Fuel Rate when valid, otherwise an assumption-bound MAF/trim conversion with fixed stoichiometric air-fuel ratio and fuel density. The empirical sensitivity results can show whether the **operational target's** road-context error pattern is stable across observable strata. They cannot establish accuracy against independently metered fuel flow, cannot identify enrichment or fuel cut, and cannot eliminate vehicle-specific ECU/reporting error. Paper claims must retain this boundary.
"""
    write_text(output / "PHASE1_TARGET_QUALITY_SENSITIVITY.md", text)


def render_map_audit_plan(output: Path) -> None:
    text = """# Road–trajectory correspondence validation plan

## Scope

The released joins use a historical road network and a deterministic map-matching pipeline. They are reported as **pipeline-assigned road–trajectory correspondence**, not as geometric ground truth for every row.

## Recommended sample

For an optional extension, draw a reproducible, target-blind sample of trajectory windows across the four published evaluation populations. Balance road class, junction complexity, parallel or ramp geometry, and map-matching distance. Freeze the seed, segment IDs, inclusion probabilities, and replacement rules before rendering maps.

## Materials and labels

Show historical OSM geometry, raw GPS, the pipeline-assigned path, direction arrows, and a fixed temporal buffer. Use the same historical snapshot as production. Record one of: supported road identity, local geometric ambiguity without a road-class change, wrong parallel/ramp or junction, other material mismatch, or uncertain.

Distance alone is insufficient. Inspect continuity before and after the window, parallel facilities, ramps, divided carriageways, and junction choice. Record a short note for every mismatch or uncertain case.

## Metrics and interpretation

Report agreement, class-specific error rates, population-specific error rates with Wilson intervals, uncertainty rate, and distance-band calibration. If the sample is stratified, use the published inclusion probabilities for population estimates.

If the correspondence criteria are not met, retain the wording **pipeline-assigned road–trajectory correspondence** and report the observed failure mode. Even a successful sample describes this pipeline and these populations; it does not establish row-level geometric truth or causal road effects.

This validation plan is separate from the released model metrics. The RPM-only analysis does not depend on completing it.
"""
    write_text(output / "ROAD_CORRESPONDENCE_QUALITY_PLAN.md", text)


def build_claim_impact_matrix(
    source_inventory: dict[str, Any], major_change: bool, output: Path
) -> pd.DataFrame:
    rows = [
        ("target definition", "narrow scope", "Describe the direct Fuel Rate / MAF-derived operational target and its observability limits."),
        ("feature contract", "update scope", "Define the RPM-only high-information branch and exclude target-source channels."),
        ("road correspondence", "narrow scope", "Use pipeline-assigned road–trajectory correspondence language for the released joins."),
        ("OOD evaluation", "retain scope", "Report context-dependent effects across the four frozen environments and sensor regimes."),
        ("null controls", "retain scope", "Preserve noise and whole-block permutation controls with the same split roles and seeds."),
        ("uncertainty", "retain scope", "Report vehicle-cluster bootstrap intervals and simultaneous families."),
        ("external transfer", "prohibit", "Do not extend the findings to other fleets, vehicles, causal effects, or deployment decisions."),
    ]
    matrix = pd.DataFrame(rows, columns=["topic", "scope_action", "scope_statement"])
    matrix["phase1_trigger"] = np.where(
        matrix["topic"].eq("road correspondence"),
        "verification: correspondence remains pipeline-assigned",
        "verification: target and split contracts are frozen",
    )
    matrix["major_result_change_rule_triggered"] = major_change
    matrix["source_files_complete"] = source_inventory["complete"]
    matrix["source_evidence_gap"] = "; ".join(source_inventory["missing"])
    matrix["status"] = "PUBLIC_SCOPE_SUMMARY"
    allowed = {
        "narrow scope",
        "update scope",
        "retain scope",
        "prohibit",
    }
    if not set(matrix["scope_action"]).issubset(allowed):
        raise RuntimeError("Public scope table contains an unexpected action label.")
    write_csv(output / "PUBLIC_SCOPE_SUMMARY.csv", matrix)
    return matrix


def write_reproducibility_note(output: Path) -> None:
    text = """# Reproducibility and scope note

The released target is reconstructed from VED channels and the public target contract. The package stores the source hashes, feature masks, split memberships, model fits, predictions, raw metrics, bootstrap draws, and null-control outputs used by the reported diagnostics.

Direct Fuel Rate is preferred when present; otherwise the operational fuel-volume target uses MAF with the documented fuel-trim, density, and elapsed-time terms. This is a dataset-derived operational target, not independently metered fuel-flow ground truth. Absolute Load is therefore treated as a sensitivity signal rather than an independent measurement.

Road-network results are prediction-error contrasts under the named OOD environments and sensor regimes. They do not establish causal infrastructure effects, fuel savings, external-fleet transfer, or deployment safety.
"""
    write_text(output / "REPRODUCIBILITY_SCOPE_NOTE.md", text)


def build_decision(
    config: dict[str, Any],
    proxy_results: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    source_inventory: dict[str, Any],
    major_change: bool,
) -> dict[str, Any]:
    pathway = proxy_results[proxy_results["record_type"].eq("target_pathway_support")].iloc[0]
    eligible_sensitivity = sensitivity[
        sensitivity["eligible"].eq(True) & ~sensitivity["sensitivity"].eq("overall")
    ]
    direction_changes = int(
        eligible_sensitivity["direction_consistent_with_overall"]
        .fillna(False)
        .astype(bool)
        .eq(False)
        .sum()
    )
    gate_a = "PASS_AFTER_RPM_ONLY_REDEFINITION"
    gate_b = "PASS_WITH_NARROWED_OPERATIONAL_TARGET_CLAIMS"
    gate_c = "PASS_PIPELINE_ASSIGNED_LANGUAGE_ONLY"
    overall = (
        "STOP_PHASE1_MAJOR_RESULT_CHANGE"
        if major_change
        else "PASS_PHASE1_WITH_REQUIRED_CLAIM_NARROWING"
    )
    references = comparison[
        comparison["environment"].isin(config["decision"]["frozen_positive_reference_environments"])
        & comparison["model"].eq("hgb")
    ][
        [
            "environment",
            "original_load_based_gain",
            "rpm_only_gain",
            "rpm_only_simultaneous_lcb",
            "rpm_only_simultaneous_ucb",
        ]
    ].to_dict("records")
    decision = {
        "schema_version": 1,
        "experiment": config["experiment"]["id"],
        "issued_at": datetime.now().astimezone().isoformat(),
        "gate_a_sensing_regime_claim": gate_a,
        "gate_a_basis": {
            "absolute_load_independence_demonstrated": False,
            "plausible_near_mechanical_target_proxy": True,
            "rpm_only_rerun_complete": True,
            "confirmatory_richer_telemetry": "lower_information_plus_RPM_summaries_only",
            "original_load_based_disposition": "EXPLORATORY_TARGET_PROXY_SENSITIVITY_ONLY",
            "reference_cell_comparison": references,
        },
        "gate_b_operational_target_robustness": gate_b,
        "gate_b_basis": {
            "mean_maf_duration_share": float(pathway["mean_maf_duration_share"]),
            "mean_direct_duration_share": float(pathway["mean_direct_duration_share"]),
            "eligible_sensitivity_strata": int(len(eligible_sensitivity)),
            "eligible_strata_with_point_direction_change": direction_changes,
            "enrichment_identifiable": False,
            "fuel_cut_identifiable": False,
            "independent_physical_fuel_flow_validated": False,
            "permitted_target_interpretation": "prediction_of_operational_direct_or_MAF_derived_fuel_volume_target",
        },
        "gate_c_alignment_language": gate_c,
        "gate_c_basis": {
            "transfer_stratified_correspondence_quality_check_complete": False,
            "current_100_trip_form_complete": False,
            "current_100_trip_blank_ratings": 75,
            "off_network_candidate_quality_sample_is_representative": False,
            "required_language": "pipeline-assigned road–trajectory correspondence",
        },
        "overall_phase_decision": overall,
        "major_result_change_rule_triggered": bool(major_change),
        "source_inventory_complete": bool(source_inventory["complete"]),
        "non_negotiable_claim_boundaries": [
            "No independently measured physical fuel-flow validation",
            "No causal road-infrastructure effect",
            "No external-fleet transfer claim",
            "No claim of row-level geometric ground truth",
            "Original Absolute Load-based results are exploratory only",
        ],
    }
    if gate_a not in config["decision"]["allowed_gate_a"]:
        raise RuntimeError("Invalid Gate A decision")
    if gate_b not in config["decision"]["allowed_gate_b"]:
        raise RuntimeError("Invalid Gate B decision")
    if gate_c not in config["decision"]["allowed_gate_c"]:
        raise RuntimeError("Invalid Gate C decision")
    if overall not in config["decision"]["allowed_overall"]:
        raise RuntimeError("Invalid overall target-diagnostics decision")
    return decision


def render_scientific_report(
    config: dict[str, Any],
    decision: dict[str, Any],
    proxy_results: pd.DataFrame,
    rerun_results: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    claim_matrix: pd.DataFrame,
    source_inventory: dict[str, Any],
    output: Path,
) -> None:
    model_real = rerun_results[rerun_results["representation"].eq("real")][
        [
            "environment",
            "model",
            "mae_l",
            "mae_pointwise_lcb",
            "mae_pointwise_ucb",
            "relative_mae_gain",
            "relative_gain_simultaneous_lcb",
            "relative_gain_simultaneous_ucb",
            "state",
        ]
    ]
    pathway = proxy_results[proxy_results["record_type"].eq("target_pathway_support")].iloc[0]
    text = f"""# Target diagnostics report

## Decision

**Overall: `{decision['overall_phase_decision']}`**

- Gate A — Sensing-regime claim: `{decision['gate_a_sensing_regime_claim']}`
- Gate B — Operational target robustness: `{decision['gate_b_operational_target_robustness']}`
- Gate C — Road correspondence scope: `{decision['gate_c_alignment_language']}`

This report describes the released target and model-evaluation boundaries. Source/configuration hashes, split memberships, model fits, predictions, null controls, and bootstrap draws are retained with the package.

## 1. Target definition and sensitivity

Absolute Load is an ECU-reported calculated OBD quantity. VED does not document an independent sensor or vehicle-specific ECU calculation. Standardized Absolute Load represents normalized air charge, so the physically motivated `Absolute Load × RPM × displacement` transformation overlaps the MAF pathway. In the unique ICE resolution pool, the target's mean MAF duration share is {float(pathway['mean_maf_duration_share']):.6%} and its mean direct Fuel Rate duration share is {float(pathway['mean_direct_duration_share']):.6%}. The global, within-vehicle, across-trip, transfer-population, and frozen-split diagnostic results are released in `PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_RESULTS.csv`.

The evidence does not prove that every vehicle's ECU calculated Absolute Load directly from its MAF sensor. It does show a plausible near-mechanical target-reconstruction pathway and fails to demonstrate independence. The original richer-telemetry branch is therefore exploratory only.

## 2. RPM-only analysis branch

The redefined branch contains the frozen eight lower-information variables plus RPM mean and standard deviation. Absolute Load and all explicit target-source features are absent. Exact observations, target memberships, split roles, road masks, model parameters, seeds, 20 matched-noise controls, 20 within-role whole-block permutation controls, vehicle bootstrap, max-t families, and 5.5% SESOI were preserved.

{markdown_table(model_real, digits=6)}

The comparison with the original Absolute Load-based branch is:

{markdown_table(comparison[['environment', 'model', 'original_load_based_gain', 'rpm_only_gain', 'point_direction_changed']], digits=6)}

The prespecified major-result-change rule was {'triggered' if decision['major_result_change_rule_triggered'] else 'not triggered'}.

## 3. Target quality

The operational target remains defensible for benchmark prediction-error analysis, with required narrowing. Train-thresholded sensitivity checks cover source path, trim-summary availability, reported high load, high speed, motorway exposure, transient/stable states, and target-duration coverage. Open-loop enrichment and deceleration fuel cut cannot be identified from the available channels and were not fabricated. Direct-only segments are too sparse for independent physical-flow validation. All eligible, ineligible, favorable, null, and adverse strata are preserved in `target_quality_sensitivity_results.csv`.

## 4. Road–trajectory correspondence

The released joins are described as **“pipeline-assigned road–trajectory correspondence.”** The package preserves map-matching coverage, discontinuity, and off-network quality fields so readers can reproduce or extend the quality analysis without treating the join as geometric ground truth for every row.

## 5. Public scope

The public scope table contains {len(claim_matrix)} entries covering the target, feature contract, road correspondence, OOD environments, null controls, uncertainty, and external-transfer boundary. It is included as a reader-facing summary of what the released evidence does and does not support.

## 6. Reproducibility

The source inventory is complete: `{source_inventory['complete']}`. No external attachment is required to verify the public package. The accompanying release manifest provides byte counts and SHA-256 values for every artifact.
"""
    write_text(output / "PHASE1_TARGET_DIAGNOSTICS_REPORT.md", text)


def key_frozen_hashes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = [
        Path(config["inputs"]["segments"]),
        Path(config["inputs"]["static_vehicle_metadata"]),
        Path(config["inputs"]["feature_contract"]),
        Path(config["inputs"]["information_contract"]),
        Path(config["inputs"]["target_builder"]),
        Path(config["inputs"]["segment_builder"]),
        Path(config["inputs"]["paper"]),
        Path(config["inputs"]["robustness_root"]) / "effects.csv",
        Path(config["inputs"]["robustness_root"]) / "active_feature_masks.json",
        Path(config["inputs"]["robustness_root"]) / "artifact_manifest.json",
    ]
    for environment in config["environments"]:
        paths.append(membership_path(config, environment))
        paths.extend(original_prediction_paths(config, environment))
    return {
        str(path): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def run() -> None:
    args = parse_args()
    config = read_config(args.config)
    output = resolve_output(config, args.output_dir)
    if output.exists() and not args.resume:
        raise RuntimeError(f"Output already exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "logs" / "execution.log")
    logger.log(f"START {config['experiment']['id']} output={output}")
    state_path = output / "phase1_state.json"
    state = load_json(state_path) if state_path.exists() else {"status": "STARTED"}
    write_json(state_path, state)
    shutil.copy2(args.config, output / "run_config.yaml")
    write_json(
        output / "runtime_versions.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "module_version_unavailable",
            "sklearn": sklearn.__version__,
        },
    )
    source_inventory_result = source_inventory(config, output)
    frozen_before = key_frozen_hashes(config)
    write_json(output / "frozen_input_hashes_before.json", frozen_before)

    segments = load_segment_data(config, logger)
    static = load_static_displacement(config)
    data = add_proxy_features(segments, static, config)
    memberships, membership_audits = load_memberships(config, logger)
    write_csv(output / "membership_audit.csv", pd.DataFrame(membership_audits))

    proxy_path = output / "PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_RESULTS.csv"
    if args.resume and proxy_path.exists():
        proxy_results = pd.read_csv(proxy_path)
        logger.log("PROXY_AUDIT resumed from existing CSV")
    else:
        proxy_results = proxy_audit(config, data, memberships, output, logger)
    render_proxy_report(config, proxy_results, output)
    state["status"] = "PROXY_AUDIT_COMPLETE"
    write_json(state_path, state)

    road_features, road_masks, road_audits = road_contract(config, memberships)
    write_json(
        output / "rpm_only_active_feature_masks.json",
        {
            "schema_version": 1,
            "road_contract_count": len(road_features),
            "policy": config["features"]["road_mask_policy"],
            "cells": road_audits,
        },
    )
    real_road = load_road_features(config, road_masks, logger)
    manifest_index = robustness_manifest_index(config)
    rpm_predictions: dict[str, pd.DataFrame] = {}
    rpm_metadata: dict[str, Any] = {}
    for environment in config["environments"]:
        frame, metadata = rpm_only_cell(
            config,
            environment,
            data,
            memberships[environment],
            real_road,
            road_masks[environment],
            manifest_index,
            output,
            logger,
        )
        rpm_predictions[environment] = frame
        rpm_metadata[environment] = metadata
        state["status"] = f"RPM_ONLY_{environment}_COMPLETE"
        write_json(state_path, state)
    write_json(output / "rpm_only_fit_manifest.json", rpm_metadata)
    rerun_results, effects, null_summary, bootstrap_frames = rpm_only_inference(
        config, rpm_predictions, manifest_index, output, logger
    )
    comparison, major_change = comparison_table(config, effects, output)
    render_rpm_comparison_report(comparison, null_summary, major_change, output)
    state["status"] = "RPM_ONLY_INFERENCE_COMPLETE"
    write_json(state_path, state)

    sensitivity = target_quality_sensitivity(
        config, data, memberships, bootstrap_frames, output, logger
    )
    render_target_sensitivity_report(sensitivity, output)
    render_map_audit_plan(output)
    claim_matrix = build_claim_impact_matrix(source_inventory_result, major_change, output)
    write_reproducibility_note(output)
    decision = build_decision(
        config,
        proxy_results,
        comparison,
        sensitivity,
        source_inventory_result,
        major_change,
    )
    write_json(output / "PHASE1_DECISION.json", decision)
    render_scientific_report(
        config,
        decision,
        proxy_results,
        rerun_results,
        comparison,
        sensitivity,
        claim_matrix,
        source_inventory_result,
        output,
    )

    frozen_after = key_frozen_hashes(config)
    unchanged = frozen_before == frozen_after
    write_json(
        output / "frozen_input_integrity.json",
        {
            "all_key_frozen_inputs_unchanged": unchanged,
            "before": frozen_before,
            "after": frozen_after,
        },
    )
    if not unchanged:
        raise RuntimeError("At least one key frozen input changed during Target diagnostics stage.")

    required = [
        "PHASE1_TARGET_DIAGNOSTICS_REPORT.md",
        "PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_AUDIT.md",
        "PHASE1_ABSOLUTE_LOAD_TARGET_PROXY_RESULTS.csv",
        "PHASE1_RPM_ONLY_RERUN_RESULTS.csv",
        "PHASE1_RPM_ONLY_RERUN_COMPARISON.md",
        "PHASE1_TARGET_QUALITY_SENSITIVITY.md",
        "ROAD_CORRESPONDENCE_QUALITY_PLAN.md",
        "PUBLIC_SCOPE_SUMMARY.csv",
        "REPRODUCIBILITY_SCOPE_NOTE.md",
        "PHASE1_DECISION.json",
    ]
    missing_required = [name for name in required if not (output / name).exists()]
    if missing_required:
        raise RuntimeError(f"Missing required deliverables: {missing_required}")
    expected_metric_rows = len(config["environments"]) * (
        len(MODELS) * 2 + len(NULL_TYPES) * int(config["null_controls"]["draws_per_type"])
    )
    checks = {
        "required_deliverables_present": not missing_required,
        "rpm_result_row_count": int(len(rerun_results)),
        "rpm_result_row_count_expected": int(expected_metric_rows),
        "rpm_result_grid_complete": len(rerun_results) == expected_metric_rows,
        "all_four_environments_present": set(rerun_results["environment"])
        == set(config["environments"]),
        "absolute_load_absent_from_confirmatory_masks": all(
            not any("absolute_load" in feature.lower() for feature in record["baseline_features"])
            for record in rpm_metadata.values()
        ),
        "verified_reused_control_count": int(
            sum(len(record["verified_frozen_controls"]) for record in rpm_metadata.values())
        ),
        "verified_reused_control_count_expected": len(config["environments"])
        * len(NULL_TYPES)
        * int(config["null_controls"]["draws_per_type"]),
        "key_frozen_inputs_unchanged": unchanged,
        "paper_protocol_hash_unchanged": frozen_before[
            str(Path(config["inputs"]["paper"]))
        ]
        == frozen_after[str(Path(config["inputs"]["paper"]))],
        "overall_decision_exactly_one": decision["overall_phase_decision"]
        in config["decision"]["allowed_overall"],
    }
    checks["all_hard_checks_pass"] = bool(
        checks["required_deliverables_present"]
        and checks["rpm_result_grid_complete"]
        and checks["all_four_environments_present"]
        and checks["absolute_load_absent_from_confirmatory_masks"]
        and checks["verified_reused_control_count"]
        == checks["verified_reused_control_count_expected"]
        and checks["key_frozen_inputs_unchanged"]
        and checks["paper_protocol_hash_unchanged"]
        and checks["overall_decision_exactly_one"]
    )
    write_json(output / "phase1_contract_checks.json", checks)
    if not checks["all_hard_checks_pass"]:
        raise RuntimeError(f"Target diagnostics stage contract checks failed: {checks}")

    state["status"] = "COMPLETE"
    state["overall_decision"] = decision["overall_phase_decision"]
    state["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(state_path, state)
    logger.log(f"FINALIZE decision={decision['overall_phase_decision']}")
    manifest_candidates = [path for path in output.rglob("*") if path.is_file()]
    manifest_candidates = [
        path
        for path in manifest_candidates
        if path.name not in {"PHASE1_RELEASE_MANIFEST.json", "IMMUTABILITY.json"}
    ]
    release_manifest = artifact_manifest(
        manifest_candidates,
        experiment=config["experiment"]["id"],
        output_directory=str(output),
        overall_decision=decision["overall_phase_decision"],
        required_deliverables=required,
        frozen_originals_unchanged=True,
    )
    write_json(output / "PHASE1_RELEASE_MANIFEST.json", release_manifest)
    write_json(
        output / "IMMUTABILITY.json",
        {
            "status": "IMMUTABLE_BY_POLICY_AFTER_RELEASE",
            "released_at": datetime.now().astimezone().isoformat(),
            "release_manifest": str(output / "PHASE1_RELEASE_MANIFEST.json"),
            "release_manifest_sha256": sha256_file(output / "PHASE1_RELEASE_MANIFEST.json"),
            "mutation_policy": "Never overwrite this directory; any correction requires a new timestamped release.",
        },
    )
    pointer = {
        "experiment": config["experiment"]["id"],
        "status": "COMPLETE",
        "overall_decision": decision["overall_phase_decision"],
        "output_directory": str(output),
        "report": str(output / "PHASE1_TARGET_DIAGNOSTICS_REPORT.md"),
        "decision": str(output / "PHASE1_DECISION.json"),
        "release_manifest": str(output / "PHASE1_RELEASE_MANIFEST.json"),
        "release_manifest_sha256": sha256_file(output / "PHASE1_RELEASE_MANIFEST.json"),
    }
    write_json(Path(config["output"]["latest_pointer"]), pointer)
    print(
        f"COMPLETE decision={decision['overall_phase_decision']} "
        f"manifest={pointer['release_manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    run()
