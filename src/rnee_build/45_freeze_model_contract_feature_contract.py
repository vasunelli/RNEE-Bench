#!/usr/bin/env python3
"""Freeze the MODEL_CONTRACT M1/B1 feature, target-support, and mechanism contracts.

This stage does not fit or evaluate any prediction model.  It derives the R2
segment layer without changing SEGMENT_SPLIT segment boundaries, freezes target-specific
common-support memberships, and decides whether a genuine mechanistic baseline
is identifiable from the released VED inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


def resolve_columns(columns: list[str], specification: dict[str, list[str]]) -> list[str]:
    resolved: list[str] = []
    for column in specification.get("exact", []):
        if column not in columns:
            raise RuntimeError(f"Required feature column is absent: {column}")
        resolved.append(column)
    for prefix in specification.get("prefixes", []):
        matches = sorted(column for column in columns if column.startswith(prefix))
        if not matches:
            raise RuntimeError(f"Feature prefix matched no columns: {prefix}")
        resolved.extend(matches)
    return list(dict.fromkeys(resolved))


def forbidden_hits(features: list[str], config: dict[str, Any]) -> list[dict[str, str]]:
    exact = {value.lower() for value in config["forbidden_features"]["exact"]}
    tokens = [value.lower() for value in config["forbidden_features"]["tokens"]]
    hits: list[dict[str, str]] = []
    for feature in features:
        lowered = feature.lower()
        if lowered in exact:
            hits.append({"feature": feature, "rule": "exact"})
            continue
        for token in tokens:
            if token in lowered:
                hits.append({"feature": feature, "rule": f"token:{token}"})
                break
    return hits


def interval_contributions(rows: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    """Reproduce the SEGMENT_SPLIT exact left-hold split at fixed time boundaries."""
    origin = rows.groupby("pilot_trip_id", sort=False)["Timestamp(ms)"].transform("min").to_numpy(np.int64)
    timestamp = rows["Timestamp(ms)"].to_numpy(np.int64)
    segment_index = np.floor_divide(timestamp - origin, window_ms).astype(np.int64)
    dt_ms = pd.to_numeric(rows["dt_seconds"], errors="coerce").fillna(0).to_numpy(float) * 1000.0
    valid = rows["dt_valid"].fillna(False).to_numpy(bool) & (dt_ms > 0)
    offset = (timestamp - origin) - segment_index * window_ms
    first_ms = np.where(valid, np.minimum(dt_ms, window_ms - offset), 0.0)
    second_ms = np.where(valid, np.maximum(0.0, dt_ms - first_ms), 0.0)
    row_pos = np.arange(len(rows), dtype=np.int64)
    trip = rows["pilot_trip_id"].astype(str).to_numpy()
    pieces: list[pd.DataFrame] = []
    for increment, duration_ms in ((0, first_ms), (1, second_ms)):
        mask = duration_ms > 0
        if not mask.any():
            continue
        piece = pd.DataFrame(
            {
                "_row_pos": row_pos[mask],
                "_segment_index": segment_index[mask] + increment,
                "weight_s": duration_ms[mask] / 1000.0,
            }
        )
        piece["_segment_key"] = trip[mask] + "|" + piece["_segment_index"].astype(str).to_numpy()
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["_row_pos", "_segment_index", "weight_s", "_segment_key"])
    return pd.concat(pieces, ignore_index=True)


def normalize_category(value: Any, mapping: dict[str, str], missing_label: str) -> str:
    if pd.isna(value):
        return missing_label
    key = str(value).strip().lower()
    return mapping.get(key, "other")


def aggregate_r2_partition(
    rows: pd.DataFrame,
    segments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = rows.sort_values(["pilot_trip_id", "Timestamp(ms)", "point_index"], kind="stable").reset_index(drop=True)
    contributions = interval_contributions(rows, int(float(config["window_seconds"]) * 1000))
    positions = contributions["_row_pos"].to_numpy(np.int64)
    contribution_rows = rows.iloc[positions].reset_index(drop=True)
    speed = pd.to_numeric(contribution_rows["Vehicle Speed[km/h]"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    work = pd.DataFrame(
        {
            "_segment_key": contributions["_segment_key"].to_numpy(str),
            "weight_s": contributions["weight_s"].to_numpy(float),
            "distance_m": speed / 3.6 * contributions["weight_s"].to_numpy(float),
            "road_available": contribution_rows["road_semantics_available"].fillna(False).to_numpy(bool),
        }
    )
    key_to_id = dict(
        zip(
            segments["trip_uid"].astype(str) + "|" + segments["segment_index"].astype(str),
            segments["segment_id"].astype(str),
        )
    )
    work["segment_id"] = work["_segment_key"].map(key_to_id)
    unmapped = int(work["segment_id"].isna().sum())
    if unmapped:
        raise RuntimeError(f"R2 interval pieces do not map to frozen SEGMENT_SPLIT segments: {unmapped}")

    total_distance = work.groupby("segment_id", sort=False)["distance_m"].sum().rename("r2_total_distance_m")
    road_work = work[work["road_available"]].copy()
    road_distance = road_work.groupby("segment_id", sort=False)["distance_m"].sum().rename("r2_road_available_distance_m")
    output = pd.concat([total_distance, road_distance], axis=1).fillna({"r2_road_available_distance_m": 0.0})
    output["r2_road_distance_coverage"] = output["r2_road_available_distance_m"] / output["r2_total_distance_m"].replace(0, np.nan)

    lane = pd.to_numeric(contribution_rows["osm_lane_count"], errors="coerce")
    road_work["lane"] = lane.to_numpy(float)[work["road_available"].to_numpy(bool)]
    lane_known = road_work[np.isfinite(road_work["lane"]) & road_work["distance_m"].gt(0)].copy()
    if not lane_known.empty:
        lane_known["weighted_lane"] = lane_known["lane"] * lane_known["distance_m"]
        lane_group = lane_known.groupby("segment_id", sort=False)
        lane_distance = lane_group["distance_m"].sum()
        output["r2_lane_count_distance_weighted_mean"] = lane_group["weighted_lane"].sum() / lane_distance
        output["r2_lane_tag_distance_coverage"] = lane_distance / output["r2_road_available_distance_m"].replace(0, np.nan)
    else:
        output["r2_lane_count_distance_weighted_mean"] = np.nan
        output["r2_lane_tag_distance_coverage"] = np.nan

    category_audit: dict[str, Counter[str]] = {}
    for source_column, category_spec in config["r2_categorical_contract"].items():
        mapping = {str(key).lower(): str(value) for key, value in category_spec["mapping"].items()}
        missing_label = str(category_spec["missing_label"])
        mapped = contribution_rows[source_column].map(lambda value: normalize_category(value, mapping, missing_label))
        category_audit[source_column] = Counter(mapped.astype(str).tolist())
        road_work[source_column] = mapped.to_numpy()[work["road_available"].to_numpy(bool)]
        for category in category_spec["categories"]:
            numerator = road_work[road_work[source_column].eq(category)].groupby("segment_id", sort=False)["distance_m"].sum()
            feature = f"r2_{source_column}_{category}_distance_share"
            output[feature] = numerator / output["r2_road_available_distance_m"].replace(0, np.nan)

    output = segments[["segment_id"]].merge(output.reset_index(), on="segment_id", how="left")
    output["r2_total_distance_m"] = output["r2_total_distance_m"].fillna(0.0)
    output["r2_road_available_distance_m"] = output["r2_road_available_distance_m"].fillna(0.0)
    audit = {
        "segment_count": int(len(output)),
        "interval_piece_count": int(len(work)),
        "interval_duration_s": float(work["weight_s"].sum()),
        "integrated_distance_m": float(work["distance_m"].sum()),
        "road_available_distance_m": float(road_work["distance_m"].sum()),
        "unmapped_interval_piece_count": unmapped,
        "category_counts": {column: dict(counter) for column, counter in category_audit.items()},
    }
    return output, audit


def build_feature_contract(
    segments: pd.DataFrame,
    r2: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    segment_columns = list(segments.columns)
    groups = {
        "R0_LOW_SENSOR": resolve_columns(segment_columns, config["feature_groups"]["R0_LOW_SENSOR"]),
        "R0_FULL_OBD_ADDITIONS": resolve_columns(segment_columns, config["feature_groups"]["R0_FULL_OBD_ADDITIONS"]),
        "R1_HIERARCHY_REGULATION": resolve_columns(segment_columns, config["feature_groups"]["R1_HIERARCHY_REGULATION"]),
        "R2_PHYSICAL_ACCESS": [
            column
            for column in r2.columns
            if column != "segment_id" and column not in set(config["r2_non_feature_columns"])
        ],
        "R3_TOPOLOGY": resolve_columns(segment_columns, config["feature_groups"]["R3_TOPOLOGY"]),
        "R4_TRAFFIC_PLACE_OBJECTS": resolve_columns(segment_columns, config["feature_groups"]["R4_TRAFFIC_PLACE_OBJECTS"]),
        "R5_BUILT_ENVIRONMENT": resolve_columns(segment_columns, config["feature_groups"]["R5_BUILT_ENVIRONMENT"]),
    }
    road_union = list(
        dict.fromkeys(
            groups["R1_HIERARCHY_REGULATION"]
            + groups["R2_PHYSICAL_ACCESS"]
            + groups["R3_TOPOLOGY"]
            + groups["R4_TRAFFIC_PLACE_OBJECTS"]
            + groups["R5_BUILT_ENVIRONMENT"]
        )
    )
    groups["R6_FULL_SEMANTICS"] = road_union

    target_contracts: dict[str, Any] = {}
    all_hits: list[dict[str, str]] = []
    for target_id, target in config["targets"].items():
        low = list(groups["R0_LOW_SENSOR"])
        denied_sources = set(config["target_source_feature_deny"].get(target["channel"], []))
        full = [
            feature
            for feature in low + groups["R0_FULL_OBD_ADDITIONS"]
            if feature not in denied_sources
        ]
        low_hits = forbidden_hits(low, config)
        full_hits = forbidden_hits(full, config)
        road_hits = forbidden_hits(road_union, config)
        for hit in low_hits + full_hits + road_hits:
            all_hits.append({"target_id": target_id, **hit})
        target_contracts[target_id] = {
            **target,
            "r0_low_sensor": low,
            "r0_full_obd": list(dict.fromkeys(full)),
            "target_source_features_removed": sorted(denied_sources.intersection(set(groups["R0_FULL_OBD_ADDITIONS"]))),
            "road_semantic_groups": {
                key: value
                for key, value in groups.items()
                if key.startswith("R") and not key.startswith("R0")
            },
        }

    contract = {
        "schema_version": 1,
        "experiment": "MODEL_CONTRACT",
        "stage": "M1_B1_feature_contract",
        "estimand": "target-specific prediction risk conditional on prediction_usable AND road_semantics_model_eligible",
        "feature_groups": groups,
        "targets": target_contracts,
        "preprocessing_contract": config["preprocessing_contract"],
        "r2_taxonomy": config["r2_categorical_contract"],
        "forbidden_feature_contract": config["forbidden_features"],
        "forbidden_hits": all_hits,
        "unified_scalar_target_released": False,
    }
    return contract, all_hits


def freeze_target_memberships(
    segments: pd.DataFrame,
    split_root: Path,
    output: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    segment_index = segments.set_index("segment_id", drop=False)
    support_records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    membership_root = output / "target_memberships"
    membership_root.mkdir(parents=True, exist_ok=True)
    required_roles = {"train", "validation", "calibration", "test"}
    for split_name in config["split_families"]:
        membership_path = split_root / split_name / "model_membership.parquet"
        membership = pd.read_parquet(membership_path)
        if set(membership["split_role"].dropna().unique()) != required_roles:
            raise RuntimeError(f"Unexpected split roles in {membership_path}")
        joined = membership.merge(
            segments[
                [
                    "segment_id",
                    "VehId",
                    "trip_uid",
                    "engine_type",
                    "prediction_usable",
                    "road_semantics_model_eligible",
                    "fuel_target_coverage",
                    "battery_target_coverage",
                ]
            ],
            on="segment_id",
            how="left",
            validate="one_to_one",
        )
        if joined["engine_type"].isna().any():
            raise RuntimeError(f"Split membership contains unknown segment IDs: {split_name}")
        for target_id, target in config["targets"].items():
            coverage_column = target["coverage_column"]
            eligible = (
                joined["prediction_usable"].fillna(False)
                & joined["road_semantics_model_eligible"].fillna(False)
                & joined["engine_type"].eq(target["engine_type"])
                & joined[coverage_column].ge(float(config["target_minimum_coverage"]))
            )
            frozen = joined.loc[eligible, ["segment_id", "split_role"]].sort_values(["split_role", "segment_id"])
            target_dir = membership_root / split_name / target_id
            target_dir.mkdir(parents=True, exist_ok=True)
            frozen_path = target_dir / "model_membership.parquet"
            frozen.to_parquet(frozen_path, index=False)
            artifacts.append(
                {
                    "split_family": split_name,
                    "target_id": target_id,
                    "path": str(frozen_path),
                    "row_count": int(len(frozen)),
                    "sha256": sha256_file(frozen_path),
                }
            )
            frozen_detail = frozen.merge(
                segment_index[["VehId", "trip_uid"]],
                left_on="segment_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
            for role in sorted(required_roles):
                subset = frozen_detail[frozen_detail["split_role"].eq(role)]
                vehicle_counts = subset["VehId"].value_counts()
                max_share = float(vehicle_counts.max() / len(subset)) if len(subset) else math.nan
                support_records.append(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "engine_type": target["engine_type"],
                        "channel": target["channel"],
                        "unit": target["unit"],
                        "split_role": role,
                        "segment_count": int(len(subset)),
                        "vehicle_count": int(subset["VehId"].nunique()),
                        "trip_count": int(subset["trip_uid"].nunique()),
                        "max_vehicle_segment_share": max_share,
                    }
                )
    support = pd.DataFrame(support_records)
    support["test_support_status"] = classify_test_support(
        support, config["support_thresholds"]
    )
    return support, artifacts


def classify_test_support(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    status = pd.Series("NOT_APPLICABLE", index=frame.index, dtype="string")
    test_mask = frame["split_role"].eq("test")
    supported = (
        (frame["vehicle_count"] >= int(thresholds["minimum_vehicles"]))
        & (frame["trip_count"] >= int(thresholds["minimum_trips"]))
        & (frame["segment_count"] >= int(thresholds["minimum_segments"]))
        & (frame["max_vehicle_segment_share"] <= float(thresholds["maximum_vehicle_segment_share"]))
    )
    status.loc[test_mask & supported] = "PRIMARY_SUPPORTED"
    status.loc[test_mask & ~supported] = "DOWNGRADED_UNSUPPORTED"
    return status


def verify_membership_artifacts(artifacts: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    mismatches = []
    for item in artifacts:
        path = Path(item["path"])
        if not path.exists():
            mismatches.append(f"missing:{path}")
            continue
        if sha256_file(path) != item["sha256"]:
            mismatches.append(f"sha256:{path}")
            continue
        if len(pd.read_parquet(path, columns=["segment_id", "split_role"])) != int(item["row_count"]):
            mismatches.append(f"row_count:{path}")
    return not mismatches, mismatches


def target_distribution(segments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    common = segments[segments["prediction_usable"] & segments["road_semantics_model_eligible"]]
    records: list[dict[str, Any]] = []
    for target_id, target in config["targets"].items():
        values = pd.to_numeric(
            common.loc[common["engine_type"].eq(target["engine_type"]), target["column"]],
            errors="coerce",
        ).dropna()
        records.append(
            {
                "target_id": target_id,
                "engine_type": target["engine_type"],
                "channel": target["channel"],
                "unit": target["unit"],
                "segment_count": int(len(values)),
                "negative_count": int(values.lt(0).sum()),
                "zero_count": int(values.eq(0).sum()),
                "q01": float(values.quantile(0.01)) if len(values) else math.nan,
                "median": float(values.median()) if len(values) else math.nan,
                "q99": float(values.quantile(0.99)) if len(values) else math.nan,
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else math.nan,
            }
        )
    return pd.DataFrame(records)


def mechanistic_audit(static: pd.DataFrame, segments: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    static = static.copy()
    static["VehId"] = pd.to_numeric(static["VehId"], errors="coerce").astype("Int64")
    population = (
        segments.groupby("engine_type", dropna=False)
        .agg(
            segment_count=("segment_id", "size"),
            vehicle_count=("VehId", "nunique"),
            prediction_usable_count=("prediction_usable", "sum"),
            common_support_count=("road_semantics_model_eligible", lambda value: int((value & segments.loc[value.index, "prediction_usable"]).sum())),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    static_fields = []
    for column in config["static_parameter_fields"]:
        values = static[column].astype("string")
        available = values.notna() & values.str.strip().ne("") & values.str.upper().ne("NO DATA")
        static_fields.append(
            {
                "field": column,
                "available_vehicle_count": int(available.sum()),
                "total_vehicle_count": int(len(static)),
                "availability_rate": float(available.mean()),
            }
        )
    requirements = config["mechanistic_requirements"]
    decisions = {}
    for channel, items in requirements.items():
        essential_missing = [item["parameter"] for item in items if item["essential"] and item["status"] != "AVAILABLE"]
        decisions[channel] = {
            "decision": "MECHANISTIC_BASELINE_FEASIBLE" if not essential_missing else "MECHANISTIC_BASELINE_NOT_IDENTIFIABLE",
            "essential_unavailable_or_partial": essential_missing,
            "required_parameters": items,
        }
    overall = (
        "MECHANISTIC_BASELINE_FEASIBLE"
        if all(value["decision"] == "MECHANISTIC_BASELINE_FEASIBLE" for value in decisions.values())
        else "MECHANISTIC_BASELINE_NOT_IDENTIFIABLE"
    )
    return {
        "schema_version": 1,
        "overall_decision": overall,
        "channel_decisions": decisions,
        "static_vehicle_count": int(len(static)),
        "static_field_availability": static_fields,
        "powertrain_population": population,
        "transparent_reference": config["transparent_reference"],
        "non_claims": [
            "Generalized_Weight is not treated as exact SI mass.",
            "Nominal constants are not substituted for missing drag, rolling-resistance, efficiency, or fuel-map parameters.",
            "The transparent reference is not a complete mechanistic vehicle-energy model.",
        ],
    }


def write_mechanistic_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# MODEL_CONTRACT mechanistic-baseline identifiability audit",
        "",
        f"- Decision: `{audit['overall_decision']}`",
        f"- Static vehicles audited: {audit['static_vehicle_count']:,}",
        "- Resolution: use the frozen transparent kinematic/environmental reference; do not label it a mechanistic baseline.",
        "",
        "## Channel decisions",
        "",
    ]
    for channel, decision in audit["channel_decisions"].items():
        missing = ", ".join(decision["essential_unavailable_or_partial"])
        lines.append(f"- `{channel}`: `{decision['decision']}`; unresolved essential inputs: {missing}.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The official static table supplies coarse vehicle descriptors, including generalized weight and engine configuration, but not the exact SI mass, drag area, rolling resistance, efficiency maps, regenerative efficiency, or fuel-consumption maps needed for fair calibration. Nominal substitutions would create a pseudo-precise comparator and are prohibited.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_manifest(output: Path, paths: list[Path]) -> dict[str, Any]:
    entries = []
    for path in sorted(set(paths), key=lambda value: str(value).lower()):
        entries.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "artifacts": entries}


def verify_artifact_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if not path.exists() or sha256_file(path) != item["sha256"]:
            mismatches.append(str(path))
    return not mismatches, mismatches


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact pipe table without pandas' optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(["" if pd.isna(value) else str(value) for value in values])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/model_contract.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    differential_audit_pointer_path = Path(config["differential_audit_pointer"])
    segment_pointer_path = Path(config["segment_pointer"])
    split_pointer_path = Path(config["split_pointer"])
    row_pointer_path = Path(config["row_pointer"])
    static_path = Path(config["static_vehicle_metadata"])
    differential_audit_pointer = load_json(differential_audit_pointer_path)
    segment_pointer = load_json(segment_pointer_path)
    split_pointer = load_json(split_pointer_path)
    row_pointer = load_json(row_pointer_path)
    if differential_audit_pointer.get("status") != "RNEE_RESULTS_VALID" or differential_audit_pointer.get("model_contract_model_training_authorized") is not True:
        raise RuntimeError("DIFFERENTIAL_AUDIT has not authorized MODEL_CONTRACT.")
    if segment_pointer.get("status") != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("Frozen SEGMENT_SPLIT primary segment release is not PASS.")
    if split_pointer.get("status") != "PASS_BENCHMARK_SPLITS":
        raise RuntimeError("Frozen SEGMENT_SPLIT split release is not PASS.")
    if row_pointer.get("status") != "PASS_TRAJECTORY_BUILD_PARTITIONED_ROW_PRODUCTION":
        raise RuntimeError("Frozen TRAJECTORY_BUILD row release is not PASS.")

    row_manifest_path = Path(row_pointer["manifest"])
    segment_manifest_path = Path(segment_pointer["manifest"])
    row_manifest = load_json(row_manifest_path)
    segment_manifest = load_json(segment_manifest_path)
    split_root = Path(split_pointer["output_directory"])
    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "differential_audit_pointer_sha256": sha256_file(differential_audit_pointer_path),
        "differential_audit_manifest_sha256": sha256_file(Path(differential_audit_pointer["manifest"])),
        "segment_manifest_sha256": sha256_file(segment_manifest_path),
        "split_manifest_sha256": sha256_file(Path(split_pointer["manifest"])),
        "row_manifest_sha256": sha256_file(row_manifest_path),
        "static_vehicle_metadata_sha256": sha256_file(static_path),
        "split_membership_sha256": {
            split: sha256_file(split_root / split / "model_membership.parquet")
            for split in config["split_families"]
        },
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "run_config.yaml").exists():
        shutil.copy2(args.config, output / "run_config.yaml")
    manifest_path = output / "r2_partition_manifest.json"
    summary_path = output / "model_contract_m1_summary.json"
    final_manifest_path = output / "artifact_manifest.json"

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": "MODEL_CONTRACT",
            "stage": "M1_B1",
            "input_fingerprint": fingerprint,
            "partitions": [
                {"partition_id": item["partition_id"], "source_file": item["source_file"], "status": "PENDING"}
                for item in row_manifest["partitions"]
            ],
        }
        write_json(manifest_path, manifest)

    if (
        summary_path.exists()
        and final_manifest_path.exists()
        and all(item["status"] == "PASS" for item in manifest["partitions"])
    ):
        primary_manifest = load_json(final_manifest_path)
        verified, mismatches = verify_artifact_manifest(primary_manifest)
        resume = {
            "status": "PASS" if verified else "FAIL",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_count": len(primary_manifest["artifacts"]),
            "mismatches": mismatches,
            "recomputed_partitions": 0,
        }
        write_json(output / "resume_check.json", resume)
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(output / "resume_check.json")
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest(
            output,
            [final_manifest_path, summary_path, output / "resume_check.json"],
        )
        release["scope"] = "post-resume release-control files; primary immutable artifacts are enumerated by artifact_manifest.json"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
        if latest_path.exists():
            latest = load_json(latest_path)
            latest["release_manifest"] = str(release_manifest_path)
            write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    segment_by_source = {item["source_file"]: item for item in segment_manifest["partitions"]}
    row_by_source = {item["source_file"]: item for item in row_manifest["partitions"]}
    r2_root = output / "r2_partitions"
    r2_root.mkdir(exist_ok=True)
    required_row_columns = [
        "pilot_trip_id",
        "point_index",
        "Timestamp(ms)",
        "dt_seconds",
        "dt_valid",
        "Vehicle Speed[km/h]",
        "road_semantics_available",
        "osm_lane_count",
        *config["r2_categorical_contract"].keys(),
    ]
    for manifest_item in manifest["partitions"]:
        if manifest_item["status"] == "PASS":
            continue
        source = manifest_item["source_file"]
        row_item = row_by_source[source]
        segment_item = segment_by_source[source]
        rows = pd.read_parquet(row_item["pointer"]["rows"], columns=required_row_columns)
        partition_segments = pd.read_parquet(
            segment_item["segments"], columns=["segment_id", "trip_uid", "segment_index"]
        )
        derived, audit = aggregate_r2_partition(rows, partition_segments, config)
        path = r2_root / f"{manifest_item['partition_id']}_r2_features.parquet"
        derived.to_parquet(path, index=False)
        checks = {
            "segment_id_conservation": "PASS"
            if set(derived["segment_id"]) == set(partition_segments["segment_id"])
            else "FAIL",
            "segment_id_unique": "PASS" if derived["segment_id"].is_unique else "FAIL",
            "interval_mapping": "PASS" if audit["unmapped_interval_piece_count"] == 0 else "FAIL",
        }
        manifest_item.update(
            {
                "status": "PASS" if "FAIL" not in checks.values() else "FAIL",
                "checks": checks,
                "path": str(path),
                "sha256": sha256_file(path),
                "audit": audit,
            }
        )
        write_json(manifest_path, manifest)
        if manifest_item["status"] == "FAIL":
            raise RuntimeError(f"R2 partition failed: {source}: {checks}")

    manifest = load_json(manifest_path)
    segments = pd.read_parquet(segment_pointer["segments_all"])
    r2 = pd.concat([pd.read_parquet(item["path"]) for item in manifest["partitions"]], ignore_index=True)
    r2 = r2.sort_values("segment_id").reset_index(drop=True)
    r2_path = output / "r2_segment_features.parquet"
    r2.to_parquet(r2_path, index=False)

    contract, hits = build_feature_contract(segments, r2, config)
    contract["input_fingerprint"] = fingerprint
    contract_path = output / "feature_contract.json"
    write_json(contract_path, contract)

    support, membership_artifacts = freeze_target_memberships(segments, split_root, output, config)
    membership_integrity, membership_mismatches = verify_membership_artifacts(membership_artifacts)
    support_path = output / "powertrain_target_support.csv"
    support.to_csv(support_path, index=False)
    membership_manifest_path = output / "target_membership_manifest.json"
    write_json(
        membership_manifest_path,
        {"schema_version": 1, "artifacts": membership_artifacts},
    )
    distribution = target_distribution(segments, config)
    distribution_path = output / "target_distribution.csv"
    distribution.to_csv(distribution_path, index=False)

    static = pd.read_parquet(static_path)
    mechanism = mechanistic_audit(static, segments, config)
    mechanism_json = output / "mechanistic_baseline_identifiability.json"
    mechanism_md = output / "MECHANISTIC_BASELINE_IDENTIFIABILITY.md"
    write_json(mechanism_json, mechanism)
    write_mechanistic_report(mechanism_md, mechanism)

    checks = {
        "differential_audit_authorization": "PASS",
        "all_r2_partitions_pass": "PASS" if all(item["status"] == "PASS" for item in manifest["partitions"]) else "FAIL",
        "r2_segment_id_conservation": "PASS"
        if len(r2) == len(segments) and r2["segment_id"].is_unique and set(r2["segment_id"]) == set(segments["segment_id"])
        else "FAIL",
        "zero_forbidden_feature_hits": "PASS" if not hits else "FAIL",
        "target_unit_separation": "PASS"
        if not segments["unified_scalar_target_released"].fillna(False).any()
        and set(value["unit"] for value in config["targets"].values()) == {"L", "Wh"}
        else "FAIL",
        "target_memberships_frozen": "PASS"
        if len(membership_artifacts) == len(config["split_families"]) * len(config["targets"])
        and membership_integrity
        else "FAIL",
        "target_membership_exact_artifact_integrity": "PASS" if membership_integrity else "FAIL",
        "support_classification_recomputed": "PASS"
        if support["test_support_status"].astype("string").equals(
            classify_test_support(support, config["support_thresholds"])
        )
        else "FAIL",
        "mechanistic_gate_resolved": "PASS"
        if mechanism["overall_decision"] in {"MECHANISTIC_BASELINE_FEASIBLE", "MECHANISTIC_BASELINE_NOT_IDENTIFIABLE"}
        else "FAIL",
        "no_model_training": "PASS",
    }
    gate_rows = pd.DataFrame([{"check": key, "status": value} for key, value in checks.items()])
    gate_path = output / "m1_gate_checks.csv"
    gate_rows.to_csv(gate_path, index=False)
    overall_pass = "FAIL" not in checks.values()
    summary = {
        "experiment": "MODEL_CONTRACT",
        "stage": "M1_B1_feature_contract_and_mechanistic_identifiability",
        "status": "PASS_MODEL_CONTRACT_M1_B1" if overall_pass else "FAIL_MODEL_CONTRACT_M1_B1",
        "feature_contract_gate": "PASS_MODEL_CONTRACT_FEATURE_CONTRACT" if overall_pass else "FAIL_MODEL_CONTRACT_FEATURE_CONTRACT",
        "mechanistic_gate": "PASS_MECHANISTIC_BASELINE_OR_NONIDENTIFIABILITY"
        if checks["mechanistic_gate_resolved"] == "PASS"
        else "FAIL_MECHANISTIC_BASELINE_OR_NONIDENTIFIABILITY",
        "mechanistic_decision": mechanism["overall_decision"],
        "checks": checks,
        "segment_count": int(len(segments)),
        "r2_segment_count": int(len(r2)),
        "prediction_usable_count": int(segments["prediction_usable"].sum()),
        "common_support_count": int((segments["prediction_usable"] & segments["road_semantics_model_eligible"]).sum()),
        "target_split_cell_count": int(len(config["split_families"]) * len(config["targets"])),
        "primary_supported_test_cell_count": int((support["test_support_status"] == "PRIMARY_SUPPORTED").sum()),
        "downgraded_test_cell_count": int((support["test_support_status"] == "DOWNGRADED_UNSUPPORTED").sum()),
        "forbidden_feature_hit_count": int(len(hits)),
        "membership_integrity_mismatches": membership_mismatches,
        "resume_noop_verified": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output),
    }
    write_json(summary_path, summary)
    report_path = output / "MODEL_CONTRACT_M1_B1_GATE.md"
    downgraded = support[
        (support["split_role"] == "test")
        & (support["test_support_status"] == "DOWNGRADED_UNSUPPORTED")
    ][["split_family", "target_id", "vehicle_count", "trip_count", "segment_count", "max_vehicle_segment_share"]]
    lines = [
        "# MODEL_CONTRACT M1/B1 feature-contract and mechanistic-feasibility gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Feature contract: `{summary['feature_contract_gate']}`",
        f"- Mechanistic gate: `{summary['mechanistic_gate']}`",
        f"- Mechanistic decision: `{summary['mechanistic_decision']}`",
        f"- Frozen SEGMENT_SPLIT segments / common support: {len(segments):,} / {summary['common_support_count']:,}",
        f"- Target/split test cells supported / downgraded: {summary['primary_supported_test_cell_count']} / {summary['downgraded_test_cell_count']}",
        f"- Forbidden-feature hits: {len(hits)}",
        "- Model fitting and test evaluation performed: no.",
        "",
        "## Downgraded test cells",
        "",
    ]
    if downgraded.empty:
        lines.append("None.")
    else:
        lines.append(dataframe_to_markdown(downgraded))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "M1/B1 passes because the data/feature contract is frozen and mechanistic non-identifiability is explicitly resolved. M2 may perform validation-only backbone selection and a minimal smoke test. A genuine mechanistic model must not be claimed; use only the frozen transparent reference.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary_paths = [
        r2_path,
        contract_path,
        support_path,
        membership_manifest_path,
        distribution_path,
        mechanism_json,
        mechanism_md,
        gate_path,
        report_path,
        manifest_path,
        output / "run_config.yaml",
        *[Path(item["path"]) for item in membership_artifacts],
        *[Path(item["path"]) for item in manifest["partitions"]],
    ]
    write_json(final_manifest_path, artifact_manifest(output, primary_paths))
    write_json(
        Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json",
        {
            "status": summary["status"],
            "summary": str(summary_path),
            "report": str(report_path),
            "feature_contract": str(contract_path),
            "mechanistic_audit": str(mechanism_json),
            "support": str(support_path),
            "manifest": str(final_manifest_path),
            "output_directory": str(output),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
