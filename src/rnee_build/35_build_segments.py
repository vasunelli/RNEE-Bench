#!/usr/bin/env python3
"""Build leakage-safe, non-overlapping SEGMENT_SPLIT benchmark segments from TRAJECTORY_BUILD rows."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from shapely import points as shapely_points
from shapely.strtree import STRtree


HERE = Path(__file__).resolve().parent
PROFILE_SPEC = importlib.util.spec_from_file_location("rnee_profile", HERE / "17_profile_chunked_context.py")
PROFILE = importlib.util.module_from_spec(PROFILE_SPEC)
assert PROFILE_SPEC.loader is not None
PROFILE_SPEC.loader.exec_module(PROFILE)


def current_working_set_gb() -> float:
    memory = PROFILE.process_tree_memory(os.getpid())
    return float(memory["working_set_bytes"] / 1024**3) if memory else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_uid(*parts: Any, length: int = 24) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def interval_contributions(rows: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    """Split each valid left-hold interval exactly at segment boundaries."""
    origin = rows.groupby("pilot_trip_id", sort=False)["Timestamp(ms)"].transform("min").to_numpy(np.int64)
    timestamp = rows["Timestamp(ms)"].to_numpy(np.int64)
    segment_index = np.floor_divide(timestamp - origin, window_ms).astype(np.int64)
    rows["_segment_index"] = segment_index
    rows["_segment_key"] = rows["pilot_trip_id"].astype(str) + "|" + pd.Series(segment_index, index=rows.index).astype(str)
    dt_ms = pd.to_numeric(rows["dt_seconds"], errors="coerce").fillna(0).to_numpy(float) * 1000.0
    valid = rows["dt_valid"].fillna(False).to_numpy(bool) & (dt_ms > 0)
    offset = (timestamp - origin) - segment_index * window_ms
    first_ms = np.where(valid, np.minimum(dt_ms, window_ms - offset), 0.0)
    second_ms = np.where(valid, np.maximum(0.0, dt_ms - first_ms), 0.0)
    row_pos = np.arange(len(rows), dtype=np.int64)
    pieces = []
    for increment, duration_ms in ((0, first_ms), (1, second_ms)):
        mask = duration_ms > 0
        piece = pd.DataFrame({
            "_row_pos": row_pos[mask],
            "_segment_index": segment_index[mask] + increment,
            "weight_s": duration_ms[mask] / 1000.0,
            "fraction": duration_ms[mask] / dt_ms[mask],
        })
        trip = rows["pilot_trip_id"].astype(str).to_numpy()[mask]
        piece["_segment_key"] = trip + "|" + piece["_segment_index"].astype(str).to_numpy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["_row_pos", "_segment_index", "weight_s", "fraction", "_segment_key"])


def weighted_numeric_features(rows: pd.DataFrame, contributions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    expanded = rows.iloc[contributions["_row_pos"].to_numpy(np.int64)][columns].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    weights = contributions["weight_s"].to_numpy(float)
    keys = contributions["_segment_key"].to_numpy(str)
    output = pd.DataFrame(index=pd.Index(pd.unique(keys), name="_segment_key"))
    for column in columns:
        values = expanded[column].to_numpy(float)
        valid = np.isfinite(values) & (weights > 0)
        frame = pd.DataFrame({"key": keys[valid], "weighted": values[valid] * weights[valid], "weighted2": values[valid] ** 2 * weights[valid], "weight": weights[valid]})
        grouped = frame.groupby("key", sort=False).sum()
        mean = grouped["weighted"] / grouped["weight"]
        variance = (grouped["weighted2"] / grouped["weight"] - mean**2).clip(lower=0)
        safe = column.replace("[", "_").replace("]", "").replace("/", "_").replace("%", "pct").replace(" ", "_").replace("(", "").replace(")", "").lower()
        output[f"{safe}_time_weighted_mean"] = mean
        output[f"{safe}_time_weighted_std"] = np.sqrt(variance)
    return output.reset_index()


def distance_mix(rows: pd.DataFrame, contributions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    positions = contributions["_row_pos"].to_numpy(np.int64)
    speed = pd.to_numeric(rows.iloc[positions]["Vehicle Speed[km/h]"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    distance_m = speed / 3.6 * contributions["weight_s"].to_numpy(float)
    base = pd.DataFrame({"_segment_key": contributions["_segment_key"].to_numpy(str), "distance_m": distance_m})
    base["road_available"] = rows.iloc[positions]["road_semantics_available"].fillna(False).to_numpy(bool)
    base["road_class"] = rows.iloc[positions]["functional_road_class"].astype("string").to_numpy()
    base["speed_limit"] = pd.to_numeric(rows.iloc[positions]["speed_limit_kmh_used"], errors="coerce").to_numpy(float)
    base["speed_limit_inferred"] = rows.iloc[positions]["speed_limit_is_inferred"].fillna(False).to_numpy(bool)
    base["speed_limit_provenance"] = rows.iloc[positions]["speed_limit_provenance"].astype("string").to_numpy()
    total = base.groupby("_segment_key", sort=False)["distance_m"].sum().rename("distance_m_speed_integrated")
    available = base[base["road_available"]].groupby("_segment_key", sort=False)["distance_m"].sum().rename("road_semantics_distance_m")
    result = pd.concat([total, available], axis=1).fillna(0)
    result["road_semantics_distance_coverage"] = np.where(result["distance_m_speed_integrated"] > 0, result["road_semantics_distance_m"] / result["distance_m_speed_integrated"], np.nan)
    class_rows = base[base["road_available"] & base["road_class"].notna() & base["distance_m"].gt(0)]
    if not class_rows.empty:
        mix = class_rows.groupby(["_segment_key", "road_class"], sort=False)["distance_m"].sum().unstack(fill_value=0)
        denominator = mix.sum(axis=1)
        shares = mix.div(denominator, axis=0)
        result["functional_road_class_dominant"] = mix.idxmax(axis=1)
        result["functional_road_class_distance_m"] = denominator
        result["functional_road_class_entropy"] = -(shares.where(shares > 0) * np.log(shares.where(shares > 0))).sum(axis=1)
        for road_class in config["functional_road_classes"]:
            result[f"functional_road_class_share_{road_class}"] = shares[road_class] if road_class in shares else 0.0
    else:
        result["functional_road_class_dominant"] = pd.NA
        result["functional_road_class_distance_m"] = 0.0
        result["functional_road_class_entropy"] = np.nan
    limit_rows = base[base["road_available"] & np.isfinite(base["speed_limit"]) & base["distance_m"].gt(0)].copy()
    if not limit_rows.empty:
        limit_rows["weighted"] = limit_rows["speed_limit"] * limit_rows["distance_m"]
        grouped = limit_rows.groupby("_segment_key", sort=False)
        limit_distance = grouped["distance_m"].sum()
        result["speed_limit_kmh_distance_weighted_mean"] = grouped["weighted"].sum() / limit_distance
        result["speed_limit_distance_coverage"] = limit_distance / result["distance_m_speed_integrated"].replace(0, np.nan)
        inferred_distance = limit_rows[limit_rows["speed_limit_inferred"]].groupby("_segment_key")["distance_m"].sum()
        result["speed_limit_inferred_distance_share"] = inferred_distance / limit_distance
        for provenance in config["speed_limit_provenance"]:
            numerator = limit_rows[limit_rows["speed_limit_provenance"].eq(provenance)].groupby("_segment_key")["distance_m"].sum()
            result[f"speed_limit_provenance_share_{provenance}"] = numerator / limit_distance
    return result.reset_index()


def transition_counts(rows: pd.DataFrame) -> pd.Series:
    work = rows[["_segment_key", "functional_road_class"]].copy()
    work["previous"] = work.groupby("_segment_key", sort=False)["functional_road_class"].shift()
    transition = work["functional_road_class"].notna() & work["previous"].notna() & work["functional_road_class"].ne(work["previous"])
    return transition.groupby(work["_segment_key"]).sum().astype(int).rename("functional_road_class_transition_count")


def build_object_trees(objects: pd.DataFrame) -> dict[str, tuple[STRtree, pd.DataFrame]]:
    trees = {}
    for snapshot, group in objects.groupby("osm_snapshot_id", sort=False):
        frame = group.reset_index(drop=True)
        trees[str(snapshot)] = (STRtree(shapely_points(frame["x_m"].to_numpy(float), frame["y_m"].to_numpy(float))), frame)
    return trees


def object_segment_features(rows: pd.DataFrame, contributions: pd.DataFrame, trees: dict[str, tuple[STRtree, pd.DataFrame]], scales: list[int]) -> pd.DataFrame:
    contribution_lookup = contributions[["_row_pos", "_segment_key", "weight_s"]].copy()
    positions = contribution_lookup["_row_pos"].to_numpy(np.int64)
    speed = pd.to_numeric(rows.iloc[positions]["Vehicle Speed[km/h]"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    contribution_lookup["distance_m"] = speed / 3.6 * contribution_lookup["weight_s"].to_numpy(float)
    records = []
    valid = rows["road_semantics_available"].fillna(False) & rows["matched_x_m"].notna() & rows["matched_y_m"].notna()
    for trip_id, trip in rows[valid].groupby("pilot_trip_id", sort=False):
        for snapshot, point_frame in trip.groupby("osm_snapshot_id", sort=False):
            if str(snapshot) not in trees:
                continue
            tree, object_frame = trees[str(snapshot)]
            point_frame = point_frame.copy()
            geometries = shapely_points(point_frame["matched_x_m"].to_numpy(float), point_frame["matched_y_m"].to_numpy(float))
            for scale in scales:
                pairs = tree.query(geometries, predicate="dwithin", distance=float(scale))
                if pairs.shape[1] == 0:
                    continue
                pair = pd.DataFrame({"point_local": pairs[0], "object_local": pairs[1]})
                pair["_row_pos"] = point_frame.index.to_numpy(np.int64)[pair["point_local"].to_numpy(np.int64)]
                pair["_segment_key"] = rows.iloc[pair["_row_pos"]]["_segment_key"].to_numpy(str)
                pair["point_index"] = rows.iloc[pair["_row_pos"]]["point_index"].to_numpy(np.int64)
                pair["object_uid"] = object_frame.iloc[pair["object_local"]]["object_uid"].to_numpy(str)
                pair["object_layer"] = object_frame.iloc[pair["object_local"]]["object_layer"].to_numpy(str)
                pair["primary_type"] = object_frame.iloc[pair["object_local"]]["primary_type"].to_numpy(str)
                pair = pair.drop_duplicates(["_segment_key", "point_index", "object_uid", "object_layer"])
                pair = pair.sort_values(["_segment_key", "object_layer", "object_uid", "point_index"])
                group_columns = ["_segment_key", "object_layer", "object_uid"]
                contiguous = pair.groupby(group_columns, sort=False)["point_index"].diff().eq(1)
                pair["entry"] = (~contiguous).astype(int)
                unique = pair.groupby(["_segment_key", "object_layer"], sort=False).agg(unique_object_count=("object_uid", "nunique"), pass_event_count=("entry", "sum"))
                exposure = pair[["_row_pos", "_segment_key", "object_layer", "object_uid"]].merge(contribution_lookup, on=["_row_pos", "_segment_key"], how="left")
                exposure = exposure.groupby(["_segment_key", "object_layer"], sort=False).agg(exposure_time_s=("weight_s", "sum"), exposure_distance_m=("distance_m", "sum"))
                combined = unique.join(exposure, how="left").reset_index()
                poi = pair[pair["object_layer"].eq("poi")].groupby("_segment_key")["primary_type"].nunique()
                for key, layer, unique_count, event_count, exposure_time, exposure_distance in combined.itertuples(index=False, name=None):
                    records.append({"_segment_key": key, "feature": f"{layer}_unique_object_count_{scale}m", "value": int(unique_count)})
                    records.append({"_segment_key": key, "feature": f"{layer}_pass_event_count_{scale}m", "value": int(event_count)})
                    records.append({"_segment_key": key, "feature": f"{layer}_exposure_time_s_{scale}m", "value": float(exposure_time)})
                    records.append({"_segment_key": key, "feature": f"{layer}_exposure_distance_m_{scale}m", "value": float(exposure_distance)})
                for key, value in poi.items():
                    records.append({"_segment_key": key, "feature": f"poi_primary_type_diversity_{scale}m", "value": int(value)})
    if not records:
        return pd.DataFrame(columns=["_segment_key"])
    return pd.DataFrame(records).pivot_table(index="_segment_key", columns="feature", values="value", aggfunc="sum", fill_value=0).reset_index().rename_axis(columns=None)


def aggregate_partition(rows: pd.DataFrame, objects: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.sort_values(["pilot_trip_id", "Timestamp(ms)", "point_index"], kind="stable").reset_index(drop=True)
    window_ms = int(float(config["window_seconds"]) * 1000)
    contributions = interval_contributions(rows, window_ms)
    keys = ["_segment_key"]
    grouped = rows.groupby(keys, sort=False)
    segments = grouped.agg(
        source_file=("source_file", "first"), trip_uid=("pilot_trip_id", "first"), VehId=("VehId", "first"), Trip=("Trip", "first"),
        engine_type=("EngineType_official", "first"), segment_index=("_segment_index", "first"), row_count=("row_id", "size"),
        observed_start_timestamp_ms=("Timestamp(ms)", "min"), observed_end_timestamp_ms=("Timestamp(ms)", "max"),
        segment_start_daynum=("DayNum", "min"), trip_target_plausible=("trip_target_plausible", "all"),
        matched_x_m_centroid=("matched_x_m", "mean"), matched_y_m_centroid=("matched_y_m", "mean"),
    )
    segments["observed_wall_duration_s"] = (segments["observed_end_timestamp_ms"] - segments["observed_start_timestamp_ms"]) / 1000.0
    origin_by_trip = rows.groupby("pilot_trip_id", sort=False)["Timestamp(ms)"].min()
    segments["segment_start_timestamp_ms"] = [int(origin_by_trip[trip]) + int(index) * window_ms for trip, index in zip(segments["trip_uid"], segments["segment_index"])]
    segments["segment_end_timestamp_ms"] = segments["segment_start_timestamp_ms"] + window_ms
    segments["segment_id"] = [stable_uid(trip, start, f"{int(config['window_seconds'])}s") for trip, start in zip(segments["trip_uid"], segments["segment_start_timestamp_ms"])]
    effective = contributions.groupby("_segment_key", sort=False)["weight_s"].sum().rename("effective_duration_s")
    segments = segments.join(effective, how="left").fillna({"effective_duration_s": 0.0})
    positions = contributions["_row_pos"].to_numpy(np.int64)
    target = pd.DataFrame({"_segment_key": contributions["_segment_key"].to_numpy(str), "weight_s": contributions["weight_s"].to_numpy(float), "fraction": contributions["fraction"].to_numpy(float)})
    target["fuel"] = pd.to_numeric(rows.iloc[positions]["fuel_volume_L"], errors="coerce").to_numpy(float)
    target["battery"] = pd.to_numeric(rows.iloc[positions]["battery_terminal_net_Wh"], errors="coerce").to_numpy(float)
    target["fuel_source"] = rows.iloc[positions]["fuel_source"].astype("string").to_numpy()
    target["fuel_value"] = target["fuel"] * target["fraction"]
    target["battery_value"] = target["battery"] * target["fraction"]
    target["fuel_valid_s"] = np.where(np.isfinite(target["fuel"]), target["weight_s"], 0.0)
    target["battery_valid_s"] = np.where(np.isfinite(target["battery"]), target["weight_s"], 0.0)
    target["fuel_direct_s"] = np.where(np.isfinite(target["fuel"]) & target["fuel_source"].eq("direct"), target["weight_s"], 0.0)
    target["fuel_maf_s"] = np.where(np.isfinite(target["fuel"]) & target["fuel_source"].eq("maf"), target["weight_s"], 0.0)
    target_agg = target.groupby("_segment_key", sort=False).agg(fuel_volume_L=("fuel_value", "sum"), battery_terminal_net_Wh=("battery_value", "sum"), fuel_valid_duration_s=("fuel_valid_s", "sum"), battery_valid_duration_s=("battery_valid_s", "sum"), fuel_direct_duration_s=("fuel_direct_s", "sum"), fuel_maf_duration_s=("fuel_maf_s", "sum"))
    segments = segments.join(target_agg, how="left")
    segments["fuel_target_coverage"] = segments["fuel_valid_duration_s"] / segments["effective_duration_s"].replace(0, np.nan)
    segments["battery_target_coverage"] = segments["battery_valid_duration_s"] / segments["effective_duration_s"].replace(0, np.nan)
    segments["fuel_direct_duration_share"] = segments["fuel_direct_duration_s"] / segments["fuel_valid_duration_s"].replace(0, np.nan)
    segments["fuel_maf_duration_share"] = segments["fuel_maf_duration_s"] / segments["fuel_valid_duration_s"].replace(0, np.nan)
    numeric_columns = [column for column in config["time_weighted_numeric_columns"] if column in rows]
    numeric = weighted_numeric_features(rows, contributions, numeric_columns)
    mix = distance_mix(rows, contributions, config)
    objects_features = object_segment_features(rows, contributions, build_object_trees(objects), config["object_scales_m"])
    segments = segments.reset_index().merge(numeric, on="_segment_key", how="left").merge(mix, on="_segment_key", how="left").merge(objects_features, on="_segment_key", how="left")
    object_columns = [column for column in segments if any(token in column for token in ("_unique_object_count_", "_pass_event_count_", "_exposure_time_s_", "_exposure_distance_m_", "poi_primary_type_diversity_"))]
    segments[object_columns] = segments[object_columns].fillna(0)
    segments = segments.merge(transition_counts(rows).reset_index(), on="_segment_key", how="left")
    speed = pd.to_numeric(rows["Vehicle Speed[km/h]"], errors="coerce")
    interval_rows = rows.iloc[positions].copy()
    ratio = pd.DataFrame({"_segment_key": contributions["_segment_key"].to_numpy(str), "weight_s": contributions["weight_s"].to_numpy(float), "idle": speed.iloc[positions].fillna(0).lt(float(config["idle_speed_kmh"])).to_numpy(), "low": speed.iloc[positions].fillna(0).lt(float(config["low_speed_kmh"])).to_numpy(), "road_available": interval_rows["road_semantics_available"].fillna(False).to_numpy(bool)})
    ratio["idle_s"] = ratio["weight_s"] * ratio["idle"]; ratio["low_s"] = ratio["weight_s"] * ratio["low"]; ratio["road_s"] = ratio["weight_s"] * ratio["road_available"]
    ratio = ratio.groupby("_segment_key", sort=False).agg(idle_duration_s=("idle_s", "sum"), low_speed_duration_s=("low_s", "sum"), road_semantics_duration_s=("road_s", "sum"))
    segments = segments.merge(ratio.reset_index(), on="_segment_key", how="left")
    segments["idle_ratio"] = segments["idle_duration_s"] / segments["effective_duration_s"].replace(0, np.nan)
    segments["low_speed_ratio"] = segments["low_speed_duration_s"] / segments["effective_duration_s"].replace(0, np.nan)
    segments["road_semantics_duration_coverage"] = segments["road_semantics_duration_s"] / segments["effective_duration_s"].replace(0, np.nan)
    base_date = datetime.fromisoformat(config["daynum_origin"])
    dates = [base_date + timedelta(days=float(day)) for day in segments["segment_start_daynum"]]
    segments["actual_start_date"] = [value.date().isoformat() for value in dates]
    segments["actual_month"] = [value.strftime("%Y-%m") for value in dates]
    segments["spatial_grid_x"] = np.floor(segments["matched_x_m_centroid"] / float(config["spatial_grid_m"])).astype("Int64")
    segments["spatial_grid_y"] = np.floor(segments["matched_y_m_centroid"] / float(config["spatial_grid_m"])).astype("Int64")
    segments["spatial_block_id"] = segments["spatial_grid_x"].astype("string") + "_" + segments["spatial_grid_y"].astype("string")
    minimum = float(config["target_minimum_coverage"])
    engine = segments["engine_type"]
    target_ok = ((engine.isin(["ICE", "HEV"]) & segments["fuel_target_coverage"].ge(minimum)) | (engine.eq("PHEV") & segments["fuel_target_coverage"].ge(minimum) & segments["battery_target_coverage"].ge(minimum)) | (engine.eq("EV") & segments["battery_target_coverage"].ge(minimum)))
    segments["qa_valid"] = segments["row_count"].ge(int(config["minimum_row_count"])) & segments["observed_wall_duration_s"].ge(float(config["minimum_observed_wall_duration_s"])) & segments["effective_duration_s"].ge(float(config["minimum_effective_duration_s"]))
    segments["prediction_usable"] = segments["qa_valid"] & segments["trip_target_plausible"].fillna(False) & target_ok
    segments["road_semantics_model_eligible"] = segments["qa_valid"] & segments["road_semantics_duration_coverage"].ge(float(config["road_semantics_minimum_coverage"]))
    segments["target_contract_status"] = "PASS_OPERATIONAL_TARGET_ONLY"
    segments["unified_scalar_target_released"] = False
    segments = segments.drop(columns=["_segment_key"])
    assignments = rows[["row_id", "source_file", "pilot_trip_id", "VehId", "Trip", "Timestamp(ms)", "_segment_key"]].copy()
    key_to_id = dict(zip((segments["trip_uid"].astype(str) + "|" + segments["segment_index"].astype(str)), segments["segment_id"]))
    assignments["segment_id"] = assignments["_segment_key"].map(key_to_id)
    assignments = assignments.drop(columns=["_segment_key"])
    return segments, assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/segment_split_segments_60s_pilot.yaml"))
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter(); args = parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    release_pointer = load_json(Path(config["trajectory_build_release_pointer"])); release = load_json(Path(release_pointer["summary"]))
    row_pointer = load_json(Path(config["row_pointer"])); row_manifest = load_json(Path(row_pointer["manifest"])); context_pointer = load_json(Path(config["context_pointer"])); objects_path = Path(context_pointer["context_objects"])
    target_contract_path = Path(config["target_contract"])
    if release["status"] != "PASS_TRAJECTORY_BUILD_RELEASE_CANDIDATE_GATE" or release["segment_split_authorized"] is not True: raise RuntimeError("TRAJECTORY_BUILD has not authorized SEGMENT_SPLIT.")
    if row_pointer["status"] != "PASS_TRAJECTORY_BUILD_PARTITIONED_ROW_PRODUCTION": raise RuntimeError("Frozen TRAJECTORY_BUILD row production is not PASS.")
    requested = config.get("selected_source_files"); partitions = row_manifest["partitions"]
    if requested:
        by_source = {item["source_file"]: item for item in partitions}; missing = sorted(set(requested) - set(by_source))
        if missing: raise RuntimeError(f"Selected sources absent from TRAJECTORY_BUILD: {missing}")
        partitions = [by_source[name] for name in requested]
    fingerprint = {"config_sha256": sha256_file(args.config), "script_sha256": sha256_file(Path(__file__)), "row_manifest_sha256": sha256_file(Path(row_pointer["manifest"])), "context_objects_sha256": sha256_file(objects_path), "target_contract_sha256": sha256_file(target_contract_path), "release_summary_sha256": sha256_file(Path(release_pointer["summary"]))}
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True); partition_root = output / "partitions"; partition_root.mkdir(exist_ok=True); manifest_path = output / "segment_manifest.json"; summary_path = output / "segment_summary.json"
    if not (output / "run_config.yaml").exists(): shutil.copy2(args.config, output / "run_config.yaml")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint: raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest = {"schema_version": 1, "experiment": config["experiment"], "input_fingerprint": fingerprint, "partitions": [{"partition_id": item["partition_id"], "source_file": item["source_file"], "status": "PENDING"} for item in partitions]}; write_json(manifest_path, manifest)
    if all(item["status"] == "PASS" for item in manifest["partitions"]) and summary_path.exists():
        summary = load_json(summary_path); summary["resume_noop_verified"] = True; summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started; summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat(); write_json(summary_path, summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    source_map = {item["source_file"]: item for item in partitions}; objects = pd.read_parquet(objects_path); peak_gb = current_working_set_gb()
    for manifest_item in manifest["partitions"]:
        if manifest_item["status"] == "PASS": continue
        item = source_map[manifest_item["source_file"]]; part_dir = partition_root / manifest_item["partition_id"]; part_dir.mkdir(exist_ok=True)
        rows = pd.read_parquet(item["pointer"]["rows"]); segments, assignments = aggregate_partition(rows, objects, config)
        segment_path = part_dir / "segments.parquet"; assignment_path = part_dir / "row_segment_assignments.parquet"; segments.to_parquet(segment_path, index=False); assignments.to_parquet(assignment_path, index=False)
        checks = {"row_assignment_conservation": "PASS" if len(assignments) == len(rows) and assignments["segment_id"].notna().all() else "FAIL", "row_id_unique": "PASS" if assignments["row_id"].is_unique else "FAIL", "segment_id_unique": "PASS" if segments["segment_id"].is_unique else "FAIL", "unified_scalar_prohibited": "PASS" if not segments["unified_scalar_target_released"].any() else "FAIL"}
        peak_gb = max(peak_gb, current_working_set_gb()); manifest_item.update({"status": "PASS" if "FAIL" not in checks.values() else "FAIL", "checks": checks, "row_count": len(rows), "segment_count": len(segments), "qa_valid_count": int(segments["qa_valid"].sum()), "prediction_usable_count": int(segments["prediction_usable"].sum()), "road_semantics_model_eligible_count": int(segments["road_semantics_model_eligible"].sum()), "segments": str(segment_path), "row_assignments": str(assignment_path), "segments_sha256": sha256_file(segment_path), "row_assignments_sha256": sha256_file(assignment_path)}); write_json(manifest_path, manifest)
        if manifest_item["status"] == "FAIL": raise RuntimeError(f"Segment partition failed: {checks}")
        del rows, segments, assignments
    manifest = load_json(manifest_path); all_segments = pd.concat([pd.read_parquet(item["segments"]) for item in manifest["partitions"]], ignore_index=True)
    all_path = output / "segments_all.parquet"; qa_path = output / "segments_qa_valid.parquet"; usable_path = output / "segments_prediction_usable.parquet"; all_segments.to_parquet(all_path, index=False); all_segments[all_segments["qa_valid"]].to_parquet(qa_path, index=False); all_segments[all_segments["prediction_usable"]].to_parquet(usable_path, index=False)
    duplicate_segments = int(all_segments["segment_id"].duplicated().sum()); ordered = all_segments.sort_values(["trip_uid", "segment_start_timestamp_ms"]); overlap = int((ordered.groupby("trip_uid")["segment_end_timestamp_ms"].shift().gt(ordered["segment_start_timestamp_ms"])).sum())
    monotonic_violations = 0
    for layer in ("intersection", "traffic_control", "public_transport", "poi"):
        columns = [f"{layer}_unique_object_count_{scale}m" for scale in config["object_scales_m"] if f"{layer}_unique_object_count_{scale}m" in all_segments]
        if len(columns) > 1: monotonic_violations += int((np.diff(all_segments[columns].fillna(0).to_numpy(float), axis=1) < 0).any(axis=1).sum())
    checks = {"all_partitions_pass": "PASS" if all(item["status"] == "PASS" for item in manifest["partitions"]) else "FAIL", "global_row_assignment_conservation": "PASS" if sum(item["row_count"] for item in manifest["partitions"]) == sum(source_map[item["source_file"]]["row_count"] for item in manifest["partitions"]) else "FAIL", "global_segment_id_unique": "PASS" if duplicate_segments == 0 else "FAIL", "non_overlapping_segments": "PASS" if overlap == 0 else "FAIL", "object_unique_count_monotonic": "PASS" if monotonic_violations == 0 else "FAIL", "unified_scalar_target_prohibited": "PASS" if not all_segments["unified_scalar_target_released"].any() else "FAIL", "memory_ceiling": "PASS" if peak_gb <= float(config["memory_ceiling_gb"]) else "FAIL"}
    dominant_counts = {("<missing>" if pd.isna(key) else str(key)): int(value) for key, value in all_segments["functional_road_class_dominant"].value_counts(dropna=False).items()}
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"; summary = {"experiment": config["experiment"], "stage": config["stage"], "status": f"{gate}_{config['status_suffix']}", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "partition_count": len(manifest["partitions"]), "source_row_count": sum(item["row_count"] for item in manifest["partitions"]), "segment_count": len(all_segments), "qa_valid_count": int(all_segments["qa_valid"].sum()), "prediction_usable_count": int(all_segments["prediction_usable"].sum()), "road_semantics_model_eligible_count": int(all_segments["road_semantics_model_eligible"].sum()), "engine_type_segment_counts": {str(key): int(value) for key, value in all_segments["engine_type"].value_counts().items()}, "engine_type_prediction_usable_counts": {str(key): int(value) for key, value in all_segments[all_segments["prediction_usable"]]["engine_type"].value_counts().items()}, "functional_road_class_dominant_counts": dominant_counts, "duplicate_segment_id_count": duplicate_segments, "overlap_count": overlap, "object_unique_count_monotonic_violation_count": monotonic_violations, "peak_working_set_gb": peak_gb, "resume_noop_verified": False, "target_contract_status": "PASS_OPERATIONAL_TARGET_ONLY", "unified_scalar_target_released": False, "segment_manifest": str(manifest_path), "segments_all": str(all_path), "segments_qa_valid": str(qa_path), "segments_prediction_usable": str(usable_path), "output_directory": str(output)}; write_json(summary_path, summary)
    report = output / "SEGMENT_SPLIT_SEGMENT_GATE.md"; report.write_text("# SEGMENT_SPLIT non-overlapping segment gate\n\n" + "\n".join([f"- Status: `{summary['status']}`", f"- Source rows / segments: {summary['source_row_count']:,} / {summary['segment_count']:,}", f"- QA-valid / prediction-usable: {summary['qa_valid_count']:,} / {summary['prediction_usable_count']:,}", f"- Duplicate IDs / overlaps: {duplicate_segments} / {overlap}", f"- Peak working set: {peak_gb:.3f} GB", "- Targets remain separate powertrain-aware fuel-volume and battery-terminal channels; no unified scalar is released."]) + "\n", encoding="utf-8")
    latest = {"status": summary["status"], "summary": str(summary_path), "report": str(report), "manifest": str(manifest_path), "segments_all": str(all_path), "segments_qa_valid": str(qa_path), "segments_prediction_usable": str(usable_path), "output_directory": str(output)}; write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", latest); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
