#!/usr/bin/env python3
"""Build exact non-overlapping distance-based SEGMENT_SPLIT sensitivity segments.

This reuses the frozen 60 s feature aggregation helpers without modifying the
primary builder or its reproducibility fingerprint.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_segment_helpers", HERE / "35_build_segments.py")
SEG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SEG)


def distance_contributions(rows: pd.DataFrame, segment_m: float) -> pd.DataFrame:
    """Split each valid left-hold interval exactly at distance boundaries."""
    records: list[dict[str, Any]] = []
    segment_index = np.zeros(len(rows), dtype=np.int64)
    segment_keys = np.empty(len(rows), dtype=object)
    tolerance = max(1e-9, segment_m * 1e-12)
    for trip_id, index in rows.groupby("pilot_trip_id", sort=False).groups.items():
        cumulative_m = 0.0
        for row_position in index:
            if abs(cumulative_m / segment_m - round(cumulative_m / segment_m)) <= tolerance / segment_m:
                cumulative_m = round(cumulative_m / segment_m) * segment_m
            current_index = int(math.floor((cumulative_m + tolerance) / segment_m))
            segment_index[row_position] = current_index
            segment_keys[row_position] = f"{trip_id}|{current_index}"
            dt_s = float(pd.to_numeric(rows.at[row_position, "dt_seconds"], errors="coerce"))
            valid_dt = bool(rows.at[row_position, "dt_valid"]) and np.isfinite(dt_s) and dt_s > 0
            if not valid_dt:
                continue
            speed_kmh = float(pd.to_numeric(rows.at[row_position, "Vehicle Speed[km/h]"], errors="coerce"))
            speed_kmh = speed_kmh if np.isfinite(speed_kmh) and speed_kmh > 0 else 0.0
            interval_m = speed_kmh / 3.6 * dt_s
            timestamp_ms = float(rows.at[row_position, "Timestamp(ms)"])
            if interval_m <= tolerance:
                records.append({
                    "_row_pos": int(row_position),
                    "_segment_index": current_index,
                    "_segment_key": f"{trip_id}|{current_index}",
                    "weight_s": dt_s,
                    "fraction": 1.0,
                    "distance_m": 0.0,
                    "piece_start_timestamp_ms": timestamp_ms,
                    "piece_end_timestamp_ms": timestamp_ms + dt_s * 1000.0,
                })
                continue
            remaining_m = interval_m
            elapsed_fraction = 0.0
            while remaining_m > tolerance:
                current_index = int(math.floor((cumulative_m + tolerance) / segment_m))
                boundary_m = (current_index + 1) * segment_m
                capacity_m = max(0.0, boundary_m - cumulative_m)
                if capacity_m <= tolerance:
                    cumulative_m = boundary_m
                    continue
                piece_m = min(remaining_m, capacity_m)
                fraction = piece_m / interval_m
                start_ms = timestamp_ms + elapsed_fraction * dt_s * 1000.0
                elapsed_fraction += fraction
                records.append({
                    "_row_pos": int(row_position),
                    "_segment_index": current_index,
                    "_segment_key": f"{trip_id}|{current_index}",
                    "weight_s": dt_s * fraction,
                    "fraction": fraction,
                    "distance_m": piece_m,
                    "piece_start_timestamp_ms": start_ms,
                    "piece_end_timestamp_ms": timestamp_ms + elapsed_fraction * dt_s * 1000.0,
                })
                cumulative_m += piece_m
                remaining_m -= piece_m
                if abs(cumulative_m - boundary_m) <= tolerance:
                    cumulative_m = boundary_m
    rows["_segment_index"] = segment_index
    rows["_segment_key"] = segment_keys
    result = pd.DataFrame.from_records(records)
    if result.empty:
        result = pd.DataFrame(columns=["_row_pos", "_segment_index", "_segment_key", "weight_s", "fraction", "distance_m", "piece_start_timestamp_ms", "piece_end_timestamp_ms"])
    distance_contributions.last_result = result
    return result


distance_contributions.last_result = pd.DataFrame()


def aggregate_distance_partition(rows: pd.DataFrame, objects: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_m = float(config["distance_segment_m"])
    original = SEG.interval_contributions

    def adapter(local_rows: pd.DataFrame, _ignored_window_ms: int) -> pd.DataFrame:
        return distance_contributions(local_rows, segment_m)

    SEG.interval_contributions = adapter
    try:
        segments, assignments = SEG.aggregate_partition(rows, objects, config)
        contributions = distance_contributions.last_result.copy()
    finally:
        SEG.interval_contributions = original
    old_ids = segments["segment_id"].astype(str).tolist()
    starts_m = segments["segment_index"].astype(float) * segment_m
    new_ids = [SEG.stable_uid(trip, f"{start:.6f}", f"{segment_m:g}m") for trip, start in zip(segments["trip_uid"], starts_m)]
    assignments["segment_id"] = assignments["segment_id"].astype(str).map(dict(zip(old_ids, new_ids)))
    segments["segment_id"] = new_ids
    segments["segment_start_distance_m"] = starts_m
    segments["segment_distance_m"] = segments["distance_m_speed_integrated"].fillna(0.0)
    segments["segment_end_distance_m"] = segments["segment_start_distance_m"] + segments["segment_distance_m"]
    segments["distance_segment_coverage"] = segments["segment_distance_m"] / segment_m
    key = segments["trip_uid"].astype(str) + "|" + segments["segment_index"].astype(str)
    if not contributions.empty:
        timing = contributions.groupby("_segment_key", sort=False).agg(
            piece_start_timestamp_ms=("piece_start_timestamp_ms", "min"),
            piece_end_timestamp_ms=("piece_end_timestamp_ms", "max"),
        )
        segments["segment_start_timestamp_ms"] = key.map(timing["piece_start_timestamp_ms"]).fillna(segments["observed_start_timestamp_ms"]).round().astype(np.int64)
        segments["segment_end_timestamp_ms"] = key.map(timing["piece_end_timestamp_ms"]).fillna(segments["observed_end_timestamp_ms"]).round().astype(np.int64)
    segments["observed_wall_duration_s"] = (segments["segment_end_timestamp_ms"] - segments["segment_start_timestamp_ms"]) / 1000.0
    base_qa = (
        segments["row_count"].ge(int(config["minimum_row_count"]))
        & segments["observed_wall_duration_s"].ge(float(config["minimum_observed_wall_duration_s"]))
        & segments["effective_duration_s"].ge(float(config["minimum_effective_duration_s"]))
        & segments["distance_segment_coverage"].ge(float(config["minimum_distance_coverage"]))
    )
    minimum = float(config["target_minimum_coverage"])
    engine = segments["engine_type"]
    target_ok = (
        (engine.isin(["ICE", "HEV"]) & segments["fuel_target_coverage"].ge(minimum))
        | (engine.eq("PHEV") & segments["fuel_target_coverage"].ge(minimum) & segments["battery_target_coverage"].ge(minimum))
        | (engine.eq("EV") & segments["battery_target_coverage"].ge(minimum))
    )
    segments["qa_valid"] = base_qa
    segments["prediction_usable"] = base_qa & segments["trip_target_plausible"].fillna(False) & target_ok
    segments["road_semantics_model_eligible"] = base_qa & segments["road_semantics_duration_coverage"].ge(float(config["road_semantics_minimum_coverage"]))
    return segments, assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    release_pointer = SEG.load_json(Path(config["trajectory_build_release_pointer"]))
    release = SEG.load_json(Path(release_pointer["summary"]))
    row_pointer = SEG.load_json(Path(config["row_pointer"]))
    row_manifest = SEG.load_json(Path(row_pointer["manifest"]))
    context_pointer = SEG.load_json(Path(config["context_pointer"]))
    objects_path = Path(context_pointer["context_objects"])
    target_contract_path = Path(config["target_contract"])
    if release["status"] != "PASS_TRAJECTORY_BUILD_RELEASE_CANDIDATE_GATE" or release["segment_split_authorized"] is not True:
        raise RuntimeError("TRAJECTORY_BUILD has not authorized SEGMENT_SPLIT.")
    if row_pointer["status"] != "PASS_TRAJECTORY_BUILD_PARTITIONED_ROW_PRODUCTION":
        raise RuntimeError("Frozen TRAJECTORY_BUILD row production is not PASS.")
    partitions = row_manifest["partitions"]
    fingerprint = {
        "config_sha256": SEG.sha256_file(config_path),
        "script_sha256": SEG.sha256_file(Path(__file__)),
        "frozen_helper_script_sha256": SEG.sha256_file(HERE / "35_build_segments.py"),
        "row_manifest_sha256": SEG.sha256_file(Path(row_pointer["manifest"])),
        "context_objects_sha256": SEG.sha256_file(objects_path),
        "target_contract_sha256": SEG.sha256_file(target_contract_path),
        "release_summary_sha256": SEG.sha256_file(Path(release_pointer["summary"])),
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    partition_root = output / "partitions"
    partition_root.mkdir(exist_ok=True)
    manifest_path = output / "segment_manifest.json"
    summary_path = output / "segment_summary.json"
    if not (output / "run_config.yaml").exists():
        shutil.copy2(config_path, output / "run_config.yaml")
    if manifest_path.exists():
        manifest = SEG.load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": "SEGMENT_SPLIT",
            "input_fingerprint": fingerprint,
            "partitions": [{"partition_id": item["partition_id"], "source_file": item["source_file"], "status": "PENDING"} for item in partitions],
        }
        SEG.write_json(manifest_path, manifest)
    if all(item["status"] == "PASS" for item in manifest["partitions"]) and summary_path.exists():
        summary = SEG.load_json(summary_path)
        summary["resume_noop_verified"] = True
        summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started
        summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
        SEG.write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    source_map = {item["source_file"]: item for item in partitions}
    objects = pd.read_parquet(objects_path)
    peak_gb = SEG.current_working_set_gb()
    for manifest_item in manifest["partitions"]:
        if manifest_item["status"] == "PASS":
            continue
        item = source_map[manifest_item["source_file"]]
        part_dir = partition_root / manifest_item["partition_id"]
        part_dir.mkdir(exist_ok=True)
        rows = pd.read_parquet(item["pointer"]["rows"])
        segments, assignments = aggregate_distance_partition(rows, objects, config)
        segment_path = part_dir / "segments.parquet"
        assignment_path = part_dir / "row_segment_assignments.parquet"
        segments.to_parquet(segment_path, index=False)
        assignments.to_parquet(assignment_path, index=False)
        checks = {
            "row_assignment_conservation": "PASS" if len(assignments) == len(rows) and assignments["segment_id"].notna().all() else "FAIL",
            "row_id_unique": "PASS" if assignments["row_id"].is_unique else "FAIL",
            "segment_id_unique": "PASS" if segments["segment_id"].is_unique else "FAIL",
            "distance_piece_conservation": "PASS" if np.isclose(segments["segment_distance_m"].sum(), (pd.to_numeric(rows["Vehicle Speed[km/h]"], errors="coerce").fillna(0).clip(lower=0) / 3.6 * pd.to_numeric(rows["dt_seconds"], errors="coerce").fillna(0).where(rows["dt_valid"].fillna(False), 0)).sum(), rtol=1e-10, atol=1e-6) else "FAIL",
            "unified_scalar_prohibited": "PASS" if not segments["unified_scalar_target_released"].any() else "FAIL",
        }
        peak_gb = max(peak_gb, SEG.current_working_set_gb())
        manifest_item.update({
            "status": "PASS" if "FAIL" not in checks.values() else "FAIL",
            "checks": checks,
            "row_count": len(rows),
            "segment_count": len(segments),
            "qa_valid_count": int(segments["qa_valid"].sum()),
            "prediction_usable_count": int(segments["prediction_usable"].sum()),
            "road_semantics_model_eligible_count": int(segments["road_semantics_model_eligible"].sum()),
            "segments": str(segment_path),
            "row_assignments": str(assignment_path),
            "segments_sha256": SEG.sha256_file(segment_path),
            "row_assignments_sha256": SEG.sha256_file(assignment_path),
        })
        SEG.write_json(manifest_path, manifest)
        if manifest_item["status"] == "FAIL":
            raise RuntimeError(f"Distance-segment partition failed: {checks}")
        del rows, segments, assignments
    manifest = SEG.load_json(manifest_path)
    all_segments = pd.concat([pd.read_parquet(item["segments"]) for item in manifest["partitions"]], ignore_index=True)
    all_path = output / "segments_all.parquet"
    qa_path = output / "segments_qa_valid.parquet"
    usable_path = output / "segments_prediction_usable.parquet"
    all_segments.to_parquet(all_path, index=False)
    all_segments[all_segments["qa_valid"]].to_parquet(qa_path, index=False)
    all_segments[all_segments["prediction_usable"]].to_parquet(usable_path, index=False)
    duplicates = int(all_segments["segment_id"].duplicated().sum())
    ordered = all_segments.sort_values(["trip_uid", "segment_index"])
    distance_overlap = int((ordered.groupby("trip_uid")["segment_end_distance_m"].shift().gt(ordered["segment_start_distance_m"] + 1e-8)).sum())
    monotonic_violations = 0
    for layer in ("intersection", "traffic_control", "public_transport", "poi"):
        columns = [f"{layer}_unique_object_count_{scale}m" for scale in config["object_scales_m"] if f"{layer}_unique_object_count_{scale}m" in all_segments]
        if len(columns) > 1:
            monotonic_violations += int((np.diff(all_segments[columns].fillna(0).to_numpy(float), axis=1) < 0).any(axis=1).sum())
    checks = {
        "all_partitions_pass": "PASS" if all(item["status"] == "PASS" for item in manifest["partitions"]) else "FAIL",
        "global_row_assignment_conservation": "PASS" if sum(item["row_count"] for item in manifest["partitions"]) == sum(source_map[item["source_file"]]["row_count"] for item in manifest["partitions"]) else "FAIL",
        "global_segment_id_unique": "PASS" if duplicates == 0 else "FAIL",
        "non_overlapping_distance_segments": "PASS" if distance_overlap == 0 else "FAIL",
        "object_unique_count_monotonic": "PASS" if monotonic_violations == 0 else "FAIL",
        "unified_scalar_target_prohibited": "PASS" if not all_segments["unified_scalar_target_released"].any() else "FAIL",
        "memory_ceiling": "PASS" if peak_gb <= float(config["memory_ceiling_gb"]) else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    summary = {
        "experiment": "SEGMENT_SPLIT",
        "stage": config["stage"],
        "status": f"{gate}_{config['status_suffix']}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "partition_count": len(manifest["partitions"]),
        "source_row_count": sum(item["row_count"] for item in manifest["partitions"]),
        "distance_segment_m": float(config["distance_segment_m"]),
        "segment_count": len(all_segments),
        "qa_valid_count": int(all_segments["qa_valid"].sum()),
        "prediction_usable_count": int(all_segments["prediction_usable"].sum()),
        "road_semantics_model_eligible_count": int(all_segments["road_semantics_model_eligible"].sum()),
        "duplicate_segment_id_count": duplicates,
        "distance_overlap_count": distance_overlap,
        "object_unique_count_monotonic_violation_count": monotonic_violations,
        "peak_working_set_gb": peak_gb,
        "resume_noop_verified": False,
        "target_contract_status": "PASS_OPERATIONAL_TARGET_ONLY",
        "unified_scalar_target_released": False,
        "segment_manifest": str(manifest_path),
        "segments_all": str(all_path),
        "segments_qa_valid": str(qa_path),
        "segments_prediction_usable": str(usable_path),
        "output_directory": str(output),
    }
    SEG.write_json(summary_path, summary)
    report = output / "SEGMENT_SPLIT_DISTANCE_SEGMENT_GATE.md"
    report.write_text("# SEGMENT_SPLIT distance-segment sensitivity gate\n\n" + "\n".join([
        f"- Status: `{summary['status']}`",
        f"- Source rows / segments: {summary['source_row_count']:,} / {summary['segment_count']:,}",
        f"- QA-valid / prediction-usable: {summary['qa_valid_count']:,} / {summary['prediction_usable_count']:,}",
        f"- Duplicate IDs / distance overlaps: {duplicates} / {distance_overlap}",
        f"- Peak working set: {peak_gb:.3f} GB",
        "- Targets remain separate powertrain-aware fuel-volume and battery-terminal channels.",
    ]) + "\n", encoding="utf-8")
    latest = {
        "status": summary["status"],
        "summary": str(summary_path),
        "report": str(report),
        "manifest": str(manifest_path),
        "segments_all": str(all_path),
        "segments_qa_valid": str(qa_path),
        "segments_prediction_usable": str(usable_path),
        "output_directory": str(output),
    }
    SEG.write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
