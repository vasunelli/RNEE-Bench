#!/usr/bin/env python3
"""Build context semantics in complete-trip chunks with deterministic resume."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_context", HERE / "12_build_context_semantics.py")
CONTEXT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONTEXT)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    CONTEXT.write_json(temporary, value)
    temporary.replace(path)


def plan_trip_chunks(points: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        raise ValueError("max_rows_per_chunk must be positive.")
    sizes = (
        points.groupby(["source_file", "pilot_trip_id", "osm_snapshot_id"], sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_rows = 0
    for record in sizes.to_dict(orient="records"):
        row_count = int(record["row_count"])
        if current and current_rows + row_count > max_rows:
            chunks.append({"trips": current, "row_count": current_rows})
            current = []
            current_rows = 0
        current.append(record)
        current_rows += row_count
        if row_count > max_rows:
            chunks.append({"trips": current, "row_count": current_rows})
            current = []
            current_rows = 0
    if current:
        chunks.append({"trips": current, "row_count": current_rows})
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{index:04d}"
        chunk["trip_count"] = len(chunk["trips"])
        chunk["source_files"] = sorted({trip["source_file"] for trip in chunk["trips"]})
        chunk["snapshot_ids"] = sorted({trip["osm_snapshot_id"] for trip in chunk["trips"]})
    return chunks


def plan_partitioned_trip_chunks(
    map_manifest: dict[str, Any], selected_profile: str, max_rows: int
) -> list[dict[str, Any]]:
    """Plan complete-trip chunks without materializing all TRAJECTORY_BUILD points."""
    chunks: list[dict[str, Any]] = []
    for partition in map_manifest["partitions"]:
        path = Path(partition["pointer"]["matched_points"])
        points = pd.read_parquet(
            path,
            columns=["profile", "source_file", "pilot_trip_id", "osm_snapshot_id", "matched_latitude", "matched_longitude"],
        )
        points = points[
            points["profile"].eq(selected_profile)
            & points["matched_latitude"].notna()
            & points["matched_longitude"].notna()
        ]
        local = plan_trip_chunks(points, max_rows)
        for chunk in local:
            chunk["partition_id"] = partition["partition_id"]
            chunk["matched_points"] = str(path)
            chunks.append(chunk)
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{index:04d}"
    return chunks


def chunk_points(points: pd.DataFrame, chunk: dict[str, Any]) -> pd.DataFrame:
    trip_ids = {trip["pilot_trip_id"] for trip in chunk["trips"]}
    result = points[points["pilot_trip_id"].isin(trip_ids)].copy()
    if len(result) != int(chunk["row_count"]):
        raise RuntimeError(f"{chunk['chunk_id']} row plan drift: {len(result)} != {chunk['row_count']}")
    if result["pilot_trip_id"].nunique() != int(chunk["trip_count"]):
        raise RuntimeError(f"{chunk['chunk_id']} trip plan drift.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/trajectory_build_context_chunk_dry_run.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    main_started = time.perf_counter()
    args = parse_args()
    config = CONTEXT.load_config(args.config)
    raw_pointer = CONTEXT.load_json(Path(config["map_match_pointer"])) if config.get("map_match_pointer") else None
    map_partition_pointer = CONTEXT.load_json(Path(config["map_partition_pointer"])) if config.get("map_partition_pointer") else None
    edge_pointer = CONTEXT.load_json(Path(config["edge_semantics_pointer"]))
    map_matching = CONTEXT.load_json(Path(config["map_matching_close_decision"]))
    if not (str(edge_pointer["status"]).startswith("PASS_") and "EDGE" in str(edge_pointer["status"])):
        raise RuntimeError("Edge semantics have not passed.")
    transformer = Transformer.from_crs(config["storage_crs"], config["metric_crs"], always_xy=True)
    map_manifest = None
    if map_partition_pointer is not None:
        map_manifest = CONTEXT.load_json(Path(map_partition_pointer["manifest"]))
        chunks = plan_partitioned_trip_chunks(map_manifest, config["selected_profile"], int(config["max_rows_per_chunk"]))
        points = None
    else:
        if raw_pointer is None:
            raise RuntimeError("Either map_match_pointer or map_partition_pointer is required.")
        points = pd.read_parquet(raw_pointer["matched_points"])
        points = points[
            points["profile"].eq(config["selected_profile"])
            & points["matched_latitude"].notna()
            & points["matched_longitude"].notna()
        ].copy()
        points["row_semantic_id"] = [CONTEXT.stable_uid(trip, index) for trip, index in zip(points["pilot_trip_id"], points["point_index"])]
        points["matched_x_m"], points["matched_y_m"] = transformer.transform(
            points["matched_longitude"].to_numpy(float), points["matched_latitude"].to_numpy(float)
        )
        chunks = plan_trip_chunks(points, int(config["max_rows_per_chunk"]))
    plan_public = [{key: value for key, value in chunk.items() if key != "trips"} | {"pilot_trip_ids": [trip["pilot_trip_id"] for trip in chunk["trips"]]} for chunk in chunks]
    total_input_rows = sum(int(chunk["row_count"]) for chunk in chunks)
    total_input_trips = sum(int(chunk["trip_count"]) for chunk in chunks)
    input_fingerprint = {
        "edge_summary_sha256": sha256_file(Path(edge_pointer["summary"])),
        "config_sha256": sha256_file(args.config),
        "chunk_plan_sha256": stable_json_hash(plan_public),
        "selected_profile": config["selected_profile"],
    }
    if map_manifest is not None:
        input_fingerprint["map_partition_manifest_sha256"] = sha256_file(Path(map_partition_pointer["manifest"]))
    else:
        input_fingerprint["matched_points_sha256"] = sha256_file(Path(raw_pointer["matched_points"]))

    if args.run_dir:
        output = args.run_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(config["output_root"]) / f"{config['run_name_prefix']}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    chunks_root = output / "chunks"
    chunks_root.mkdir(exist_ok=True)
    manifest_path = output / "chunk_manifest.json"
    summary_path = output / "chunked_context_summary.json"
    if not (output / "run_config.yaml").exists():
        shutil.copy2(args.config, output / "run_config.yaml")

    if manifest_path.exists():
        manifest = CONTEXT.load_json(manifest_path)
        if manifest["input_fingerprint"] != input_fingerprint:
            raise RuntimeError("Resume manifest fingerprint does not match current inputs/config.")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": input_fingerprint,
            "max_rows_per_chunk": int(config["max_rows_per_chunk"]),
            "total_row_count": int(total_input_rows),
            "total_trip_count": int(total_input_trips),
            "chunks": [{**item, "status": "PENDING"} for item in plan_public],
        }
        write_json_atomic(manifest_path, manifest)

    if all(chunk["status"] == "PASS" for chunk in manifest["chunks"]) and summary_path.exists():
        summary = CONTEXT.load_json(summary_path)
        summary["resume_noop_verified"] = True
        summary["resume_noop_elapsed_seconds"] = time.perf_counter() - main_started
        summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    handlers = {
        snapshot_id: CONTEXT.parse_snapshot(snapshot_id, Path(pbf), config, transformer)
        for snapshot_id, pbf in config["historical_source_pbfs"].items()
    }
    objects = pd.DataFrame([record for handler in handlers.values() for record in handler.objects])
    objects = objects.drop_duplicates(["osm_snapshot_id", "object_layer", "element_type", "osm_id"], keep="first").reset_index(drop=True)
    objects_path = output / "context_objects.parquet"
    if not objects_path.exists():
        objects.to_parquet(objects_path, index=False)

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    cached_partition_id = None
    cached_partition_points = None
    for manifest_chunk in manifest["chunks"]:
        if manifest_chunk["status"] == "PASS":
            continue
        chunk = chunk_lookup[manifest_chunk["chunk_id"]]
        chunk_dir = chunks_root / chunk["chunk_id"]
        chunk_dir.mkdir(exist_ok=True)
        manifest_chunk["status"] = "RUNNING"
        manifest_chunk["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)

        if map_manifest is not None:
            if cached_partition_id != chunk["partition_id"]:
                cached_partition_points = pd.read_parquet(chunk["matched_points"])
                cached_partition_points = cached_partition_points[
                    cached_partition_points["profile"].eq(config["selected_profile"])
                    & cached_partition_points["matched_latitude"].notna()
                    & cached_partition_points["matched_longitude"].notna()
                ].copy()
                cached_partition_points["row_semantic_id"] = [CONTEXT.stable_uid(trip, index) for trip, index in zip(cached_partition_points["pilot_trip_id"], cached_partition_points["point_index"])]
                cached_partition_points["matched_x_m"], cached_partition_points["matched_y_m"] = transformer.transform(
                    cached_partition_points["matched_longitude"].to_numpy(float), cached_partition_points["matched_latitude"].to_numpy(float)
                )
                cached_partition_id = chunk["partition_id"]
            current = chunk_points(cached_partition_points, chunk)
        else:
            current = chunk_points(points, chunk)
        current = CONTEXT.build_topology(current, handlers, float(config["topology_scale_m"]))
        current = CONTEXT.polygon_coverage(current, handlers, config["built_environment"]["coverage_scales_m"])
        row_context, object_exposure, trip_exposure = CONTEXT.build_object_context(current, objects, config["scales_m"])
        monotonic_count = CONTEXT.monotonic_violations(row_context, config["scales_m"])
        coverage_columns = [column for column in row_context if "coverage_ratio" in column]
        coverage_bounds_ok = bool(((row_context[coverage_columns] >= 0) & (row_context[coverage_columns] <= 1)).all(axis=None))
        row_path = chunk_dir / "row_context_semantics.parquet"
        object_path = chunk_dir / "object_exposure.parquet"
        trip_path = chunk_dir / "trip_context_exposure.parquet"
        row_context.to_parquet(row_path, index=False)
        object_exposure.to_parquet(object_path, index=False)
        trip_exposure.to_parquet(trip_path, index=False)
        checks = {
            "row_conservation": "PASS" if len(row_context) == int(chunk["row_count"]) else "FAIL",
            "complete_trip_partition": "PASS" if row_context["pilot_trip_id"].nunique() == int(chunk["trip_count"]) else "FAIL",
            "row_semantic_id_unique": "PASS" if not row_context["row_semantic_id"].duplicated().any() else "FAIL",
            "monotonic_counts": "PASS" if monotonic_count == 0 else "FAIL",
            "coverage_ratio_bounds": "PASS" if coverage_bounds_ok else "FAIL",
        }
        manifest_chunk.update({
            "status": "PASS" if "FAIL" not in checks.values() else "FAIL",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "row_context_count": int(len(row_context)),
            "object_exposure_count": int(len(object_exposure)),
            "trip_exposure_count": int(len(trip_exposure)),
            "outputs": {
                "row_context": str(row_path),
                "object_exposure": str(object_path),
                "trip_exposure": str(trip_path),
            },
            "sha256": {
                "row_context": sha256_file(row_path),
                "object_exposure": sha256_file(object_path),
                "trip_exposure": sha256_file(trip_path),
            },
        })
        write_json_atomic(manifest_path, manifest)
        if manifest_chunk["status"] != "PASS":
            raise RuntimeError(f"Per-chunk QA failed for {chunk['chunk_id']}: {checks}")
        del current, row_context, object_exposure, trip_exposure
        gc.collect()

    manifest = CONTEXT.load_json(manifest_path)
    all_pass = all(chunk["status"] == "PASS" for chunk in manifest["chunks"])
    total_rows = sum(int(chunk.get("row_context_count", 0)) for chunk in manifest["chunks"])
    total_object_exposure = sum(int(chunk.get("object_exposure_count", 0)) for chunk in manifest["chunks"])
    total_trip_exposure = sum(int(chunk.get("trip_exposure_count", 0)) for chunk in manifest["chunks"])
    checks = {
        "all_chunks_pass": "PASS" if all_pass else "FAIL",
        "global_row_conservation": "PASS" if total_rows == total_input_rows else "FAIL",
        "manifest_complete": "PASS" if all("sha256" in chunk for chunk in manifest["chunks"]) else "FAIL",
        "trip_partition_complete": "PASS" if sum(int(chunk["trip_count"]) for chunk in manifest["chunks"]) == total_input_trips else "FAIL",
        "map_matching_waiver_carried": "PASS" if map_matching["status"] == "PASS_WITH_DOCUMENTED_MANUAL_WAIVER" else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    summary = {
        "experiment": config["experiment"],
        "stage": config["stage"],
        "status": f"{gate}_{config['status_suffix']}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(manifest["chunks"]),
        "max_rows_per_chunk": int(config["max_rows_per_chunk"]),
        "row_context_count": int(total_rows),
        "trip_count": int(total_input_trips),
        "object_exposure_row_count": int(total_object_exposure),
        "trip_exposure_row_count": int(total_trip_exposure),
        "context_object_count": int(len(objects)),
        "checks": checks,
        "partitioned_outputs_only": not bool(config["write_global_row_outputs"]),
        "resume_manifest": str(manifest_path),
        "context_objects": str(objects_path),
        "output_directory": str(output),
        "resume_noop_verified": False,
    }
    write_json_atomic(summary_path, summary)
    report_path = output / "TRAJECTORY_BUILD_CHUNKED_CONTEXT_DRY_RUN.md"
    report_path.write_text(
        "# TRAJECTORY_BUILD chunked context dry run\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Chunks: {summary['chunk_count']}\n"
        f"- Rows / trips: {summary['row_context_count']:,} / {summary['trip_count']:,}\n"
        f"- Object/trip exposure rows: {total_object_exposure:,} / {total_trip_exposure:,}\n"
        f"- Partitioned outputs only: `{summary['partitioned_outputs_only']}`\n"
        f"- Resume manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )
    latest_prefix = config["latest_prefix"]
    CONTEXT.write_json(Path(config["output_root"]) / f"{latest_prefix}_latest_pointer.json", {
        "status": summary["status"],
        "summary": str(summary_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "output_directory": str(output),
        "context_objects": str(objects_path),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # PyOsmium/GEOS objects can stall CPython finalization on Windows after all
    # outputs are safely closed. Exit directly only at the executable boundary.
    os._exit(exit_code)
