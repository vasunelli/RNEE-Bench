#!/usr/bin/env python3
"""Resume pending frozen time-segment partitions with bounded parallel workers.

The canonical builder remains 35_build_segments.py.  This helper writes the
same per-partition artifacts and leaves final consolidation to the canonical
builder, preserving its manifest fingerprint and no-op contract.
"""
from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("canonical_time_segments", HERE / "35_build_segments.py")
SEG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SEG)

WORKER_CONFIG: dict[str, Any] | None = None
WORKER_OBJECTS: pd.DataFrame | None = None


def initialize_worker(config_path: str, objects_path: str) -> None:
    global WORKER_CONFIG, WORKER_OBJECTS
    WORKER_CONFIG = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    WORKER_OBJECTS = pd.read_parquet(objects_path)


def process_partition(task: dict[str, Any]) -> dict[str, Any]:
    assert WORKER_CONFIG is not None and WORKER_OBJECTS is not None
    rows = pd.read_parquet(task["rows_path"])
    segments, assignments = SEG.aggregate_partition(rows, WORKER_OBJECTS, WORKER_CONFIG)
    part_dir = Path(task["part_dir"])
    part_dir.mkdir(parents=True, exist_ok=True)
    segment_path = part_dir / "segments.parquet"
    assignment_path = part_dir / "row_segment_assignments.parquet"
    segments.to_parquet(segment_path, index=False)
    assignments.to_parquet(assignment_path, index=False)
    checks = {
        "row_assignment_conservation": "PASS" if len(assignments) == len(rows) and assignments["segment_id"].notna().all() else "FAIL",
        "row_id_unique": "PASS" if assignments["row_id"].is_unique else "FAIL",
        "segment_id_unique": "PASS" if segments["segment_id"].is_unique else "FAIL",
        "unified_scalar_prohibited": "PASS" if not segments["unified_scalar_target_released"].any() else "FAIL",
    }
    peak_gb = SEG.current_working_set_gb()
    return {
        "partition_id": task["partition_id"],
        "source_file": task["source_file"],
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
        "worker_peak_working_set_gb": peak_gb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "segment_manifest.json"
    manifest = SEG.load_json(manifest_path)
    row_pointer = SEG.load_json(Path(config["row_pointer"]))
    row_manifest = SEG.load_json(Path(row_pointer["manifest"]))
    context_pointer = SEG.load_json(Path(config["context_pointer"]))
    objects_path = Path(context_pointer["context_objects"])
    expected = {
        "config_sha256": SEG.sha256_file(config_path),
        "script_sha256": SEG.sha256_file(HERE / "35_build_segments.py"),
        "row_manifest_sha256": SEG.sha256_file(Path(row_pointer["manifest"])),
        "context_objects_sha256": SEG.sha256_file(objects_path),
        "target_contract_sha256": SEG.sha256_file(Path(config["target_contract"])),
        "release_summary_sha256": SEG.sha256_file(Path(SEG.load_json(Path(config["trajectory_build_release_pointer"]))["summary"])),
    }
    if manifest["input_fingerprint"] != expected:
        raise RuntimeError("Canonical manifest fingerprint mismatch.")
    source_map = {item["source_file"]: item for item in row_manifest["partitions"]}
    tasks = []
    for item in manifest["partitions"]:
        if item["status"] == "PASS":
            continue
        source = source_map[item["source_file"]]
        tasks.append({
            "partition_id": item["partition_id"],
            "source_file": item["source_file"],
            "rows_path": source["pointer"]["rows"],
            "part_dir": str(run_dir / "partitions" / item["partition_id"]),
        })
    if not tasks:
        print("No pending partitions.")
        return 0
    by_id = {item["partition_id"]: item for item in manifest["partitions"]}
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=initialize_worker, initargs=(str(config_path), str(objects_path))) as pool:
        futures = {pool.submit(process_partition, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            by_id[result["partition_id"]].update(result)
            SEG.write_json(manifest_path, manifest)
            print(f"{result['partition_id']}: {result['status']} peak={result['worker_peak_working_set_gb']:.3f} GB", flush=True)
            if result["status"] != "PASS":
                raise RuntimeError(f"Parallel partition failed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
