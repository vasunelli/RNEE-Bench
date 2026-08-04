#!/usr/bin/env python3
"""Run Valhalla map matching as resumable source-file partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
PROFILE_SPEC = importlib.util.spec_from_file_location("rnee_profile", HERE / "17_profile_chunked_context.py")
PROFILE = importlib.util.module_from_spec(PROFILE_SPEC)
assert PROFILE_SPEC.loader is not None
PROFILE_SPEC.loader.exec_module(PROFILE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def safe_name(source_file: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in source_file).strip("_")


def partition_plan(source_files: list[str]) -> list[dict[str, Any]]:
    if len(source_files) != len(set(source_files)):
        raise ValueError("source_files contains duplicates.")
    return [
        {"partition_id": f"partition_{index:04d}", "source_file": source_file}
        for index, source_file in enumerate(source_files)
    ]


def request_failures_within_bound(count: int, total: int, maximum_count: int, maximum_rate: float) -> bool:
    return total > 0 and count <= maximum_count and count / total <= maximum_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/trajectory_build_map_match_partition_dry_run.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    main_started = time.perf_counter()
    args = parse_args()
    overlay = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = Path(overlay["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    frozen_plan = None
    if overlay.get("preflight_pointer"):
        preflight_pointer = load_json(Path(overlay["preflight_pointer"]))
        if preflight_pointer["status"] != "PASS_TRAJECTORY_BUILD_PRODUCTION_PREFLIGHT":
            raise RuntimeError("TRAJECTORY_BUILD production preflight has not passed.")
        frozen_plan = load_json(Path(preflight_pointer["partition_plan"]))
        plan = frozen_plan["partitions"]
        source_files = [item["source_file"] for item in plan]
    else:
        source_files = list(overlay["source_files"])
        plan = partition_plan(source_files)
    ved_dir = Path(base["ved_dynamic_dir"])
    source_hashes = {source: sha256_file(ved_dir / source) for source in source_files}
    if frozen_plan:
        expected_hashes = {item["source_file"]: item["sha256"] for item in plan}
        if source_hashes != expected_hashes:
            raise RuntimeError("Current VED source hashes differ from the frozen TRAJECTORY_BUILD partition plan.")
    fingerprint = {
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "base_config_sha256": sha256_file(base_path),
        "profiles_sha256": sha256_file(Path(overlay["profiles"])),
        "remediation_sha256": sha256_file(Path(overlay["remediation"])),
        "source_sha256": source_hashes,
        "production_partition_plan_sha256": sha256_file(Path(preflight_pointer["partition_plan"])) if frozen_plan else None,
    }
    if args.run_dir:
        output = args.run_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(overlay["output_root"]) / f"{overlay['run_name_prefix']}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    partitions_root = output / "partitions"
    partitions_root.mkdir(exist_ok=True)
    profile_root = output / "profile_output"
    profile_root.mkdir(exist_ok=True)
    manifest_path = output / "partition_manifest.json"
    summary_path = output / "partitioned_map_match_summary.json"
    if not (output / "run_config.yaml").exists():
        shutil.copy2(args.config, output / "run_config.yaml")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": overlay["experiment"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint,
            "partitions": [{**item, "status": "PENDING"} for item in plan],
        }
        write_json_atomic(manifest_path, manifest)
    if all(item["status"] == "PASS" for item in manifest["partitions"]) and summary_path.exists():
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = True
        summary["resume_noop_elapsed_seconds"] = time.perf_counter() - main_started
        summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    plan_by_id = {item["partition_id"]: item for item in plan}
    for item in manifest["partitions"]:
        if item["status"] == "PASS":
            continue
        planned = plan_by_id[item["partition_id"]]
        partition_root = partitions_root / f"{item['partition_id']}_{safe_name(item['source_file'])}"
        partition_root.mkdir(exist_ok=True)
        child_output_root = partition_root / "runs"
        child_output_root.mkdir(exist_ok=True)
        child_config = dict(base)
        child_config.update({
            "experiment": overlay["experiment"],
            "output_root": str(child_output_root),
            "run_name_prefix": item["partition_id"],
            "latest_raw_pointer_name": f"{item['partition_id']}_latest_pointer.json",
            "all_trips_in_source_files": True,
            "source_files": [planned["source_file"]],
            "map_snap_failure_fallback": overlay.get("map_snap_failure_fallback", {"enabled": False}),
        })
        child_config_path = partition_root / "partition_config.yaml"
        child_config_path.write_text(yaml.safe_dump(child_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        stdout_path = partition_root / "child_stdout.log"
        command = [
            sys.executable,
            str(HERE / "06_map_match_trips.py"),
            "--config", str(child_config_path),
            "--profiles", str(Path(overlay["profiles"])),
            "--remediation", str(Path(overlay["remediation"])),
        ]
        item["status"] = "RUNNING"
        item["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
        samples = []
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout:
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=HERE.parents[1])
            while process.poll() is None:
                memory = PROFILE.process_tree_memory(process.pid)
                if memory:
                    samples.append({"elapsed_seconds": time.perf_counter() - started, **memory})
                time.sleep(max(0.05, float(overlay["sample_interval_seconds"])))
            return_code = int(process.returncode)
        pointer_path = child_output_root / f"{item['partition_id']}_latest_pointer.json"
        if return_code != 0 or not pointer_path.exists():
            item.update({"status": "FAIL", "return_code": return_code, "stdout_log": str(stdout_path)})
            write_json_atomic(manifest_path, manifest)
            raise RuntimeError(f"Map-match partition failed: {item['partition_id']}")
        pointer = load_json(pointer_path)
        child_summary = load_json(Path(pointer["summary"]))
        peak_ws = max((sample["working_set_bytes"] for sample in samples), default=0) / (1024**3)
        peak_private = max((sample["private_bytes"] for sample in samples), default=0) / (1024**3)
        sample_path = profile_root / f"{item['partition_id']}_memory_samples.csv"
        with sample_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["elapsed_seconds", "process_count", "working_set_bytes", "peak_working_set_bytes", "private_bytes", "peak_pagefile_bytes"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(samples)
        checks = {
            "return_code": "PASS" if return_code == 0 else "FAIL",
            "row_cardinality": "PASS" if child_summary["output_point_count"] == child_summary["expected_output_point_count"] else "FAIL",
            "request_failure_bound": "PASS" if request_failures_within_bound(
                int(child_summary["request_failure_count"]), int(child_summary["request_count"]),
                int(overlay.get("maximum_request_failures_per_partition", 0)),
                float(overlay.get("maximum_request_failure_rate_per_partition", 0.0)),
            ) else "FAIL",
            "response_cardinality": "PASS" if child_summary["cardinality_failure_count"] == 0 else "FAIL",
            "source_partition": "PASS" if child_summary["input_point_count_per_profile"] > 0 else "FAIL",
            "inventory_row_count": "PASS" if not frozen_plan or child_summary["output_point_count"] == int(planned["data_rows"]) else "FAIL",
            "memory_ceiling": "PASS" if peak_ws <= float(overlay["memory_ceiling_gb"]) else "FAIL",
        }
        item.update({
            "status": "PASS" if "FAIL" not in checks.values() else "FAIL",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "return_code": return_code,
            "checks": checks,
            "trip_count": int(child_summary["pilot_trip_count"]),
            "point_count": int(child_summary["output_point_count"]),
            "request_count": int(child_summary["request_count"]),
            "request_failure_count": int(child_summary["request_failure_count"]),
            "request_failure_rate": float(child_summary["request_failure_count"] / child_summary["request_count"]),
            "peak_working_set_gb": peak_ws,
            "peak_private_gb": peak_private,
            "sample_count": len(samples),
            "pointer": pointer,
            "stdout_log": str(stdout_path),
            "memory_samples": str(sample_path),
            "sha256": {key: sha256_file(Path(pointer[key])) for key in ["selection", "matched_points", "matched_edges", "trip_summary"]},
        })
        write_json_atomic(manifest_path, manifest)
        if item["status"] != "PASS":
            raise RuntimeError(f"Partition QA failed: {item['partition_id']}: {checks}")

    manifest = load_json(manifest_path)
    total_requests = sum(item["request_count"] for item in manifest["partitions"])
    total_request_failures = sum(item["request_failure_count"] for item in manifest["partitions"])
    checks = {
        "all_partitions_pass": "PASS" if all(item["status"] == "PASS" for item in manifest["partitions"]) else "FAIL",
        "source_file_coverage": "PASS" if {item["source_file"] for item in manifest["partitions"]} == set(source_files) else "FAIL",
        "manifest_hashes_complete": "PASS" if all("sha256" in item for item in manifest["partitions"]) else "FAIL",
        "global_memory_ceiling": "PASS" if max(item["peak_working_set_gb"] for item in manifest["partitions"]) <= float(overlay["memory_ceiling_gb"]) else "FAIL",
        "global_request_failure_bound": "PASS" if total_requests > 0 and total_request_failures / total_requests <= float(overlay.get("maximum_request_failure_rate_global", 0.0)) else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    summary = {
        "experiment": overlay["experiment"],
        "stage": overlay.get("summary_stage", "partitioned_map_match_dry_run"),
        "status": f"{gate}_{overlay.get('status_suffix', 'TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_DRY_RUN')}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition_count": len(manifest["partitions"]),
        "source_file_count": len(source_files),
        "trip_count": sum(item["trip_count"] for item in manifest["partitions"]),
        "point_count": sum(item["point_count"] for item in manifest["partitions"]),
        "request_count": total_requests,
        "request_failure_count": total_request_failures,
        "request_failure_rate": total_request_failures / total_requests,
        "peak_working_set_gb": max(item["peak_working_set_gb"] for item in manifest["partitions"]),
        "peak_private_gb": max(item["peak_private_gb"] for item in manifest["partitions"]),
        "memory_ceiling_gb": float(overlay["memory_ceiling_gb"]),
        "checks": checks,
        "resume_noop_verified": False,
        "partition_manifest": str(manifest_path),
        "output_directory": str(output),
    }
    write_json_atomic(summary_path, summary)
    report_path = output / overlay.get("report_filename", "TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_DRY_RUN.md")
    report_path.write_text(
        "# TRAJECTORY_BUILD partitioned map-match dry run\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Files / trips / points: {len(source_files)} / {summary['trip_count']:,} / {summary['point_count']:,}\n"
        f"- Peak working set/private: {summary['peak_working_set_gb']:.2f} / {summary['peak_private_gb']:.2f} GB\n"
        f"- Memory ceiling: {summary['memory_ceiling_gb']:.2f} GB\n\n"
        "## Instrumentation changelog\n\n"
        "| File | Change type | What was added/modified |\n|---|---|---|\n"
        "| `src/rnee_build/20_run_map_match_partitioned.py` | created | Source-file partition wrapper, process-tree memory sampling, atomic resume manifest and per-partition QA |\n"
        "| `profile_output/partition_*_memory_samples.csv` | generated | Per-partition process-tree memory time series |\n",
        encoding="utf-8",
    )
    latest = Path(overlay["output_root"]) / f"{overlay['latest_prefix']}_latest_pointer.json"
    write_json(latest, {"status": summary["status"], "summary": str(summary_path), "report": str(report_path), "manifest": str(manifest_path), "output_directory": str(output)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
