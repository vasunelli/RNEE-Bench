#!/usr/bin/env python3
"""Run edge semantics as resumable map-match partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_profile", HERE / "17_profile_chunked_context.py")
PROFILE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROFILE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); write_json(temporary, value); temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/trajectory_build_edge_partition_dry_run.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--force-partition", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    started_main = time.perf_counter()
    args = parse_args(); overlay = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(overlay["base_config"]).read_text(encoding="utf-8"))
    map_pointer = load_json(Path(overlay["map_partition_pointer"])); map_manifest = load_json(Path(map_pointer["manifest"]))
    fingerprint = {"config_sha256": sha256_file(args.config), "base_config_sha256": sha256_file(Path(overlay["base_config"])), "map_manifest_sha256": sha256_file(Path(map_pointer["manifest"]))}
    if overlay.get("osm_way_tag_cache_pointer"):
        fingerprint["osm_way_tag_cache_pointer_sha256"] = sha256_file(Path(overlay["osm_way_tag_cache_pointer"]))
    output = args.run_dir.resolve() if args.run_dir else Path(overlay["output_root"]) / f"{overlay['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True); partitions_root = output / "partitions"; partitions_root.mkdir(exist_ok=True); profile_root = output / "profile_output"; profile_root.mkdir(exist_ok=True)
    manifest_path = output / "partition_manifest.json"; summary_path = output / "partitioned_edge_summary.json"
    if not (output / "run_config.yaml").exists(): shutil.copy2(args.config, output / "run_config.yaml")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint: raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest = {"schema_version": 1, "experiment": "TRAJECTORY_BUILD-DRYRUN", "input_fingerprint": fingerprint, "partitions": [{"partition_id": item["partition_id"], "source_file": item["source_file"], "status": "PENDING"} for item in map_manifest["partitions"]]}
        write_atomic(manifest_path, manifest)
    force_partitions = set(args.force_partition)
    if not force_partitions and all(item["status"] == "PASS" for item in manifest["partitions"]) and summary_path.exists():
        summary = load_json(summary_path); summary["resume_noop_verified"] = True; summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started_main; summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat(); write_atomic(summary_path, summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    map_by_id = {item["partition_id"]: item for item in map_manifest["partitions"]}
    for item in manifest["partitions"]:
        if item["status"] == "PASS" and item["partition_id"] not in force_partitions: continue
        source = map_by_id[item["partition_id"]]; partition_root = partitions_root / item["partition_id"]; partition_root.mkdir(exist_ok=True); child_root = partition_root / "runs"; child_root.mkdir(exist_ok=True)
        map_pointer_path = partition_root / "map_pointer.json"; write_json(map_pointer_path, source["pointer"])
        child_config = dict(base); child_config.update({"experiment": overlay["experiment"], "stage": "partitioned_edge_semantics", "status_suffix": "EDGE_SEMANTICS_PARTITION", "run_name_prefix": item["partition_id"], "latest_prefix": f"{item['partition_id']}_edge", "map_match_pointer": str(map_pointer_path), "output_root": str(child_root), "require_edge_semantics_authorization": False})
        if overlay.get("osm_way_tag_cache_pointer"):
            child_config["osm_way_tag_cache_pointer"] = overlay["osm_way_tag_cache_pointer"]
        child_config_path = partition_root / "partition_config.yaml"; child_config_path.write_text(yaml.safe_dump(child_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        child_env = os.environ.copy(); child_env["PYTHONPATH"] = str(Path(sys.prefix) / "Lib" / "site-packages") + os.pathsep + child_env.get("PYTHONPATH", "")
        interpreter = str(getattr(sys, "_base_executable", sys.executable))
        stdout_path = partition_root / "child_stdout.log"; command = [interpreter, str(HERE / "10_extract_edge_semantics.py"), "--config", str(child_config_path)]
        item["status"] = "RUNNING"; write_atomic(manifest_path, manifest); samples = []; started = time.perf_counter()
        pointer_path = child_root / f"{item['partition_id']}_edge_latest_pointer.json"
        if pointer_path.exists():
            pointer_path.replace(pointer_path.with_name(pointer_path.stem + f".superseded_{datetime.now().strftime('%Y%m%d_%H%M%S')}" + pointer_path.suffix))
        controlled_post_output_reap = False; completion_seen_at = None
        with stdout_path.open("w", encoding="utf-8") as stdout:
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=HERE.parents[1], env=child_env)
            while process.poll() is None:
                memory = PROFILE.process_tree_memory(process.pid)
                if memory: samples.append({"elapsed_seconds": time.perf_counter() - started, **memory})
                if pointer_path.exists():
                    completion_seen_at = completion_seen_at or time.perf_counter()
                    if time.perf_counter() - completion_seen_at >= 1.0:
                        try: process._handle.Close()
                        except Exception: pass
                        process.returncode = 0
                        controlled_post_output_reap = True
                        break
                time.sleep(max(0.05, float(overlay["sample_interval_seconds"])))
            return_code = int(process.returncode)
        if return_code != 0 or not pointer_path.exists(): item.update({"status": "FAIL", "return_code": return_code}); write_atomic(manifest_path, manifest); raise RuntimeError(f"Edge partition failed: {item['partition_id']}")
        pointer = load_json(pointer_path); child_summary = load_json(Path(pointer["summary"])); peak_ws = max((sample["working_set_bytes"] for sample in samples), default=0)/(1024**3); peak_private = max((sample["private_bytes"] for sample in samples), default=0)/(1024**3)
        sample_path = profile_root / f"{item['partition_id']}_memory_samples.csv"
        with sample_path.open("w", newline="", encoding="utf-8") as handle:
            fields=["elapsed_seconds","process_count","working_set_bytes","peak_working_set_bytes","private_bytes","peak_pagefile_bytes"]; writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(samples)
        checks = {"status": "PASS" if str(child_summary["status"]).startswith("PASS_EDGE_SEMANTICS") else "FAIL", "osm_join": "PASS" if child_summary["osm_join_coverage"] == 1.0 else "FAIL", "point_edge_join": "PASS" if child_summary["point_to_edge_semantic_join_coverage"] == 1.0 else "FAIL", "memory_ceiling": "PASS" if peak_ws <= float(overlay["memory_ceiling_gb"]) else "FAIL"}
        item.update({"status": "PASS" if "FAIL" not in checks.values() else "FAIL", "checks": checks, "controlled_post_output_reap": controlled_post_output_reap, "edge_count": child_summary["matched_edge_row_count"], "point_count": child_summary["matched_point_with_edge_count"], "peak_working_set_gb": peak_ws, "peak_private_gb": peak_private, "sample_count": len(samples), "pointer": pointer, "memory_samples": str(sample_path), "sha256": {key: sha256_file(Path(pointer[key])) for key in ["edge_semantics","point_edge_semantics","osm_way_tags","anomalies"]}}); write_atomic(manifest_path, manifest)
        if item["status"] != "PASS": raise RuntimeError(f"Edge partition QA failed: {checks}")
    manifest = load_json(manifest_path); checks = {"all_partitions_pass": "PASS" if all(item["status"] == "PASS" for item in manifest["partitions"]) else "FAIL", "manifest_hashes_complete": "PASS" if all("sha256" in item for item in manifest["partitions"]) else "FAIL", "memory_ceiling": "PASS" if max(item["peak_working_set_gb"] for item in manifest["partitions"]) <= float(overlay["memory_ceiling_gb"]) else "FAIL"}; gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    summary = {"experiment":overlay["experiment"],"stage":overlay.get("stage","partitioned_edge_semantics_dry_run"),"status":f"{gate}_{overlay.get('status_suffix','TRAJECTORY_BUILD_PARTITIONED_EDGE_DRY_RUN')}","completed_at_utc":datetime.now(timezone.utc).isoformat(),"partition_count":len(manifest["partitions"]),"edge_count":sum(item["edge_count"] for item in manifest["partitions"]),"point_count":sum(item["point_count"] for item in manifest["partitions"]),"peak_working_set_gb":max(item["peak_working_set_gb"] for item in manifest["partitions"]),"peak_private_gb":max(item["peak_private_gb"] for item in manifest["partitions"]),"checks":checks,"resume_noop_verified":False,"partition_manifest":str(manifest_path),"output_directory":str(output)}; write_atomic(summary_path, summary)
    report_path=output/overlay.get("report_filename","TRAJECTORY_BUILD_PARTITIONED_EDGE_DRY_RUN.md"); report_path.write_text(f"# TRAJECTORY_BUILD partitioned edge semantics\n\n- Status: `{summary['status']}`\n- Partitions / edges / points: {summary['partition_count']} / {summary['edge_count']:,} / {summary['point_count']:,}\n- Peak working set/private: {summary['peak_working_set_gb']:.2f} / {summary['peak_private_gb']:.2f} GB\n",encoding="utf-8")
    write_json(Path(overlay["output_root"])/f"{overlay['latest_prefix']}_latest_pointer.json",{"status":summary["status"],"summary":str(summary_path),"report":str(report_path),"manifest":str(manifest_path),"output_directory":str(output)}); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if gate=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
