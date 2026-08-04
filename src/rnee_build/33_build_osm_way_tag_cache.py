#!/usr/bin/env python3
"""Extract all TRAJECTORY_BUILD matched OSM way tags once per historical snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_edge", HERE / "10_extract_edge_semantics.py")
EDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EDGE)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/trajectory_build_osm_way_tag_cache.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    edge_config = yaml.safe_load(Path(config["edge_config"]).read_text(encoding="utf-8"))
    map_pointer = load(Path(config["map_partition_pointer"]))
    manifest = load(Path(map_pointer["manifest"]))
    network = load(Path(config["network_pointer"]))

    ids_by_snapshot: dict[str, set[int]] = {}
    for partition in manifest["partitions"]:
        pointer = partition["pointer"]
        edges = pd.read_parquet(pointer["matched_edges"], columns=["profile", "osm_snapshot_id", "way_id"])
        edges = edges[edges["profile"].eq(config["selected_profile"])]
        edges["way_id"] = pd.to_numeric(edges["way_id"], errors="coerce")
        for snapshot_id, group in edges[edges["way_id"].notna()].groupby("osm_snapshot_id"):
            ids_by_snapshot.setdefault(str(snapshot_id), set()).update(group["way_id"].astype(int).tolist())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    output = output_root / f"{config['run_name_prefix']}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "run_config.yaml")
    frames = []
    snapshot_stats = {}
    for snapshot_id, target_ids in sorted(ids_by_snapshot.items()):
        frame = EDGE.extract_target_ways(Path(network["cropped_pbfs"][snapshot_id]), target_ids, edge_config["osm_tags"])
        frame.insert(0, "osm_snapshot_id", snapshot_id)
        frames.append(frame)
        snapshot_stats[snapshot_id] = {
            "requested_way_count": len(target_ids),
            "extracted_way_count": len(frame),
            "coverage": len(frame) / len(target_ids) if target_ids else 1.0,
        }
    cache = pd.concat(frames, ignore_index=True)
    cache_path = output / "target_osm_way_tags_all_snapshots.parquet"
    cache.to_parquet(cache_path, index=False)
    unique = cache[["osm_snapshot_id", "way_id"]].drop_duplicates()
    requested = sum(len(ids_) for ids_ in ids_by_snapshot.values())
    checks = {
        "all_snapshots_present": "PASS" if set(ids_by_snapshot) == set(snapshot_stats) else "FAIL",
        "unique_snapshot_way_keys": "PASS" if len(unique) == len(cache) else "FAIL",
        "all_target_ways_extracted": "PASS" if len(cache) == requested else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    summary = {
        "experiment": config["experiment"],
        "stage": "snapshot_level_osm_way_tag_cache",
        "status": f"{gate}_TRAJECTORY_BUILD_OSM_WAY_TAG_CACHE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "map_partition_count": len(manifest["partitions"]),
        "snapshot_count": len(ids_by_snapshot),
        "requested_way_count": requested,
        "cached_way_count": len(cache),
        "snapshot_stats": snapshot_stats,
        "checks": checks,
        "cache_sha256": sha256_file(cache_path),
        "output_directory": str(output),
    }
    dump(output / "osm_way_tag_cache_summary.json", summary)
    report = output / "TRAJECTORY_BUILD_OSM_WAY_TAG_CACHE.md"
    report.write_text(
        "# TRAJECTORY_BUILD snapshot-level OSM way-tag cache\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Map partitions / historical snapshots: {len(manifest['partitions'])} / {len(ids_by_snapshot)}\n"
        f"- Requested / cached unique snapshot-way pairs: {requested:,} / {len(cache):,}\n"
        "- Each historical cropped PBF is parsed once; partitioned edge construction performs exact snapshot-way filtering from this immutable cache.\n",
        encoding="utf-8",
    )
    pointer = {"status": summary["status"], "summary": str(output / "osm_way_tag_cache_summary.json"), "report": str(report), "osm_way_tags": str(cache_path), "output_directory": str(output)}
    dump(output_root / f"{config['latest_prefix']}_latest_pointer.json", pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
