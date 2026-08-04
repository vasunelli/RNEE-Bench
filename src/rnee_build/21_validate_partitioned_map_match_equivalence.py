#!/usr/bin/env python3
"""Compare partitioned map matching with the accepted END_TO_END two-file run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_equivalence", HERE / "18_validate_chunked_context_equivalence.py")
EQUIV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EQUIV)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-pointer", type=Path, default=Path(r"results/trajectory_build_map_match_partition_dry_run_latest_pointer.json"))
    parser.add_argument("--baseline-pointer", type=Path, default=Path(r"results/end_to_end_latest_raw_pointer.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    partition_pointer = load_json(args.partition_pointer)
    baseline_pointer = load_json(args.baseline_pointer)
    summary = load_json(Path(partition_pointer["summary"]))
    manifest = load_json(Path(partition_pointer["manifest"]))
    baseline = {
        "selection": pd.read_parquet(baseline_pointer["selection"]),
        "matched_points": pd.read_parquet(baseline_pointer["matched_points"]),
        "matched_edges": pd.read_parquet(baseline_pointer["matched_edges"]),
        "trip_summary": pd.read_parquet(baseline_pointer["trip_summary"]),
    }
    sort_keys = {
        "selection": ["source_file", "VehId", "Trip"],
        "matched_points": ["profile", "pilot_trip_id", "point_index"],
        "matched_edges": ["profile", "pilot_trip_id", "edge_index"],
        "trip_summary": ["profile", "pilot_trip_id"],
    }
    results = []
    for partition in manifest["partitions"]:
        source_file = partition["source_file"]
        item = {"partition_id": partition["partition_id"], "source_file": source_file, "comparisons": {}}
        for name in ["selection", "matched_points", "matched_edges", "trip_summary"]:
            expected = baseline[name][baseline[name]["source_file"].eq(source_file)].copy()
            actual = pd.read_parquet(partition["pointer"][name])
            if name == "trip_summary" and "response_path" in expected.columns:
                expected["response_path"] = expected["response_path"].map(lambda value: Path(value).name)
                actual["response_path"] = actual["response_path"].map(lambda value: Path(value).name)
            ok, error, expected_hash, actual_hash = EQUIV.compare_frame(expected, actual, sort_keys[name])
            item["comparisons"][name] = {"equivalent": ok, "error": error, "expected_hash": expected_hash, "actual_hash": actual_hash, "row_count": len(actual)}
        results.append(item)
    all_equivalent = all(comparison["equivalent"] for item in results for comparison in item["comparisons"].values())
    checks = {
        "all_partition_frames_exact": "PASS" if all_equivalent else "FAIL",
        "global_point_count": "PASS" if sum(item["comparisons"]["matched_points"]["row_count"] for item in results) == len(baseline["matched_points"]) else "FAIL",
        "resume_noop_verified": "PASS" if summary.get("resume_noop_verified") is True else "FAIL",
        "response_path_relocation_normalized": "PASS",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    output = Path(partition_pointer["output_directory"]) / "equivalence_audit"
    output.mkdir(exist_ok=True)
    result = {
        "experiment": "TRAJECTORY_BUILD-DRYRUN",
        "stage": "partitioned_map_match_equivalence",
        "status": f"{gate}_TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_EQUIVALENCE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "partition_count": len(results),
        "baseline_point_count": len(baseline["matched_points"]),
        "partition_point_count": sum(item["comparisons"]["matched_points"]["row_count"] for item in results),
        "partition_results": results,
        "output_directory": str(output),
    }
    summary_path = output / "equivalence_summary.json"
    report_path = output / "EQUIVALENCE_REPORT.md"
    write_json(summary_path, result)
    report_path.write_text(
        "# TRAJECTORY_BUILD partitioned map-match equivalence\n\n"
        f"- Status: `{result['status']}`\n"
        f"- Partitions: {len(results)}\n"
        f"- Baseline/partition points: {len(baseline['matched_points']):,} / {result['partition_point_count']:,}\n"
        f"- Selection, points, edges and trip summaries exact after response-directory relocation normalization: `{checks['all_partition_frames_exact']}`\n"
        f"- Resume no-op: `{checks['resume_noop_verified']}`\n",
        encoding="utf-8",
    )
    write_json(args.partition_pointer.parent / "trajectory_build_map_match_partition_dry_run_latest_equivalence_pointer.json", {"status": result["status"], "summary": str(summary_path), "report": str(report_path), "output_directory": str(output)})
    print(json.dumps({key: value for key, value in result.items() if key != "partition_results"}, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
