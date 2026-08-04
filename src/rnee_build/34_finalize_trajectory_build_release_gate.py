#!/usr/bin/env python3
"""Consolidate all full-corpus TRAJECTORY_BUILD evidence into the release-candidate gate."""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/trajectory_build_release_gate.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = ("preflight", "map", "map_gate", "abstention", "tag_cache", "edge", "context", "row")
    pointers = {name: load(Path(cfg[f"{name}_pointer"])) for name in names}
    summaries = {name: load(Path(pointer["summary"])) for name, pointer in pointers.items()}
    row_manifest = load(Path(pointers["row"]["manifest"]))
    map_manifest = load(Path(pointers["map"]["manifest"]))
    edge_manifest = load(Path(pointers["edge"]["manifest"]))
    context_manifest = load(Path(pointers["context"]["manifest"]))
    context_profile = load(Path(pointers["context"]["output_directory"]) / "profile_output" / "profile_summary.json")

    reason_counts: collections.Counter[str] = collections.Counter()
    partition_rows = []
    raw_hash_samples = 0
    available_rows = 0
    unavailable_rows = 0
    forbidden_hits = 0
    missing_allowlist = 0
    for item in row_manifest["partitions"]:
        child = load(Path(item["pointer"]["summary"]))
        reason_counts.update({str(key): int(value) for key, value in child["road_semantics_exclusion_reason_counts"].items()})
        available_rows += int(child["road_semantics_available_row_count"])
        unavailable_rows += int(child["road_semantics_unavailable_row_count"])
        raw_hash_samples += int(child["raw_hash_sample_count"])
        forbidden_hits += int(child["forbidden_feature_hit_count"])
        missing_allowlist += int(child["missing_explicit_allowlist_column_count"])
        partition_rows.append({
            "partition_id": item["partition_id"], "source_file": item["source_file"], "row_count": item["row_count"],
            "expected_abstain_trip_count": item["expected_abstain_trip_count"], "observed_abstain_trip_count": item["observed_abstain_trip_count"],
            "abstention_row_count": item["abstention_row_count"], "schema_signature": item["schema_signature"],
            "raw_hash_sample_count": child["raw_hash_sample_count"], "raw_hash_match_rate": child["raw_hash_match_rate"],
            "forbidden_feature_hit_count": child["forbidden_feature_hit_count"], "peak_working_set_gb": item["peak_working_set_gb"],
        })

    expected_rows = int(cfg["expected_row_count"])
    expected_files = int(cfg["expected_source_file_count"])
    expected_abstain_trips = int(cfg["expected_abstain_trip_count"])
    abstention_reason = "off_network_open_lot_quality_1_abstention"
    stage_summaries = (summaries["map"], summaries["edge"], summaries["context"], summaries["row"])
    peak_working_set_gb_by_stage = {
        "map": float(summaries["map"]["peak_working_set_gb"]), "edge": float(summaries["edge"]["peak_working_set_gb"]),
        "context": float(context_profile["peak_working_set_gb"]), "row": float(summaries["row"]["peak_working_set_gb"]),
    }
    map_gate_nonquality_check = [value for key, value in summaries["map_gate"]["checks"].items() if key not in {"distance_p95", "distance_p99"}]
    test = subprocess.run(
        [str(Path.cwd() / r".venv-rnee311/Scripts/python.exe"), "-m", "pytest", "tests/rnee_build", "-q"],
        capture_output=True, text=True,
    )
    checks = {
        "ved_source_contract": "PASS" if pointers["preflight"]["status"] == "PASS_TRAJECTORY_BUILD_PRODUCTION_PREFLIGHT" and all(value == "PASS" for value in summaries["preflight"]["checks"].values()) else "FAIL",
        "source_file_and_row_conservation": "PASS" if summaries["preflight"]["source_file_count"] == expected_files and summaries["preflight"]["source_row_count"] == expected_rows and summaries["row"]["partition_count"] == expected_files and summaries["row"]["row_count"] == expected_rows else "FAIL",
        "production_stages_pass": "PASS" if all(str(summary["status"]).startswith("PASS_TRAJECTORY_BUILD_") for summary in stage_summaries) else "FAIL",
        "all_stage_resume_noops": "PASS" if all(summary.get("resume_noop_verified") is True for summary in stage_summaries) else "FAIL",
        "partition_manifests_complete": "PASS" if len(map_manifest["partitions"]) == len(edge_manifest["partitions"]) == len(row_manifest["partitions"]) == expected_files and len(context_manifest["chunks"]) == int(cfg["expected_context_chunk_count"]) else "FAIL",
        "map_quality_check_remediated_by_frozen_policy": "PASS" if pointers["map_gate"]["status"] == "CAUTION_TRAJECTORY_BUILD_MAP_MATCH_PRODUCTION_GATE" and all(value == "PASS" for value in map_gate_nonquality_check) and pointers["abstention"]["status"] == "PASS_TRAJECTORY_BUILD_QUALITY_CHECK1_OFF_NETWORK_ABSTENTION_POLICY" and summaries["abstention"]["abstain_trip_count"] == expected_abstain_trips and summaries["abstention"]["uncertain_trip_count"] == 0 else "FAIL",
        "historical_osm_way_cache": "PASS" if pointers["tag_cache"]["status"] == "PASS_TRAJECTORY_BUILD_OSM_WAY_TAG_CACHE" and all(value == "PASS" for value in summaries["tag_cache"]["checks"].values()) else "FAIL",
        "edge_cardinality": "PASS" if summaries["edge"]["point_count"] == summaries["map_gate"]["status_counts"]["matched_with_edge"] else "FAIL",
        "context_conservation": "PASS" if summaries["context"]["checks"]["global_row_conservation"] == "PASS" and summaries["context"]["row_context_count"] + summaries["map_gate"]["status_counts"]["request_error"] + summaries["map_gate"]["status_counts"]["input_break_short_chunk"] + summaries["map_gate"]["status_counts"]["input_break_singleton"] == expected_rows else "FAIL",
        "stable_output_dtypes": summaries["row"]["checks"]["stable_output_dtypes"],
        "raw_lineage_hashes": "PASS" if raw_hash_samples > 0 and all(float(row["raw_hash_match_rate"]) == 1.0 for row in partition_rows) else "FAIL",
        "forbidden_feature_and_allowlist_gate": "PASS" if forbidden_hits == 0 and missing_allowlist == 0 else "FAIL",
        "semantic_reason_conservation": "PASS" if available_rows + unavailable_rows == expected_rows and sum(reason_counts.values()) == expected_rows and reason_counts.get("", 0) == available_rows else "FAIL",
        "quality_1_abstention_rows_null_verified": "PASS" if summaries["row"]["checks"]["quality_1_abstention"] == "PASS" and summaries["row"]["expected_abstain_trip_count"] == summaries["row"]["observed_abstain_trip_count"] == expected_abstain_trips and reason_counts[abstention_reason] == summaries["row"]["abstention_row_count"] else "FAIL",
        "memory_ceiling": "PASS" if max(peak_working_set_gb_by_stage.values()) <= float(cfg["memory_ceiling_gb"]) else "FAIL",
        "rnee_test_suite": "PASS" if test.returncode == 0 and f"{cfg['expected_test_count']} passed" in test.stdout else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["output_root"]) / f"{cfg['run_name_prefix']}_{stamp}"
    output.mkdir(parents=True)
    shutil.copy2(args.config, output / "run_config.yaml")
    pd.DataFrame(partition_rows).to_csv(output / "partition_release_audit.csv", index=False)
    pd.DataFrame([{"road_semantics_exclusion_reason": key, "row_count": value} for key, value in sorted(reason_counts.items())]).to_csv(output / "semantic_reason_counts.csv", index=False)
    status = f"{gate}_TRAJECTORY_BUILD_RELEASE_CANDIDATE_GATE"
    summary = {
        "experiment": "TRAJECTORY_BUILD", "stage": "release_candidate_gate", "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "release_candidate_authorized": gate == "PASS", "segment_split_authorized": gate == "PASS", "model_training_authorized": False,
        "source_file_count": expected_files, "row_count": expected_rows, "trip_count": summaries["map"]["trip_count"],
        "map_matched_coverage": summaries["map_gate"]["matched_coverage"], "edge_association_coverage": summaries["map_gate"]["edge_association_coverage"],
        "context_row_count": summaries["context"]["row_context_count"], "road_semantics_available_row_count": available_rows,
        "road_semantics_unavailable_row_count": unavailable_rows, "road_semantics_available_rate": available_rows / expected_rows,
        "road_semantics_exclusion_reason_counts": dict(reason_counts), "quality_1_abstain_trip_count": expected_abstain_trips,
        "quality_1_abstention_row_count": reason_counts[abstention_reason], "raw_hash_sample_count": raw_hash_samples,
        "forbidden_feature_hit_count": forbidden_hits, "peak_working_set_gb_by_stage": peak_working_set_gb_by_stage,
        "maximum_peak_working_set_gb": max(peak_working_set_gb_by_stage.values()),
        "test_output": test.stdout.strip(), "partition_release_audit": str(output / "partition_release_audit.csv"),
        "semantic_reason_counts": str(output / "semantic_reason_counts.csv"),
        "evidence": {name: pointer["summary"] for name, pointer in pointers.items()}, "output_directory": str(output),
    }
    dump(output / "trajectory_build_release_gate_summary.json", summary)
    report = output / "TRAJECTORY_BUILD_RELEASE_CANDIDATE_GATE.md"
    report.write_text("# TRAJECTORY_BUILD release-candidate gate\n\n" + "\n".join([
        f"- Status: `{status}`", f"- Files / rows / trips: {expected_files} / {expected_rows:,} / {summary['trip_count']:,}",
        f"- Road semantics available: {available_rows:,} ({summary['road_semantics_available_rate']:.4%}); unavailable: {unavailable_rows:,}.",
        f"- Quality check 1 abstention: {expected_abstain_trips} trips / {reason_counts[abstention_reason]:,} preserved rows with model-facing road semantics null.",
        f"- Raw lineage hashes: {raw_hash_samples:,} sampled rows at 100% match; forbidden feature hits: {forbidden_hits}.",
        f"- Maximum stage working set: {summary['maximum_peak_working_set_gb']:.3f} GB; test suite: `{test.stdout.strip()}`.",
        f"- SEGMENT_SPLIT authorized: `{summary['segment_split_authorized']}`. Model training remains blocked until SEGMENT_SPLIT leakage-controlled segment and split gates pass.",
    ]) + "\n", encoding="utf-8")
    latest = {"status": status, "release_candidate_authorized": gate == "PASS", "segment_split_authorized": gate == "PASS", "summary": str(output / "trajectory_build_release_gate_summary.json"), "report": str(report), "partition_release_audit": str(output / "partition_release_audit.csv"), "semantic_reason_counts": str(output / "semantic_reason_counts.csv"), "output_directory": str(output)}
    dump(Path(cfg["output_root"]) / f"{cfg['latest_prefix']}_latest_pointer.json", latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
