#!/usr/bin/env python3
"""Consolidate the END_TO_END two-file end-to-end quality gate."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/end_to_end_gate.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_pointer = load_json(Path(config["raw_pointer"]))
    qa_pointer = load_json(Path(config["qa_pointer"]))
    quality_check_pointer = load_json(Path(config["quality_1_pointer"]))
    edge_pointer = load_json(Path(config["edge_pointer"]))
    context_pointer = load_json(Path(config["context_pointer"]))
    rows_pointer = load_json(Path(config["rows_pointer"]))

    raw = load_json(Path(raw_pointer["summary"]))
    qa = load_json(Path(qa_pointer["summary"]))
    quality_check = load_json(Path(quality_check_pointer["summary"]))
    edge = load_json(Path(edge_pointer["summary"]))
    context = load_json(Path(context_pointer["summary"]))
    rows = load_json(Path(rows_pointer["summary"]))
    selection = pd.read_parquet(raw_pointer["selection"])

    source_count = int(selection["source_file"].nunique())
    snapshot_ids = sorted(selection["osm_snapshot_id"].unique().tolist())
    automated_checks = qa["profiles"][0]["checks"]
    automated_has_fail = "FAIL" in automated_checks.values()
    memory_policy = config["trajectory_build_memory_policy"]
    observed_peak = float(config["observed_context_peak_working_set_gb"])
    checks = {
        "two_source_files": "PASS" if source_count == int(config["expected_source_file_count"]) else "FAIL",
        "row_cardinality": "PASS" if raw["output_point_count"] == rows["output_row_count"] == int(config["expected_row_count"]) else "FAIL",
        "trip_cardinality": "PASS" if raw["pilot_trip_count"] == context["trip_count"] == int(config["expected_trip_count"]) else "FAIL",
        "historical_snapshot_allowlist": "PASS" if snapshot_ids == sorted(config["allowed_snapshot_ids"]) else "FAIL",
        "map_automated_no_fail": "PASS" if not automated_has_fail else "FAIL",
        "quality_1_manual_gate": "PASS" if str(quality_check["status"]).startswith("PASS_") else "FAIL",
        "quality_check2_not_used": "PASS" if quality_check["quality_check_2_used"] is False else "FAIL",
        "edge_semantics": "PASS" if str(edge["status"]).startswith("PASS_EDGE_SEMANTICS") else "FAIL",
        "context_semantics": "PASS" if str(context["status"]).startswith("PASS_CONTEXT_SEMANTICS") else "FAIL",
        "row_assembly": "PASS" if str(rows["status"]).startswith("PASS_ROW_ASSEMBLY") else "FAIL",
        "row_conservation": "PASS" if rows["output_input_row_ratio"] == 1.0 and rows["duplicate_row_id_count"] == 0 else "FAIL",
        "raw_lineage_hash": "PASS" if rows["raw_hash_match_rate"] == 1.0 else "FAIL",
        "forbidden_feature_control": "PASS" if rows["forbidden_feature_hit_count"] == 0 else "FAIL",
        "legacy_eved_not_used": "PASS" if raw["selection_uses_energy_or_eved_fields"] is False and qa["profile_selection_uses_energy_target_model_or_eved"] is False else "FAIL",
    }
    quality_gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    chunking_required = bool(memory_policy["require_chunked_context_processing"] and observed_peak > float(memory_policy["maximum_working_set_gb"]))
    if quality_gate == "PASS" and chunking_required:
        status = "PASS_END_TO_END_WITH_TRAJECTORY_BUILD_CHUNKING_REQUIRED"
        trajectory_build_authorized = False
        stop_reason = "END_TO_END quality gates passed, but the two-file context stage peaked above the TRAJECTORY_BUILD memory ceiling; chunked/resumable processing must pass a dry run before the full build."
    elif quality_gate == "PASS":
        status = "PASS_END_TO_END_AUTHORIZE_TRAJECTORY_BUILD"
        trajectory_build_authorized = True
        stop_reason = None
    else:
        status = "FAIL_END_TO_END_BLOCK_TRAJECTORY_BUILD"
        trajectory_build_authorized = False
        stop_reason = "At least one END_TO_END end-to-end quality gate failed."

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    output = output_root / f"end_to_end_gate_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "run_config.yaml")
    summary = {
        "experiment": "END_TO_END",
        "stage": "two_file_end_to_end_gate",
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_gate": quality_gate,
        "checks": checks,
        "source_file_count": source_count,
        "row_count": int(rows["output_row_count"]),
        "trip_count": int(raw["pilot_trip_count"]),
        "snapshot_ids": snapshot_ids,
        "map_automated_gate": qa["selected_profile_automated_gate"],
        "quality_1_incorrect_rate": quality_check["incorrect_rate"],
        "quality_1_incorrect_rate_pass_max": quality_check["incorrect_rate_pass_max"],
        "edge_semantic_coverage_all_rows": rows["edge_semantic_join_coverage_all_rows"],
        "context_semantic_coverage_all_rows": rows["context_semantic_join_coverage_all_rows"],
        "observed_context_peak_working_set_gb": observed_peak,
        "trajectory_build_memory_ceiling_gb": float(memory_policy["maximum_working_set_gb"]),
        "trajectory_build_chunking_required": chunking_required,
        "trajectory_build_require_resume_manifest": bool(memory_policy["require_resume_manifest"]),
        "trajectory_build_require_per_chunk_qa": bool(memory_policy["require_per_chunk_qa"]),
        "trajectory_build_authorized": trajectory_build_authorized,
        "stop_reason": stop_reason,
        "output_directory": str(output),
    }
    summary_path = output / "end_to_end_end_to_end_summary.json"
    report_path = output / "END_TO_END_END_TO_END_GATE.md"
    write_json(summary_path, summary)
    report_path.write_text(
        "# END_TO_END two-file end-to-end gate\n\n"
        f"- Status: `{status}`\n"
        f"- Quality checks: {sum(value == 'PASS' for value in checks.values())}/{len(checks)} PASS\n"
        f"- Files / trips / rows: {source_count} / {raw['pilot_trip_count']:,} / {rows['output_row_count']:,}\n"
        f"- Quality check 1 incorrect rate: {quality_check['incorrect_rate']:.1%} (inclusive maximum {quality_check['incorrect_rate_pass_max']:.1%})\n"
        f"- Edge/context row coverage: {rows['edge_semantic_join_coverage_all_rows']:.2%} / {rows['context_semantic_join_coverage_all_rows']:.2%}\n"
        f"- Observed context peak working set: {observed_peak:.2f} GB; TRAJECTORY_BUILD ceiling: {memory_policy['maximum_working_set_gb']:.2f} GB\n"
        f"- TRAJECTORY_BUILD authorized: `{trajectory_build_authorized}`\n"
        f"- Stop reason: {stop_reason or 'none'}\n",
        encoding="utf-8",
    )
    shutil.copy2(summary_path, output_root / "end_to_end_latest_end_to_end_summary.json")
    shutil.copy2(report_path, output_root / "end_to_end_latest_end_to_end_report.md")
    write_json(output_root / "end_to_end_latest_end_to_end_pointer.json", {
        "status": status,
        "summary": str(summary_path),
        "report": str(report_path),
        "output_directory": str(output),
        "trajectory_build_authorized": trajectory_build_authorized,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if quality_gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
