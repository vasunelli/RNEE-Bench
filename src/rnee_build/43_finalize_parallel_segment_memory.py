#!/usr/bin/env python3
"""Correct consolidated memory evidence after bounded parallel partition work."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("segment_helpers", HERE / "35_build_segments.py")
SEG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SEG)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers-used", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    manifest = SEG.load_json(run_dir / "segment_manifest.json")
    summary_path = run_dir / "segment_summary.json"
    summary = SEG.load_json(summary_path)
    worker_peaks = [float(item["worker_peak_working_set_gb"]) for item in manifest["partitions"] if item.get("worker_peak_working_set_gb") is not None]
    if not worker_peaks:
        raise RuntimeError("No parallel-worker memory evidence is present.")
    summary["peak_working_set_gb"] = max(float(summary.get("peak_working_set_gb", 0.0)), max(worker_peaks))
    summary["parallel_partition_workers_used"] = int(args.workers_used)
    summary["parallel_worker_memory_observations"] = len(worker_peaks)
    summary["checks"]["memory_ceiling"] = "PASS" if summary["peak_working_set_gb"] <= float(config["memory_ceiling_gb"]) else "FAIL"
    if summary["checks"]["memory_ceiling"] == "FAIL":
        summary["status"] = summary["status"].replace("PASS_", "FAIL_", 1)
    SEG.write_json(summary_path, summary)
    if "distance_segment_m" in summary:
        report_name = "SEGMENT_SPLIT_DISTANCE_SEGMENT_GATE.md"
        title = "# SEGMENT_SPLIT distance-segment sensitivity gate"
        overlap_text = f"- Duplicate IDs / distance overlaps: {summary['duplicate_segment_id_count']} / {summary['distance_overlap_count']}"
    else:
        report_name = "SEGMENT_SPLIT_SEGMENT_GATE.md"
        title = "# SEGMENT_SPLIT non-overlapping segment gate"
        overlap_text = f"- Duplicate IDs / overlaps: {summary['duplicate_segment_id_count']} / {summary['overlap_count']}"
    report = run_dir / report_name
    report.write_text(title + "\n\n" + "\n".join([
        f"- Status: `{summary['status']}`",
        f"- Source rows / segments: {summary['source_row_count']:,} / {summary['segment_count']:,}",
        f"- QA-valid / prediction-usable: {summary['qa_valid_count']:,} / {summary['prediction_usable_count']:,}",
        overlap_text,
        f"- Peak per-worker working set: {summary['peak_working_set_gb']:.3f} GB across {args.workers_used} bounded workers",
        "- Targets remain separate powertrain-aware fuel-volume and battery-terminal channels; no unified scalar is released.",
    ]) + "\n", encoding="utf-8")
    latest = {
        "status": summary["status"],
        "summary": str(summary_path),
        "report": str(report),
        "manifest": str(run_dir / "segment_manifest.json"),
        "segments_all": summary["segments_all"],
        "segments_qa_valid": summary["segments_qa_valid"],
        "segments_prediction_usable": summary["segments_prediction_usable"],
        "output_directory": str(run_dir),
    }
    SEG.write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
