#!/usr/bin/env python3
"""Verify chunked TRAJECTORY_BUILD dry-run outputs against accepted END_TO_END context outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def logical_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def compare_frame(expected: pd.DataFrame, actual: pd.DataFrame, sort_keys: list[str]) -> tuple[bool, str | None, str, str]:
    if list(expected.columns) != list(actual.columns):
        return False, "column order/schema mismatch", logical_hash(expected), logical_hash(actual)
    expected = expected.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    actual = actual.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    expected_hash = logical_hash(expected)
    actual_hash = logical_hash(actual)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_exact=True)
        return True, None, expected_hash, actual_hash
    except AssertionError as error:
        return False, str(error)[:1000], expected_hash, actual_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-pointer",
        type=Path,
        default=Path(r"results/trajectory_build_context_chunk_dry_run_latest_pointer.json"),
    )
    parser.add_argument(
        "--baseline-pointer",
        type=Path,
        default=Path(r"results/end_to_end_context_latest_pointer.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunk_pointer = load_json(args.chunk_pointer)
    baseline_pointer = load_json(args.baseline_pointer)
    manifest = load_json(Path(chunk_pointer["manifest"]))
    chunk_summary = load_json(Path(chunk_pointer["summary"]))
    if not str(chunk_summary["status"]).startswith("PASS_"):
        raise RuntimeError("Chunked dry run has not passed.")

    baseline_rows = pd.read_parquet(baseline_pointer["row_context"])
    baseline_objects = pd.read_parquet(baseline_pointer["objects"])
    baseline_object_exposure = pd.read_parquet(baseline_pointer["object_exposure"])
    baseline_trip_exposure = pd.read_parquet(baseline_pointer["trip_exposure"])
    actual_objects = pd.read_parquet(chunk_pointer["context_objects"])

    object_ok, object_error, object_expected_hash, object_actual_hash = compare_frame(
        baseline_objects,
        actual_objects,
        ["osm_snapshot_id", "object_layer", "element_type", "osm_id"],
    )
    results = []
    for chunk in manifest["chunks"]:
        if chunk["status"] != "PASS":
            raise RuntimeError(f"Manifest contains non-PASS chunk: {chunk['chunk_id']}")
        trip_ids = set(chunk["pilot_trip_ids"])
        expected_rows = baseline_rows[baseline_rows["pilot_trip_id"].isin(trip_ids)].copy()
        actual_rows = pd.read_parquet(chunk["outputs"]["row_context"])
        row_ok, row_error, row_expected_hash, row_actual_hash = compare_frame(
            expected_rows, actual_rows, ["pilot_trip_id", "point_index"]
        )
        expected_objects = baseline_object_exposure[baseline_object_exposure["pilot_trip_id"].isin(trip_ids)].copy()
        actual_object = pd.read_parquet(chunk["outputs"]["object_exposure"])
        exposure_ok, exposure_error, exposure_expected_hash, exposure_actual_hash = compare_frame(
            expected_objects,
            actual_object,
            ["pilot_trip_id", "scale_m", "object_layer", "object_uid"],
        )
        expected_trips = baseline_trip_exposure[baseline_trip_exposure["pilot_trip_id"].isin(trip_ids)].copy()
        actual_trip = pd.read_parquet(chunk["outputs"]["trip_exposure"])
        trip_ok, trip_error, trip_expected_hash, trip_actual_hash = compare_frame(
            expected_trips,
            actual_trip,
            ["pilot_trip_id", "scale_m", "object_layer"],
        )
        results.append({
            "chunk_id": chunk["chunk_id"],
            "row_count": int(len(actual_rows)),
            "row_equivalent": row_ok,
            "object_exposure_equivalent": exposure_ok,
            "trip_exposure_equivalent": trip_ok,
            "row_error": row_error,
            "object_exposure_error": exposure_error,
            "trip_exposure_error": trip_error,
            "logical_hashes": {
                "row_expected": row_expected_hash,
                "row_actual": row_actual_hash,
                "object_exposure_expected": exposure_expected_hash,
                "object_exposure_actual": exposure_actual_hash,
                "trip_exposure_expected": trip_expected_hash,
                "trip_exposure_actual": trip_actual_hash,
            },
        })

    checks = {
        "context_objects_exact": "PASS" if object_ok else "FAIL",
        "all_row_chunks_exact": "PASS" if all(item["row_equivalent"] for item in results) else "FAIL",
        "all_object_exposure_chunks_exact": "PASS" if all(item["object_exposure_equivalent"] for item in results) else "FAIL",
        "all_trip_exposure_chunks_exact": "PASS" if all(item["trip_exposure_equivalent"] for item in results) else "FAIL",
        "row_count_equal": "PASS" if sum(item["row_count"] for item in results) == len(baseline_rows) else "FAIL",
        "resume_noop_verified": "PASS" if chunk_summary.get("resume_noop_verified") is True else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    output = Path(chunk_pointer["output_directory"]) / "equivalence_audit"
    output.mkdir(exist_ok=True)
    summary = {
        "experiment": "TRAJECTORY_BUILD-DRYRUN",
        "stage": "chunked_context_equivalence_audit",
        "status": f"{gate}_TRAJECTORY_BUILD_CHUNKED_CONTEXT_EQUIVALENCE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "baseline_row_count": int(len(baseline_rows)),
        "chunk_row_count": int(sum(item["row_count"] for item in results)),
        "chunk_count": len(results),
        "context_object_error": object_error,
        "context_object_logical_hash_expected": object_expected_hash,
        "context_object_logical_hash_actual": object_actual_hash,
        "chunk_results": results,
        "output_directory": str(output),
    }
    summary_path = output / "equivalence_summary.json"
    report_path = output / "EQUIVALENCE_REPORT.md"
    write_json(summary_path, summary)
    report_path.write_text(
        "# TRAJECTORY_BUILD chunked-context equivalence audit\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Chunks: {len(results)}\n"
        f"- Baseline/chunk rows: {len(baseline_rows):,} / {sum(item['row_count'] for item in results):,}\n"
        f"- Context objects exact: `{checks['context_objects_exact']}`\n"
        f"- Row chunks exact: `{checks['all_row_chunks_exact']}`\n"
        f"- Object/trip exposure exact: `{checks['all_object_exposure_chunks_exact']}` / `{checks['all_trip_exposure_chunks_exact']}`\n"
        f"- Resume no-op: `{checks['resume_noop_verified']}`\n",
        encoding="utf-8",
    )
    latest = args.chunk_pointer.parent / "trajectory_build_context_chunk_dry_run_latest_equivalence_pointer.json"
    write_json(latest, {
        "status": summary["status"],
        "summary": str(summary_path),
        "report": str(report_path),
        "output_directory": str(output),
    })
    print(json.dumps({key: value for key, value in summary.items() if key != "chunk_results"}, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
