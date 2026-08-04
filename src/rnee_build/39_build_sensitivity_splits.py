#!/usr/bin/env python3
"""Build six-family split memberships for an SEGMENT_SPLIT sensitivity segment release."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_primary_split_builder", HERE / "36_build_splits.py")
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pointer = BASE.load_json(Path(config["segment_pointer"]))
    if pointer["status"] != config["required_segment_status"]:
        raise RuntimeError(f"Sensitivity segment gate mismatch: {pointer['status']}")
    segment_summary = BASE.load_json(Path(pointer["summary"]))
    if segment_summary.get("resume_noop_verified") is not True:
        raise RuntimeError("Sensitivity segment no-op resume has not been verified.")
    segments_path = Path(pointer["segments_prediction_usable"])
    fingerprint = {
        "config_sha256": BASE.sha256_file(config_path),
        "script_sha256": BASE.sha256_file(Path(__file__)),
        "frozen_primary_split_builder_sha256": BASE.sha256_file(HERE / "36_build_splits.py"),
        "segment_summary_sha256": BASE.sha256_file(Path(pointer["summary"])),
        "segments_prediction_usable_sha256": BASE.sha256_file(segments_path),
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.json"
    summary_path = output / "split_build_summary.json"
    if manifest_path.exists():
        manifest = BASE.load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
        if summary_path.exists() and manifest.get("status") == config["build_pass_status"]:
            summary = BASE.load_json(summary_path)
            summary["resume_noop_verified"] = True
            summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started
            summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
            BASE.write_json(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
    shutil.copy2(config_path, output / "run_config.yaml")
    frame = pd.read_parquet(segments_path)
    missing = sorted(set(BASE.AUDIT_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Sensitivity segment input misses split columns: {missing}")
    if not frame["segment_id"].is_unique or not frame["prediction_usable"].all():
        raise RuntimeError("Split input must contain unique, prediction-usable segments only.")
    family_summaries = [BASE.build_family(frame, family, config, output) for family in BASE.FAMILIES]
    contract = {
        "schema_version": 1,
        "purpose": "Sensitivity split membership only; no model features are released by SEGMENT_SPLIT.",
        "model_membership_columns": ["segment_id", "split_role"],
        "join_only_columns": ["segment_id"],
        "role_only_columns": ["split_role"],
        "model_feature_columns": [],
        "audit_only_assignment_columns": ["split_family", "split_role", "disposition"] + BASE.AUDIT_COLUMNS,
        "forbidden_model_feature_exact": config["forbidden_model_feature_exact"],
        "forbidden_model_feature_substrings": config["forbidden_model_feature_substrings"],
    }
    contract_path = output / "model_membership_contract.json"
    BASE.write_json(contract_path, contract)
    manifest = {
        "schema_version": 1,
        "experiment": "SEGMENT_SPLIT",
        "stage": config["stage"],
        "status": config["build_pass_status"],
        "input_fingerprint": fingerprint,
        "families": family_summaries,
        "membership_contract": str(contract_path),
        "membership_contract_sha256": BASE.sha256_file(contract_path),
    }
    BASE.write_json(manifest_path, manifest)
    summary = {
        "experiment": "SEGMENT_SPLIT",
        "stage": config["stage"],
        "status": config["build_pass_status"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_segments": int(len(frame)),
        "input_vehicles": int(frame["VehId"].nunique()),
        "input_trips": int(frame["trip_uid"].nunique()),
        "families": family_summaries,
        "manifest": str(manifest_path),
        "membership_contract": str(contract_path),
        "output_directory": str(output),
        "resume_noop_verified": False,
        "model_training_authorized": False,
    }
    BASE.write_json(summary_path, summary)
    build_pointer = {
        "status": summary["status"],
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "output_directory": str(output),
        "model_training_authorized": False,
    }
    BASE.write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_build_latest_pointer.json", build_pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
