#!/usr/bin/env python3
"""Validate Quality check 1 labels and freeze the TRAJECTORY_BUILD road-semantic abstention policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


KEYS = ["source_file", "VehId", "Trip", "pilot_trip_id"]


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_quality_check(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    required = {*KEYS, "validation_category", "quality_rating"}
    missing = sorted(required - set(frame.columns))
    ratings = frame["quality_rating"].fillna("").astype(str).str.strip().str.lower()
    allowed = set(config["allowed_ratings"])
    blank_count = int(ratings.eq("").sum())
    invalid_count = int((~ratings.isin(allowed) & ratings.ne("")).sum())
    duplicate_key_count = int(frame.duplicated(KEYS).sum())
    candidate = frame["validation_category"].eq("abstention_candidate")
    candidate_count = int(candidate.sum())
    candidate_abstain_count = int((candidate & ratings.eq("abstain")).sum())
    candidate_precision = candidate_abstain_count / candidate_count if candidate_count else 0.0
    uncertain_count = int(ratings.eq("uncertain").sum())
    checks = {
        "required_columns": "PASS" if not missing else "FAIL",
        "complete_quality_check": "PASS" if not config["require_complete_quality_check"] or blank_count == 0 else "FAIL",
        "allowed_ratings": "PASS" if invalid_count == 0 else "FAIL",
        "unique_trip_keys": "PASS" if duplicate_key_count == 0 else "FAIL",
        "candidate_precision": "PASS" if candidate_precision >= float(config["minimum_candidate_precision"]) else "FAIL",
        "zero_uncertain": "PASS" if not config["require_zero_uncertain"] or uncertain_count == 0 else "FAIL",
    }
    return {
        "ratings": ratings,
        "missing_columns": missing,
        "blank_count": blank_count,
        "invalid_count": invalid_count,
        "duplicate_key_count": duplicate_key_count,
        "candidate_count": candidate_count,
        "candidate_abstain_count": candidate_abstain_count,
        "candidate_precision": candidate_precision,
        "uncertain_count": uncertain_count,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/trajectory_build_off_network_abstention.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    quality_check_path = Path(config["quality_check_form"])
    frame = pd.read_csv(quality_check_path)
    validation = validate_quality_check(frame, config)
    ratings = validation.pop("ratings")
    gate = "PASS" if "FAIL" not in validation["checks"].values() else "FAIL"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    output = output_root / f"{config['run_name_prefix']}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "run_config.yaml")
    shutil.copy2(quality_check_path, output / "quality_1_off_network_validation_frozen.csv")

    policy = frame.loc[:, KEYS + ["validation_category"]].copy()
    policy["quality_rating"] = ratings
    policy["road_semantics_available"] = ~ratings.eq("abstain")
    policy["road_semantics_exclusion_reason"] = ""
    policy.loc[ratings.eq("abstain"), "road_semantics_exclusion_reason"] = "off_network_open_lot_quality_1_abstention"
    policy["policy_basis"] = "trajectory_build_quality_1_manual_resolution"
    policy.to_csv(output / "road_semantics_trip_policy.csv", index=False, encoding="utf-8-sig")

    abstain_count = int(ratings.eq("abstain").sum())
    retain_count = int(ratings.eq("retain").sum())
    summary = {
        "experiment": config["experiment"],
        "stage": "quality_1_off_network_abstention_finalization",
        "status": f"{gate}_TRAJECTORY_BUILD_QUALITY_CHECK1_OFF_NETWORK_ABSTENTION_POLICY",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_check_form": str(quality_check_path),
        "quality_check_form_sha256": sha256_file(quality_check_path),
        "quality_check_count": int(len(frame)),
        "abstain_trip_count": abstain_count,
        "retain_trip_count": retain_count,
        "uncertain_trip_count": validation["uncertain_count"],
        "candidate_count": validation["candidate_count"],
        "candidate_abstain_count": validation["candidate_abstain_count"],
        "candidate_precision": validation["candidate_precision"],
        "known_incorrect_miss_abstain_count": int(((frame["validation_category"] == "known_incorrect_not_candidate") & ratings.eq("abstain")).sum()),
        "rows_deleted": 0,
        "trips_deleted": 0,
        "raw_data_modified": False,
        "release_policy": "Preserve every VED row. Null derived road-network semantic model fields for the 40 manually abstained trips; retain map-match fields for audit only.",
        "checks": validation["checks"],
        "output_directory": str(output),
    }
    dump(output / "off_network_abstention_summary.json", summary)
    report = output / "TRAJECTORY_BUILD_OFF_NETWORK_ABSTENTION_POLICY.md"
    report.write_text(
        "# TRAJECTORY_BUILD Quality check 1 off-network abstention policy\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Complete labels: {len(frame) - validation['blank_count']}/{len(frame)}; invalid: {validation['invalid_count']}; uncertain: {validation['uncertain_count']}.\n"
        f"- Decisions: {abstain_count} abstain, {retain_count} retain.\n"
        f"- Candidate precision after fresh quality_check: {validation['candidate_abstain_count']}/{validation['candidate_count']} = {validation['candidate_precision']:.2%}.\n"
        f"- Known incorrect rule misses recovered by quality_check: {summary['known_incorrect_miss_abstain_count']}.\n"
        "- Release action: preserve all original rows and targets; withhold only derived road-network semantic model fields for abstained trips.\n"
        "- Scope: this is an TRAJECTORY_BUILD corpus-specific, Quality check 1-resolved policy. The compact geometric rule remains a screening rule, not an automatic universal deletion rule.\n",
        encoding="utf-8",
    )
    pointer = {
        "status": summary["status"],
        "summary": str(output / "off_network_abstention_summary.json"),
        "report": str(report),
        "trip_policy": str(output / "road_semantics_trip_policy.csv"),
        "output_directory": str(output),
    }
    dump(output_root / f"{config['latest_prefix']}_latest_pointer.json", pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

