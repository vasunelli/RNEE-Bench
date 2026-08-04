#!/usr/bin/env python3
"""Independently validate the SEGMENT_SPLIT six-family split release and G8."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ACTIVE_ROLES = ("train", "validation", "calibration", "test")
FAMILIES = (
    "random_trip_blocked",
    "cold_vehicle",
    "cold_trip",
    "cold_month",
    "cold_spatial",
    "cold_functional_road_class",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_block(value: Any) -> tuple[int, int] | None:
    if pd.isna(value) or "_" not in str(value):
        return None
    left, right = str(value).split("_", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def block_within_buffer(candidate: Any, test_blocks: set[str], radius: int) -> bool:
    parsed = parse_block(candidate)
    if parsed is None:
        return False
    x, y = parsed
    for value in test_blocks:
        test = parse_block(value)
        if test is not None and max(abs(x - test[0]), abs(y - test[1])) <= radius:
            return True
    return False


def check(checks: list[dict[str, Any]], family: str, name: str, passed: bool, evidence: Any) -> None:
    checks.append({"family": family, "check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def pair_overlap(frame: pd.DataFrame, column: str, left: str, right: str) -> int:
    a = set(frame.loc[frame["split_role"].eq(left), column].dropna().astype(str))
    b = set(frame.loc[frame["split_role"].eq(right), column].dropna().astype(str))
    return len(a & b)


def validate_family(
    family: str,
    assignments: pd.DataFrame,
    membership: pd.DataFrame,
    input_ids: set[str],
    config: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    active = assignments[assignments["split_role"].isin(ACTIVE_ROLES)].copy()
    check(checks, family, "assignment_row_conservation", len(assignments) == len(input_ids), {"rows": len(assignments), "expected": len(input_ids)})
    check(checks, family, "assignment_segment_id_unique", assignments["segment_id"].is_unique, int(assignments["segment_id"].duplicated().sum()))
    assignment_ids = set(assignments["segment_id"].astype(str))
    check(checks, family, "assignment_segment_id_exact_set", assignment_ids == input_ids, {"missing": len(input_ids - assignment_ids), "extra": len(assignment_ids - input_ids)})
    check(checks, family, "all_four_active_roles_present", set(ACTIVE_ROLES).issubset(set(active["split_role"])), sorted(set(active["split_role"])))
    expected_membership = active[["segment_id", "split_role"]].sort_values(["segment_id", "split_role"]).reset_index(drop=True)
    actual_membership = membership.sort_values(["segment_id", "split_role"]).reset_index(drop=True)
    check(checks, family, "membership_exactly_matches_active_assignments", expected_membership.equals(actual_membership), {"expected": len(expected_membership), "actual": len(actual_membership)})
    check(checks, family, "membership_schema", list(membership.columns) == contract["model_membership_columns"], list(membership.columns))
    proposed_features = set(membership.columns) - set(contract["join_only_columns"]) - set(contract["role_only_columns"])
    forbidden_hits = sorted(
        column for column in proposed_features
        if column in set(contract["forbidden_model_feature_exact"])
        or any(token.lower() in column.lower() for token in contract["forbidden_model_feature_substrings"])
    )
    check(checks, family, "forbidden_model_feature_hits_zero", not forbidden_hits and not proposed_features, {"proposed_features": sorted(proposed_features), "forbidden_hits": forbidden_hits})
    for left, right in combinations(ACTIVE_ROLES, 2):
        check(checks, family, f"segment_overlap_{left}_vs_{right}", pair_overlap(active, "segment_id", left, right) == 0, pair_overlap(active, "segment_id", left, right))
        check(checks, family, f"trip_overlap_{left}_vs_{right}", pair_overlap(active, "trip_uid", left, right) == 0, pair_overlap(active, "trip_uid", left, right))
    if family == "cold_vehicle":
        for left, right in combinations(ACTIVE_ROLES, 2):
            overlap = pair_overlap(active, "VehId", left, right)
            check(checks, family, f"vehicle_overlap_{left}_vs_{right}", overlap == 0, overlap)
    if family == "cold_month":
        month_order = {role: sorted(active.loc[active["split_role"].eq(role), "actual_month"].astype(str).unique()) for role in ACTIVE_ROLES}
        chronological = all(month_order[role] for role in ACTIVE_ROLES)
        chronological = chronological and max(month_order["train"]) < min(month_order["validation"]) <= max(month_order["validation"]) < min(month_order["calibration"]) <= max(month_order["calibration"]) < min(month_order["test"])
        check(checks, family, "chronological_boundaries", chronological, month_order)
    if family == "cold_spatial":
        family_cfg = config["splits"][family]
        test_blocks = set(active.loc[active["split_role"].eq("test"), "spatial_block_id"].dropna().astype(str))
        train_like = active[active["split_role"].isin(("train", "validation", "calibration"))]
        crossings = int(train_like["spatial_block_id"].map(lambda value: block_within_buffer(value, test_blocks, int(family_cfg["buffer_cells"]))).sum())
        check(checks, family, "spatial_buffer_crossing_zero", crossings == 0, {"crossing_segments": crossings, "test_blocks": sorted(test_blocks)})
        test_trips = set(active.loc[active["split_role"].eq("test"), "trip_uid"].astype(str))
        overlap = int(train_like["trip_uid"].astype(str).isin(test_trips).sum())
        check(checks, family, "test_trip_absent_from_train_validation_calibration", overlap == 0, overlap)
    if family == "cold_functional_road_class":
        heldout = str(config["splits"][family]["heldout_class"])
        test_classes = set(active.loc[active["split_role"].eq("test"), "functional_road_class_dominant"].dropna().astype(str))
        train_like_classes = set(active.loc[active["split_role"].isin(("train", "validation", "calibration")), "functional_road_class_dominant"].dropna().astype(str))
        check(checks, family, "test_contains_only_heldout_functional_class", test_classes == {heldout}, sorted(test_classes))
        check(checks, family, "heldout_functional_class_absent_from_train_validation_calibration", heldout not in train_like_classes, sorted(train_like_classes))
    test = active[active["split_role"].eq("test")]
    vehicle_counts = test["VehId"].value_counts()
    maximum_share = float(vehicle_counts.iloc[0] / len(test)) if len(test) else 1.0
    thresholds = config["support_thresholds"]
    support = {
        "segments": int(len(test)),
        "vehicles": int(test["VehId"].nunique()),
        "trips": int(test["trip_uid"].nunique()),
        "maximum_vehicle_segment_share": maximum_share,
        "engine_type_counts": {str(key): int(value) for key, value in test["engine_type"].value_counts().items()},
    }
    support_pass = (
        support["segments"] >= int(thresholds["minimum_test_segments"])
        and support["vehicles"] >= int(thresholds["minimum_test_vehicles"])
        and support["trips"] >= int(thresholds["minimum_test_trips"])
        and support["maximum_vehicle_segment_share"] <= float(thresholds["maximum_single_vehicle_test_share"])
    )
    check(checks, family, "main_test_support", support_pass, support)
    input_engine_types = set(assignments["engine_type"].dropna().astype(str))
    missing_test_engine_types = sorted(input_engine_types - set(support["engine_type_counts"]))
    warnings = []
    if missing_test_engine_types:
        warnings.append(f"Target-specific evaluation is unsupported for test powertrains: {', '.join(missing_test_engine_types)}")
    return checks, {"family": family, "benchmark_tier": "main" if support_pass else "appendix", "test_support": support, "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    build_pointer = load_json(Path(config["output_root"]) / f"{config['latest_prefix']}_build_latest_pointer.json")
    run_dir = args.run_dir.resolve() if args.run_dir else Path(build_pointer["output_directory"])
    manifest = load_json(run_dir / "split_manifest.json")
    if manifest["status"] != "PASS_SEGMENT_SPLIT_SPLIT_BUILD":
        raise RuntimeError("Split build did not pass its construction gate.")
    contract = load_json(Path(manifest["membership_contract"]))
    segment_pointer = load_json(Path(config["segment_pointer"]))
    segments_path = Path(segment_pointer["segments_prediction_usable"])
    segments = pd.read_parquet(segments_path, columns=["segment_id", "prediction_usable", "unified_scalar_target_released"])
    input_ids = set(segments["segment_id"].astype(str))
    global_checks: list[dict[str, Any]] = []
    check(global_checks, "GLOBAL", "frozen_input_sha256", sha256_file(segments_path) == manifest["input_fingerprint"]["segments_prediction_usable_sha256"], sha256_file(segments_path))
    check(global_checks, "GLOBAL", "prediction_usable_input_only", bool(segments["prediction_usable"].all()), int((~segments["prediction_usable"]).sum()))
    check(global_checks, "GLOBAL", "unified_scalar_target_prohibited", not bool(segments["unified_scalar_target_released"].any()), int(segments["unified_scalar_target_released"].sum()))
    manifest_families = {item["family"]: item for item in manifest["families"]}
    check(global_checks, "GLOBAL", "six_required_families_present", set(manifest_families) == set(FAMILIES), sorted(manifest_families))
    all_checks = list(global_checks)
    family_gates = []
    for family in FAMILIES:
        item = manifest_families[family]
        assignment_path = Path(item["assignment_path"])
        membership_path = Path(item["membership_path"])
        assignments = pd.read_parquet(assignment_path)
        membership = pd.read_parquet(membership_path)
        checks, family_gate = validate_family(family, assignments, membership, input_ids, config, contract)
        all_checks.extend(checks)
        family_gate["failed_checks"] = [row["check"] for row in checks if row["status"] == "FAIL"]
        family_gate["leakage_status"] = "PASS" if not family_gate["failed_checks"] else "FAIL"
        family_gates.append(family_gate)
    failed = [row for row in all_checks if row["status"] == "FAIL"]
    warnings = [{"family": item["family"], "warning": warning} for item in family_gates for warning in item["warnings"]]
    status = "PASS_BENCHMARK_SPLITS" if not failed else "FAIL_BENCHMARK_SPLITS"
    all_stats_rows = []
    split_cards: dict[str, str] = {}
    family_gate_map = {item["family"]: item for item in family_gates}
    for family in FAMILIES:
        family_item = manifest_families[family]
        gate_item = family_gate_map[family]
        for row in family_item["role_stats"]:
            all_stats_rows.append({
                "family": family,
                **{key: value for key, value in row.items() if key != "engine_type_counts"},
                "engine_type_counts": json.dumps(row["engine_type_counts"], ensure_ascii=False, sort_keys=True),
            })
        card_lines = [
            f"# SEGMENT_SPLIT split card: {family}",
            "",
            f"- Benchmark tier: `{gate_item['benchmark_tier']}`",
            f"- Leakage status: `{gate_item['leakage_status']}`",
            f"- Design: `{json.dumps(family_item['design'], ensure_ascii=False, sort_keys=True)}`",
            "- Rich assignments retain identifiers only for leakage checks; model membership contains no model feature columns.",
            "",
            "## Role support",
            "",
            "| Role | Segments | Vehicles | Trips | Max vehicle share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in family_item["role_stats"]:
            card_lines.append(f"| {row['split_role']} | {row['segments']:,} | {row['vehicles']:,} | {row['trips']:,} | {row['maximum_vehicle_segment_share']:.3%} |")
        card_lines.extend(["", "## Target-channel support by powertrain", ""])
        for row in family_item["role_stats"]:
            if row["split_role"] in ACTIVE_ROLES:
                card_lines.append(f"- `{row['split_role']}`: {json.dumps(row['engine_type_counts'], ensure_ascii=False, sort_keys=True)}")
        card_lines.append("")
        card_path = run_dir / family / "SPLIT_CARD.md"
        card_path.write_text("\n".join(card_lines), encoding="utf-8")
        split_cards[family] = str(card_path)
    all_stats_path = run_dir / "all_split_stats.csv"
    pd.DataFrame(all_stats_rows).to_csv(all_stats_path, index=False)
    checks_frame = pd.DataFrame([{**row, "evidence": json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True)} for row in all_checks])
    checks_path = run_dir / "all_leakage_checks.csv"
    checks_frame.to_csv(checks_path, index=False)
    gate_summary = {
        "experiment": "SEGMENT_SPLIT",
        "stage": "G8_leakage_and_support_validation",
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_total": len(all_checks),
        "checks_passed": len(all_checks) - len(failed),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "warnings": warnings,
        "family_gates": family_gates,
        "split_manifest": str(run_dir / "split_manifest.json"),
        "all_leakage_checks": str(checks_path),
        "all_split_stats": str(all_stats_path),
        "split_cards": split_cards,
        "output_directory": str(run_dir),
        "model_training_authorized": False,
        "training_block_reason": "DIFFERENTIAL_AUDIT G9 is still required even after SEGMENT_SPLIT G8 passes.",
    }
    summary_path = run_dir / "split_gate_summary.json"
    write_json(summary_path, gate_summary)
    report_lines = [
        "# SEGMENT_SPLIT G8 leakage-controlled split gate",
        "",
        f"- Status: `{status}`",
        f"- Checks: {gate_summary['checks_passed']} PASS / {gate_summary['checks_failed']} FAIL",
        f"- Frozen prediction-usable input: {len(segments):,} segments",
        "- Model training remains blocked pending DIFFERENTIAL_AUDIT G9.",
        "",
        "## Family support",
        "",
    ]
    for item in family_gates:
        support = item["test_support"]
        report_lines.append(f"- `{item['family']}` ({item['benchmark_tier']}): {support['segments']:,} segments, {support['vehicles']:,} vehicles, {support['trips']:,} trips, max vehicle share {support['maximum_vehicle_segment_share']:.3%}; leakage {item['leakage_status']}.")
        for warning in item["warnings"]:
            report_lines.append(f"  - Target-support warning: {warning}.")
    report_lines.extend(["", "## Boundary", "", "Identifiers in rich assignment files are grouping-only. Model membership files contain only `segment_id` and `split_role`; both are excluded from model features.", ""])
    report_path = run_dir / "SEGMENT_SPLIT_SPLIT_GATE.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    pointer = {
        "status": status,
        "summary": str(summary_path),
        "report": str(report_path),
        "manifest": str(run_dir / "split_manifest.json"),
        "checks": str(checks_path),
        "stats": str(all_stats_path),
        "split_cards": split_cards,
        "output_directory": str(run_dir),
        "model_training_authorized": False,
    }
    write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", pointer)
    print(json.dumps(gate_summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS_BENCHMARK_SPLITS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
