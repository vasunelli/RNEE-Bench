#!/usr/bin/env python3
"""Apply the frozen G8 leakage/support checks to a sensitivity split release."""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_primary_split_validator", HERE / "37_validate_leakage.py")
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    build_pointer = AUDIT.load_json(Path(config["output_root"]) / f"{config['latest_prefix']}_build_latest_pointer.json")
    run_dir = args.run_dir.resolve() if args.run_dir else Path(build_pointer["output_directory"])
    manifest = AUDIT.load_json(run_dir / "split_manifest.json")
    if manifest["status"] != config["build_pass_status"]:
        raise RuntimeError("Sensitivity split build did not pass.")
    contract = AUDIT.load_json(Path(manifest["membership_contract"]))
    segment_pointer = AUDIT.load_json(Path(config["segment_pointer"]))
    segments_path = Path(segment_pointer["segments_prediction_usable"])
    segments = pd.read_parquet(segments_path, columns=["segment_id", "prediction_usable", "unified_scalar_target_released"])
    input_ids = set(segments["segment_id"].astype(str))
    global_checks: list[dict] = []
    AUDIT.check(global_checks, "GLOBAL", "frozen_input_sha256", AUDIT.sha256_file(segments_path) == manifest["input_fingerprint"]["segments_prediction_usable_sha256"], AUDIT.sha256_file(segments_path))
    AUDIT.check(global_checks, "GLOBAL", "prediction_usable_input_only", bool(segments["prediction_usable"].all()), int((~segments["prediction_usable"]).sum()))
    AUDIT.check(global_checks, "GLOBAL", "unified_scalar_target_prohibited", not bool(segments["unified_scalar_target_released"].any()), int(segments["unified_scalar_target_released"].sum()))
    manifest_families = {item["family"]: item for item in manifest["families"]}
    AUDIT.check(global_checks, "GLOBAL", "six_required_families_present", set(manifest_families) == set(AUDIT.FAMILIES), sorted(manifest_families))
    all_checks = list(global_checks)
    family_gates = []
    for family in AUDIT.FAMILIES:
        item = manifest_families[family]
        assignments = pd.read_parquet(item["assignment_path"])
        membership = pd.read_parquet(item["membership_path"])
        checks, family_gate = AUDIT.validate_family(family, assignments, membership, input_ids, config, contract)
        all_checks.extend(checks)
        family_gate["failed_checks"] = [row["check"] for row in checks if row["status"] == "FAIL"]
        family_gate["leakage_status"] = "PASS" if not family_gate["failed_checks"] else "FAIL"
        family_gates.append(family_gate)
    failed = [row for row in all_checks if row["status"] == "FAIL"]
    warnings = [{"family": item["family"], "warning": warning} for item in family_gates for warning in item["warnings"]]
    status = config["validation_pass_status"] if not failed else config["validation_fail_status"]
    all_stats_rows = []
    split_cards = {}
    gate_map = {item["family"]: item for item in family_gates}
    for family in AUDIT.FAMILIES:
        family_item = manifest_families[family]
        gate_item = gate_map[family]
        for row in family_item["role_stats"]:
            all_stats_rows.append({"family": family, **{key: value for key, value in row.items() if key != "engine_type_counts"}, "engine_type_counts": json.dumps(row["engine_type_counts"], ensure_ascii=False, sort_keys=True)})
        card_lines = [
            f"# SEGMENT_SPLIT sensitivity split card: {family}", "",
            f"- Sensitivity branch: `{config['sensitivity_name']}`",
            f"- Benchmark tier: `{gate_item['benchmark_tier']}`",
            f"- Leakage status: `{gate_item['leakage_status']}`",
            f"- Design: `{json.dumps(family_item['design'], ensure_ascii=False, sort_keys=True)}`",
            "- Audit identifiers are never model features.", "", "## Role support", "",
            "| Role | Segments | Vehicles | Trips | Max vehicle share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in family_item["role_stats"]:
            card_lines.append(f"| {row['split_role']} | {row['segments']:,} | {row['vehicles']:,} | {row['trips']:,} | {row['maximum_vehicle_segment_share']:.3%} |")
        card_lines.append("")
        card_path = run_dir / family / "SPLIT_CARD.md"
        card_path.write_text("\n".join(card_lines), encoding="utf-8")
        split_cards[family] = str(card_path)
    stats_path = run_dir / "all_split_stats.csv"
    pd.DataFrame(all_stats_rows).to_csv(stats_path, index=False)
    checks_path = run_dir / "all_leakage_checks.csv"
    pd.DataFrame([{**row, "evidence": json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True)} for row in all_checks]).to_csv(checks_path, index=False)
    summary = {
        "experiment": "SEGMENT_SPLIT",
        "stage": config["stage"],
        "sensitivity_name": config["sensitivity_name"],
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
        "all_split_stats": str(stats_path),
        "split_cards": split_cards,
        "output_directory": str(run_dir),
        "model_training_authorized": False,
        "training_block_reason": "DIFFERENTIAL_AUDIT G9 is still required.",
    }
    summary_path = run_dir / "split_gate_summary.json"
    AUDIT.write_json(summary_path, summary)
    report_lines = [
        f"# SEGMENT_SPLIT {config['sensitivity_name']} split gate", "",
        f"- Status: `{status}`",
        f"- Checks: {summary['checks_passed']} PASS / {summary['checks_failed']} FAIL",
        f"- Frozen prediction-usable input: {len(segments):,} segments",
        "- Model training remains blocked pending DIFFERENTIAL_AUDIT G9.", "", "## Family support", "",
    ]
    for item in family_gates:
        support = item["test_support"]
        report_lines.append(f"- `{item['family']}` ({item['benchmark_tier']}): {support['segments']:,} segments, {support['vehicles']:,} vehicles, {support['trips']:,} trips, max vehicle share {support['maximum_vehicle_segment_share']:.3%}; leakage {item['leakage_status']}.")
        for warning in item["warnings"]:
            report_lines.append(f"  - Target-support warning: {warning}.")
    report_path = run_dir / "SEGMENT_SPLIT_SENSITIVITY_SPLIT_GATE.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    pointer = {
        "status": status,
        "summary": str(summary_path),
        "report": str(report_path),
        "manifest": str(run_dir / "split_manifest.json"),
        "checks": str(checks_path),
        "stats": str(stats_path),
        "split_cards": split_cards,
        "output_directory": str(run_dir),
        "model_training_authorized": False,
    }
    AUDIT.write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
