#!/usr/bin/env python3
"""Evaluate MAP_MATCHING map-matching profiles against the frozen quality gates."""

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


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metric_level(value: float, pass_test: bool, quality_check_test: bool) -> str:
    if pass_test:
        return "PASS"
    if quality_check_test:
        return "CAUTION"
    return "FAIL"


def evaluate_profile(
    name: str,
    points: pd.DataFrame,
    trips: pd.DataFrame,
    thresholds: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    matched = points["match_status"].isin(
        ["matched_with_edge", "matched_without_edge"]
    )
    distance = pd.to_numeric(
        points["distance_from_trace_point_m"], errors="coerce"
    ).dropna()
    metrics = {
        "request_count": int(len(trips)),
        "request_success_rate": float(trips["request_succeeded"].mean()),
        "response_cardinality_rate": float(trips["cardinality_ok"].mean()),
        "row_cardinality_rate": float(trips["row_cardinality_ok"].mean()),
        "status_reason_coverage": float(points["match_status"].notna().mean()),
        "matched_coverage": float(matched.mean()),
        "edge_association_coverage": float(points["edge_index_valid"].mean()),
        "distance_p50_m": float(distance.quantile(0.50)),
        "distance_p95_m": float(distance.quantile(0.95)),
        "distance_p99_m": float(distance.quantile(0.99)),
        "trip_route_discontinuity_rate": float(
            trips["has_route_discontinuity"].mean()
        ),
        "silent_fallback_count": int(trips["silent_fallback_count"].sum()),
        "eligible_short_time_pairs": int(
            trips["eligible_short_time_pairs"].sum()
        ),
        "matched_edge_jump_over_500m_within_10s_count": int(
            trips["edge_jump_over_500m_within_10s_count"].sum()
        ),
    }
    metrics["matched_edge_jump_over_500m_within_10s_rate"] = (
        metrics["matched_edge_jump_over_500m_within_10s_count"]
        / max(metrics["eligible_short_time_pairs"], 1)
    )
    distance_gate = thresholds["distance_from_trace_point_m"]
    checks = {
        "request_success": (
            "PASS" if metrics["request_success_rate"] == 1.0 else "FAIL"
        ),
        "response_cardinality": (
            "PASS" if metrics["response_cardinality_rate"] == 1.0 else "FAIL"
        ),
        "row_cardinality": (
            "PASS" if metrics["row_cardinality_rate"] == 1.0 else "FAIL"
        ),
        "status_reason_coverage": (
            "PASS"
            if metrics["status_reason_coverage"]
            >= thresholds["input_point_status_coverage"]["pass_min"]
            else "FAIL"
        ),
        "matched_coverage": metric_level(
            metrics["matched_coverage"],
            metrics["matched_coverage"]
            >= thresholds["matched_coverage"]["pass_min"],
            metrics["matched_coverage"]
            >= thresholds["matched_coverage"]["quality_check_min"],
        ),
        "distance_p50": metric_level(
            metrics["distance_p50_m"],
            metrics["distance_p50_m"] <= distance_gate["p50_pass_max"],
            metrics["distance_p50_m"] <= distance_gate["p50_quality_check_max"],
        ),
        "distance_p95": metric_level(
            metrics["distance_p95_m"],
            metrics["distance_p95_m"] <= distance_gate["p95_pass_max"],
            metrics["distance_p95_m"] <= distance_gate["p95_quality_check_max"],
        ),
        "distance_p99": metric_level(
            metrics["distance_p99_m"],
            metrics["distance_p99_m"] <= distance_gate["p99_pass_max"],
            metrics["distance_p99_m"] <= distance_gate["p99_quality_check_max"],
        ),
        "trip_route_discontinuity_rate": metric_level(
            metrics["trip_route_discontinuity_rate"],
            metrics["trip_route_discontinuity_rate"]
            <= thresholds["trip_route_discontinuity_rate"]["pass_max"],
            metrics["trip_route_discontinuity_rate"]
            <= thresholds["trip_route_discontinuity_rate"]["quality_check_max"],
        ),
        "silent_fallback_count": (
            "PASS"
            if metrics["silent_fallback_count"]
            <= thresholds["silent_fallback_count"]["pass_max"]
            else "FAIL"
        ),
        "matched_edge_jump_rate": metric_level(
            metrics["matched_edge_jump_over_500m_within_10s_rate"],
            metrics["matched_edge_jump_over_500m_within_10s_rate"]
            <= thresholds[
                "matched_edge_jump_over_500m_within_10s_rate"
            ]["pass_max"],
            metrics["matched_edge_jump_over_500m_within_10s_rate"]
            <= thresholds[
                "matched_edge_jump_over_500m_within_10s_rate"
            ]["quality_check_max"],
        ),
    }
    if "FAIL" in checks.values():
        gate = "FAIL"
    elif "CAUTION" in checks.values():
        gate = "CAUTION"
    else:
        gate = "PASS"
    return {
        "profile": name,
        "gps_accuracy_m": profile["gps_accuracy_m"],
        "search_radius_m": profile["search_radius_m"],
        "metrics": metrics,
        "checks": checks,
        "automated_gate": gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-pointer",
        type=Path,
        default=Path(r"results/map_matching_latest_raw_pointer.json"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(r"configs/rnee_build/qa_thresholds.yaml"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(r"configs/rnee_build/map_match_profiles.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"results/rnee_build"),
    )
    parser.add_argument("--experiment", default="MAP_MATCHING")
    parser.add_argument("--output-prefix", default="map_matching_map_match_qa")
    parser.add_argument("--latest-prefix", default="map_matching_latest_qa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pointer = load_json(args.raw_pointer)
    thresholds = load_yaml(args.thresholds)["map_matching"]
    profiles = load_yaml(args.profiles)["profiles"]
    trips = pd.read_parquet(pointer["trip_summary"])
    points = pd.read_parquet(pointer["matched_points"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"{args.output_prefix}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    results = []
    for name, profile in profiles.items():
        results.append(
            evaluate_profile(
                name,
                points[points["profile"].eq(name)].copy(),
                trips[trips["profile"].eq(name)].copy(),
                thresholds,
                profile,
            )
        )
    rank = {"PASS": 2, "CAUTION": 1, "FAIL": 0}
    selected = sorted(
        results,
        key=lambda item: (
            -rank[item["automated_gate"]],
            -item["metrics"]["matched_coverage"],
            item["metrics"]["distance_p95_m"],
            item["metrics"]["trip_route_discontinuity_rate"],
            item["search_radius_m"],
        ),
    )[0]
    any_eligible = selected["automated_gate"] in {"PASS", "CAUTION"}
    status = (
        "PENDING_MANUAL_SPATIAL_AUDIT"
        if any_eligible
        else "STOP_REQUIRES_MANUAL_DIAGNOSTIC"
    )
    summary = {
        "experiment": args.experiment,
        "stage": "automated_map_match_quality_gate",
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_output_directory": pointer["output_directory"],
        "profile_selection_uses_energy_target_model_or_eved": False,
        "selected_profile_for_manual_audit": selected["profile"],
        "selected_profile_automated_gate": selected["automated_gate"],
        "profiles": results,
        "manual_audit_required": True,
        "full_build_authorized": False,
    }
    write_json(output / "map_match_qa_summary.json", summary)
    rows = []
    for result in results:
        row = {
            "profile": result["profile"],
            "automated_gate": result["automated_gate"],
            **result["metrics"],
        }
        row.update(
            {f"check_{key}": value for key, value in result["checks"].items()}
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output / "profile_quality_metrics.csv", index=False
    )
    lines = [
        f"# {args.experiment} automated map-matching quality gate",
        "",
        f"- Status: `{status}`",
        f"- Raw run: `{pointer['output_directory']}`",
        f"- Diagnostic profile: `{selected['profile']}`",
        f"- Automated gate: `{selected['automated_gate']}`",
        "- Profile selection used only map-matching quality metrics.",
        "- Energy, model performance, and legacy eVED fields were not used.",
        "- Full construction remains blocked until the manual spatial audit is completed.",
        "",
        "## Profile results",
        "",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.extend(
            [
                f"### {result['profile']}: {result['automated_gate']}",
                "",
                f"- matched coverage: {metrics['matched_coverage']:.4f}",
                f"- edge association coverage: {metrics['edge_association_coverage']:.4f}",
                f"- distance p50/p95/p99: {metrics['distance_p50_m']:.2f} / "
                f"{metrics['distance_p95_m']:.2f} / {metrics['distance_p99_m']:.2f} m",
                f"- trip discontinuity rate: {metrics['trip_route_discontinuity_rate']:.4f}",
                f"- >500 m within 10 s jump rate: "
                f"{metrics['matched_edge_jump_over_500m_within_10s_rate']:.6f}",
                f"- checks: `{json.dumps(result['checks'], ensure_ascii=False)}`",
                "",
            ]
        )
    report = output / f"{args.experiment}_AUTOMATED_MAP_MATCH_GATE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_summary = args.output_root / f"{args.latest_prefix}_summary.json"
    latest_report = args.output_root / f"{args.latest_prefix}_report.md"
    shutil.copy2(output / "map_match_qa_summary.json", latest_summary)
    shutil.copy2(report, latest_report)
    write_json(
        args.output_root / f"{args.latest_prefix}_pointer.json",
        {
            "status": status,
            "output_directory": str(output),
            "summary": str(output / "map_match_qa_summary.json"),
            "report": str(report),
            "selected_profile": selected["profile"],
            "raw_pointer": str(args.raw_pointer),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
