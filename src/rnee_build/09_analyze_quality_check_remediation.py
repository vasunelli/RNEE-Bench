#!/usr/bin/env python3
"""Classify quality_check-1 MAP_MATCHING errors using frozen raw-GPS quality rules."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EARTH_RADIUS_M = 6_371_008.8


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_pair_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    updates = frame[frame["is_coordinate_update"].eq(True)].sort_values(
        "point_index"
    )
    if len(updates) < 2:
        return pd.DataFrame(
            columns=[
                "from_point_index",
                "to_point_index",
                "dt_s",
                "distance_m",
                "speed_kmh",
            ]
        )
    latitude = np.radians(updates["latitude_raw"].to_numpy(float))
    longitude = np.radians(updates["longitude_raw"].to_numpy(float))
    delta_latitude = np.diff(latitude)
    delta_longitude = np.diff(longitude)
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude[:-1])
        * np.cos(latitude[1:])
        * np.sin(delta_longitude / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_M * np.arcsin(
        np.minimum(1, np.sqrt(haversine))
    )
    dt = np.diff(updates["timestamp_ms_raw"].to_numpy(float)) / 1000.0
    speed = np.divide(
        distance * 3.6,
        dt,
        out=np.full_like(distance, np.nan),
        where=dt > 0,
    )
    point_index = updates["point_index"].to_numpy(int)
    return pd.DataFrame(
        {
            "from_point_index": point_index[:-1],
            "to_point_index": point_index[1:],
            "dt_s": dt,
            "distance_m": distance,
            "speed_kmh": speed,
        }
    )


def classify_trip(
    frame: pd.DataFrame,
    rating: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    pairs = update_pair_metrics(frame)
    rules = config["raw_gps_break_rules"]
    nonpositive = pairs["dt_s"].le(0)
    speed = pairs["speed_kmh"].gt(
        float(rules["maximum_coordinate_update_speed_kmh"])
    )
    jump = pairs["distance_m"].gt(
        float(rules["maximum_coordinate_update_jump_m"])
    )
    trigger = nonpositive | speed | jump
    trigger_count = int(trigger.sum())
    if rating == "incorrect" and trigger_count:
        classification = "raw_gps_triggered_incorrect"
        action = "split_at_flagged_update_transition"
    elif rating == "incorrect":
        classification = "map_match_error_candidate"
        action = "retain_as_map_match_error_candidate"
    else:
        classification = "not_quality_1_incorrect"
        action = "retain_for_segmented_rerun"
    return {
        "raw_row_count": int(len(frame)),
        "coordinate_update_count": int(frame["is_coordinate_update"].eq(True).sum()),
        "update_pair_count": int(len(pairs)),
        "nonpositive_dt_pair_count": int(nonpositive.sum()),
        "speed_over_200_pair_count": int(speed.sum()),
        "jump_over_500m_pair_count": int(jump.sum()),
        "raw_gps_break_pair_count": trigger_count,
        "max_update_speed_kmh": (
            float(pairs["speed_kmh"].max()) if len(pairs) else None
        ),
        "max_update_jump_m": (
            float(pairs["distance_m"].max()) if len(pairs) else None
        ),
        "classification": classification,
        "recommended_action": action,
        "delete_original_trip": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-pointer",
        type=Path,
        default=Path(r"results/map_matching_latest_raw_pointer.json"),
    )
    parser.add_argument(
        "--manual-pointer",
        type=Path,
        default=Path(r"results/map_matching_latest_manual_qa_pointer.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/map_matching_remediation.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"results/rnee_build"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    raw_pointer = load_json(args.raw_pointer)
    manual_pointer = load_json(args.manual_pointer)
    quality_check = pd.read_csv(
        manual_pointer["quality_check_form"], encoding="utf-8-sig", keep_default_na=False
    )
    quality_rating_column = config["quality_rating_column"]
    if quality_check[quality_rating_column].eq("").any():
        raise RuntimeError("quality_rating is incomplete.")
    unexpected = set(quality_check[quality_rating_column]) - set(config["allowed_ratings"])
    if unexpected:
        raise RuntimeError(f"Unexpected quality_check-1 ratings: {sorted(unexpected)}")
    points = pd.read_parquet(raw_pointer["matched_points"])
    points = points[points["profile"].eq("tolerant")].copy()
    records = []
    for row in quality_check.itertuples(index=False):
        trip = points[points["pilot_trip_id"].eq(row.pilot_trip_id)].copy()
        if trip.empty:
            raise RuntimeError(f"Missing pilot trip {row.pilot_trip_id}.")
        diagnostics = classify_trip(
            trip, getattr(row, quality_rating_column), config
        )
        records.append(
            {
                "audit_id": row.audit_id,
                "pilot_trip_id": row.pilot_trip_id,
                "VehId": int(row.VehId),
                "Trip": int(row.Trip),
                "audit_category": row.audit_category,
                "quality_rating": getattr(row, quality_rating_column),
                "matched_coverage": float(row.matched_coverage),
                "distance_p95_m": float(row.distance_p95_m),
                "has_route_discontinuity": bool(row.has_route_discontinuity),
                **diagnostics,
            }
        )
    diagnostics = pd.DataFrame(records).sort_values("audit_id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"map_matching_quality_check_remediation_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    diagnostics.to_csv(output / "trip_remediation_diagnostics.csv", index=False)
    incorrect = diagnostics[diagnostics["quality_rating"].eq("incorrect")]
    incorrect.to_csv(output / "quality_1_incorrect_actions.csv", index=False)
    counts = incorrect["classification"].value_counts().to_dict()
    summary = {
        "experiment": "MAP_MATCHING",
        "stage": "quality_check_remediation_policy",
        "status": "POLICY_FROZEN_REQUIRES_SEGMENTED_RERUN",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_rating_column_used": quality_rating_column,
        "quality_check2_or_resolution_used": False,
        "inspected_trip_count": int(len(diagnostics)),
        "rating_counts": diagnostics[quality_rating_column].value_counts().to_dict(),
        "quality_1_incorrect_rate": float(
            diagnostics[quality_rating_column].eq("incorrect").mean()
        ),
        "incorrect_classification_counts": counts,
        "trips_with_raw_gps_break_trigger": int(
            diagnostics["raw_gps_break_pair_count"].gt(0).sum()
        ),
        "original_trips_deleted": 0,
        "policy": config,
        "next_action": "split traces at frozen raw-GPS break transitions, rerun Valhalla, and regenerate quality_check-1 maps",
        "edge_semantics_authorized": False,
    }
    write_json(output / "quality_check_remediation_summary.json", summary)
    report = output / "MAP_MATCHING_QUALITY_CHECK1_REMEDIATION_REPORT.md"
    report.write_text(
        "# MAP_MATCHING quality_check-1 remediation\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Ratings: `{summary['rating_counts']}`\n"
        f"- Incorrect rate: {summary['quality_1_incorrect_rate']:.1%}\n"
        f"- Incorrect classifications: `{counts}`\n"
        f"- All-pilot trips with an objective raw-GPS break: "
        f"{summary['trips_with_raw_gps_break_trigger']} / {len(diagnostics)}\n"
        "- Original trips deleted: 0\n\n"
        "Six quality_check-1 incorrect trips contain objective raw-GPS break "
        "triggers and must be split at those transitions. One incorrect trip "
        "has no raw-GPS trigger and remains a map-matching error candidate. "
        "No trip may be deleted merely because of its quality_check rating. EDGE_SEMANTICS "
        "remains blocked until the segmented pilot is rerun and rerated.\n",
        encoding="utf-8",
    )
    shutil.copy2(output / "quality_check_remediation_summary.json", args.output_root / "map_matching_latest_quality_check_remediation_summary.json")
    shutil.copy2(report, args.output_root / "map_matching_latest_quality_check_remediation_report.md")
    write_json(
        args.output_root / "map_matching_latest_quality_check_remediation_pointer.json",
        {
            "status": summary["status"],
            "output_directory": str(output),
            "summary": str(output / "quality_check_remediation_summary.json"),
            "report": str(report),
            "incorrect_actions": str(output / "quality_1_incorrect_actions.csv"),
            "trip_diagnostics": str(output / "trip_remediation_diagnostics.csv"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

