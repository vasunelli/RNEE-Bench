#!/usr/bin/env python3
"""DIFFERENTIAL_AUDIT: reference-only eVED versus VED-native RNEE differential audit.

This script never trains a model. Legacy eVED fields are read only to quantify
contract drift; they are never copied into the RNEE release or used as targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ACTIVE_ROLES = {"train", "validation", "calibration", "test"}
SPLIT_MAP = {
    "random_trip_blocked": "random_trip_blocked",
    "cold_vehicle": "cold_vehicle",
    "cold_trip": "cold_trip",
    "cold_month": "cold_month",
    "cold_spatial": "cold_spatial",
    "cold_road_class": "cold_functional_road_class",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_source_name(value: str) -> str:
    name = Path(str(value)).name
    return "VED_" + name[len("eVED_") :] if name.startswith("eVED_") else name


def strict_integer_id(values: pd.Series, field_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Null or non-numeric observation key in {field_name}")
    rounded = numeric.round()
    if not np.allclose(numeric.to_numpy(float), rounded.to_numpy(float), rtol=0.0, atol=1e-9):
        raise ValueError(f"Non-integer observation key in {field_name}")
    return rounded.astype("Int64")


def align_observation_rows(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for key in keys:
        old[key] = strict_integer_id(old[key], f"legacy.{key}")
        new[key] = strict_integer_id(new[key], f"rnee.{key}")
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Observation keys must be unique within each source file")
    outer = old.merge(new, on=keys, how="outer", validate="one_to_one", indicator=True)
    unmatched = outer.loc[outer["_merge"].ne("both"), keys + ["_merge"]].copy()
    return outer.loc[outer["_merge"].eq("both")].drop(columns="_merge"), unmatched


def categorical(values: pd.Series, missing: str = "<missing>") -> pd.Series:
    result = values.astype("string").fillna(missing)
    return result.replace({"nan": missing, "None": missing, "": missing})


def add_crosstab(counter: Counter[tuple[str, str]], left: pd.Series, right: pd.Series) -> None:
    table = pd.crosstab(categorical(left), categorical(right), dropna=False)
    for left_value, row in table.iterrows():
        for right_value, count in row.items():
            if int(count):
                counter[(str(left_value), str(right_value))] += int(count)


def counter_frame(counter: Counter[tuple[str, str]], left_name: str, right_name: str) -> pd.DataFrame:
    rows = [
        {left_name: left, right_name: right, "row_count": count}
        for (left, right), count in sorted(counter.items())
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["row_share_within_legacy"] = frame["row_count"] / frame.groupby(left_name)["row_count"].transform("sum")
    return frame


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def sampled_quantiles(parts: list[np.ndarray]) -> dict[str, float | int | None]:
    if not parts:
        return {"sample_n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"sample_n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "sample_n": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def binary_confusion(old_present: np.ndarray, new_present: np.ndarray) -> dict[str, int]:
    return {
        "old_absent_new_absent": int((~old_present & ~new_present).sum()),
        "old_absent_new_present": int((~old_present & new_present).sum()),
        "old_present_new_absent": int((old_present & ~new_present).sum()),
        "old_present_new_present": int((old_present & new_present).sum()),
    }


def safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    paired = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return None
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def decide_g9(source_failures: list[str], benchmark_failures: list[str]) -> str:
    if source_failures:
        return "RNEE_NO_GO"
    if benchmark_failures:
        return "RNEE_DATASET_ONLY"
    return "RNEE_RESULTS_VALID"


def exact_segment_gate(summary: dict[str, Any], expected_status: str) -> bool:
    checks = summary.get("checks", {})
    return (
        summary.get("status") == expected_status
        and bool(checks)
        and all(value == "PASS" for value in checks.values())
        and summary.get("resume_noop_verified") is True
    )


def exact_split_gate(summary: dict[str, Any], expected_status: str, required_checks: int = 135) -> bool:
    families = summary.get("family_gates", [])
    return (
        summary.get("status") == expected_status
        and int(summary.get("checks_total", -1)) == required_checks
        and int(summary.get("checks_passed", -1)) == required_checks
        and int(summary.get("checks_failed", -1)) == 0
        and not summary.get("failed_checks")
        and len(families) == 6
        and all(item.get("leakage_status") == "PASS" and not item.get("failed_checks") for item in families)
    )


def refresh_artifact_manifest(output: Path, decision: str) -> None:
    artifacts = sorted(path for path in output.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    manifest = {
        "g9_decision": decision,
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    write_json(output / "artifact_manifest.json", manifest)


def audit_sensitivity_releases(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected_segments = {
        "PASS_SEGMENT_SPLIT_30S_SEGMENT_SENSITIVITY_GATE": "30s",
        "PASS_SEGMENT_SPLIT_500M_SEGMENT_SENSITIVITY_GATE": "500m",
    }
    expected_splits = {
        "PASS_SEGMENT_SPLIT_30S_SENSITIVITY_SPLITS": "30s",
        "PASS_SEGMENT_SPLIT_500M_SENSITIVITY_SPLITS": "500m",
    }
    by_granularity: dict[str, dict[str, Any]] = {}
    for pointer_path in map(Path, config.get("sensitivity_segment_pointers", [])):
        pointer = load_json(pointer_path)
        summary = load_json(Path(pointer["summary"]))
        granularity = expected_segments.get(pointer.get("status"), pointer_path.stem)
        by_granularity.setdefault(granularity, {}).update({
            "granularity": granularity,
            "segment_status": pointer.get("status"),
            "segment_gate_exact_pass": exact_segment_gate(summary, str(pointer.get("status"))),
            "segment_count": int(summary.get("segment_count", 0)),
            "prediction_usable_count": int(summary.get("prediction_usable_count", 0)),
        })
    for pointer_path in map(Path, config.get("sensitivity_split_pointers", [])):
        pointer = load_json(pointer_path)
        summary = load_json(Path(pointer["summary"]))
        granularity = expected_splits.get(pointer.get("status"), pointer_path.stem)
        by_granularity.setdefault(granularity, {}).update({
            "granularity": granularity,
            "split_status": pointer.get("status"),
            "split_gate_exact_pass": exact_split_gate(summary, str(pointer.get("status")), int(config["required_split_check_passes"])),
            "split_checks_passed": int(summary.get("checks_passed", 0)),
            "split_checks_failed": int(summary.get("checks_failed", 0)),
            "warnings": summary.get("warnings", []),
        })
    return [by_granularity[key] for key in sorted(by_granularity)]


def audit_rows(config: dict[str, Any], row_manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    legacy_dir = Path(config["legacy_eved_dir"])
    class_to_valhalla: Counter[tuple[str, str]] = Counter()
    class_to_functional: Counter[tuple[str, str]] = Counter()
    per_file: list[dict[str, Any]] = []
    object_totals: dict[str, Counter[str]] = {
        name: Counter() for name in ("intersection", "public_transport", "poi_focus")
    }
    speed_abs_samples: list[np.ndarray] = []
    coordinate_samples: list[np.ndarray] = []
    speed_pair_count = 0
    speed_within_5_count = 0
    speed_within_10_count = 0
    old_match_count = 0
    new_match_count = 0
    new_semantics_count = 0
    total_rows = 0
    aligned_rows = 0
    key_mismatch_rows = 0
    unmatched_parts: list[pd.DataFrame] = []
    rng_seed = int(config["random_seed"])
    sample_n = int(config["row_quantile_sample_per_partition"])
    scale = int(config["object_scale_m"])
    new_columns = [
        "VehId", "Trip", "Timestamp(ms)", "road_class", "functional_road_class",
        "speed_limit_kmh_used", "matched_latitude", "matched_longitude",
        "road_semantics_available", f"intersection_unique_count_{scale}m",
        f"public_transport_unique_count_{scale}m", f"poi_unique_count_{scale}m",
    ]
    old_required = [
        "VehId", "Trip", "Timestamp(ms)", "Matchted Latitude[deg]", "Matched Longitude[deg]",
        "Class of Speed Limit", config["legacy_speed_limit_column"], "Intersection", "Bus Stops", "Focus Points",
    ]
    for index, item in enumerate(row_manifest["partitions"]):
        source_file = str(item["source_file"])
        legacy_path = legacy_dir / source_file.replace("VED_", "eVED_", 1)
        if not legacy_path.exists():
            raise FileNotFoundError(f"Missing legacy counterpart: {legacy_path}")
        old = pd.read_csv(legacy_path, low_memory=False)
        old.columns = [str(column).strip().rstrip(";") for column in old.columns]
        missing = sorted(set(old_required) - set(old.columns))
        if missing:
            raise KeyError(f"Legacy columns missing in {legacy_path.name}: {missing}")
        old = old[old_required]
        new = pd.read_parquet(Path(item["pointer"]["rows"]), columns=new_columns)
        if len(old) != len(new):
            raise ValueError(f"Row count mismatch for {source_file}: {len(old)} versus {len(new)}")
        input_row_count = len(old)
        keys = ["VehId", "Trip", "Timestamp(ms)"]
        # TRAJECTORY_BUILD deliberately stores rows in deterministic trajectory-key order,
        # while legacy eVED preserves weekly-file physical order. Join by the
        # audited unique observation key; never compare by row position.
        merged, unmatched = align_observation_rows(old, new, keys)
        if not unmatched.empty:
            unmatched.insert(0, "source_file", source_file)
            unmatched_parts.append(unmatched)
        aligned = len(merged)
        mismatched = len(unmatched)
        old = merged[old_required]
        new = merged[new_columns]
        total_rows += input_row_count
        aligned_rows += aligned
        key_mismatch_rows += mismatched
        add_crosstab(class_to_valhalla, old["Class of Speed Limit"], new["road_class"])
        add_crosstab(class_to_functional, old["Class of Speed Limit"], new["functional_road_class"])

        old_speed = pd.to_numeric(old[config["legacy_speed_limit_column"]], errors="coerce").to_numpy(float)
        new_speed = pd.to_numeric(new["speed_limit_kmh_used"], errors="coerce").to_numpy(float)
        speed_ok = np.isfinite(old_speed) & np.isfinite(new_speed)
        differences = np.abs(old_speed[speed_ok] - new_speed[speed_ok])
        speed_pair_count += int(speed_ok.sum())
        speed_within_5_count += int((differences <= 5.0).sum())
        speed_within_10_count += int((differences <= 10.0).sum())

        old_lat = pd.to_numeric(old["Matchted Latitude[deg]"], errors="coerce").to_numpy(float)
        old_lon = pd.to_numeric(old["Matched Longitude[deg]"], errors="coerce").to_numpy(float)
        new_lat = pd.to_numeric(new["matched_latitude"], errors="coerce").to_numpy(float)
        new_lon = pd.to_numeric(new["matched_longitude"], errors="coerce").to_numpy(float)
        old_match = np.isfinite(old_lat) & np.isfinite(old_lon)
        new_match = np.isfinite(new_lat) & np.isfinite(new_lon)
        coordinate_ok = old_match & new_match
        old_match_count += int(old_match.sum())
        new_match_count += int(new_match.sum())
        new_semantics_count += int(new["road_semantics_available"].fillna(False).astype(bool).sum())

        sample_rng = np.random.default_rng(rng_seed + index)
        if differences.size:
            take = sample_rng.choice(differences.size, size=min(sample_n, differences.size), replace=False)
            speed_abs_samples.append(differences[take])
        coordinate_indices = np.flatnonzero(coordinate_ok)
        if coordinate_indices.size:
            take = sample_rng.choice(coordinate_indices, size=min(sample_n, coordinate_indices.size), replace=False)
            coordinate_samples.append(haversine_m(old_lat[take], old_lon[take], new_lat[take], new_lon[take]))

        old_presence = {
            "intersection": old["Intersection"].notna().to_numpy(bool),
            "public_transport": old["Bus Stops"].notna().to_numpy(bool),
            "poi_focus": old["Focus Points"].notna().to_numpy(bool),
        }
        new_presence = {
            "intersection": pd.to_numeric(new[f"intersection_unique_count_{scale}m"], errors="coerce").fillna(0).gt(0).to_numpy(bool),
            "public_transport": pd.to_numeric(new[f"public_transport_unique_count_{scale}m"], errors="coerce").fillna(0).gt(0).to_numpy(bool),
            "poi_focus": pd.to_numeric(new[f"poi_unique_count_{scale}m"], errors="coerce").fillna(0).gt(0).to_numpy(bool),
        }
        for name in old_presence:
            object_totals[name].update(binary_confusion(old_presence[name], new_presence[name]))
        per_file.append({
            "source_file": source_file,
            "rows": input_row_count,
            "aligned_rows": aligned,
            "key_mismatch_rows": mismatched,
            "legacy_matched_rows": int(old_match.sum()),
            "rnee_matched_rows": int(new_match.sum()),
            "rnee_semantics_available_rows": int(new["road_semantics_available"].fillna(False).astype(bool).sum()),
        })
        del old, new

    valhalla_frame = counter_frame(class_to_valhalla, "legacy_speed_source_class", "rnee_valhalla_road_class")
    functional_frame = counter_frame(class_to_functional, "legacy_speed_source_class", "rnee_functional_road_class")
    valhalla_frame.to_csv(output / "row_legacy_class_vs_valhalla_class.csv", index=False)
    functional_frame.to_csv(output / "row_legacy_class_vs_functional_class.csv", index=False)
    pd.DataFrame(per_file).to_csv(output / "row_alignment_by_file.csv", index=False)
    unmatched_columns = ["source_file", "VehId", "Trip", "Timestamp(ms)", "_merge"]
    unmatched_frame = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame(columns=unmatched_columns)
    unmatched_frame.to_csv(output / "row_unmatched_keys.csv", index=False)
    object_rows = []
    for metric, counts in object_totals.items():
        row = {"legacy_metric": metric, "rnee_metric": f"unique_object_count_{scale}m", **dict(counts)}
        overlap = counts["old_present_new_present"]
        union = overlap + counts["old_present_new_absent"] + counts["old_absent_new_present"]
        row["presence_jaccard"] = overlap / union if union else None
        object_rows.append(row)
    pd.DataFrame(object_rows).to_csv(output / "row_object_presence_comparison.csv", index=False)
    return {
        "source_files": len(per_file),
        "total_rows": total_rows,
        "aligned_rows": aligned_rows,
        "key_mismatch_rows": key_mismatch_rows,
        "alignment_rate": aligned_rows / total_rows if total_rows else 0.0,
        "legacy_matched_rows": old_match_count,
        "legacy_match_rate": old_match_count / total_rows if total_rows else 0.0,
        "rnee_matched_rows": new_match_count,
        "rnee_match_rate": new_match_count / total_rows if total_rows else 0.0,
        "rnee_semantics_available_rows": new_semantics_count,
        "rnee_semantics_available_rate": new_semantics_count / total_rows if total_rows else 0.0,
        "speed_limit_paired_rows": speed_pair_count,
        "speed_limit_within_5_kmh_rate": speed_within_5_count / speed_pair_count if speed_pair_count else None,
        "speed_limit_within_10_kmh_rate": speed_within_10_count / speed_pair_count if speed_pair_count else None,
        "speed_limit_absolute_difference_kmh": sampled_quantiles(speed_abs_samples),
        "matched_coordinate_difference_m": sampled_quantiles(coordinate_samples),
        "object_presence": {name: dict(counts) for name, counts in object_totals.items()},
    }


def audit_segments(config: dict[str, Any], segment_pointer: dict[str, Any], output: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    old_columns = [
        "segment_id", "source_file", "VehId", "Trip", "window_id", "rows",
        "start_timestamp_ms", "end_timestamp_ms", "prediction_usable", "dominant_road_class",
        "speed_limit_kmh_mean", "intersection_count", "bus_stop_count", "focus_point_count", "energy_wh",
    ]
    new_columns = [
        "segment_id", "source_file", "VehId", "Trip", "segment_index", "row_count",
        "observed_start_timestamp_ms", "observed_end_timestamp_ms", "prediction_usable",
        "road_semantics_model_eligible", "functional_road_class_dominant",
        "speed_limit_kmh_distance_weighted_mean", "intersection_unique_object_count_25m",
        "intersection_pass_event_count_25m", "public_transport_unique_object_count_25m",
        "public_transport_pass_event_count_25m", "poi_unique_object_count_25m", "poi_pass_event_count_25m",
        "fuel_volume_L", "battery_terminal_net_Wh", "engine_type",
    ]
    old = pd.read_csv(Path(config["legacy_segment_csv"]), usecols=old_columns, low_memory=False)
    new = pd.read_parquet(Path(segment_pointer["segments_all"]), columns=new_columns)
    old["source_file"] = old["source_file"].map(normalize_source_name)
    old = old.rename(columns={"window_id": "segment_index", "segment_id": "legacy_segment_id"})
    new = new.rename(columns={"segment_id": "rnee_segment_id"})
    keys = ["source_file", "VehId", "Trip", "segment_index"]
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Segment comparison keys must be unique")
    old_key = old[["legacy_segment_id", *keys]].copy()
    new_key = new[["rnee_segment_id", *keys]].copy()
    outer = old.merge(new, on=keys, how="outer", suffixes=("_legacy", "_rnee"), validate="one_to_one", indicator=True)
    disposition = outer[keys + ["_merge"]].rename(columns={"_merge": "segment_key_disposition"})
    disposition.to_csv(output / "segment_key_disposition.csv", index=False)
    merged = outer.loc[outer["_merge"].eq("both")].drop(columns="_merge").copy()
    time_equal = (
        merged["start_timestamp_ms"].eq(merged["observed_start_timestamp_ms"])
        & merged["end_timestamp_ms"].eq(merged["observed_end_timestamp_ms"])
    )
    row_count_equal = pd.to_numeric(merged["rows"], errors="coerce").eq(pd.to_numeric(merged["row_count"], errors="coerce"))
    eligibility = pd.crosstab(
        merged["prediction_usable_legacy"].astype(bool),
        merged["prediction_usable_rnee"].astype(bool),
        dropna=False,
    )
    eligibility_rows = []
    for old_value in (False, True):
        for new_value in (False, True):
            eligibility_rows.append({
                "legacy_prediction_usable": old_value,
                "rnee_prediction_usable": new_value,
                "segment_count": int(eligibility.loc[old_value, new_value]) if old_value in eligibility.index and new_value in eligibility.columns else 0,
            })
    pd.DataFrame(eligibility_rows).to_csv(output / "segment_eligibility_transition.csv", index=False)
    add_crosstab_counter: Counter[tuple[str, str]] = Counter()
    add_crosstab(add_crosstab_counter, merged["dominant_road_class"], merged["functional_road_class_dominant"])
    segment_class = counter_frame(add_crosstab_counter, "legacy_dominant_speed_source_class", "rnee_distance_dominant_functional_class")
    segment_class.to_csv(output / "segment_legacy_vs_functional_class.csv", index=False)

    target_rows = []
    for engine, frame in merged.groupby("engine_type", dropna=False):
        old_target = pd.to_numeric(frame["energy_wh"], errors="coerce")
        if str(engine) in {"PHEV", "EV"}:
            diagnostic = pd.to_numeric(frame["battery_terminal_net_Wh"], errors="coerce")
            diagnostic_name = "battery_terminal_net_Wh"
        else:
            diagnostic = pd.to_numeric(frame["fuel_volume_L"], errors="coerce") / 0.1123 * 1000.0
            diagnostic_name = "fuel_L_divided_by_legacy_0.1123_times_1000_reference_only"
        valid = old_target.notna() & diagnostic.notna()
        ratios = old_target[valid] / diagnostic[valid].replace(0, np.nan)
        target_rows.append({
            "engine_type": str(engine),
            "paired_segments": int(valid.sum()),
            "legacy_target": "eVED_energy_wh_unified_reference_only",
            "rnee_diagnostic": diagnostic_name,
            "pearson_correlation": safe_correlation(old_target[valid], diagnostic[valid]),
            "mean_absolute_difference_reference_scale": float((old_target[valid] - diagnostic[valid]).abs().mean()) if valid.any() else None,
            "median_legacy_to_rnee_reference_ratio": float(ratios.median()) if ratios.notna().any() else None,
            "legacy_prediction_usable_segments": int(frame["prediction_usable_legacy"].sum()),
            "rnee_prediction_usable_segments": int(frame["prediction_usable_rnee"].sum()),
            "physical_cross_powertrain_comparison_allowed": False,
        })
    pd.DataFrame(target_rows).to_csv(output / "segment_target_reference_comparison.csv", index=False)

    object_rows = []
    pairs = [
        ("intersection_count", "intersection_unique_object_count_25m", "intersection_pass_event_count_25m"),
        ("bus_stop_count", "public_transport_unique_object_count_25m", "public_transport_pass_event_count_25m"),
        ("focus_point_count", "poi_unique_object_count_25m", "poi_pass_event_count_25m"),
    ]
    for legacy, unique, events in pairs:
        for rnee in (unique, events):
            left = pd.to_numeric(merged[legacy], errors="coerce").fillna(0)
            right = pd.to_numeric(merged[rnee], errors="coerce").fillna(0)
            object_rows.append({
                "legacy_metric": legacy,
                "rnee_metric": rnee,
                "paired_segments": len(merged),
                "pearson_correlation": safe_correlation(left, right),
                "legacy_nonzero_rate": float(left.gt(0).mean()),
                "rnee_nonzero_rate": float(right.gt(0).mean()),
                "metric_definition_identical": False,
            })
    pd.DataFrame(object_rows).to_csv(output / "segment_object_metric_comparison.csv", index=False)
    summary = {
        "legacy_segments": len(old),
        "rnee_segments": len(new),
        "common_segment_keys": len(merged),
        "legacy_segment_key_coverage": len(merged) / len(old) if len(old) else 0.0,
        "legacy_only_segments": int(outer["_merge"].eq("left_only").sum()),
        "rnee_only_segments": int(outer["_merge"].eq("right_only").sum()),
        "rnee_extra_segments": len(new) - len(merged),
        "time_exact_segments": int(time_equal.sum()),
        "time_agreement_rate": float(time_equal.mean()) if len(merged) else 0.0,
        "row_count_exact_segments": int(row_count_equal.sum()),
        "row_count_agreement_rate": float(row_count_equal.mean()) if len(merged) else 0.0,
        "legacy_prediction_usable_segments": int(old["prediction_usable"].sum()),
        "rnee_prediction_usable_segments": int(new["prediction_usable"].sum()),
        "common_legacy_usable_rnee_not_usable": int((merged["prediction_usable_legacy"] & ~merged["prediction_usable_rnee"]).sum()),
        "common_legacy_not_usable_rnee_usable": int((~merged["prediction_usable_legacy"] & merged["prediction_usable_rnee"]).sum()),
        "rnee_road_semantics_model_eligible_segments": int(new["road_semantics_model_eligible"].sum()),
        "rnee_road_semantics_model_eligible_rate": float(new["road_semantics_model_eligible"].mean()),
        "legacy_unified_target_transferable": False,
        "legacy_semantic_metrics_transferable": False,
    }
    return summary, old_key, new_key


def audit_splits(config: dict[str, Any], split_pointer: dict[str, Any], old_key: pd.DataFrame, new_key: pd.DataFrame, output: Path) -> dict[str, Any]:
    legacy_root = Path(config["legacy_split_dir"])
    new_root = Path(split_pointer["output_directory"])
    rows = []
    for old_family, new_family in SPLIT_MAP.items():
        old_assignment = pd.read_csv(legacy_root / old_family / "assignments.csv", usecols=["segment_id", "split"])
        old_assignment = old_assignment.merge(old_key, left_on="segment_id", right_on="legacy_segment_id", how="inner")
        new_assignment = pd.read_parquet(new_root / new_family / "assignments.parquet", columns=["segment_id", "split_role"])
        new_assignment = new_assignment.merge(new_key, left_on="segment_id", right_on="rnee_segment_id", how="inner")
        keys = ["source_file", "VehId", "Trip", "segment_index"]
        paired = old_assignment[keys + ["split"]].merge(
            new_assignment[keys + ["split_role"]], on=keys, how="inner", validate="one_to_one"
        )
        old_active = old_assignment[old_assignment["split"].isin(ACTIVE_ROLES)]
        new_active = new_assignment[new_assignment["split_role"].isin(ACTIVE_ROLES)]
        old_test = set(map(tuple, old_active.loc[old_active["split"].eq("test"), keys].itertuples(index=False, name=None)))
        new_test = set(map(tuple, new_active.loc[new_active["split_role"].eq("test"), keys].itertuples(index=False, name=None)))
        intersection = len(old_test & new_test)
        union = len(old_test | new_test)
        rows.append({
            "legacy_family": old_family,
            "rnee_family": new_family,
            "legacy_active_segments": len(old_active),
            "rnee_active_segments": len(new_active),
            "paired_common_segments": len(paired),
            "paired_role_agreement_rate": float(paired["split"].eq(paired["split_role"]).mean()) if len(paired) else None,
            "legacy_test_segments": len(old_test),
            "rnee_test_segments": len(new_test),
            "common_test_segments": intersection,
            "test_membership_jaccard": intersection / union if union else None,
            "split_contract_identical": False,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "split_membership_comparison.csv", index=False)
    return {
        "families_compared": len(frame),
        "mean_role_agreement_on_common_segments": float(frame["paired_role_agreement_rate"].mean()),
        "mean_test_membership_jaccard": float(frame["test_membership_jaccard"].mean()),
        "cold_road_class_replaced_by_true_functional_class": True,
        "legacy_split_memberships_transferable": False,
    }


def audit_legacy_road_results(config: dict[str, Any], output: Path) -> dict[str, Any]:
    metrics = pd.read_csv(Path(config["legacy_road_ablation_csv"]))
    metrics = metrics[(metrics["split"].eq("test")) & metrics["model"].str.startswith("g")].copy()
    rows = []
    for split_name, frame in metrics.groupby("split_name"):
        g0 = frame.loc[frame["model"].eq("g0_road_free")].iloc[0]
        best = frame.loc[frame["mae"].idxmin()]
        g5 = frame.loc[frame["model"].eq("g5_full_road_semantics")].iloc[0]
        rows.append({
            "legacy_split": split_name,
            "legacy_g0_mae_wh": float(g0["mae"]),
            "legacy_best_model": str(best["model"]),
            "legacy_best_mae_improvement_pct_vs_g0": float(best["mae_improvement_pct_vs_g0"]),
            "legacy_g5_mae_improvement_pct_vs_g0": float(g5["mae_improvement_pct_vs_g0"]),
            "rnee_comparable_gain_available": False,
            "transfer_status": "HISTORICAL_ONLY_REQUIRES_MODEL_CONTRACT_RERUN",
        })
    frame = pd.DataFrame(rows).sort_values("legacy_split")
    frame.to_csv(output / "legacy_road_gain_inventory.csv", index=False)
    return {
        "legacy_splits_inventoried": len(frame),
        "legacy_best_gain_range_pct": [
            float(frame["legacy_best_mae_improvement_pct_vs_g0"].min()),
            float(frame["legacy_best_mae_improvement_pct_vs_g0"].max()),
        ],
        "rnee_road_gain_comparison_status": "NOT_RUN_MODEL_TRAINING_PROHIBITED_DURING_G9_AUDIT",
        "legacy_results_transferable": False,
        "required_next_stage": "MODEL_CONTRACT_CONTROLLED_TARGET_SEPARATED_RERUN",
    }


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_summary_path = Path(config["target_summary"])
    target_contract_path = Path(config["target_contract"])
    release_pointer_path = Path(config["trajectory_build_release_pointer"])
    row_manifest_path = Path(config["row_manifest"])
    segment_pointer_path = Path(config["primary_segment_pointer"])
    split_pointer_path = Path(config["primary_split_pointer"])
    target_summary = load_json(target_summary_path)
    target_contract = load_json(target_contract_path)
    release_pointer = load_json(release_pointer_path)
    release_summary_path = Path(release_pointer["summary"])
    release_summary = load_json(release_summary_path)
    row_manifest = load_json(row_manifest_path)
    segment_pointer = load_json(segment_pointer_path)
    segment_summary_path = Path(segment_pointer["summary"])
    segment_summary = load_json(segment_summary_path)
    split_pointer = load_json(split_pointer_path)
    split_summary_path = Path(split_pointer["summary"])
    split_summary = load_json(split_summary_path)
    legacy_segment_summary = load_json(Path(config["legacy_segment_summary"]))
    legacy_split_summary = load_json(Path(config["legacy_split_summary"]))
    legacy_road_summary = load_json(Path(config["legacy_road_ablation_summary"]))
    input_paths = [
        target_summary_path, target_contract_path, release_pointer_path, row_manifest_path,
        segment_pointer_path, split_pointer_path, Path(config["legacy_segment_summary"]),
        Path(config["legacy_split_summary"]), Path(config["legacy_road_ablation_summary"]),
        release_summary_path, segment_summary_path, Path(segment_pointer["manifest"]),
        Path(segment_pointer["segments_all"]), Path(segment_pointer["segments_prediction_usable"]),
        split_summary_path, Path(split_pointer["manifest"]), Path(config["legacy_segment_csv"]),
        Path(config["legacy_road_ablation_csv"]),
    ]
    for item in row_manifest["partitions"]:
        input_paths.append(Path(item["pointer"]["rows"]))
        input_paths.append(Path(config["legacy_eved_dir"]) / str(item["source_file"]).replace("VED_", "eVED_", 1))
    for old_family, new_family in SPLIT_MAP.items():
        input_paths.append(Path(config["legacy_split_dir"]) / old_family / "assignments.csv")
        input_paths.append(Path(split_pointer["output_directory"]) / new_family / "assignments.parquet")
    for pointer_value in config.get("sensitivity_segment_pointers", []) + config.get("sensitivity_split_pointers", []):
        pointer_path = Path(pointer_value)
        pointer = load_json(pointer_path)
        input_paths.extend([pointer_path, Path(pointer["summary"]), Path(pointer["manifest"])])
        if "segments_all" in pointer:
            input_paths.append(Path(pointer["segments_all"]))
        if "output_directory" in pointer and "splits" in pointer_path.stem:
            for family in ("random_trip_blocked", "cold_vehicle", "cold_trip", "cold_month", "cold_spatial", "cold_functional_road_class"):
                input_paths.append(Path(pointer["output_directory"]) / family / "assignments.parquet")
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Fingerprint inputs are missing: {missing_inputs}")
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {str(path): sha256_file(path) for path in input_paths},
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "differential_audit_summary.json"
    fingerprint_path = output / "input_fingerprint.json"
    if summary_path.exists() and fingerprint_path.exists():
        if load_json(fingerprint_path) != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = True
        summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started
        summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
        write_json(output / "resume_check.json", {
            "status": "PASS_NOOP_RESUME",
            "verified_at_utc": summary["last_resume_check_at_utc"],
            "input_fingerprint_unchanged": True,
            "model_training_performed": False,
        })
        refresh_artifact_manifest(output, str(summary["g9_decision"]))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    write_json(fingerprint_path, fingerprint)
    shutil.copy2(config_path, output / "run_config.yaml")

    row_audit = audit_rows(config, row_manifest, output)
    segment_audit, old_key, new_key = audit_segments(config, segment_pointer, output)
    split_audit = audit_splits(config, split_pointer, old_key, new_key, output)
    legacy_results = audit_legacy_road_results(config, output)
    sensitivity_audit = audit_sensitivity_releases(config)
    pd.DataFrame([
        {**item, "warnings": json.dumps(item.get("warnings", []), ensure_ascii=False)}
        for item in sensitivity_audit
    ]).to_csv(output / "sensitivity_release_audit.csv", index=False)

    source_checks = {
        "target_audit_operational_target_gate": (
            target_summary.get("overall_gate") == "PASS_OPERATIONAL_TARGET_ONLY"
            and all(target_summary["checks"].get(name) is True for class_name in ("blocking_source_integrity", "blocking_target_validity") for name in target_summary["check_classes"][class_name])
        ),
        "trajectory_build_release_gate": (
            release_summary.get("status") == "PASS_TRAJECTORY_BUILD_RELEASE_CANDIDATE_GATE"
            and all(value == "PASS" for value in release_summary.get("checks", {}).values())
        ),
        "row_manifest_exact_pass": all(item.get("status") == "PASS" and item.get("checks", {}).get("status") == "PASS" for item in row_manifest.get("partitions", [])),
        "row_alignment": row_audit["alignment_rate"] >= float(config["minimum_row_alignment_rate"]),
        "row_unmatched_keys_zero": row_audit["key_mismatch_rows"] == 0,
        "row_count_matches_frozen_source": row_audit["total_rows"] == int(target_summary["counts"]["rows"]),
        "target_channels_separate": target_contract["unified_scalar"]["status"] == "not released",
    }
    benchmark_checks = {
        "segment_split_segment_gate": exact_segment_gate(segment_summary, "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE"),
        "segment_split_split_gate": exact_split_gate(split_summary, "PASS_BENCHMARK_SPLITS", int(config["required_split_check_passes"])),
        "legacy_segment_key_coverage": segment_audit["legacy_segment_key_coverage"] >= float(config["minimum_legacy_segment_key_coverage"]),
        "segment_time_agreement": segment_audit["time_agreement_rate"] >= float(config["minimum_segment_time_agreement_rate"]),
        "segment_row_count_agreement": segment_audit["row_count_agreement_rate"] == 1.0,
        "segment_new_only_accounted": segment_audit["legacy_only_segments"] == 0 and segment_audit["rnee_only_segments"] == segment_audit["rnee_extra_segments"],
        "legacy_segment_summary_reproduced": (
            segment_audit["legacy_segments"] == int(legacy_segment_summary["segments"]["segments_total"])
            and segment_audit["legacy_prediction_usable_segments"] == int(legacy_segment_summary["segments"]["segments_prediction_usable"])
        ),
        "legacy_split_summary_valid": legacy_split_summary.get("decision_gate", {}).get("status") == "PASS" and len(legacy_split_summary.get("splits", [])) == 6,
        "legacy_road_summary_valid": legacy_road_summary.get("decision_gate", {}).get("status") == "PASS",
        "prediction_support": segment_audit["rnee_prediction_usable_segments"] >= int(config["minimum_prediction_usable_segments"]),
        "road_semantics_support": (
            segment_audit["rnee_road_semantics_model_eligible_segments"] >= int(config["minimum_road_semantics_model_eligible_segments"])
            and segment_audit["rnee_road_semantics_model_eligible_rate"] >= float(config["minimum_road_semantics_eligible_rate"])
        ),
        "split_check_count": int(split_summary.get("checks_passed", 0)) == int(config["required_split_check_passes"]),
        "six_split_families": split_audit["families_compared"] == int(config["required_split_families"]),
        "sensitivity_release_gates": len(sensitivity_audit) == 2 and all(item.get("segment_gate_exact_pass") is True and item.get("split_gate_exact_pass") is True for item in sensitivity_audit),
    }
    source_failures = [name for name, passed in source_checks.items() if not passed]
    benchmark_failures = [name for name, passed in benchmark_checks.items() if not passed]
    decision = decide_g9(source_failures, benchmark_failures)
    checks_frame = pd.DataFrame(
        [{"check_class": "source", "check": key, "status": "PASS" if value else "FAIL"} for key, value in source_checks.items()]
        + [{"check_class": "benchmark", "check": key, "status": "PASS" if value else "FAIL"} for key, value in benchmark_checks.items()]
    )
    checks_frame.to_csv(output / "g9_checks.csv", index=False)
    summary = {
        "experiment": config["experiment"],
        "stage": config["stage"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "g9_decision": decision,
        "interpretation": "RNEE data and benchmark contracts authorize controlled target-separated result generation; no legacy model result is carried over." if decision == "RNEE_RESULTS_VALID" else "See failed checks.",
        "source_checks": source_checks,
        "benchmark_checks": benchmark_checks,
        "source_failures": source_failures,
        "benchmark_failures": benchmark_failures,
        "row_differential": row_audit,
        "segment_differential": segment_audit,
        "split_differential": split_audit,
        "sensitivity_release_audit": sensitivity_audit,
        "legacy_result_inventory": legacy_results,
        "legacy_eved_role": "reference_only",
        "legacy_target_transferable": False,
        "legacy_semantics_transferable": False,
        "legacy_split_memberships_transferable": False,
        "legacy_model_results_transferable": False,
        "model_contract_model_training_authorized": decision == "RNEE_RESULTS_VALID",
        "authorization_scope": {
            "primary_60s": "authorized_for_MODEL_CONTRACT_controlled_target_separated_rerun" if decision == "RNEE_RESULTS_VALID" else "not_authorized",
            "sensitivity_30s_500m": "release_integrity_verified; execute only as prespecified MODEL_CONTRACT sensitivity after primary evidence freeze" if decision == "RNEE_RESULTS_VALID" else "not_authorized",
            "unsupported_target_split_cells": config.get("unsupported_target_split_cells", []),
        },
        "model_training_performed_in_differential_audit": False,
        "next_stage": "MODEL_CONTRACT" if decision == "RNEE_RESULTS_VALID" else "STOP_AND_CAUTION",
        "resume_noop_verified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "output_directory": str(output),
    }
    write_json(summary_path, summary)
    report_lines = [
        "# DIFFERENTIAL_AUDIT eVED-versus-RNEE differential audit and G9",
        "",
        f"- G9 decision: `{decision}`",
        f"- Hard checks: {len(source_checks) + len(benchmark_checks) - len(source_failures) - len(benchmark_failures)}/{len(source_checks) + len(benchmark_checks)} PASS",
        f"- Row alignment: {row_audit['aligned_rows']:,}/{row_audit['total_rows']:,} ({row_audit['alignment_rate']:.6%})",
        f"- Common legacy segment keys: {segment_audit['common_segment_keys']:,}/{segment_audit['legacy_segments']:,}",
        f"- New-only / legacy-only segment keys: {segment_audit['rnee_only_segments']:,} / {segment_audit['legacy_only_segments']:,}",
        f"- Segment row-count/time agreement: {segment_audit['row_count_agreement_rate']:.6%} / {segment_audit['time_agreement_rate']:.6%}",
        f"- Legacy / RNEE prediction-usable segments: {segment_audit['legacy_prediction_usable_segments']:,} / {segment_audit['rnee_prediction_usable_segments']:,}",
        f"- RNEE road-semantics-model-eligible segments: {segment_audit['rnee_road_semantics_model_eligible_segments']:,}",
        f"- Split families compared: {split_audit['families_compared']}",
        f"- Mean common-segment role agreement / test Jaccard: {split_audit['mean_role_agreement_on_common_segments']:.3f} / {split_audit['mean_test_membership_jaccard']:.3f}",
        "",
        "## Material differential findings",
        "",
        f"- RNEE map-match and road-semantic row coverage are {row_audit['rnee_match_rate']:.3%} and {row_audit['rnee_semantics_available_rate']:.3%}.",
        f"- Paired old/new speed limits are within 5 km/h for {row_audit['speed_limit_within_5_kmh_rate']:.3%} of rows, while the sampled P99 absolute difference is {row_audit['speed_limit_absolute_difference_kmh']['p99']:.2f} km/h.",
        "- The legacy `Class of Speed Limit` is a speed-source class, not a functional hierarchy; the row and segment confusion tables show that it maps across multiple RNEE functional classes.",
        "- Legacy row-presence object fields and RNEE unique/event/exposure metrics are definitionally different and have low presence overlap; they cannot be renamed or carried forward.",
        f"- Of common segments, {segment_audit['common_legacy_not_usable_rnee_usable']:,} become prediction-usable under the audited RNEE target, while {segment_audit['common_legacy_usable_rnee_not_usable']:,} move in the opposite direction.",
        "- The legacy unified target is not a physical cross-powertrain target. Fuel L and battery-terminal Wh remain separate; the reference-scale comparison is diagnostic only.",
        "",
        "## Decision boundary",
        "",
        "Legacy `Energy_Consumption`, speed-source classes, nearest-neighbour speed limits, row-presence object counts, split memberships and road-aware model numbers are not transferable to RNEE. They remain historical reference evidence only.",
        "",
        "`RNEE_RESULTS_VALID` authorizes MODEL_CONTRACT to generate new, target-separated results on the frozen primary 60 s RNEE memberships. It does not validate or carry forward any old eVED model metric.",
        "",
        "The 30 s and 500 m segment/split release gates were independently rechecked here; they remain prespecified sensitivity branches to execute only after the MODEL_CONTRACT primary evidence freeze.",
        "",
        "The EV cold-vehicle cell is unsupported for the primary 60 s and 500 m releases and must be omitted or explicitly downgraded. The 30 s release does not share that specific support warning.",
        "",
        "## Model execution",
        "",
        "No model training or prediction was performed in DIFFERENTIAL_AUDIT.",
    ]
    (output / "DIFFERENTIAL_AUDIT_DIFFERENTIAL_AUDIT_G9.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    refresh_artifact_manifest(output, decision)
    latest = {
        "status": decision,
        "summary": str(summary_path),
        "report": str(output / "DIFFERENTIAL_AUDIT_DIFFERENTIAL_AUDIT_G9.md"),
        "manifest": str(output / "artifact_manifest.json"),
        "output_directory": str(output),
        "legacy_model_results_transferable": False,
        "model_contract_model_training_authorized": decision == "RNEE_RESULTS_VALID",
    }
    write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json", latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision == "RNEE_RESULTS_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
