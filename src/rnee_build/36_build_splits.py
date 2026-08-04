#!/usr/bin/env python3
"""Build the six frozen SEGMENT_SPLIT leakage-controlled split families.

The rich assignment tables are audit artifacts.  Model-facing membership files
contain only a join key and a role; neither column is a model feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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
AUDIT_COLUMNS = [
    "segment_id",
    "trip_uid",
    "VehId",
    "Trip",
    "engine_type",
    "actual_month",
    "spatial_block_id",
    "spatial_grid_x",
    "spatial_grid_y",
    "functional_road_class_dominant",
    "road_semantics_model_eligible",
    "fuel_volume_L",
    "battery_terminal_net_Wh",
]


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


def stable_hash(value: Any, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()[:16], 16)


def parse_block(value: Any) -> tuple[int, int] | None:
    if pd.isna(value):
        return None
    text = str(value)
    if "_" not in text:
        return None
    left, right = text.split("_", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def neighboring_blocks(blocks: Iterable[str], radius: int, include_selected: bool = False) -> set[str]:
    selected = {str(value) for value in blocks}
    output: set[str] = set()
    for block in selected:
        parsed = parse_block(block)
        if parsed is None:
            continue
        x, y = parsed
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                candidate = f"{x + dx}_{y + dy}"
                if include_selected or candidate not in selected:
                    output.add(candidate)
    return output


def _fraction_labels(fractions: dict[str, float]) -> list[tuple[str, float]]:
    total = sum(float(fractions.get(role, 0.0)) for role in ACTIVE_ROLES)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Active role fractions must sum to 1, got {total}")
    cumulative = 0.0
    result = []
    for role in ACTIVE_ROLES:
        cumulative += float(fractions.get(role, 0.0))
        result.append((role, cumulative))
    return result


def assign_groups_by_rank(
    frame: pd.DataFrame,
    eligible: pd.Series,
    group_column: str,
    fractions: dict[str, float],
    seed: int,
    strata_column: str | None = None,
) -> pd.Series:
    """Assign whole groups, optionally using within-stratum hash ranks."""
    labels = _fraction_labels(fractions)
    result = pd.Series("excluded_unassigned", index=frame.index, dtype="string")
    work = frame.loc[eligible, [group_column] + ([strata_column] if strata_column else [])].drop_duplicates(group_column)
    if work.empty:
        return result
    strata = work.groupby(strata_column, sort=False, dropna=False) if strata_column else [(None, work)]
    mapping: dict[str, str] = {}
    for stratum, group in strata:
        keys = group[group_column].astype(str).tolist()
        keys.sort(key=lambda key: (stable_hash(f"{stratum}|{key}", seed), key))
        n = len(keys)
        for index, key in enumerate(keys):
            quantile = (index + 0.5) / n
            role = next(role for role, boundary in labels if quantile <= boundary + 1e-12)
            mapping[key] = role
    result.loc[eligible] = frame.loc[eligible, group_column].astype(str).map(mapping).astype("string")
    return result


def assign_groups_balanced_by_rows(
    frame: pd.DataFrame,
    eligible: pd.Series,
    group_column: str,
    fractions: dict[str, float],
    seed: int,
) -> pd.Series:
    """Greedily balance segment counts while keeping each group intact."""
    _fraction_labels(fractions)
    result = pd.Series("excluded_unassigned", index=frame.index, dtype="string")
    counts = frame.loc[eligible].groupby(group_column, dropna=False).size().reset_index(name="segments")
    counts["tie"] = counts[group_column].map(lambda value: stable_hash(value, seed))
    counts = counts.sort_values(["segments", "tie", group_column], ascending=[False, True, True], kind="stable")
    total = int(counts["segments"].sum())
    targets = {role: total * float(fractions.get(role, 0.0)) for role in ACTIVE_ROLES}
    assigned = {role: 0 for role in ACTIVE_ROLES}
    mapping: dict[str, str] = {}
    for group, size, _ in counts.itertuples(index=False, name=None):
        candidates = [role for role in ACTIVE_ROLES if targets[role] > 0]
        role = max(candidates, key=lambda name: ((targets[name] - assigned[name]) / targets[name], -ACTIVE_ROLES.index(name)))
        mapping[str(group)] = role
        assigned[role] += int(size)
    result.loc[eligible] = frame.loc[eligible, group_column].astype(str).map(mapping).astype("string")
    return result


def select_connected_test_blocks(frame: pd.DataFrame, cfg: dict[str, Any]) -> set[str]:
    counts = frame.dropna(subset=["spatial_block_id"]).groupby("spatial_block_id").size().astype(int)
    if counts.empty:
        raise RuntimeError("No spatial blocks are available for cold_spatial.")
    anchor = str(cfg["test_anchor_block"])
    if anchor not in counts.index:
        raise RuntimeError(f"Configured spatial anchor is absent: {anchor}")
    target = int(np.ceil(len(frame) * float(cfg["test_target_fraction"])))
    selected = {anchor}
    selected_count = int(counts.loc[anchor])
    seed = int(cfg["seed"])
    while selected_count < target:
        frontier = neighboring_blocks(selected, 1) & set(counts.index.astype(str)) - selected
        if not frontier:
            raise RuntimeError("Could not grow a connected cold_spatial test region to its target support.")
        candidate = min(frontier, key=lambda block: (-int(counts.loc[block]), stable_hash(block, seed), block))
        selected.add(candidate)
        selected_count += int(counts.loc[candidate])
    return selected


def build_random_trip_blocked(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    eligible = pd.Series(True, index=frame.index)
    role = assign_groups_by_rank(frame, eligible, "trip_uid", cfg["fractions"], int(cfg["seed"]))
    return role, {"grouping": "trip_uid", "trip_blocked": True}


def build_cold_vehicle(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    eligible = pd.Series(True, index=frame.index)
    role = assign_groups_balanced_by_rows(frame, eligible, "VehId", cfg["fractions"], int(cfg["seed"]))
    return role, {"grouping": "VehId", "vehicle_blocked": True, "balance_basis": "segment_count"}


def build_cold_trip(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    trip_counts = frame[["VehId", "trip_uid"]].drop_duplicates().groupby("VehId").size()
    eligible_vehicles = set(trip_counts[trip_counts >= int(cfg["minimum_trips_per_vehicle"])].index)
    eligible = frame["VehId"].isin(eligible_vehicles)
    role = assign_groups_by_rank(frame, eligible, "trip_uid", cfg["fractions"], int(cfg["seed"]), "VehId")
    role.loc[~eligible] = "excluded_low_trip_vehicle"
    return role, {
        "grouping": "trip_uid_within_vehicle",
        "minimum_trips_per_vehicle": int(cfg["minimum_trips_per_vehicle"]),
        "excluded_vehicle_count": int(frame.loc[~eligible, "VehId"].nunique()),
    }


def build_cold_month(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    role = pd.Series("excluded_outside_temporal_window", index=frame.index, dtype="string")
    month_sets = {role_name: {str(value) for value in cfg[f"{role_name}_months"]} for role_name in ACTIVE_ROLES}
    for role_name, months in month_sets.items():
        role.loc[frame["actual_month"].astype(str).isin(months)] = role_name
    return role, {"month_sets": {key: sorted(value) for key, value in month_sets.items()}, "chronological": True}


def build_cold_spatial(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    test_blocks = select_connected_test_blocks(frame, cfg)
    buffer_blocks = neighboring_blocks(test_blocks, int(cfg["buffer_cells"]))
    role = pd.Series("excluded_unassigned", index=frame.index, dtype="string")
    is_test = frame["spatial_block_id"].astype("string").isin(test_blocks)
    test_trips = set(frame.loc[is_test, "trip_uid"].astype(str))
    same_test_trip = frame["trip_uid"].astype(str).isin(test_trips)
    role.loc[is_test] = "test"
    role.loc[same_test_trip & ~is_test] = "excluded_same_test_trip_spatial"
    is_buffer = frame["spatial_block_id"].astype("string").isin(buffer_blocks)
    buffer_trips = set(frame.loc[is_buffer & ~same_test_trip, "trip_uid"].astype(str))
    same_buffer_trip = frame["trip_uid"].astype(str).isin(buffer_trips)
    role.loc[same_buffer_trip] = "excluded_spatial_buffer_trip"
    missing = frame["spatial_block_id"].isna()
    role.loc[missing & role.eq("excluded_unassigned")] = "excluded_missing_spatial"
    eligible = role.eq("excluded_unassigned")
    secondary = assign_groups_by_rank(
        frame,
        eligible,
        "trip_uid",
        cfg["remaining_fractions"],
        int(cfg["seed"]) + 1,
    )
    role.loc[eligible] = secondary.loc[eligible]
    return role, {
        "test_anchor_block": str(cfg["test_anchor_block"]),
        "test_blocks": sorted(test_blocks),
        "buffer_cells": int(cfg["buffer_cells"]),
        "buffer_blocks": sorted(buffer_blocks),
        "test_trip_count": len(test_trips),
        "buffer_trip_count": len(buffer_trips),
    }


def build_cold_functional_road_class(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    heldout = str(cfg["heldout_class"])
    road_class = frame["functional_road_class_dominant"].astype("string")
    is_test = road_class.eq(heldout)
    if not is_test.any():
        raise RuntimeError(f"Held-out functional road class has no support: {heldout}")
    test_trips = set(frame.loc[is_test, "trip_uid"].astype(str))
    same_test_trip = frame["trip_uid"].astype(str).isin(test_trips)
    role = pd.Series("excluded_unassigned", index=frame.index, dtype="string")
    role.loc[is_test] = "test"
    role.loc[same_test_trip & ~is_test] = "excluded_same_test_trip_road_class"
    role.loc[road_class.isna() & role.eq("excluded_unassigned")] = "excluded_missing_functional_class"
    eligible = role.eq("excluded_unassigned")
    secondary = assign_groups_by_rank(frame, eligible, "trip_uid", cfg["remaining_fractions"], int(cfg["seed"]))
    role.loc[eligible] = secondary.loc[eligible]
    return role, {"heldout_class": heldout, "test_trip_count": len(test_trips), "test_is_segment_dominant_class_only": True}


BUILDERS = {
    "random_trip_blocked": build_random_trip_blocked,
    "cold_vehicle": build_cold_vehicle,
    "cold_trip": build_cold_trip,
    "cold_month": build_cold_month,
    "cold_spatial": build_cold_spatial,
    "cold_functional_road_class": build_cold_functional_road_class,
}


def role_stats(assignments: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for role, subset in assignments.groupby("split_role", sort=True, dropna=False):
        vehicle_counts = subset["VehId"].value_counts()
        rows.append({
            "split_role": str(role),
            "segments": int(len(subset)),
            "vehicles": int(subset["VehId"].nunique()),
            "trips": int(subset["trip_uid"].nunique()),
            "maximum_vehicle_segment_share": float(vehicle_counts.iloc[0] / len(subset)) if len(subset) else None,
            "road_semantics_model_eligible": int(subset["road_semantics_model_eligible"].fillna(False).sum()),
            "fuel_target_segments": int(subset["engine_type"].isin(["ICE", "HEV", "PHEV"]).sum()),
            "battery_target_segments": int(subset["engine_type"].isin(["EV", "PHEV"]).sum()),
            "engine_type_counts": {str(key): int(value) for key, value in subset["engine_type"].value_counts().items()},
        })
    return rows


def build_family(frame: pd.DataFrame, family: str, config: dict[str, Any], output: Path) -> dict[str, Any]:
    role, design = BUILDERS[family](frame, config["splits"][family])
    assignments = frame[AUDIT_COLUMNS].copy()
    assignments.insert(0, "split_family", family)
    assignments.insert(1, "split_role", role.astype(str))
    assignments["disposition"] = np.where(assignments["split_role"].isin(ACTIVE_ROLES), "active", "excluded")
    family_dir = output / family
    family_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = family_dir / "assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)
    membership = assignments.loc[assignments["disposition"].eq("active"), ["segment_id", "split_role"]].copy()
    membership_path = family_dir / "model_membership.parquet"
    membership.to_parquet(membership_path, index=False)
    for role_name in ACTIVE_ROLES:
        membership.loc[membership["split_role"].eq(role_name), ["segment_id"]].to_parquet(family_dir / f"{role_name}_ids.parquet", index=False)
    stats = role_stats(assignments)
    pd.DataFrame([{key: value for key, value in row.items() if key != "engine_type_counts"} for row in stats]).to_csv(family_dir / "split_stats.csv", index=False)
    return {
        "family": family,
        "design": design,
        "assignment_path": str(assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "membership_path": str(membership_path),
        "membership_sha256": sha256_file(membership_path),
        "assignment_rows": int(len(assignments)),
        "active_rows": int(len(membership)),
        "role_stats": stats,
    }


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
    pointer = load_json(Path(config["segment_pointer"]))
    if pointer["status"] != "PASS_SEGMENT_SPLIT_60S_SEGMENT_PRODUCTION_GATE":
        raise RuntimeError("Frozen primary 60 s segment gate is not PASS.")
    segment_summary = load_json(Path(pointer["summary"]))
    if segment_summary.get("resume_noop_verified") is not True:
        raise RuntimeError("Primary 60 s no-op resume has not been verified.")
    segments_path = Path(pointer["segments_prediction_usable"])
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "segment_summary_sha256": sha256_file(Path(pointer["summary"])),
        "segments_prediction_usable_sha256": sha256_file(segments_path),
    }
    output = args.run_dir.resolve() if args.run_dir else Path(config["output_root"]) / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.json"
    summary_path = output / "split_build_summary.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
        if summary_path.exists() and manifest.get("status") == "PASS_SEGMENT_SPLIT_SPLIT_BUILD":
            summary = load_json(summary_path)
            summary["resume_noop_verified"] = True
            summary["resume_noop_elapsed_seconds"] = time.perf_counter() - started
            summary["last_resume_check_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
    shutil.copy2(config_path, output / "run_config.yaml")
    frame = pd.read_parquet(segments_path)
    missing = sorted(set(AUDIT_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Frozen segment input misses split columns: {missing}")
    if not frame["segment_id"].is_unique or not frame["prediction_usable"].all():
        raise RuntimeError("Split input must contain unique, prediction-usable segments only.")
    family_summaries = [build_family(frame, family, config, output) for family in FAMILIES]
    contract = {
        "schema_version": 1,
        "purpose": "Split membership only; no model features are released by SEGMENT_SPLIT.",
        "model_membership_columns": ["segment_id", "split_role"],
        "join_only_columns": ["segment_id"],
        "role_only_columns": ["split_role"],
        "model_feature_columns": [],
        "audit_only_assignment_columns": ["split_family", "split_role", "disposition"] + AUDIT_COLUMNS,
        "forbidden_model_feature_exact": config["forbidden_model_feature_exact"],
        "forbidden_model_feature_substrings": config["forbidden_model_feature_substrings"],
    }
    contract_path = output / "model_membership_contract.json"
    write_json(contract_path, contract)
    manifest = {
        "schema_version": 1,
        "experiment": "SEGMENT_SPLIT",
        "stage": "leakage_controlled_split_build",
        "status": "PASS_SEGMENT_SPLIT_SPLIT_BUILD",
        "input_fingerprint": fingerprint,
        "families": family_summaries,
        "membership_contract": str(contract_path),
        "membership_contract_sha256": sha256_file(contract_path),
    }
    write_json(manifest_path, manifest)
    summary = {
        "experiment": "SEGMENT_SPLIT",
        "stage": "leakage_controlled_split_build",
        "status": "PASS_SEGMENT_SPLIT_SPLIT_BUILD",
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
    write_json(summary_path, summary)
    build_pointer = {
        "status": summary["status"],
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "output_directory": str(output),
        "model_training_authorized": False,
    }
    write_json(Path(config["output_root"]) / f"{config['latest_prefix']}_build_latest_pointer.json", build_pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
