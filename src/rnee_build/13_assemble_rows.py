#!/usr/bin/env python3
"""Assemble ROW_ASSEMBLY pilot rows with strict row conservation and field roles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
MAP_SPEC = importlib.util.spec_from_file_location("rnee_map_match", ROOT / "src" / "rnee_build" / "06_map_match_trips.py")
MAP_MODULE = importlib.util.module_from_spec(MAP_SPEC)
assert MAP_SPEC.loader is not None
MAP_SPEC.loader.exec_module(MAP_MODULE)
TARGET_SPEC = importlib.util.spec_from_file_location("rnee_target", ROOT / "src" / "rnee_build" / "02_audit_energy_target.py")
TARGET_MODULE = importlib.util.module_from_spec(TARGET_SPEC)
assert TARGET_SPEC.loader is not None
TARGET_SPEC.loader.exec_module(TARGET_MODULE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("base_config"):
        base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
        base.update({key: value for key, value in config.items() if key != "base_config"})
        return base
    return config


def row_id(source_file: str, vehicle: Any, trip: Any, point_index: Any) -> str:
    text = f"{source_file}|{vehicle}|{trip}|{point_index}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def value_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(lambda value: "<NA>" if pd.isna(value) else repr(value))
    return normalized.astype(str).agg("|".join, axis=1).map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())


def apply_road_semantics_availability(
    frame: pd.DataFrame,
    semantic_columns: list[str],
    trip_policy: pd.DataFrame | None,
) -> pd.DataFrame:
    """Withhold derived semantic fields without deleting rows or audit lineage."""
    result = frame.copy()
    edge_available = result.get("edge_semantic_join_found", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    result["road_semantics_available"] = edge_available
    result["road_semantics_exclusion_reason"] = ""
    result.loc[~edge_available, "road_semantics_exclusion_reason"] = "map_match_or_edge_semantics_unavailable"
    if trip_policy is not None and not trip_policy.empty:
        keys = ["source_file", "VehId", "Trip"]
        if trip_policy.duplicated(keys).any():
            raise RuntimeError("Road-semantic trip policy contains duplicate trip keys.")
        policy = trip_policy.loc[:, keys + ["road_semantics_available", "road_semantics_exclusion_reason"]].copy()
        policy = policy.rename(columns={
            "road_semantics_available": "quality_check_policy_available",
            "road_semantics_exclusion_reason": "quality_check_policy_reason",
        })
        result = result.merge(policy, on=keys, how="left", validate="many_to_one")
        inspected_abstain = result["quality_check_policy_available"].eq(False)
        result.loc[inspected_abstain, "road_semantics_available"] = False
        result.loc[inspected_abstain, "road_semantics_exclusion_reason"] = result.loc[inspected_abstain, "quality_check_policy_reason"]
        result = result.drop(columns=["quality_check_policy_available", "quality_check_policy_reason"])
    unavailable = ~result["road_semantics_available"]
    for column in semantic_columns:
        if column in result.columns:
            if pd.api.types.is_bool_dtype(result[column].dtype):
                result[column] = result[column].astype("boolean")
            elif pd.api.types.is_integer_dtype(result[column].dtype):
                result[column] = result[column].astype("Int64")
            result.loc[unavailable, column] = pd.NA
    return result


def unit_for(column: str) -> str | None:
    lower = column.lower()
    if lower.endswith("_m") or "distance_m" in lower:
        return "m"
    if lower.endswith("_s") or "time_s" in lower:
        return "s"
    if "speed" in lower and "kmh" in lower:
        return "km/h"
    if "latitude" in lower or "longitude" in lower or "[deg]" in lower:
        return "degree"
    if "ratio" in lower or "coverage" in lower:
        return "fraction"
    if "density_m_per_km2" in lower:
        return "m/km2"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/row_assembly.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); config = load_config(args.config)
    raw_pointer = load_json(Path(config["map_match_pointer"])); edge_pointer = load_json(Path(config["edge_semantics_pointer"])); context_pointer = load_json(Path(config["context_semantics_pointer"])); map_matching = load_json(Path(config["map_matching_close_decision"])); valhalla = yaml.safe_load(Path(config["valhalla_config"]).read_text(encoding="utf-8")); target_config = yaml.safe_load(Path(config["target_config"]).read_text(encoding="utf-8"))
    if not str(edge_pointer["status"]).startswith("PASS_EDGE_SEMANTICS") or not str(context_pointer["status"]).startswith("PASS_CONTEXT_SEMANTICS"):
        raise RuntimeError("EDGE_SEMANTICS and CONTEXT_SEMANTICS pilot gates must pass before ROW_ASSEMBLY.")
    selection = pd.read_parquet(raw_pointer["selection"])
    raw_frames = []
    for source_file, selected_file in selection.groupby("source_file"):
        source = pd.read_csv(Path(config["ved_dynamic_dir"]) / source_file)
        for selected in selected_file.itertuples(index=False):
            frame = source[
                source[valhalla["columns"]["vehicle"]].eq(selected.VehId)
                & source[valhalla["columns"]["trip"]].eq(selected.Trip)
            ].copy().reset_index(drop=True)
            if len(frame) != int(selected.row_count):
                raise RuntimeError(f"Full raw cardinality mismatch for {selected.pilot_trip_id}: {len(frame)} != {selected.row_count}.")
            frame.insert(0, "point_index", np.arange(len(frame), dtype=np.int64))
            frame.insert(0, "pilot_trip_id", selected.pilot_trip_id)
            frame.insert(0, "source_file", selected.source_file)
            raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)
    official_ved_columns = [column for column in raw.columns if column not in {"source_file", "pilot_trip_id", "point_index"}]
    lineage_columns = ["source_file", "pilot_trip_id", "point_index", *official_ved_columns]
    raw["row_id"] = [row_id(source, vehicle, trip, index) for source, vehicle, trip, index in zip(raw["source_file"], raw[valhalla["columns"]["vehicle"]], raw[valhalla["columns"]["trip"]], raw["point_index"])]
    if raw["row_id"].duplicated().any():
        raise RuntimeError("Raw pilot row_id is not unique.")

    timestamp = pd.to_numeric(raw[TARGET_MODULE.COL["timestamp"]], errors="coerce")
    next_timestamp = timestamp.groupby(raw["pilot_trip_id"], sort=False).shift(-1)
    dt = (next_timestamp - timestamp) / 1000.0
    stft = TARGET_MODULE.mean_available(raw, [TARGET_MODULE.COL["stft1"], TARGET_MODULE.COL["stft2"]])
    ltft = TARGET_MODULE.mean_available(raw, [TARGET_MODULE.COL["ltft1"], TARGET_MODULE.COL["ltft2"]])
    correction = 1.0 + stft / 100.0 + ltft / 100.0
    target = TARGET_MODULE.compute_interval_targets(
        dt=dt,
        fuel_rate=pd.to_numeric(raw[TARGET_MODULE.COL["fuel"]], errors="coerce"),
        maf=pd.to_numeric(raw[TARGET_MODULE.COL["maf"]], errors="coerce"),
        correction=correction,
        current=pd.to_numeric(raw[TARGET_MODULE.COL["current"]], errors="coerce"),
        voltage=pd.to_numeric(raw[TARGET_MODULE.COL["voltage"]], errors="coerce"),
        max_dt=float(target_config["integration"]["maximum_dt_seconds_inclusive"]),
        afr=float(target_config["fuel"]["stoichiometric_afr_e10"]),
        density=float(target_config["fuel"]["nominal_density_g_per_l"]),
        plausibility=target_config["quality_gates"]["plausibility"],
    )
    raw["dt_seconds"] = dt
    raw["dt_valid"] = target["valid_dt"]
    raw["fuel_volume_L"] = target["fuel_l"]
    raw["fuel_source"] = target["fuel_source"]
    raw["battery_terminal_net_Wh"] = target["battery_wh"].where(target["battery_ok"])
    trip_target = pd.read_parquet(config["trip_target_summary"])[["source_file", "VehId", "Trip", "EngineType_official", "trip_target_plausible", "target_release_eligible", "target_release_exclusion_reason"]].copy()
    trip_target["VehId"] = pd.to_numeric(trip_target["VehId"], errors="raise")
    trip_target["Trip"] = pd.to_numeric(trip_target["Trip"], errors="raise")
    raw = raw.merge(trip_target, on=["source_file", "VehId", "Trip"], how="left", validate="many_to_one")
    if raw["EngineType_official"].isna().any():
        raise RuntimeError("TARGET_AUDIT trip target metadata does not cover every pilot row.")
    required = pd.Series(False, index=raw.index)
    engine = raw["EngineType_official"]
    required.loc[engine.isin(["ICE", "HEV"])] = raw.loc[engine.isin(["ICE", "HEV"]), "fuel_volume_L"].notna()
    required.loc[engine.eq("PHEV")] = raw.loc[engine.eq("PHEV"), ["fuel_volume_L", "battery_terminal_net_Wh"]].notna().all(axis=1)
    required.loc[engine.eq("EV")] = raw.loc[engine.eq("EV"), "battery_terminal_net_Wh"].notna()
    raw["target_exclusion_reason"] = ""
    raw.loc[~raw["dt_valid"], "target_exclusion_reason"] = "invalid_or_terminal_dt"
    raw.loc[raw["dt_valid"] & ~required, "target_exclusion_reason"] = "missing_or_out_of_range_required_target_channel"

    points = pd.read_parquet(raw_pointer["matched_points"])
    points = points[points["profile"].eq(config["selected_profile"])].copy()
    point_keys = ["pilot_trip_id", "point_index"]
    if points.duplicated(point_keys).any():
        raise RuntimeError("Map-match point keys are duplicated.")
    point_columns = [column for column in points.columns if column not in {"source_file", "VehId", "Trip"}]
    assembled = raw.merge(points[point_columns], on=point_keys, how="left", validate="one_to_one", indicator="map_match_join")

    edge = pd.read_parquet(edge_pointer["point_edge_semantics"])
    if edge.duplicated(point_keys).any():
        raise RuntimeError("Edge semantic point keys are duplicated.")
    edge_keep = point_keys + [column for column in ["way_id", "edge_id", "road_class", "highway", "functional_road_class", "road_class_source", "speed_limit_kmh_used", "speed_limit_provenance", "speed_limit_is_inferred", "speed_limit_selected_tag", "speed_limit_parse_status", "osm_lane_count", "lane_parse_status", "surface_osm", "smoothness", "oneway", "access", "motor_vehicle", "bridge_osm", "tunnel_osm", "roundabout_osm", "speed_anomaly", "lane_anomaly", "edge_semantic_join_found"] if column in edge.columns]
    assembled = assembled.merge(edge[edge_keep], on=point_keys, how="left", validate="one_to_one", indicator="edge_semantic_join")

    context = pd.read_parquet(context_pointer["row_context"])
    if context.duplicated(point_keys).any():
        raise RuntimeError("Context semantic point keys are duplicated.")
    context_exclude = set(raw.columns) | set(points.columns) | {"row_semantic_id", "profile"}
    context_keep = point_keys + [column for column in context.columns if column not in context_exclude and column not in point_keys]
    assembled = assembled.merge(context[context_keep], on=point_keys, how="left", validate="one_to_one", indicator="context_semantic_join")

    trip_policy = None
    abstention_pointer_path = config.get("road_semantics_abstention_pointer")
    if abstention_pointer_path:
        abstention_pointer = load_json(Path(abstention_pointer_path))
        if not str(abstention_pointer.get("status", "")).startswith("PASS_TRAJECTORY_BUILD_QUALITY_CHECK1_OFF_NETWORK_ABSTENTION_POLICY"):
            raise RuntimeError("TRAJECTORY_BUILD road-semantic abstention policy has not passed validation.")
        trip_policy = pd.read_csv(abstention_pointer["trip_policy"])
    semantic_model_columns = sorted((set(edge_keep) | set(context_keep)) - set(point_keys) - {
        "edge_semantic_join_found", "edge_semantic_join", "context_semantic_join"
    })
    assembled = apply_road_semantics_availability(assembled, semantic_model_columns, trip_policy)

    # Keep partition schemas stable when a nullable numeric field happens to have
    # no missing values in one partition (otherwise pandas may narrow it to int).
    dtype_contract = config.get("output_dtype_contract", {})
    missing_dtype_contract_columns = sorted(set(dtype_contract) - set(assembled.columns))
    if missing_dtype_contract_columns:
        raise RuntimeError(f"Output dtype contract columns are missing: {missing_dtype_contract_columns}")
    for column, dtype in dtype_contract.items():
        assembled[column] = assembled[column].astype(dtype)

    input_count = len(raw); output_count = len(assembled); duplicate_row_ids = int(assembled["row_id"].duplicated().sum())
    sample = assembled.sample(n=min(len(assembled), int(config["raw_hash_sample_count"])), random_state=int(config["raw_hash_seed"]))
    raw_sample = raw.set_index("row_id").loc[sample["row_id"]]
    raw_columns = lineage_columns
    hash_match_rate = float((value_hash(sample.set_index("row_id").loc[:, raw_columns], raw_columns).to_numpy() == value_hash(raw_sample, raw_columns).to_numpy()).mean())

    target_set = {"dt_seconds", "dt_valid", "fuel_volume_L", "fuel_source", "battery_terminal_net_Wh", "target_exclusion_reason", "EngineType_official", "trip_target_plausible", "target_release_eligible", "target_release_exclusion_reason"}
    raw_set = set(raw.columns) - target_set; map_set = set(points.columns); edge_set = set(edge_keep); context_set = set(context_keep)
    forbidden = set(config["forbidden_model_columns"]) | set(config["target_source_forbidden_columns"])
    explicit_allowlist = set(config["explicit_model_allowlist"])
    missing_allowlist_columns = sorted(explicit_allowlist - set(assembled.columns))
    catalog = []
    model_allowlist = []
    audit_only = []
    for column in assembled.columns:
        if column in target_set:
            source = "target_audit_operational_target_contract"
        elif column in raw_set:
            source = "official_ved_raw"
        elif column in edge_set:
            source = "historical_osm_or_valhalla_edge_semantics"
        elif column in context_set:
            source = "historical_osm_context_semantics"
        elif column in map_set:
            source = "historical_valhalla_map_match"
        else:
            source = "assembly_qa"
        is_forbidden = column in forbidden
        is_audit = is_forbidden or column not in explicit_allowlist
        role = "forbidden" if is_forbidden else ("candidate_model_feature" if column in explicit_allowlist else "audit_only")
        if role == "candidate_model_feature": model_allowlist.append(column)
        else: audit_only.append(column)
        catalog.append({"field": column, "dtype": str(assembled[column].dtype), "unit": unit_for(column), "source": source, "role": role, "missing_rate": float(assembled[column].isna().mean()), "derivation_version": str(config.get("derivation_version", "ROW_ASSEMBLY-v1")), "missing_policy": "preserve_nullable_no_silent_fill"})
    forbidden_hits = sorted(set(model_allowlist) & forbidden)

    checks = {
        "row_ratio": "PASS" if output_count / input_count == config["qa"]["output_input_row_ratio"] else "FAIL",
        "row_id_unique": "PASS" if duplicate_row_ids <= config["qa"]["duplicate_row_id_count_max"] else "FAIL",
        "raw_hash_match": "PASS" if hash_match_rate >= config["qa"]["raw_hash_match_rate_min"] else "FAIL",
        "map_match_join_complete": "PASS" if assembled["map_match_join"].eq("both").all() else "FAIL",
        "forbidden_feature_hits": "PASS" if len(forbidden_hits) <= config["qa"]["forbidden_feature_hits_max"] else "FAIL",
        "explicit_allowlist_complete": "PASS" if len(missing_allowlist_columns) <= config["qa"]["explicit_allowlist_missing_count_max"] else "FAIL",
        "output_dtype_contract": "PASS" if all(str(assembled[column].dtype) == dtype for column, dtype in dtype_contract.items()) else "FAIL",
        "map_matching_waiver_carried": "PASS" if map_matching["status"] == "PASS_WITH_DOCUMENTED_MANUAL_WAIVER" else "FAIL",
    }
    gate = "FAIL" if "FAIL" in checks.values() else "PASS"
    experiment = str(config.get("experiment", "ROW_ASSEMBLY")); stage = str(config.get("stage", "row_assembly_pilot")); status_suffix = str(config.get("status_suffix", "ROW_ASSEMBLY_PILOT")); run_prefix = str(config.get("run_name_prefix", "row_assembly_row_assembly_pilot")); latest_prefix = str(config.get("latest_prefix", "row_assembly"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); output_root = Path(config["output_root"]); output = output_root / f"{run_prefix}_{stamp}"; output.mkdir(parents=True, exist_ok=False); shutil.copy2(args.config, output / "run_config.yaml")
    assembled.to_parquet(output / "enriched_trajectory_rows.parquet", index=False)
    pd.DataFrame(catalog).to_csv(output / "field_catalog.csv", index=False)
    write_json(output / "feature_roles.json", {"model_allowlist": model_allowlist, "audit_only_or_forbidden": audit_only, "explicit_forbidden": sorted(forbidden), "target_source_forbidden": config["target_source_forbidden_columns"], "forbidden_hits_in_model_allowlist": forbidden_hits, "missing_explicit_allowlist_columns": missing_allowlist_columns})
    summary = {"experiment": experiment, "stage": stage, "status": f"{gate}_{status_suffix}", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "input_row_count": input_count, "output_row_count": output_count, "output_input_row_ratio": output_count / input_count, "duplicate_row_id_count": duplicate_row_ids, "official_ved_source_column_count": len(official_ved_columns), "target_contract_field_count": len(target_set), "raw_hash_sample_count": len(sample), "raw_hash_match_rate": hash_match_rate, "map_match_join_coverage": float(assembled["map_match_join"].eq("both").mean()), "edge_semantic_join_coverage_all_rows": float(assembled["edge_semantic_join"].eq("both").mean()), "context_semantic_join_coverage_all_rows": float(assembled["context_semantic_join"].eq("both").mean()), "road_semantics_available_row_count": int(assembled["road_semantics_available"].sum()), "road_semantics_unavailable_row_count": int((~assembled["road_semantics_available"]).sum()), "road_semantics_exclusion_reason_counts": assembled["road_semantics_exclusion_reason"].value_counts(dropna=False).to_dict(), "field_count": len(assembled.columns), "model_allowlist_count": len(model_allowlist), "missing_explicit_allowlist_column_count": len(missing_allowlist_columns), "forbidden_feature_hit_count": len(forbidden_hits), "checks": checks, "map_matching_close_status": map_matching["status"], "full_build_authorized": False, "output_directory": str(output)}
    write_json(output / "row_assembly_summary.json", summary)
    report = output / f"{experiment}_{status_suffix}.md"; report.write_text(f"# {experiment} {stage.replace('_', ' ')}\n\n" + "\n".join([f"- Status: `{summary['status']}`", f"- Input/output rows: {input_count:,} / {output_count:,}", f"- Duplicate row IDs: {duplicate_row_ids}", f"- Raw hash sample match: {hash_match_rate:.2%} ({len(sample):,} rows)", f"- Map-match join coverage: {summary['map_match_join_coverage']:.2%}", f"- Edge semantics coverage across all source rows: {summary['edge_semantic_join_coverage_all_rows']:.2%}", f"- Context semantics coverage across all source rows: {summary['context_semantic_join_coverage_all_rows']:.2%}", f"- Forbidden fields admitted to model allowlist: {len(forbidden_hits)}", f"- MAP_MATCHING closure carried as: `{map_matching['status']}`."]) + "\n", encoding="utf-8")
    shutil.copy2(output / "row_assembly_summary.json", output_root / f"{latest_prefix}_latest_summary.json"); shutil.copy2(report, output_root / f"{latest_prefix}_latest_report.md")
    write_json(output_root / f"{latest_prefix}_latest_pointer.json", {"status": summary["status"], "output_directory": str(output), "summary": str(output / "row_assembly_summary.json"), "report": str(report), "rows": str(output / "enriched_trajectory_rows.parquet"), "field_catalog": str(output / "field_catalog.csv"), "feature_roles": str(output / "feature_roles.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
