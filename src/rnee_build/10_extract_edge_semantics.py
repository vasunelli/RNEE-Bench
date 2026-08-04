#!/usr/bin/env python3
"""Build EDGE_SEMANTICS pilot edge semantics from matched way IDs and historical OSM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import osmium
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_osm_parser", HERE / "11_parse_osm_tags.py")
PARSER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSER)


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


class TargetWayHandler(osmium.SimpleHandler):
    def __init__(self, target_ids: set[int], tag_names: list[str]) -> None:
        super().__init__()
        self.target_ids = target_ids
        self.tag_names = tag_names
        self.records: list[dict[str, Any]] = []

    def way(self, way: Any) -> None:
        if int(way.id) not in self.target_ids:
            return
        tags = {name: way.tags.get(name) for name in self.tag_names}
        self.records.append({"way_id": int(way.id), **tags})


def extract_target_ways(pbf: Path, target_ids: set[int], tag_names: list[str]) -> pd.DataFrame:
    handler = TargetWayHandler(target_ids, tag_names)
    handler.apply_file(str(pbf), locations=False)
    frame = pd.DataFrame(handler.records)
    if frame.empty:
        return pd.DataFrame(columns=["way_id", *tag_names])
    return frame.drop_duplicates("way_id", keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/edge_semantics.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    close = load_json(Path(config["map_matching_close_decision"]))
    if config.get("require_edge_semantics_authorization", True) and close.get("edge_semantics_authorized") is not True:
        raise RuntimeError("MAP_MATCHING has not authorized EDGE_SEMANTICS.")
    raw = load_json(Path(config["map_match_pointer"]))
    network = load_json(Path(config["network_pointer"]))
    edges = pd.read_parquet(raw["matched_edges"])
    edges = edges[edges["profile"].eq(config["selected_profile"])].copy()
    if edges.empty:
        raise RuntimeError("Selected MAP_MATCHING profile has no matched edges.")
    edges["way_id"] = pd.to_numeric(edges["way_id"], errors="coerce").astype("Int64")
    matched = edges[edges["way_id"].notna()].copy()
    tags_by_snapshot = []
    cached_tags = None
    if config.get("osm_way_tag_cache_pointer"):
        cache_pointer = load_json(Path(config["osm_way_tag_cache_pointer"]))
        if not str(cache_pointer.get("status", "")).startswith("PASS_TRAJECTORY_BUILD_OSM_WAY_TAG_CACHE"):
            raise RuntimeError("TRAJECTORY_BUILD OSM way-tag cache has not passed validation.")
        cached_tags = pd.read_parquet(cache_pointer["osm_way_tags"])
    for snapshot_id, group in matched.groupby("osm_snapshot_id"):
        target_ids = set(group["way_id"].astype(int))
        if cached_tags is None:
            frame = extract_target_ways(Path(network["cropped_pbfs"][snapshot_id]), target_ids, config["osm_tags"])
            frame.insert(0, "osm_snapshot_id", snapshot_id)
        else:
            frame = cached_tags[
                cached_tags["osm_snapshot_id"].eq(snapshot_id)
                & cached_tags["way_id"].isin(target_ids)
            ].copy()
            if len(frame) != len(target_ids):
                raise RuntimeError(f"OSM tag cache does not cover all requested ways for {snapshot_id}.")
            # Preserve the object/string-null representation produced by direct
            # PyOsmium extraction so cached and uncached paths remain exact.
            for tag_name in config["osm_tags"]:
                values = frame[tag_name].astype(object)
                missing_value = float("nan") if values.notna().any() else None
                frame[tag_name] = values.where(values.notna(), missing_value)
        tags_by_snapshot.append(frame)
    osm_tags = pd.concat(tags_by_snapshot, ignore_index=True)
    joined = matched.merge(
        osm_tags,
        on=["osm_snapshot_id", "way_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_valhalla", "_osm"),
    )
    joined["osm_join_found"] = joined["_merge"].eq("both")
    joined = joined.drop(columns="_merge")
    mapping = config["functional_road_class"]
    joined["functional_road_class"] = joined["highway"].map(lambda value: PARSER.functional_class(value, mapping))
    joined["road_class_source"] = joined["functional_road_class"].map(lambda value: "osm_highway" if pd.notna(value) else "missing")
    joined_records = joined.to_dict(orient="records")
    speed_records = [
        PARSER.choose_speed(row, bool(row.get("forward")), row.get("speed_limit_kmh"), config["speed"])
        for row in joined_records
    ]
    joined = pd.concat([joined.reset_index(drop=True), pd.DataFrame(speed_records)], axis=1)
    lane_records = []
    for tags in joined.to_dict(orient="records"):
        directional = tags.get("lanes:forward") if bool(tags.get("forward")) else tags.get("lanes:backward")
        parsed = PARSER.parse_lane_count(directional if directional not in {None, ""} else tags.get("lanes"))
        lane_records.append({"osm_lane_count": parsed["lane_count"], "lane_parse_status": parsed["status"], "lane_raw": parsed["raw"]})
    joined = pd.concat([joined.reset_index(drop=True), pd.DataFrame(lane_records)], axis=1)
    joined["bridge_osm_raw"] = joined["bridge_osm"]
    joined["tunnel_osm_raw"] = joined["tunnel_osm"]
    joined["bridge_osm"] = joined["bridge_osm_raw"].map(PARSER.truthy_osm)
    joined["tunnel_osm"] = joined["tunnel_osm_raw"].map(PARSER.truthy_osm)
    joined["roundabout_osm"] = joined["junction"].astype("string").str.lower().eq("roundabout")
    speed = pd.to_numeric(joined["speed_limit_kmh_used"], errors="coerce")
    lanes = pd.to_numeric(joined["osm_lane_count"], errors="coerce")
    joined["speed_anomaly"] = speed.notna() & ((speed < config["speed"]["anomaly_below_kmh"]) | (speed > config["speed"]["anomaly_above_kmh"]))
    joined["lane_anomaly"] = lanes.notna() & ((lanes < config["lane"]["anomaly_below"]) | (lanes > config["lane"]["anomaly_above"]))

    points = pd.read_parquet(raw["matched_points"])
    points = points[
        points["profile"].eq(config["selected_profile"])
        & points["match_status"].eq("matched_with_edge")
    ].copy()
    edge_keys = ["pilot_trip_id", "osm_snapshot_id", "profile", "edge_index"]
    if joined.duplicated(edge_keys).any():
        raise RuntimeError("Matched edge indexes are not unique within trip/profile.")
    point_edge = points.merge(
        joined,
        left_on=["pilot_trip_id", "osm_snapshot_id", "profile", "matched_edge_index"],
        right_on=edge_keys,
        how="left",
        validate="many_to_one",
        indicator="edge_semantic_join",
        suffixes=("_point", "_edge"),
    )
    point_edge["edge_semantic_join_found"] = point_edge["edge_semantic_join"].eq("both")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    experiment = str(config.get("experiment", "EDGE_SEMANTICS"))
    run_prefix = str(config.get("run_name_prefix", "edge_semantics_edge_semantics_pilot"))
    latest_prefix = str(config.get("latest_prefix", "edge_semantics"))
    stage = str(config.get("stage", "edge_semantics_pilot"))
    status_suffix = str(config.get("status_suffix", "EDGE_SEMANTICS_PILOT"))
    output = output_root / f"{run_prefix}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "run_config.yaml")
    osm_tags.to_parquet(output / "target_osm_way_tags.parquet", index=False)
    joined.to_parquet(output / "matched_edge_semantics.parquet", index=False)
    point_edge.to_parquet(output / "matched_point_edge_semantics.parquet", index=False)
    joined[joined["speed_anomaly"] | joined["lane_anomaly"]].to_csv(output / "edge_semantic_anomalies.csv", index=False)
    confusion = pd.crosstab(joined["road_class"], joined["functional_road_class"], dropna=False)
    confusion.to_csv(output / "valhalla_osm_road_class_confusion.csv")

    missing_rate = float(point_edge["functional_road_class"].isna().mean())
    speed_nonmissing = point_edge["speed_limit_kmh_used"].notna()
    provenance_coverage = float(point_edge.loc[speed_nonmissing, "speed_limit_provenance"].notna().mean()) if speed_nonmissing.any() else 1.0
    qa = config["qa"]
    if missing_rate <= qa["functional_class_missing_rate_pass_max"]:
        class_gate = "PASS"
    elif missing_rate <= qa["functional_class_missing_rate_quality_check_max"]:
        class_gate = "CAUTION"
    else:
        class_gate = "FAIL"
    provenance_gate = "PASS" if provenance_coverage >= qa["speed_provenance_coverage_min"] else "FAIL"
    gate = "FAIL" if "FAIL" in {class_gate, provenance_gate} else ("CAUTION" if "CAUTION" in {class_gate, provenance_gate} else "PASS")
    summary = {
        "experiment": experiment,
        "stage": stage,
        "status": f"{gate}_{status_suffix}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": config["selected_profile"],
        "matched_edge_row_count": int(len(joined)),
        "matched_point_with_edge_count": int(len(point_edge)),
        "unique_target_way_count": int(matched[["osm_snapshot_id", "way_id"]].drop_duplicates().shape[0]),
        "osm_join_coverage": float(joined["osm_join_found"].mean()),
        "point_to_edge_semantic_join_coverage": float(point_edge["edge_semantic_join_found"].mean()),
        "functional_class_missing_rate": missing_rate,
        "functional_class_gate": class_gate,
        "speed_nonmissing_count": int(speed_nonmissing.sum()),
        "speed_provenance_coverage": provenance_coverage,
        "speed_provenance_gate": provenance_gate,
        "speed_provenance_counts": point_edge["speed_limit_provenance"].value_counts(dropna=False).to_dict(),
        "speed_anomaly_count": int(joined["speed_anomaly"].sum()),
        "lane_anomaly_count": int(joined["lane_anomaly"].sum()),
        "coordinate_nearest_join_used": False,
        "map_matching_close_status": close["status"],
        "full_build_authorized": False,
        "output_directory": str(output),
    }
    write_json(output / "edge_semantics_summary.json", summary)
    report = output / f"{experiment}_{status_suffix}.md"
    report.write_text(
        f"# {experiment} {stage.replace('_', ' ')}\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Matched edge rows: {len(joined):,}\n"
        f"- Historical OSM join coverage: {summary['osm_join_coverage']:.2%}\n"
        f"- Matched point-to-edge semantic join coverage: {summary['point_to_edge_semantic_join_coverage']:.2%}\n"
        f"- Row-level functional-class missing rate: {missing_rate:.2%} (`{class_gate}`)\n"
        f"- Speed provenance coverage among non-missing speeds: {provenance_coverage:.2%} (`{provenance_gate}`)\n"
        f"- Speed anomalies: {summary['speed_anomaly_count']:,}\n"
        f"- Lane anomalies: {summary['lane_anomaly_count']:,}\n\n"
        "Join keys are historical snapshot ID and matched OSM way ID. No coordinate nearest join or legacy eVED semantic field is used.\n",
        encoding="utf-8",
    )
    shutil.copy2(output / "edge_semantics_summary.json", output_root / f"{latest_prefix}_latest_summary.json")
    shutil.copy2(report, output_root / f"{latest_prefix}_latest_report.md")
    write_json(output_root / f"{latest_prefix}_latest_pointer.json", {"status": summary["status"], "output_directory": str(output), "summary": str(output / "edge_semantics_summary.json"), "report": str(report), "edge_semantics": str(output / "matched_edge_semantics.parquet"), "point_edge_semantics": str(output / "matched_point_edge_semantics.parquet"), "osm_way_tags": str(output / "target_osm_way_tags.parquet"), "anomalies": str(output / "edge_semantic_anomalies.csv"), "confusion_matrix": str(output / "valhalla_osm_road_class_confusion.csv")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid a PyOsmium finalization stall on Windows after all outputs close.
    os._exit(exit_code)
