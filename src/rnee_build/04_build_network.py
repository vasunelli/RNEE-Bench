#!/usr/bin/env python3
"""Crop historical OSM snapshots and build separate Valhalla graphs for NETWORK_FREEZE."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import osmium
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_from_geojson(path: Path) -> tuple[float, float, float, float]:
    data = load_json(path)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def point_in_bbox(
    lon: float, lat: float, bbox: tuple[float, float, float, float]
) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def segment_intersects_bbox(
    a: tuple[float, float],
    b: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Liang-Barsky line clipping test."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - min_lon, max_lon - x0, y0 - min_lat, max_lat - y0)
    lower, upper = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


class IntersectingWaySelector(osmium.SimpleHandler):
    def __init__(self, bbox: tuple[float, float, float, float]):
        super().__init__()
        self.bbox = bbox
        self.way_ids: set[int] = set()
        self.node_ids: set[int] = set()
        self.invalid_location_way_count = 0

    def way(self, way: osmium.osm.Way) -> None:
        if "highway" not in way.tags:
            return
        points: list[tuple[float, float]] = []
        try:
            for node in way.nodes:
                if node.location.valid():
                    points.append((node.lon, node.lat))
                else:
                    self.invalid_location_way_count += 1
                    return
        except osmium.InvalidLocationError:
            self.invalid_location_way_count += 1
            return
        intersects = any(point_in_bbox(*point, self.bbox) for point in points)
        if not intersects:
            intersects = any(
                segment_intersects_bbox(a, b, self.bbox)
                for a, b in zip(points[:-1], points[1:])
            )
        if intersects:
            self.way_ids.add(way.id)
            self.node_ids.update(node.ref for node in way.nodes)


class RestrictionRelationSelector(osmium.SimpleHandler):
    def __init__(self, selected_way_ids: set[int], allowed_types: set[str]):
        super().__init__()
        self.selected_way_ids = selected_way_ids
        self.allowed_types = allowed_types
        self.relation_ids: set[int] = set()
        self.related_way_ids: set[int] = set()
        self.related_node_ids: set[int] = set()

    def relation(self, relation: osmium.osm.Relation) -> None:
        relation_type = relation.tags.get("type", "")
        if relation_type not in self.allowed_types:
            return
        way_members = {
            member.ref for member in relation.members if member.type == "w"
        }
        if not way_members.intersection(self.selected_way_ids):
            return
        self.relation_ids.add(relation.id)
        self.related_way_ids.update(way_members)
        self.related_node_ids.update(
            member.ref for member in relation.members if member.type == "n"
        )


class RelatedWayNodeCollector(osmium.SimpleHandler):
    def __init__(self, way_ids: set[int]):
        super().__init__()
        self.way_ids = way_ids
        self.node_ids: set[int] = set()
        self.found_way_ids: set[int] = set()

    def way(self, way: osmium.osm.Way) -> None:
        if way.id in self.way_ids:
            self.found_way_ids.add(way.id)
            self.node_ids.update(node.ref for node in way.nodes)


class CropWriter(osmium.SimpleHandler):
    def __init__(
        self,
        writer: osmium.SimpleWriter,
        node_ids: set[int],
        way_ids: set[int],
        relation_ids: set[int],
    ):
        super().__init__()
        self.writer = writer
        self.node_ids = node_ids
        self.way_ids = way_ids
        self.relation_ids = relation_ids
        self.counts = {"nodes": 0, "ways": 0, "relations": 0}

    def node(self, node: osmium.osm.Node) -> None:
        if node.id in self.node_ids:
            self.writer.add_node(node)
            self.counts["nodes"] += 1

    def way(self, way: osmium.osm.Way) -> None:
        if way.id in self.way_ids:
            self.writer.add_way(way)
            self.counts["ways"] += 1

    def relation(self, relation: osmium.osm.Relation) -> None:
        if relation.id in self.relation_ids:
            self.writer.add_relation(relation)
            self.counts["relations"] += 1


class CropAudit(osmium.SimpleHandler):
    def __init__(self, tags: list[str]):
        super().__init__()
        self.counts = {"nodes": 0, "ways": 0, "relations": 0}
        self.highway_ways = 0
        self.tag_counts = {tag: 0 for tag in tags}
        self.zero_or_single_node_highway_ways = 0

    def node(self, _: osmium.osm.Node) -> None:
        self.counts["nodes"] += 1

    def way(self, way: osmium.osm.Way) -> None:
        self.counts["ways"] += 1
        if "highway" in way.tags:
            self.highway_ways += 1
            if len(way.nodes) < 2:
                self.zero_or_single_node_highway_ways += 1
            for tag in self.tag_counts:
                if tag in way.tags:
                    self.tag_counts[tag] += 1

    def relation(self, _: osmium.osm.Relation) -> None:
        self.counts["relations"] += 1


def crop_snapshot(
    source: Path,
    destination: Path,
    bbox: tuple[float, float, float, float],
    relation_types: set[str],
    audit_tags: list[str],
) -> dict[str, Any]:
    selector = IntersectingWaySelector(bbox)
    selector.apply_file(str(source), locations=True, idx="flex_mem")

    relations = RestrictionRelationSelector(selector.way_ids, relation_types)
    relations.apply_file(str(source))
    all_way_ids = selector.way_ids | relations.related_way_ids

    related_only = relations.related_way_ids - selector.way_ids
    related_nodes = RelatedWayNodeCollector(related_only)
    if related_only:
        related_nodes.apply_file(str(source))
    all_node_ids = (
        selector.node_ids | relations.related_node_ids | related_nodes.node_ids
    )

    writer = osmium.SimpleWriter(str(destination))
    crop_writer = CropWriter(
        writer, all_node_ids, all_way_ids, relations.relation_ids
    )
    crop_writer.apply_file(str(source))
    writer.close()

    audit = CropAudit(audit_tags)
    audit.apply_file(str(destination))
    result = {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "crop_path": str(destination),
        "crop_bytes": destination.stat().st_size,
        "crop_sha256": sha256_file(destination),
        "bbox": {
            "min_lon": bbox[0],
            "min_lat": bbox[1],
            "max_lon": bbox[2],
            "max_lat": bbox[3],
        },
        "selection": {
            "intersecting_highway_way_count": len(selector.way_ids),
            "restriction_relation_count": len(relations.relation_ids),
            "related_way_count": len(related_only),
            "required_node_count": len(all_node_ids),
            "missing_related_way_count": len(
                related_only - related_nodes.found_way_ids
            ),
            "invalid_location_way_count": selector.invalid_location_way_count,
        },
        "written_counts": crop_writer.counts,
        "audit_counts": audit.counts,
        "highway_way_count": audit.highway_ways,
        "zero_or_single_node_highway_ways": audit.zero_or_single_node_highway_ways,
        "tag_counts": audit.tag_counts,
    }
    del selector, relations, related_nodes, crop_writer, audit
    gc.collect()
    return result


def generate_valhalla_config(
    base: dict[str, Any],
    network_config: dict[str, Any],
    graph_dir: Path,
    config_path: Path,
) -> None:
    runtime_root = Path(base["valhalla"]["runtime_root"])
    generator = runtime_root / "valhalla" / "valhalla_build_config.py"
    tile_dir = graph_dir / "tiles"
    quality_dir = graph_dir / "data_quality"
    tile_dir.mkdir(parents=True, exist_ok=False)
    quality_dir.mkdir(parents=True, exist_ok=False)
    settings = network_config["valhalla"]
    command = [
        sys.executable,
        str(generator),
        "--mjolnir-tile-dir",
        str(tile_dir),
        "--mjolnir-tile-extract",
        str(settings["tile_extract"]),
        "--mjolnir-admin",
        str(settings["admin"]),
        "--mjolnir-timezone",
        str(settings["timezone"]),
        "--mjolnir-traffic-extract",
        str(settings["traffic_extract"]),
        "--mjolnir-data-quality-dir",
        str(quality_dir),
        "--mjolnir-concurrency",
        str(settings["concurrency"]),
        "--mjolnir-include-driving",
        str(settings["include_driving"]).lower(),
        "--mjolnir-include-bicycle",
        str(settings["include_bicycle"]).lower(),
        "--mjolnir-include-pedestrian",
        str(settings["include_pedestrian"]).lower(),
        "--mjolnir-data-processing-use-admin-db",
        str(settings["use_admin_db"]).lower(),
        "--additional-data-elevation",
        str(settings["elevation"]),
        "--logging-type",
        "std_out",
        "--output",
        str(config_path),
    ]
    subprocess.run(command, check=True, timeout=120)


def runtime_environment(base: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        str(Path(base["valhalla"]["dll_directory"]))
        + os.pathsep
        + env.get("PATH", "")
    )
    return env


def build_graph(
    base: dict[str, Any],
    config: dict[str, Any],
    pbf_path: Path,
    graph_dir: Path,
) -> dict[str, Any]:
    valhalla_config = graph_dir / "valhalla.json"
    generate_valhalla_config(base, config, graph_dir, valhalla_config)
    executable = Path(base["valhalla"]["build_tiles_executable"])
    log_path = graph_dir / "valhalla_build.log"
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [
                str(executable),
                "-c",
                str(valhalla_config),
                "-j",
                str(config["valhalla"]["concurrency"]),
                str(pbf_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=4 * 60 * 60,
            env=runtime_environment(base),
            check=False,
        )
    completed = datetime.now(timezone.utc)
    tile_dir = graph_dir / "tiles"
    artifact_files = [path for path in tile_dir.rglob("*") if path.is_file()]
    graph_tile_files = [path for path in artifact_files if path.suffix == ".gph"]
    temporary_files = [
        path
        for path in artifact_files
        if path.suffix in {".tmp", ".temp"} or ".tmp" in path.name
    ]
    return {
        "config_path": str(valhalla_config),
        "config_sha256": sha256_file(valhalla_config),
        "build_log_path": str(log_path),
        "returncode": process.returncode,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "tile_count": len(graph_tile_files),
        "artifact_file_count": len(artifact_files),
        "temporary_file_count": len(temporary_files),
        "graph_bytes": sum(path.stat().st_size for path in graph_tile_files),
        "tile_directory": str(tile_dir),
    }


def locate_graph_samples(
    base: dict[str, Any],
    config: dict[str, Any],
    snapshot_id: str,
    graph_config_path: Path,
    gps_trips: pd.DataFrame,
) -> dict[str, Any]:
    candidates = gps_trips[
        (gps_trips["osm_snapshot_id"] == snapshot_id)
        & gps_trips["gps_basic_valid"].astype(bool)
    ].copy()
    count = int(config["valhalla"]["locate_sample_count_per_snapshot"])
    if len(candidates) > count:
        positions = [
            round(index * (len(candidates) - 1) / (count - 1))
            for index in range(count)
        ]
        candidates = candidates.sort_values(
            ["trip_mid_date", "source_file", "VehId", "Trip"]
        ).iloc[positions]
    locations = [
        {
            "lat": float((row.min_lat + row.max_lat) / 2),
            "lon": float((row.min_lon + row.max_lon) / 2),
        }
        for row in candidates.itertuples(index=False)
    ]
    request = json.dumps(
        {"verbose": True, "locations": locations, "costing": "auto"},
        separators=(",", ":"),
    )
    service = Path(base["valhalla"]["service_executable"])
    process = subprocess.run(
        [str(service), str(graph_config_path), "locate", request],
        capture_output=True,
        text=True,
        timeout=180,
        env=runtime_environment(base),
        check=False,
    )
    response = None
    parse_error = None
    if process.returncode == 0:
        try:
            response = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            parse_error = str(error)
    successful = 0
    minimum_reachability = None
    if isinstance(response, list):
        location_max_reachability: list[int] = []
        for item in response:
            edges = item.get("edges", [])
            if edges:
                successful += 1
                location_max_reachability.append(
                    max(
                        min(
                            int(edge.get("inbound_reach", 0)),
                            int(edge.get("outbound_reach", 0)),
                        )
                        for edge in edges
                    )
                )
        if location_max_reachability:
            minimum_reachability = min(location_max_reachability)
    return {
        "request_count": len(locations),
        "returncode": process.returncode,
        "response_parse_error": parse_error,
        "successful_location_count": successful,
        "success_rate": successful / len(locations) if locations else 0.0,
        "minimum_reported_edge_reachability": minimum_reachability,
        "stderr_tail": process.stderr[-2000:],
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NETWORK_FREEZE historical road-network and Valhalla graph gate",
        "",
        f"Decision: `{summary['status']}`",
        "",
        "## Snapshot graphs",
        "",
        "| Snapshot | Cropped highway ways | Tiles | Graph MB | Locate success | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["snapshots"]:
        lines.append(
            "| {snapshot_id} | {ways:,} | {tiles:,} | {mb:.2f} | "
            "{success:.2%} | `{status}` |".format(
                snapshot_id=item["snapshot_id"],
                ways=item["crop"]["highway_way_count"],
                tiles=item["graph"]["tile_count"],
                mb=item["graph"]["graph_bytes"] / (1024**2),
                success=item["locate"]["success_rate"],
                status=item["status"],
            )
        )
    lines.extend(
        [
            "",
            "Each graph was built only from its assigned frozen historical PBF.",
            "Graph success does not imply map-matching success; MAP_MATCHING remains a",
            "separate trip-level matching-quality gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/network_build.yaml"),
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    base = load_yaml(Path(config["base_contract"]))
    preflight = load_json(Path(config["preflight_summary"]))
    if preflight.get("status") != "PASS_CONTRACT_AND_ENVIRONMENT":
        raise RuntimeError("RUNTIME_FREEZE must pass before historical graph construction.")

    disk = shutil.disk_usage(Path(base["runtime"]["disk"]["build_drive"]))
    free_disk_gb = disk.free / (1024**3)
    minimum_free = float(base["runtime"]["disk"]["stop_full_build_below_free_gb"])
    if free_disk_gb < minimum_free:
        raise RuntimeError(
            f"Free disk {free_disk_gb:.2f} GB is below {minimum_free:.2f} GB."
        )

    snapshot_manifest = load_json(Path(config["snapshot_manifest"]))
    bbox = bbox_from_geojson(Path(config["research_bbox"]))
    gps_trips = pd.read_parquet(config["gps_trip_quality"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    output_dir = output_root / f"network_freeze_network_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_dir / "run_config.yaml")
    shutil.copy2(config["base_contract"], output_dir / "base_contract.yaml")

    results: list[dict[str, Any]] = []
    qa = config["qa"]
    for snapshot in snapshot_manifest:
        snapshot_id = snapshot["snapshot_id"]
        snapshot_dir = output_dir / snapshot_id
        snapshot_dir.mkdir()
        crop_path = snapshot_dir / "study_area.osm.pbf"
        crop = crop_snapshot(
            Path(snapshot["local_path"]),
            crop_path,
            bbox,
            set(config["crop"]["include_relation_types"]),
            list(qa["required_tag_coverage_fields"]),
        )
        graph_dir = snapshot_dir / "valhalla"
        graph_dir.mkdir()
        graph = build_graph(base, config, crop_path, graph_dir)
        locate = locate_graph_samples(
            base,
            config,
            snapshot_id,
            Path(graph["config_path"]),
            gps_trips,
        )
        checks = {
            "source_hash_matches_snapshot_manifest": (
                crop["source_sha256"] == snapshot["sha256"]
            ),
            "crop_has_minimum_highway_ways": (
                crop["highway_way_count"]
                >= int(qa["minimum_cropped_highway_ways"])
            ),
            "crop_relation_way_closure": (
                crop["selection"]["missing_related_way_count"] == 0
            ),
            "crop_has_no_degenerate_highway_ways": (
                crop["zero_or_single_node_highway_ways"] == 0
            ),
            "valhalla_build_returncode": graph["returncode"] == 0,
            "graph_has_minimum_tiles": (
                graph["tile_count"] >= int(qa["minimum_graph_tile_count"])
            ),
            "graph_has_minimum_bytes": (
                graph["graph_bytes"] >= int(qa["minimum_graph_bytes"])
            ),
            "graph_has_no_temporary_files": graph["temporary_file_count"] == 0,
            "locate_success_rate": (
                locate["success_rate"] >= float(qa["minimum_locate_success_rate"])
            ),
            "locate_reachability": (
                locate["minimum_reported_edge_reachability"] is not None
                and locate["minimum_reported_edge_reachability"]
                >= int(qa["minimum_locate_edge_reachability"])
            ),
        }
        item = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot["snapshot_date"],
            "crop": crop,
            "graph": graph,
            "locate": locate,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
        write_json(snapshot_dir / "network_qa.json", item)
        results.append(item)

    passed = all(item["status"] == "PASS" for item in results)
    summary = {
        "experiment": "NETWORK_FREEZE",
        "status": "PASS_OSM_SNAPSHOT_AND_NETWORK" if passed else "FAIL_OSM_NETWORK",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_freeze_fingerprint": preflight["environment_fingerprint_sha256"],
        "free_disk_gb_at_start": free_disk_gb,
        "bbox": {
            "min_lon": bbox[0],
            "min_lat": bbox[1],
            "max_lon": bbox[2],
            "max_lat": bbox[3],
        },
        "snapshots": results,
        "output_directory": str(output_dir),
    }
    write_json(output_dir / "network_summary.json", summary)
    report = report_markdown(summary)
    (output_dir / "NETWORK_FREEZE_HISTORICAL_NETWORK_GATE.md").write_text(
        report, encoding="utf-8"
    )
    pointer = {
        "status": summary["status"],
        "output_directory": str(output_dir),
        "summary": str(output_dir / "network_summary.json"),
        "report": str(output_dir / "NETWORK_FREEZE_HISTORICAL_NETWORK_GATE.md"),
        "graphs": {
            item["snapshot_id"]: item["graph"]["tile_directory"]
            for item in results
        },
        "valhalla_configs": {
            item["snapshot_id"]: item["graph"]["config_path"] for item in results
        },
        "cropped_pbfs": {
            item["snapshot_id"]: item["crop"]["crop_path"] for item in results
        },
    }
    write_json(output_root / "network_freeze_latest_network_pointer.json", pointer)
    shutil.copy2(
        output_dir / "network_summary.json",
        output_root / "network_freeze_latest_network_summary.json",
    )
    shutil.copy2(
        output_dir / "NETWORK_FREEZE_HISTORICAL_NETWORK_GATE.md",
        output_root / "network_freeze_latest_network_report.md",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
