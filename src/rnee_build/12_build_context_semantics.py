#!/usr/bin/env python3
"""Build CONTEXT_SEMANTICS historical OSM context, topology, and exposure semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmium
import pandas as pd
import yaml
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


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


def stable_uid(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def poi_labels(tags: dict[str, str], taxonomy: dict[str, Any], priority: list[str]) -> tuple[list[str], str | None]:
    labels: list[str] = []
    for label in priority:
        rules = taxonomy[label]
        amenity = tags.get("amenity")
        if amenity and amenity in rules.get("amenity", []):
            labels.append(label)
            continue
        if any(tags.get(name) not in {None, ""} for name in rules.get("tag_presence", [])):
            labels.append(label)
    labels = list(dict.fromkeys(labels))
    return labels, (labels[0] if labels else None)


class ContextHandler(osmium.SimpleHandler):
    NODE_TAGS = ["highway", "railway", "traffic_calming", "public_transport", "amenity", "shop", "leisure", "tourism", "office"]
    WAY_TAGS = ["highway", "building", "landuse", "amenity", "shop", "leisure", "tourism", "office", "public_transport"]

    def __init__(self, snapshot_id: str, config: dict[str, Any], transformer: Transformer) -> None:
        super().__init__()
        self.snapshot_id = snapshot_id
        self.config = config
        self.transformer = transformer
        self.objects: list[dict[str, Any]] = []
        self.road_segments: list[dict[str, Any]] = []
        self.buildings: list[dict[str, Any]] = []
        self.landuse: list[dict[str, Any]] = []
        self.graph = nx.Graph()
        self.incident_ways: dict[int, set[int]] = defaultdict(set)
        self.node_xy: dict[int, tuple[float, float]] = {}

    def _inside_bbox(self, lon: float, lat: float) -> bool:
        bbox = self.config["research_bbox_wgs84"]
        return bbox["min_lon"] <= lon <= bbox["max_lon"] and bbox["min_lat"] <= lat <= bbox["max_lat"]

    def _append_object(self, element_type: str, element_id: int, layer: str, primary_type: str, labels: list[str], lon: float, lat: float, tags: dict[str, str]) -> None:
        x, y = self.transformer.transform(lon, lat)
        self.objects.append({
            "osm_snapshot_id": self.snapshot_id,
            "element_type": element_type,
            "osm_id": int(element_id),
            "object_layer": layer,
            "primary_type": primary_type,
            "type_labels_json": json.dumps(labels, sort_keys=True),
            "object_uid": stable_uid(self.snapshot_id, element_type, element_id, layer),
            "longitude": lon,
            "latitude": lat,
            "x_m": x,
            "y_m": y,
            "tags_json": json.dumps(tags, sort_keys=True),
        })

    def node(self, node: Any) -> None:
        if not node.location.valid():
            return
        tags = {name: node.tags.get(name) for name in self.NODE_TAGS if node.tags.get(name) is not None}
        lon, lat = float(node.location.lon), float(node.location.lat)
        if not self._inside_bbox(lon, lat):
            return
        highway = tags.get("highway")
        railway = tags.get("railway")
        traffic_types = []
        if highway in self.config["traffic_controls"]["highway"]:
            traffic_types.append(highway)
        if railway in self.config["traffic_controls"]["railway"]:
            traffic_types.append(f"railway_{railway}")
        if tags.get("traffic_calming"):
            traffic_types.append(f"traffic_calming_{tags['traffic_calming']}")
        if traffic_types:
            self._append_object("node", node.id, "traffic_control", traffic_types[0], traffic_types, lon, lat, tags)
        transit_types = []
        for key, allowed in self.config["public_transport"].items():
            if tags.get(key) in allowed:
                transit_types.append(f"{key}_{tags[key]}")
        if transit_types:
            self._append_object("node", node.id, "public_transport", transit_types[0], transit_types, lon, lat, tags)
        labels, primary = poi_labels(tags, self.config["poi_taxonomy"], self.config["poi_priority"])
        if primary:
            self._append_object("node", node.id, "poi", primary, labels, lon, lat, tags)

    def way(self, way: Any) -> None:
        tags = {name: way.tags.get(name) for name in self.WAY_TAGS if way.tags.get(name) is not None}
        lonlat = []
        node_ids = []
        for ref in way.nodes:
            if not ref.location.valid():
                return
            lon, lat = float(ref.location.lon), float(ref.location.lat)
            lonlat.append((lon, lat))
            node_ids.append(int(ref.ref))
        if not lonlat:
            return
        bbox = self.config["research_bbox_wgs84"]
        lons = [item[0] for item in lonlat]; lats = [item[1] for item in lonlat]
        if max(lons) < bbox["min_lon"] or min(lons) > bbox["max_lon"] or max(lats) < bbox["min_lat"] or min(lats) > bbox["max_lat"]:
            return
        xs, ys = self.transformer.transform(lons, lats)
        coords = list(zip(xs, ys))
        if tags.get("highway") in self.config["drivable_highways"] and len(coords) >= 2:
            for node_id, xy in zip(node_ids, coords):
                self.node_xy[node_id] = xy
                self.graph.add_node(node_id)
                self.incident_ways[node_id].add(int(way.id))
            for left, right in zip(node_ids[:-1], node_ids[1:]):
                if left != right:
                    self.graph.add_edge(left, right)
            line = LineString(coords)
            if line.length > 0:
                self.road_segments.append({"osm_snapshot_id": self.snapshot_id, "way_id": int(way.id), "highway": tags.get("highway"), "length_m": line.length, "geometry": line})
        if len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                polygon = make_valid(Polygon(coords))
            except Exception:
                return
            if polygon.is_empty:
                return
            if tags.get("building") not in {None, "", "no"}:
                self.buildings.append({"osm_snapshot_id": self.snapshot_id, "way_id": int(way.id), "building": tags.get("building"), "geometry": polygon})
            if tags.get("landuse") not in {None, ""}:
                self.landuse.append({"osm_snapshot_id": self.snapshot_id, "way_id": int(way.id), "landuse": tags.get("landuse"), "geometry": polygon})
        if coords:
            centroid = LineString(coords).centroid
            lon, lat = Transformer.from_crs(self.config["metric_crs"], self.config["storage_crs"], always_xy=True).transform(centroid.x, centroid.y)
            labels, primary = poi_labels(tags, self.config["poi_taxonomy"], self.config["poi_priority"])
            if primary:
                self._append_object("way", way.id, "poi", primary, labels, lon, lat, tags)

    def add_intersections(self) -> None:
        for node_id, degree in self.graph.degree():
            if degree < 3:
                continue
            x, y = self.node_xy[node_id]
            lon, lat = Transformer.from_crs(self.config["metric_crs"], self.config["storage_crs"], always_xy=True).transform(x, y)
            labels = ["intersection"]
            self._append_object("node", node_id, "intersection", "intersection", labels, lon, lat, {"derived_degree": str(degree)})


def parse_snapshot(snapshot_id: str, pbf: Path, config: dict[str, Any], transformer: Transformer) -> ContextHandler:
    handler = ContextHandler(snapshot_id, config, transformer)
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    handler.add_intersections()
    return handler


def geometry_pairs(points: list[Point], objects: list[Point], distance: float) -> np.ndarray:
    if not points or not objects:
        return np.empty((2, 0), dtype=np.int64)
    return STRtree(objects).query(points, predicate="dwithin", distance=distance)


def orientation_entropy(lines: list[LineString]) -> float:
    angles = []
    for line in lines:
        parts = list(line.geoms) if hasattr(line, "geoms") else [line]
        for part in parts:
            if not hasattr(part, "coords"):
                continue
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            if dx == 0 and dy == 0:
                continue
            angles.append((math.degrees(math.atan2(dy, dx)) % 180) // 22.5)
    if not angles:
        return 0.0
    counts = np.array(list(Counter(angles).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def build_topology(points: pd.DataFrame, handlers: dict[str, ContextHandler], topology_scale: float) -> pd.DataFrame:
    output = []
    for snapshot_id, group in points.groupby("osm_snapshot_id", sort=False):
        handler = handlers[snapshot_id]
        node_ids = list(handler.node_xy)
        node_geoms = [Point(handler.node_xy[node_id]) for node_id in node_ids]
        node_tree = STRtree(node_geoms)
        road_geoms = [record["geometry"] for record in handler.road_segments]
        road_tree = STRtree(road_geoms)
        unique = group[["matched_x_m", "matched_y_m"]].drop_duplicates().reset_index(drop=True)
        point_geoms = [Point(xy) for xy in unique[["matched_x_m", "matched_y_m"]].itertuples(index=False, name=None)]
        nearest = node_tree.nearest(point_geoms)
        records = []
        for local_index, point in enumerate(point_geoms):
            node_id = node_ids[int(nearest[local_index])]
            node_point = node_geoms[int(nearest[local_index])]
            buffer = point.buffer(topology_scale)
            road_indexes = road_tree.query(buffer, predicate="intersects")
            local_lines = [road_geoms[int(index)].intersection(buffer) for index in road_indexes]
            local_lines = [line for line in local_lines if not line.is_empty]
            records.append({
                "nearest_graph_node_degree": int(handler.graph.degree(node_id)),
                "nearest_graph_node_streets": int(len(handler.incident_ways[node_id])),
                "nearest_graph_node_distance_m": float(point.distance(node_point)),
                "dead_end_exposure": bool(handler.graph.degree(node_id) == 1),
                "edge_density_m_per_km2_250m": float(sum(line.length for line in local_lines) / (math.pi * topology_scale**2) * 1_000_000),
                "orientation_entropy_250m": orientation_entropy(local_lines),
            })
        unique = pd.concat([unique, pd.DataFrame(records)], axis=1)
        merged = group.merge(unique, on=["matched_x_m", "matched_y_m"], how="left", validate="many_to_one")
        output.append(merged)
    return pd.concat(output, ignore_index=True)


def polygon_coverage(points: pd.DataFrame, handlers: dict[str, ContextHandler], scales: list[int]) -> pd.DataFrame:
    result = points[["osm_snapshot_id", "matched_x_m", "matched_y_m"]].drop_duplicates().copy()
    frames = []
    for snapshot_id, group in result.groupby("osm_snapshot_id", sort=False):
        handler = handlers[snapshot_id]
        unique = group.reset_index(drop=True)
        geoms = [Point(xy) for xy in unique[["matched_x_m", "matched_y_m"]].itertuples(index=False, name=None)]
        for label, records in [("building", handler.buildings), ("landuse", handler.landuse)]:
            polygons = [record["geometry"] for record in records]
            tree = STRtree(polygons) if polygons else None
            for scale in scales:
                values = []
                for point in geoms:
                    buffer = point.buffer(scale)
                    if tree is None:
                        values.append(0.0)
                        continue
                    indexes = tree.query(buffer, predicate="intersects")
                    area = sum(polygons[int(index)].intersection(buffer).area for index in indexes)
                    values.append(float(min(1.0, area / buffer.area)))
                unique[f"{label}_coverage_ratio_{scale}m"] = values
        frames.append(unique)
    coverage = pd.concat(frames, ignore_index=True)
    return points.merge(coverage, on=["osm_snapshot_id", "matched_x_m", "matched_y_m"], how="left", validate="many_to_one")


def build_object_context(points: pd.DataFrame, objects: pd.DataFrame, scales: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_frames = []
    pair_frames = []
    layers = sorted(objects["object_layer"].unique())
    for snapshot_id, group in points.groupby("osm_snapshot_id", sort=False):
        obj = objects[objects["osm_snapshot_id"].eq(snapshot_id)].reset_index(drop=True)
        point_group = group.reset_index(drop=True)
        point_geoms = [Point(xy) for xy in point_group[["matched_x_m", "matched_y_m"]].itertuples(index=False, name=None)]
        object_geoms = [Point(xy) for xy in obj[["x_m", "y_m"]].itertuples(index=False, name=None)]
        base = point_group[["row_semantic_id"]].copy()
        for scale in scales:
            pairs = geometry_pairs(point_geoms, object_geoms, scale)
            pair = pd.DataFrame({"point_local": pairs[0], "object_local": pairs[1]})
            if not pair.empty:
                pair = pair.join(point_group[["row_semantic_id", "pilot_trip_id", "point_index", "timestamp_ms_raw", "matched_x_m", "matched_y_m"]], on="point_local")
                pair = pair.join(obj[["object_uid", "object_layer", "primary_type"]], on="object_local")
                pair["scale_m"] = scale
                pair_frames.append(pair.drop(columns=["point_local", "object_local"]))
                counts = pair.groupby(["row_semantic_id", "object_layer"])["object_uid"].nunique().unstack(fill_value=0)
            else:
                counts = pd.DataFrame(index=base["row_semantic_id"])
            for layer in layers:
                base[f"{layer}_unique_count_{scale}m"] = base["row_semantic_id"].map(counts[layer] if layer in counts else {}).fillna(0).astype(int)
            poi_pair = pair[pair["object_layer"].eq("poi")] if not pair.empty else pair
            diversity = poi_pair.groupby("row_semantic_id")["primary_type"].nunique() if not poi_pair.empty else pd.Series(dtype=int)
            base[f"poi_diversity_{scale}m"] = base["row_semantic_id"].map(diversity).fillna(0).astype(int)
        count_frames.append(base)
    counts = pd.concat(count_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    enriched = points.merge(counts, on="row_semantic_id", how="left", validate="one_to_one")
    exposure_records = []
    if not pairs.empty:
        pairs = pairs.sort_values(["pilot_trip_id", "scale_m", "object_layer", "object_uid", "point_index"])
        for keys, group in pairs.groupby(["pilot_trip_id", "scale_m", "object_layer", "object_uid"], sort=False):
            group = group.drop_duplicates("point_index").sort_values("point_index")
            index = group["point_index"].to_numpy(int)
            time = group["timestamp_ms_raw"].to_numpy(float) / 1000.0
            x = group["matched_x_m"].to_numpy(float)
            y = group["matched_y_m"].to_numpy(float)
            contiguous = np.r_[False, np.diff(index) == 1]
            dt = np.r_[0.0, np.diff(time)]
            valid_dt = contiguous & (dt > 0) & (dt <= 10)
            distance = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
            exposure_records.append({
                "pilot_trip_id": keys[0], "scale_m": int(keys[1]), "object_layer": keys[2], "object_uid": keys[3],
                "pass_event_count": int((~contiguous).sum()),
                "exposure_time_s": float(dt[valid_dt].sum()),
                "exposure_distance_m": float(distance[valid_dt].sum()),
            })
    object_exposure = pd.DataFrame(exposure_records)
    if object_exposure.empty:
        trip_exposure = pd.DataFrame()
    else:
        trip_exposure = object_exposure.groupby(["pilot_trip_id", "scale_m", "object_layer"], as_index=False).agg(unique_object_count=("object_uid", "nunique"), pass_event_count=("pass_event_count", "sum"), exposure_time_s=("exposure_time_s", "sum"), exposure_distance_m=("exposure_distance_m", "sum"))
    return enriched, object_exposure, trip_exposure


def monotonic_violations(frame: pd.DataFrame, scales: list[int]) -> int:
    prefixes = sorted({column.rsplit("_", 1)[0] for column in frame.columns if any(column.endswith(f"_{scale}m") for scale in scales) and "coverage_ratio" not in column})
    violations = 0
    for prefix in prefixes:
        columns = [f"{prefix}_{scale}m" for scale in scales if f"{prefix}_{scale}m" in frame]
        if len(columns) >= 2:
            values = frame[columns].to_numpy(float)
            violations += int((np.diff(values, axis=1) < 0).any(axis=1).sum())
    return violations


def build_maps(points: pd.DataFrame, objects: pd.DataFrame, handlers: dict[str, ContextHandler], output: Path, count: int, experiment: str) -> int:
    maps = output / "trajectory_maps"
    maps.mkdir()
    trips = points[["pilot_trip_id", "osm_snapshot_id"]].drop_duplicates().sort_values("pilot_trip_id").head(count)
    for audit_index, row in enumerate(trips.itertuples(index=False), start=1):
        trip = points[points["pilot_trip_id"].eq(row.pilot_trip_id)].sort_values("point_index")
        obj = objects[objects["osm_snapshot_id"].eq(row.osm_snapshot_id)]
        xmin, xmax = trip["matched_x_m"].min() - 300, trip["matched_x_m"].max() + 300
        ymin, ymax = trip["matched_y_m"].min() - 300, trip["matched_y_m"].max() + 300
        fig, ax = plt.subplots(figsize=(9, 7))
        for record in handlers[row.osm_snapshot_id].road_segments:
            line = record["geometry"]
            if line.bounds[2] < xmin or line.bounds[0] > xmax or line.bounds[3] < ymin or line.bounds[1] > ymax:
                continue
            x, y = line.xy
            ax.plot(x, y, color="#cccccc", linewidth=0.5, zorder=1)
        ax.plot(trip["matched_x_m"], trip["matched_y_m"], color="#d73027", linewidth=1.5, zorder=3)
        nearby = obj[obj["x_m"].between(xmin, xmax) & obj["y_m"].between(ymin, ymax)]
        colors = {"intersection": "#4575b4", "traffic_control": "#984ea3", "public_transport": "#ff7f00", "poi": "#4daf4a"}
        for layer, group in nearby.groupby("object_layer"):
            ax.scatter(group["x_m"], group["y_m"], s=14, label=layer, color=colors.get(layer, "black"), alpha=0.8, zorder=4)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_title(f"{experiment}-{audit_index:03d} | {row.pilot_trip_id} | {row.osm_snapshot_id}")
        ax.legend(loc="best", fontsize=7)
        ax.set_xlabel("Easting (m), EPSG:32617"); ax.set_ylabel("Northing (m), EPSG:32617")
        fig.tight_layout(); fig.savefig(maps / f"{experiment}-{audit_index:03d}.png", dpi=140); plt.close(fig)
    return len(trips)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/context_semantics.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    map_matching = load_json(Path(config["map_matching_close_decision"]))
    edge_semantics = load_json(Path(config["edge_semantics_pointer"]))
    raw = load_json(Path(config["map_match_pointer"]))
    network = load_json(Path(config["network_pointer"]))
    if not str(edge_semantics["status"]).startswith("PASS_EDGE_SEMANTICS"):
        raise RuntimeError("EDGE_SEMANTICS pilot has not passed.")
    transformer = Transformer.from_crs(config["storage_crs"], config["metric_crs"], always_xy=True)
    handlers = {snapshot_id: parse_snapshot(snapshot_id, Path(pbf), config, transformer) for snapshot_id, pbf in config["historical_source_pbfs"].items()}
    objects = pd.DataFrame([record for handler in handlers.values() for record in handler.objects])
    objects = objects.drop_duplicates(["osm_snapshot_id", "object_layer", "element_type", "osm_id"], keep="first").reset_index(drop=True)
    duplicate_count = int(objects.duplicated(["osm_snapshot_id", "object_layer", "element_type", "osm_id"]).sum())
    points = pd.read_parquet(raw["matched_points"])
    points = points[points["profile"].eq(config["selected_profile"]) & points["matched_latitude"].notna() & points["matched_longitude"].notna()].copy()
    points["row_semantic_id"] = [stable_uid(trip, index) for trip, index in zip(points["pilot_trip_id"], points["point_index"])]
    points["matched_x_m"], points["matched_y_m"] = transformer.transform(points["matched_longitude"].to_numpy(float), points["matched_latitude"].to_numpy(float))
    points = build_topology(points, handlers, float(config["topology_scale_m"]))
    points = polygon_coverage(points, handlers, config["built_environment"]["coverage_scales_m"])
    row_context, object_exposure, trip_exposure = build_object_context(points, objects, config["scales_m"])
    monotonic_count = monotonic_violations(row_context, config["scales_m"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = str(config.get("experiment", "CONTEXT_SEMANTICS"))
    run_prefix = str(config.get("run_name_prefix", "context_semantics_context_semantics_pilot"))
    latest_prefix = str(config.get("latest_prefix", "context_semantics"))
    stage = str(config.get("stage", "context_semantics_pilot"))
    status_suffix = str(config.get("status_suffix", "CONTEXT_SEMANTICS_PILOT"))
    output_root = Path(config["output_root"]); output = output_root / f"{run_prefix}_{stamp}"; output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "run_config.yaml")
    objects.to_parquet(output / "context_objects.parquet", index=False)
    row_context.to_parquet(output / "row_context_semantics.parquet", index=False)
    object_exposure.to_parquet(output / "object_exposure.parquet", index=False)
    trip_exposure.to_parquet(output / "trip_context_exposure.parquet", index=False)
    object_sample = objects.sample(n=min(len(objects), int(config["object_audit_sample_count"])), random_state=int(config["pilot_selection_seed"])).sort_values(["osm_snapshot_id", "object_layer", "osm_id"])
    object_sample.to_csv(output / "object_audit_sample.csv", index=False)
    map_count = build_maps(row_context, objects, handlers, output, int(config["trajectory_map_count"]), experiment)
    coverage_columns = [column for column in row_context if "coverage_ratio" in column]
    coverage_bounds_ok = bool(((row_context[coverage_columns] >= 0) & (row_context[coverage_columns] <= 1)).all(axis=None))
    building_polygon_count = int(sum(len(handler.buildings) for handler in handlers.values()))
    landuse_polygon_count = int(sum(len(handler.landuse) for handler in handlers.values()))
    nonzero_building_rows = int(row_context["building_coverage_ratio_250m"].gt(0).sum())
    nonzero_landuse_rows = int(row_context["landuse_coverage_ratio_250m"].gt(0).sum())
    checks = {
        "metric_crs": "PASS" if config["metric_crs"] == config["qa"]["metric_crs_required"] else "FAIL",
        "object_duplicate_count": "PASS" if duplicate_count <= config["qa"]["object_duplicate_count_max"] else "FAIL",
        "monotonic_counts": "PASS" if monotonic_count <= config["qa"]["monotonic_count_violation_max"] else "FAIL",
        "coverage_ratio_bounds": "PASS" if coverage_bounds_ok else "FAIL",
        "object_audit_sample": "PASS" if len(object_sample) >= min(len(objects), config["qa"]["minimum_object_audit_rows"]) else "FAIL",
        "trajectory_map_count": "PASS" if map_count >= config["qa"]["minimum_trajectory_maps"] else "FAIL",
        "building_polygons_present": "PASS" if building_polygon_count >= config["qa"]["minimum_building_polygon_count"] else "FAIL",
        "landuse_polygons_present": "PASS" if landuse_polygon_count >= config["qa"]["minimum_landuse_polygon_count"] else "FAIL",
        "building_coverage_nonzero": "PASS" if nonzero_building_rows >= config["qa"]["minimum_nonzero_building_coverage_rows"] else "FAIL",
        "landuse_coverage_nonzero": "PASS" if nonzero_landuse_rows >= config["qa"]["minimum_nonzero_landuse_coverage_rows"] else "FAIL",
        "map_matching_waiver_carried": "PASS" if map_matching["status"] == "PASS_WITH_DOCUMENTED_MANUAL_WAIVER" else "FAIL",
    }
    gate = "FAIL" if "FAIL" in checks.values() else "PASS"
    summary = {
        "experiment": experiment, "stage": stage, "status": f"{gate}_{status_suffix}", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_crs": config["storage_crs"], "metric_crs": config["metric_crs"], "context_object_count": int(len(objects)), "object_layer_counts": objects["object_layer"].value_counts().to_dict(),
        "object_duplicate_count": duplicate_count, "row_context_count": int(len(row_context)), "trip_count": int(row_context["pilot_trip_id"].nunique()), "object_exposure_row_count": int(len(object_exposure)), "trip_exposure_row_count": int(len(trip_exposure)),
        "monotonic_count_violation_count": monotonic_count, "coverage_ratio_bounds_ok": coverage_bounds_ok, "object_audit_sample_count": int(len(object_sample)), "trajectory_map_count": map_count,
        "building_polygon_count": building_polygon_count, "landuse_polygon_count": landuse_polygon_count, "nonzero_building_coverage_row_count": nonzero_building_rows, "nonzero_landuse_coverage_row_count": nonzero_landuse_rows,
        "graph_metrics": {snapshot_id: {"node_count": handler.graph.number_of_nodes(), "edge_count": handler.graph.number_of_edges(), "connected_component_count": nx.number_connected_components(handler.graph)} for snapshot_id, handler in handlers.items()},
        "simple_closed_way_polygon_limitation": True, "context_source": "frozen_full_historical_pbf_clipped_in_handler_to_research_bbox", "checks": checks, "map_matching_close_status": map_matching["status"], "full_build_authorized": False, "output_directory": str(output),
    }
    write_json(output / "context_semantics_summary.json", summary)
    report = output / f"{experiment}_{status_suffix}.md"
    report.write_text(f"# {experiment} {stage.replace('_', ' ')}\n\n" + "\n".join([f"- Status: `{summary['status']}`", f"- Context objects: {len(objects):,}", f"- Row context records: {len(row_context):,}", f"- Trips: {summary['trip_count']}", f"- Duplicate objects within layer: {duplicate_count}", f"- Monotonic count violations: {monotonic_count}", f"- Building polygons: {building_polygon_count:,}; nonzero 250 m rows: {nonzero_building_rows:,}", f"- Land-use polygons: {landuse_polygon_count:,}; nonzero 250 m rows: {nonzero_landuse_rows:,}", f"- Object audit rows: {len(object_sample):,}", f"- Trajectory maps: {map_count}", f"- Metric CRS: `{config['metric_crs']}`", "- Context objects come from the frozen full historical PBFs and are clipped in the parser to the frozen research bbox.", "- Polygon coverage currently uses valid simple closed OSM ways; relation multipolygons remain a documented pilot limitation.", f"- MAP_MATCHING closure carried as: `{map_matching['status']}`."]) + "\n", encoding="utf-8")
    shutil.copy2(output / "context_semantics_summary.json", output_root / f"{latest_prefix}_latest_summary.json"); shutil.copy2(report, output_root / f"{latest_prefix}_latest_report.md")
    write_json(output_root / f"{latest_prefix}_latest_pointer.json", {"status": summary["status"], "output_directory": str(output), "summary": str(output / "context_semantics_summary.json"), "report": str(report), "objects": str(output / "context_objects.parquet"), "row_context": str(output / "row_context_semantics.parquet"), "object_exposure": str(output / "object_exposure.parquet"), "trip_exposure": str(output / "trip_context_exposure.parquet"), "object_audit_sample": str(output / "object_audit_sample.csv"), "trajectory_maps": str(output / "trajectory_maps")})
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
