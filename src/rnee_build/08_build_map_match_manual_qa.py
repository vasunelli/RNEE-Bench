#!/usr/bin/env python3
"""Build a historical-OSM manual spatial audit package for MAP_MATCHING."""

from __future__ import annotations

import argparse
import gc
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import osmium
import pandas as pd
import yaml

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from shapely.geometry import LineString, box


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class HighwayHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def way(self, way: Any) -> None:
        highway = way.tags.get("highway")
        if not highway:
            return
        try:
            coordinates = [
                (node.lon, node.lat)
                for node in way.nodes
                if node.location.valid()
            ]
        except osmium.InvalidLocationError:
            return
        if len(coordinates) < 2:
            return
        self.records.append(
            {
                "way_id": int(way.id),
                "highway": highway,
                "geometry": LineString(coordinates),
            }
        )


def read_historical_roads(path: Path) -> gpd.GeoDataFrame:
    handler = HighwayHandler()
    handler.apply_file(str(path), locations=True)
    return gpd.GeoDataFrame(handler.records, geometry="geometry", crs="EPSG:4326")


def contiguous_matched_lines(points: pd.DataFrame) -> list[list[list[float]]]:
    lines: list[list[list[float]]] = []
    current: list[list[float]] = []
    for row in points.itertuples(index=False):
        valid = (
            row.match_status in {"matched_with_edge", "matched_without_edge"}
            and pd.notna(row.matched_longitude)
            and pd.notna(row.matched_latitude)
        )
        if valid:
            coordinate = [
                float(row.matched_longitude),
                float(row.matched_latitude),
            ]
            if not current or coordinate != current[-1]:
                current.append(coordinate)
        elif len(current) >= 2:
            lines.append(current)
            current = []
        else:
            current = []
    if len(current) >= 2:
        lines.append(current)
    return lines


def contiguous_raw_lines(points: pd.DataFrame) -> list[list[list[float]]]:
    lines: list[list[list[float]]] = []
    current: list[list[float]] = []
    has_break_field = "input_quality_break_before" in points.columns
    for row in points.itertuples(index=False):
        break_before = bool(
            has_break_field and getattr(row, "input_quality_break_before", False)
        )
        if break_before and len(current) >= 2:
            lines.append(current)
            current = []
        coordinate = [float(row.longitude_raw), float(row.latitude_raw)]
        if not current or coordinate != current[-1]:
            current.append(coordinate)
    if len(current) >= 2:
        lines.append(current)
    return lines


def feature_collection(points: pd.DataFrame) -> dict[str, Any]:
    raw_lines = contiguous_raw_lines(points)
    matched_lines = contiguous_matched_lines(points)
    unmatched = points.loc[
        points["match_status"].eq("unmatched"),
        ["longitude_raw", "latitude_raw"],
    ].astype(float).values.tolist()
    discontinuity = points.loc[
        points["begin_route_discontinuity"]
        | points["end_route_discontinuity"],
        ["matched_longitude", "matched_latitude"],
    ].dropna().astype(float).values.tolist()
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {"layer": "raw_ved_trace"},
            "geometry": {"type": "MultiLineString", "coordinates": raw_lines},
        },
        {
            "type": "Feature",
            "properties": {"layer": "matched_trace_segments"},
            "geometry": {
                "type": "MultiLineString",
                "coordinates": matched_lines,
            },
        },
        {
            "type": "Feature",
            "properties": {"layer": "unmatched_raw_points"},
            "geometry": {"type": "MultiPoint", "coordinates": unmatched},
        },
        {
            "type": "Feature",
            "properties": {"layer": "route_discontinuities"},
            "geometry": {"type": "MultiPoint", "coordinates": discontinuity},
        },
    ]
    return {"type": "FeatureCollection", "features": features}


def select_audit_trips(
    trips: pd.DataFrame,
    edges: pd.DataFrame,
    thresholds: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    edge_complexity = (
        edges.groupby("pilot_trip_id")
        .agg(
            road_class_count=("road_class", "nunique"),
            edge_use_count=("use", "nunique"),
        )
        .reset_index()
    )
    frame = trips.merge(edge_complexity, on="pilot_trip_id", how="left")
    frame["road_class_count"] = frame["road_class_count"].fillna(0)
    frame["edge_use_count"] = frame["edge_use_count"].fillna(0)
    high_count = int(thresholds["highest_distance_trip_count"])
    complex_count = int(thresholds["complex_geometry_trip_count"])
    random_count = int(thresholds["random_trip_count"])
    high = frame.sort_values(
        ["distance_p95_m", "matched_coverage", "pilot_trip_id"],
        ascending=[False, True, True],
    ).head(high_count).copy()
    high["audit_category"] = "highest_distance"
    remaining = frame[~frame["pilot_trip_id"].isin(high["pilot_trip_id"])].copy()
    remaining["complexity_score"] = (
        remaining["has_route_discontinuity"].astype(int) * 1_000_000
        + (1 - remaining["matched_coverage"]) * 10_000
        + remaining["road_class_count"] * 100
        + remaining["edge_use_count"] * 10
        + remaining["edge_count"]
    )
    complex_trips = remaining.sort_values(
        ["complexity_score", "pilot_trip_id"], ascending=[False, True]
    ).head(complex_count).copy()
    complex_trips["audit_category"] = "complex_geometry"
    remaining = remaining[
        ~remaining["pilot_trip_id"].isin(complex_trips["pilot_trip_id"])
    ]
    random_trips = remaining.sample(n=random_count, random_state=seed).copy()
    random_trips["audit_category"] = "random"
    selected = pd.concat(
        [random_trips, high, complex_trips], ignore_index=True
    )
    order = {"random": 0, "highest_distance": 1, "complex_geometry": 2}
    selected["_category_order"] = selected["audit_category"].map(order)
    return selected.sort_values(
        ["_category_order", "pilot_trip_id"]
    ).drop(columns="_category_order").reset_index(drop=True)


def plot_trip(
    points: pd.DataFrame,
    roads: gpd.GeoDataFrame,
    output: Path,
    title: str,
) -> None:
    lon = points["longitude_raw"].astype(float)
    lat = points["latitude_raw"].astype(float)
    span_x = max(float(lon.max() - lon.min()), 0.003)
    span_y = max(float(lat.max() - lat.min()), 0.003)
    margin_x = max(span_x * 0.12, 0.002)
    margin_y = max(span_y * 0.12, 0.002)
    bounds = (
        float(lon.min() - margin_x),
        float(lat.min() - margin_y),
        float(lon.max() + margin_x),
        float(lat.max() + margin_y),
    )
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    cosine_latitude = max(np.cos(np.deg2rad(center_y)), 0.2)
    x_span = bounds[2] - bounds[0]
    y_span = bounds[3] - bounds[1]
    target_x_span = y_span * 1.25 / cosine_latitude
    if x_span < target_x_span:
        bounds = (
            center_x - target_x_span / 2,
            bounds[1],
            center_x + target_x_span / 2,
            bounds[3],
        )
    road_indices = list(roads.sindex.query(box(*bounds)))
    local_roads = roads.iloc[road_indices]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for geometry in local_roads.geometry:
        x, y = geometry.xy
        ax.plot(x, y, color="#c7c7c7", linewidth=0.45, zorder=1)
    for line_index, coordinates in enumerate(contiguous_raw_lines(points)):
        values = np.asarray(coordinates)
        ax.plot(
            values[:, 0],
            values[:, 1],
            color="#2271b2",
            linewidth=1.4,
            alpha=0.75,
            label="Raw VED GPS segments" if line_index == 0 else None,
            zorder=2,
        )
    for line_index, coordinates in enumerate(contiguous_matched_lines(points)):
        values = np.asarray(coordinates)
        ax.plot(
            values[:, 0],
            values[:, 1],
            color="#d7301f",
            linewidth=1.5,
            label="Valhalla matched" if line_index == 0 else None,
            zorder=3,
        )
    unmatched = points[points["match_status"].eq("unmatched")]
    if not unmatched.empty:
        ax.scatter(
            unmatched["longitude_raw"],
            unmatched["latitude_raw"],
            marker="x",
            s=18,
            color="#7a0177",
            label="Unmatched",
            zorder=4,
        )
    discontinuity = points[
        points["begin_route_discontinuity"]
        | points["end_route_discontinuity"]
    ].dropna(subset=["matched_longitude", "matched_latitude"])
    if not discontinuity.empty:
        ax.scatter(
            discontinuity["matched_longitude"],
            discontinuity["matched_latitude"],
            marker="s",
            s=32,
            facecolor="#ffbf00",
            edgecolor="black",
            linewidth=0.5,
            label="Route discontinuity",
            zorder=5,
        )
    if "input_quality_break_before" in points.columns:
        input_breaks = points[
            points["input_quality_break_before"].fillna(False)
        ]
        if not input_breaks.empty:
            ax.scatter(
                input_breaks["longitude_raw"],
                input_breaks["latitude_raw"],
                marker="^",
                s=34,
                facecolor="#00bfc4",
                edgecolor="black",
                linewidth=0.5,
                label="Frozen input-quality break",
                zorder=6,
            )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1 / cosine_latitude, adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude (WGS84)")
    ax.set_ylabel("Latitude (WGS84)")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-pointer",
        type=Path,
        default=Path(r"results/map_matching_latest_qa_pointer.json"),
    )
    parser.add_argument(
        "--network-pointer",
        type=Path,
        default=Path(r"results/network_freeze_latest_network_pointer.json"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(r"configs/rnee_build/qa_thresholds.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"results/rnee_build"),
    )
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--experiment", default="MAP_MATCHING")
    parser.add_argument("--output-prefix", default="map_matching_manual_spatial_qa")
    parser.add_argument("--latest-pointer-name", default="map_matching_latest_manual_qa_pointer.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qa_pointer = load_json(args.qa_pointer)
    diagnostic_only = qa_pointer["status"] == "STOP_REQUIRES_MANUAL_DIAGNOSTIC"
    raw_pointer = load_json(Path(qa_pointer["raw_pointer"]))
    network_pointer = load_json(args.network_pointer)
    thresholds = load_yaml(args.thresholds)["map_matching"][
        "manual_spatial_audit"
    ]
    profile = qa_pointer["selected_profile"]
    trips = pd.read_parquet(raw_pointer["trip_summary"])
    trips = trips[trips["profile"].eq(profile)].copy()
    points = pd.read_parquet(raw_pointer["matched_points"])
    points = points[points["profile"].eq(profile)].copy()
    edges = pd.read_parquet(raw_pointer["matched_edges"])
    edges = edges[edges["profile"].eq(profile)].copy()
    selected = select_audit_trips(trips, edges, thresholds, args.seed)
    expected_count = sum(
        int(thresholds[key])
        for key in [
            "random_trip_count",
            "highest_distance_trip_count",
            "complex_geometry_trip_count",
        ]
    )
    if len(selected) != expected_count or selected["pilot_trip_id"].duplicated().any():
        raise RuntimeError("Manual audit selection is not cardinality-safe.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"{args.output_prefix}_{timestamp}"
    maps_dir = output / "maps"
    geojson_dir = output / "geojson"
    maps_dir.mkdir(parents=True, exist_ok=False)
    geojson_dir.mkdir(parents=True, exist_ok=False)
    road_frames = {
        snapshot_id: read_historical_roads(Path(pbf_path))
        for snapshot_id, pbf_path in network_pointer["cropped_pbfs"].items()
    }

    quality_check_rows = []
    cards = []
    for audit_index, row in enumerate(selected.itertuples(index=False), start=1):
        audit_id = f"{args.experiment}-{audit_index:03d}"
        trip_points = points[
            points["pilot_trip_id"].eq(row.pilot_trip_id)
        ].sort_values("point_index")
        png_name = f"{audit_id}_{row.audit_category}.png"
        geojson_name = f"{audit_id}_{row.audit_category}.geojson"
        title = (
            f"{audit_id} | {row.audit_category} | Veh {row.VehId}, Trip {row.Trip}\n"
            f"{row.osm_snapshot_id} | coverage={row.matched_coverage:.3f}, "
            f"p95={row.distance_p95_m:.1f} m, "
            f"discontinuity={bool(row.has_route_discontinuity)}"
        )
        plot_trip(
            trip_points,
            road_frames[row.osm_snapshot_id],
            maps_dir / png_name,
            title,
        )
        write_json(
            geojson_dir / geojson_name,
            feature_collection(trip_points),
        )
        quality_check_rows.append(
            {
                "audit_id": audit_id,
                "audit_category": row.audit_category,
                "profile": profile,
                "pilot_trip_id": row.pilot_trip_id,
                "VehId": int(row.VehId),
                "Trip": int(row.Trip),
                "trip_mid_date": row.trip_mid_date,
                "osm_snapshot_id": row.osm_snapshot_id,
                "matched_coverage": row.matched_coverage,
                "edge_association_coverage": row.edge_association_coverage,
                "distance_p95_m": row.distance_p95_m,
                "has_route_discontinuity": bool(row.has_route_discontinuity),
                "map_png": f"maps/{png_name}",
                "geojson": f"geojson/{geojson_name}",
                "quality_rating": "",
                "quality_notes": "",
            }
        )
        cards.append(
            "<article><a href='maps/{png}'><img loading='lazy' src='maps/{png}' "
            "alt='{audit}'></a><p><b>{audit}</b> · {category} · coverage "
            "{coverage:.3f} · p95 {p95:.1f} m · discontinuity {disc}</p></article>".format(
                png=html.escape(png_name),
                audit=html.escape(audit_id),
                category=html.escape(row.audit_category),
                coverage=row.matched_coverage,
                p95=row.distance_p95_m,
                disc=bool(row.has_route_discontinuity),
            )
        )
    quality_check = pd.DataFrame(quality_check_rows)
    quality_check_path = output / "manual_quality_check_form.csv"
    quality_check.to_csv(quality_check_path, index=False, encoding="utf-8-sig")
    html_text = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>MAP_MATCHING 历史路网匹配人工质量评估</title>
<style>body{font-family:Arial,sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}article{border:1px solid #ddd;padding:8px}img{width:100%;height:auto}p{font-size:13px}</style>
</head><body><h1>MAP_MATCHING 历史路网匹配人工质量评估</h1>
<p>灰色为对应日期的历史 OSM 路网，蓝色为已切分的 VED 原始 GPS，红色为匹配轨迹，紫色叉号为未匹配点，黄色方块为 Valhalla 断裂，青色三角为冻结的输入质量切分点。</p>
<main>""" + "\n".join(cards) + "</main></body></html>\n"
    (output / "index.html").write_text(html_text, encoding="utf-8")
    package_status = (
        "DIAGNOSTIC_ONLY_REQUIRES_QUALITY_CHECK1_RERATING"
        if diagnostic_only
        else "REQUIRES_QUALITY_CHECK1_RERATING"
    )
    readme = f"""# {args.experiment} manual spatial audit

Status: `{package_status}`

Open `index.html`, inspect all {len(quality_check)} maps, and record ratings in
`manual_quality_check_form.csv`.

Allowed ratings:

- `correct`: matched trace follows the plausible historical road path.
- `minor`: local ambiguity exists but road identity and route remain usable.
- `incorrect`: wrong parallel road, impossible jump, wrong turn, or material
  off-network assignment.

Only `quality_rating` is used. The frozen pass condition is a quality_check-1
`incorrect` rate no greater than {thresholds['incorrect_rate_pass_max']:.1%}.

This package uses the selected `{profile}` profile and the trip-specific
historical OSM snapshot. It does not use energy labels, model results, current
maps, or legacy eVED semantics.

The automated profile gate is {"not passed; this package is diagnostic only"
if diagnostic_only else "eligible for manual confirmation"}. Completing this
form {"cannot by itself authorize EDGE_SEMANTICS or the full build" if args.experiment == "MAP_MATCHING" else "is required before TRAJECTORY_BUILD edge-semantic production may begin"}.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "experiment": args.experiment,
        "stage": "manual_spatial_audit_package",
        "status": package_status,
        "automated_gate_passed": not diagnostic_only,
        "diagnostic_only": diagnostic_only,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "audit_trip_count": len(quality_check),
        "category_counts": quality_check["audit_category"].value_counts().to_dict(),
        "historical_snapshot_ids": sorted(quality_check["osm_snapshot_id"].unique()),
        "incorrect_rate_pass_max": thresholds["incorrect_rate_pass_max"],
        "quality_rating_column": "quality_rating",
        "quality_check2_or_resolution_required": False,
        "maps_directory": str(maps_dir),
        "quality_check_form": str(quality_check_path),
        "index_html": str(output / "index.html"),
        "full_build_authorized": False,
    }
    write_json(output / "manual_audit_package_summary.json", summary)
    write_json(
        args.output_root / args.latest_pointer_name,
        {
            "status": summary["status"],
            "output_directory": str(output),
            "summary": str(output / "manual_audit_package_summary.json"),
            "quality_check_form": str(quality_check_path),
            "index_html": str(output / "index.html"),
        },
    )
    road_frames.clear()
    del points, edges, trips, selected
    gc.collect()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
