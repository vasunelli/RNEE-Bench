#!/usr/bin/env python3
"""NETWORK_FREEZE: audit original VED GPS and freeze historical-only OSM snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download configured historical snapshots if they are absent.",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def daynum_to_date(value: float, origin: date) -> date:
    return origin + timedelta(days=math.floor(float(value)))


def buffered_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    buffer_km: float,
) -> dict[str, float]:
    middle_lat = (min_lat + max_lat) / 2.0
    lat_delta = buffer_km / 111.32
    lon_delta = buffer_km / (111.32 * math.cos(math.radians(middle_lat)))
    return {
        "min_lon": min_lon - lon_delta,
        "min_lat": min_lat - lat_delta,
        "max_lon": max_lon + lon_delta,
        "max_lat": max_lat + lat_delta,
    }


def bbox_geojson(bbox: dict[str, float]) -> dict[str, Any]:
    west, south, east, north = (
        bbox["min_lon"],
        bbox["min_lat"],
        bbox["max_lon"],
        bbox["max_lat"],
    )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"crs": "EPSG:4326", "purpose": "NETWORK_FREEZE research bbox"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                },
            }
        ],
    }


def latest_snapshot_on_or_before(
    trip_date: date, snapshots: list[dict[str, Any]]
) -> tuple[str, int]:
    candidates: list[tuple[date, str]] = []
    for item in snapshots:
        snapshot_date = date.fromisoformat(item["snapshot_date"])
        if snapshot_date <= trip_date:
            candidates.append((snapshot_date, item["snapshot_id"]))
    if not candidates:
        raise ValueError(
            f"No historical snapshot exists on or before trip date {trip_date}."
        )
    snapshot_date, snapshot_id = max(candidates)
    return snapshot_id, (trip_date - snapshot_date).days


def validate_historical_policy(config: dict[str, Any]) -> None:
    policy = config["policy"]
    if not policy.get("historical_only"):
        raise ValueError("NETWORK_FREEZE requires historical_only=true")
    if policy.get("current_osm_fallback") != "forbidden":
        raise ValueError("Current OSM fallback must be forbidden")
    maximum = date.fromisoformat(policy["maximum_allowed_snapshot_date"])
    for snapshot in config["osm"]["snapshots"]:
        if "latest" in snapshot["url"].lower() or "latest" in snapshot["filename"].lower():
            raise ValueError(f"Current/latest OSM URL is forbidden: {snapshot['url']}")
        if date.fromisoformat(snapshot["snapshot_date"]) > maximum:
            raise ValueError(f"Snapshot exceeds historical cutoff: {snapshot['snapshot_id']}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RNEE-Bench-NETWORK_FREEZE/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    temporary.replace(destination)


def pbf_tag_audit(path: Path) -> dict[str, Any]:
    try:
        import osmium
    except ImportError as exc:
        raise RuntimeError(
            "The isolated RNEE environment must provide pyosmium/osmium."
        ) from exc

    class Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes = 0
            self.ways = 0
            self.relations = 0
            self.highway_ways = 0
            self.tags = Counter()

        def node(self, _) -> None:
            self.nodes += 1

        def way(self, way) -> None:
            self.ways += 1
            if "highway" in way.tags:
                self.highway_ways += 1
                for key in (
                    "highway",
                    "maxspeed",
                    "source:maxspeed",
                    "lanes",
                    "oneway",
                    "access",
                    "surface",
                    "smoothness",
                    "bridge",
                    "tunnel",
                    "junction",
                    "traffic_calming",
                    "lit",
                ):
                    if key in way.tags:
                        self.tags[key] += 1

        def relation(self, _) -> None:
            self.relations += 1

    reader = osmium.io.Reader(str(path))
    header = reader.header()
    boxes = []
    try:
        box = header.box()
        if box.valid():
            boxes.append(
                {
                    "min_lon": float(box.bottom_left.lon),
                    "min_lat": float(box.bottom_left.lat),
                    "max_lon": float(box.top_right.lon),
                    "max_lat": float(box.top_right.lat),
                }
            )
    finally:
        reader.close()

    handler = Handler()
    handler.apply_file(str(path))
    return {
        "nodes": handler.nodes,
        "ways": handler.ways,
        "relations": handler.relations,
        "highway_ways": handler.highway_ways,
        "highway_tag_counts": dict(handler.tags),
        "header_boxes": boxes,
    }


def bbox_inside_any(
    bbox: dict[str, float], boxes: list[dict[str, float]]
) -> bool:
    return any(
        bbox["min_lon"] >= box["min_lon"]
        and bbox["min_lat"] >= box["min_lat"]
        and bbox["max_lon"] <= box["max_lon"]
        and bbox["max_lat"] <= box["max_lat"]
        for box in boxes
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_historical_policy(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_dir / "run_config.yaml")

    columns = config["columns"]
    usecols = list(columns.values())
    source_dir = Path(config["inputs"]["ved_dynamic_dir"])
    files = sorted(source_dir.glob(config["inputs"]["ved_pattern"]))
    chunk_rows = int(config["gps_quality"]["chunk_rows"])
    origin = date.fromisoformat(config["gps_quality"]["daynum_origin"])
    expected_region = config["gps_quality"]["expected_region"]
    warning_speed = float(config["gps_quality"]["implied_speed_warning_kmh"])
    repeat_eps = float(config["gps_quality"]["repeated_coordinate_epsilon_deg"])

    total_rows = 0
    valid_rows = 0
    missing_rows = 0
    zero_rows = 0
    out_of_range_rows = 0
    repeated_pairs = 0
    positive_dt_pairs = 0
    row_interval_speed_warning_pairs = 0
    gps_update_pairs = 0
    gps_update_speed_warning_pairs = 0
    nonpositive_dt_pairs = 0
    exact_min_lat = math.inf
    exact_max_lat = -math.inf
    exact_min_lon = math.inf
    exact_max_lon = -math.inf
    file_rows: list[dict[str, Any]] = []
    trip_rows: list[dict[str, Any]] = []

    for path in files:
        file_stats = Counter()
        file_trip_parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows):
            total_rows += len(chunk)
            file_stats["rows"] += len(chunk)
            for key in ("daynum", "timestamp_ms", "latitude", "longitude"):
                chunk[columns[key]] = pd.to_numeric(chunk[columns[key]], errors="coerce")

            lat = chunk[columns["latitude"]]
            lon = chunk[columns["longitude"]]
            missing = lat.isna() | lon.isna()
            zero = lat.eq(0) & lon.eq(0)
            in_world = lat.between(-90, 90) & lon.between(-180, 180)
            in_region = (
                lat.between(expected_region["min_lat"], expected_region["max_lat"])
                & lon.between(expected_region["min_lon"], expected_region["max_lon"])
            )
            valid = ~missing & ~zero & in_world & in_region
            missing_rows += int(missing.sum())
            zero_rows += int(zero.sum())
            out_of_range_rows += int((~missing & ~zero & (~in_world | ~in_region)).sum())
            valid_rows += int(valid.sum())
            file_stats["valid_rows"] += int(valid.sum())

            if valid.any():
                exact_min_lat = min(exact_min_lat, float(lat[valid].min()))
                exact_max_lat = max(exact_max_lat, float(lat[valid].max()))
                exact_min_lon = min(exact_min_lon, float(lon[valid].min()))
                exact_max_lon = max(exact_max_lon, float(lon[valid].max()))

            work = chunk[
                [
                    columns["vehicle"],
                    columns["trip"],
                    columns["daynum"],
                    columns["timestamp_ms"],
                    columns["latitude"],
                    columns["longitude"],
                ]
            ].copy()
            work["_valid"] = valid.to_numpy()
            group = [columns["vehicle"], columns["trip"]]
            previous_lat = work.groupby(group, sort=False)[columns["latitude"]].shift()
            previous_lon = work.groupby(group, sort=False)[columns["longitude"]].shift()
            previous_time = work.groupby(group, sort=False)[columns["timestamp_ms"]].shift()
            previous_valid = work.groupby(group, sort=False)["_valid"].shift(fill_value=False)
            pair_valid = work["_valid"] & previous_valid
            dt_s = (work[columns["timestamp_ms"]] - previous_time) / 1000.0
            positive = pair_valid & dt_s.gt(0)
            nonpositive_dt_pairs += int((pair_valid & ~dt_s.gt(0)).sum())
            positive_dt_pairs += int(positive.sum())
            distance = np.full(len(work), np.nan)
            indices = np.flatnonzero(positive.to_numpy())
            if len(indices):
                distance[indices] = haversine_km(
                    previous_lat.iloc[indices].to_numpy(),
                    previous_lon.iloc[indices].to_numpy(),
                    work[columns["latitude"]].iloc[indices].to_numpy(),
                    work[columns["longitude"]].iloc[indices].to_numpy(),
                )
            implied_speed = distance / dt_s.to_numpy() * 3600.0
            speed_warning = positive.to_numpy() & (implied_speed > warning_speed)
            row_interval_speed_warning_pairs += int(speed_warning.sum())
            repeated = (
                pair_valid
                & (work[columns["latitude"]] - previous_lat).abs().le(repeat_eps)
                & (work[columns["longitude"]] - previous_lon).abs().le(repeat_eps)
            )
            repeated_pairs += int(repeated.sum())

            coordinate_update = work["_valid"] & (~previous_valid | ~repeated)
            updates = work.loc[
                coordinate_update,
                group
                + [
                    columns["timestamp_ms"],
                    columns["latitude"],
                    columns["longitude"],
                ],
            ].copy()
            update_previous_lat = updates.groupby(group, sort=False)[
                columns["latitude"]
            ].shift()
            update_previous_lon = updates.groupby(group, sort=False)[
                columns["longitude"]
            ].shift()
            update_previous_time = updates.groupby(group, sort=False)[
                columns["timestamp_ms"]
            ].shift()
            update_dt_s = (
                updates[columns["timestamp_ms"]] - update_previous_time
            ) / 1000.0
            update_pair = update_previous_time.notna() & update_dt_s.gt(0)
            update_distance = np.full(len(updates), np.nan)
            update_indices = np.flatnonzero(update_pair.to_numpy())
            if len(update_indices):
                update_distance[update_indices] = haversine_km(
                    update_previous_lat.iloc[update_indices].to_numpy(),
                    update_previous_lon.iloc[update_indices].to_numpy(),
                    updates[columns["latitude"]].iloc[update_indices].to_numpy(),
                    updates[columns["longitude"]].iloc[update_indices].to_numpy(),
                )
            update_speed = update_distance / update_dt_s.to_numpy() * 3600.0
            update_speed_warning = update_pair.to_numpy() & (
                update_speed > warning_speed
            )
            gps_update_pairs += int(update_pair.sum())
            gps_update_speed_warning_pairs += int(update_speed_warning.sum())

            work["_positive_pair"] = positive.to_numpy()
            work["_row_interval_speed_warning"] = speed_warning
            work["_repeated_pair"] = repeated.to_numpy()
            work["_gps_update_pair"] = False
            work["_gps_update_speed_warning"] = False
            work.loc[updates.index, "_gps_update_pair"] = update_pair.to_numpy()
            work.loc[updates.index, "_gps_update_speed_warning"] = (
                update_speed_warning
            )
            file_trip_parts.append(
                work.groupby(group, sort=False)
                .agg(
                    row_count=("_valid", "size"),
                    valid_coordinate_rows=("_valid", "sum"),
                    positive_dt_pairs=("_positive_pair", "sum"),
                    row_interval_speed_warning_pairs=(
                        "_row_interval_speed_warning",
                        "sum",
                    ),
                    repeated_coordinate_pairs=("_repeated_pair", "sum"),
                    gps_update_pairs=("_gps_update_pair", "sum"),
                    gps_update_speed_warning_pairs=(
                        "_gps_update_speed_warning",
                        "sum",
                    ),
                    min_daynum=(columns["daynum"], "min"),
                    max_daynum=(columns["daynum"], "max"),
                    min_lat=(columns["latitude"], "min"),
                    max_lat=(columns["latitude"], "max"),
                    min_lon=(columns["longitude"], "min"),
                    max_lon=(columns["longitude"], "max"),
                )
                .reset_index()
            )

        trips = pd.concat(file_trip_parts, ignore_index=True)
        trips = (
            trips.groupby([columns["vehicle"], columns["trip"]], as_index=False)
            .agg(
                row_count=("row_count", "sum"),
                valid_coordinate_rows=("valid_coordinate_rows", "sum"),
                positive_dt_pairs=("positive_dt_pairs", "sum"),
                row_interval_speed_warning_pairs=(
                    "row_interval_speed_warning_pairs",
                    "sum",
                ),
                repeated_coordinate_pairs=("repeated_coordinate_pairs", "sum"),
                gps_update_pairs=("gps_update_pairs", "sum"),
                gps_update_speed_warning_pairs=(
                    "gps_update_speed_warning_pairs",
                    "sum",
                ),
                min_daynum=("min_daynum", "min"),
                max_daynum=("max_daynum", "max"),
                min_lat=("min_lat", "min"),
                max_lat=("max_lat", "max"),
                min_lon=("min_lon", "min"),
                max_lon=("max_lon", "max"),
            )
        )
        trips.insert(0, "source_file", path.name)
        trip_rows.extend(trips.to_dict("records"))
        file_rows.append(
            {
                "source_file": path.name,
                "rows": int(file_stats["rows"]),
                "valid_coordinate_rows": int(file_stats["valid_rows"]),
                "valid_coordinate_rate": (
                    file_stats["valid_rows"] / file_stats["rows"]
                    if file_stats["rows"]
                    else 0.0
                ),
            }
        )

    trip_df = pd.DataFrame(trip_rows)
    snapshots = config["osm"]["snapshots"]
    mid_daynum = (trip_df["min_daynum"] + trip_df["max_daynum"]) / 2.0
    trip_dates = mid_daynum.map(lambda value: daynum_to_date(value, origin))
    assignments = trip_dates.map(
        lambda value: latest_snapshot_on_or_before(value, snapshots)
    )
    trip_df["trip_mid_date"] = trip_dates.map(str)
    trip_df["osm_snapshot_id"] = assignments.map(lambda item: item[0])
    trip_df["snapshot_age_days"] = assignments.map(lambda item: item[1])
    trip_df["snapshot_lag_days"] = trip_df["snapshot_age_days"]
    snapshot_dates = {
        item["snapshot_id"]: date.fromisoformat(item["snapshot_date"])
        for item in snapshots
    }
    trip_df["snapshot_is_future"] = [
        snapshot_dates[snapshot_id] > trip_date
        for snapshot_id, trip_date in zip(
            trip_df["osm_snapshot_id"], trip_dates
        )
    ]
    trip_df["valid_coordinate_rate"] = (
        trip_df["valid_coordinate_rows"] / trip_df["row_count"]
    )
    trip_df["gps_basic_valid"] = trip_df["valid_coordinate_rows"].gt(0)

    raw_bbox = {
        "min_lon": exact_min_lon,
        "min_lat": exact_min_lat,
        "max_lon": exact_max_lon,
        "max_lat": exact_max_lat,
    }
    research_bbox = buffered_bbox(
        exact_min_lon,
        exact_min_lat,
        exact_max_lon,
        exact_max_lat,
        float(config["gps_quality"]["research_bbox_buffer_km"]),
    )
    (output_dir / "research_bbox.geojson").write_text(
        json.dumps(bbox_geojson(research_bbox), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(file_rows).to_csv(output_dir / "gps_file_summary.csv", index=False)
    trip_df.to_parquet(output_dir / "gps_trip_quality.parquet", index=False)
    trip_df.to_csv(output_dir / "gps_trip_quality.csv", index=False)

    raw_dir = Path(config["osm"]["raw_dir"])
    snapshot_manifest = []
    for snapshot in snapshots:
        destination = raw_dir / snapshot["filename"]
        if args.download and not destination.exists():
            download_file(snapshot["url"], destination)
        exists = destination.exists()
        size = destination.stat().st_size if exists else None
        size_matches = size == int(snapshot["expected_bytes"]) if exists else False
        digest = sha256_file(destination) if exists else None
        pbf_audit = pbf_tag_audit(destination) if exists else None
        inside = (
            bbox_inside_any(research_bbox, pbf_audit["header_boxes"])
            if pbf_audit and pbf_audit["header_boxes"]
            else False
        )
        snapshot_manifest.append(
            {
                **snapshot,
                "local_path": str(destination.resolve()),
                "downloaded": exists,
                "actual_bytes": size,
                "exact_size_match": size_matches,
                "sha256": digest,
                "pbf_audit": pbf_audit,
                "research_bbox_inside_pbf_header": inside,
            }
        )

    assignment_counts = (
        trip_df.groupby("osm_snapshot_id", as_index=False)
        .agg(
            trips=("Trip", "size"),
            rows=("row_count", "sum"),
            maximum_lag_days=("snapshot_lag_days", "max"),
            median_lag_days=("snapshot_lag_days", "median"),
            future_snapshot_trips=("snapshot_is_future", "sum"),
        )
        .to_dict("records")
    )
    gates = config["quality_gates"]
    checks = {
        "historical_only_policy": config["policy"]["historical_only"]
        and config["policy"]["current_osm_fallback"] == "forbidden",
        "raw_file_count": len(files) == int(gates["raw_file_count"]),
        "raw_row_count": total_rows == int(gates["raw_row_count"]),
        "valid_coordinate_rate": (valid_rows / total_rows)
        >= float(gates["valid_coordinate_rate_min"]),
        "trips_have_valid_coordinates": int((~trip_df["gps_basic_valid"]).sum())
        <= int(gates["trips_without_valid_coordinates_max"]),
        "snapshot_count": len(snapshot_manifest)
        == int(gates["downloaded_snapshot_count"]),
        "snapshots_downloaded": all(item["downloaded"] for item in snapshot_manifest),
        "snapshot_sizes": all(item["exact_size_match"] for item in snapshot_manifest),
        "snapshot_hashes": all(bool(item["sha256"]) for item in snapshot_manifest),
        "pbf_parse": all(item["pbf_audit"] is not None for item in snapshot_manifest),
        "highway_way_count": all(
            item["pbf_audit"]["highway_ways"]
            >= int(gates["require_highway_ways_min"])
            for item in snapshot_manifest
            if item["pbf_audit"] is not None
        )
        and all(item["pbf_audit"] is not None for item in snapshot_manifest),
        "research_bbox_covered": all(
            item["research_bbox_inside_pbf_header"] for item in snapshot_manifest
        ),
        "assignment_lag": int(trip_df["snapshot_lag_days"].max())
        <= int(config["policy"]["maximum_trip_to_snapshot_lag_days"]),
        "no_future_snapshot_assignment": not bool(
            trip_df["snapshot_is_future"].any()
        ),
    }
    overall_gate = (
        "PASS_HISTORICAL_OSM_SNAPSHOT"
        if all(checks.values())
        else "FAIL_HISTORICAL_OSM_SNAPSHOT"
    )
    summary = {
        "stage": "NETWORK_FREEZE",
        "overall_gate": overall_gate,
        "policy": config["policy"],
        "gps": {
            "files": len(files),
            "rows": total_rows,
            "valid_coordinate_rows": valid_rows,
            "valid_coordinate_rate": valid_rows / total_rows,
            "missing_coordinate_rows": missing_rows,
            "zero_coordinate_rows": zero_rows,
            "out_of_range_rows": out_of_range_rows,
            "positive_dt_pairs": positive_dt_pairs,
            "nonpositive_dt_pairs": nonpositive_dt_pairs,
            "repeated_coordinate_pairs": repeated_pairs,
            "row_interval_implied_speed_warning_pairs": (
                row_interval_speed_warning_pairs
            ),
            "gps_update_pairs": gps_update_pairs,
            "gps_update_implied_speed_warning_pairs": (
                gps_update_speed_warning_pairs
            ),
            "gps_update_implied_speed_warning_rate": (
                gps_update_speed_warning_pairs / gps_update_pairs
                if gps_update_pairs
                else None
            ),
            "trips": int(len(trip_df)),
            "trips_without_valid_coordinates": int((~trip_df["gps_basic_valid"]).sum()),
            "raw_bbox_wgs84": raw_bbox,
            "research_bbox_wgs84": research_bbox,
        },
        "snapshot_assignment": assignment_counts,
        "snapshots": snapshot_manifest,
        "checks": checks,
        "runtime": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    (output_dir / "historical_snapshot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "historical_snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest, indent=2), encoding="utf-8"
    )

    report = [
        "# NETWORK_FREEZE Historical-Only OSM Snapshot Attempt",
        "",
        f"Final gate: `{overall_gate}`",
        "",
        "## Policy",
        "",
        "- Only timestamped historical OSM extracts are allowed.",
        "- Current/latest OSM fallback is forbidden.",
        (
            "- Trips are assigned to the latest configured historical snapshot "
            "on or before the trip midpoint date."
        ),
        "",
        "## GPS audit",
        "",
        f"- Files: {len(files)}",
        f"- Rows: {total_rows:,}",
        f"- Valid coordinate rate: {valid_rows / total_rows:.8%}",
        f"- Trips: {len(trip_df):,}",
        f"- Trips without valid coordinates: {(~trip_df['gps_basic_valid']).sum():,}",
        f"- Raw bbox: `{raw_bbox}`",
        f"- Buffered research bbox: `{research_bbox}`",
        (
            f"- GPS-update implied-speed warnings above {warning_speed:g} km/h: "
            f"{gps_update_speed_warning_pairs:,} / {gps_update_pairs:,}"
        ),
        (
            "- Row-interval apparent warnings (diagnostic only; coordinates are "
            f"sample-and-held): {row_interval_speed_warning_pairs:,}"
        ),
        "",
        "## Historical snapshots",
        "",
    ]
    for item in snapshot_manifest:
        audit = item["pbf_audit"] or {}
        report.extend(
            [
                f"### {item['snapshot_id']}",
                "",
                f"- Date: {item['snapshot_date']}",
                f"- File: `{item['local_path']}`",
                f"- Bytes: {item['actual_bytes']}",
                f"- SHA-256: `{item['sha256']}`",
                f"- Highway ways: {audit.get('highway_ways')}",
                f"- Research bbox covered: {item['research_bbox_inside_pbf_header']}",
                "",
            ]
        )
    report.extend(
        [
            "## Gate checks",
            "",
            *[f"- {name}: `{passed}`" for name, passed in checks.items()],
            "",
            "No current OSM data were downloaded or used.",
        ]
    )
    (output_dir / "NETWORK_FREEZE_HISTORICAL_OSM_ATTEMPT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    exit_code = 0 if overall_gate == "PASS_HISTORICAL_OSM_SNAPSHOT" else 2
    if sys.platform == "win32":
        # pyosmium 4.2.0 may block while native PBF objects are destroyed on
        # Windows. All outputs are closed at this point, so bypass teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
