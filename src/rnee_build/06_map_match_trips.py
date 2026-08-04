#!/usr/bin/env python3
"""Run the MAP_MATCHING quality-only Valhalla map-matching profile pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


UINT64_MAX = 2**64 - 1


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def runtime_environment(base: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        str(Path(base["valhalla"]["dll_directory"]))
        + os.pathsep
        + env.get("PATH", "")
    )
    return env


def allocate_month_targets(
    counts: pd.Series, total: int, minimum_per_month: int = 3
) -> dict[str, int]:
    months = list(counts.index)
    if total < minimum_per_month * len(months):
        raise ValueError("Pilot count is too small for the month coverage contract.")
    targets = {month: minimum_per_month for month in months}
    remaining = total - sum(targets.values())
    weights = counts / counts.sum()
    raw = weights * remaining
    floors = np.floor(raw).astype(int)
    for month, value in floors.items():
        targets[month] += int(value)
    leftover = total - sum(targets.values())
    remainders = sorted(
        months,
        key=lambda month: (-(raw[month] - floors[month]), month),
    )
    for month in remainders[:leftover]:
        targets[month] += 1
    return targets


def round_robin_stratified_sample(
    frame: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    data = frame.copy()
    data["row_bin"] = pd.qcut(
        data["row_count"].rank(method="first"), 4, labels=False
    )
    jump_rate = data["gps_update_speed_warning_pairs"] / data[
        "gps_update_pairs"
    ].clip(lower=1)
    data["jump_bin"] = pd.cut(
        jump_rate,
        bins=[-np.inf, 0, 1e-4, 1e-3, np.inf],
        labels=False,
    ).fillna(0)
    mid_lat = (data["min_lat"] + data["max_lat"]) / 2
    mid_lon = (data["min_lon"] + data["max_lon"]) / 2
    data["spatial_bin"] = (
        (mid_lat >= mid_lat.median()).astype(int) * 2
        + (mid_lon >= mid_lon.median()).astype(int)
    )
    data["month"] = data["trip_mid_date"].str.slice(0, 7)
    data["selection_hash"] = [
        stable_hash(seed, row.source_file, row.VehId, row.Trip)
        for row in data.itertuples(index=False)
    ]
    month_targets = allocate_month_targets(
        data.groupby("month").size(), count
    )
    selected: list[pd.DataFrame] = []
    for month, target in month_targets.items():
        month_data = data[data["month"] == month]
        groups = [
            group.sort_values("selection_hash")
            for _, group in month_data.groupby(
                ["osm_snapshot_id", "row_bin", "jump_bin", "spatial_bin"],
                sort=True,
            )
        ]
        positions = [0] * len(groups)
        rows: list[pd.Series] = []
        while len(rows) < target:
            progressed = False
            for index, group in enumerate(groups):
                if positions[index] < len(group) and len(rows) < target:
                    rows.append(group.iloc[positions[index]])
                    positions[index] += 1
                    progressed = True
            if not progressed:
                break
        selected.append(pd.DataFrame(rows))
    result = pd.concat(selected, ignore_index=True)
    if len(result) != count:
        raise RuntimeError(
            f"Pilot selection cardinality {len(result)} != {count}."
        )
    result["pilot_trip_id"] = [
        stable_hash(row.source_file, row.VehId, row.Trip)[:20]
        for row in result.itertuples(index=False)
    ]
    return result.sort_values(
        ["month", "osm_snapshot_id", "selection_hash"]
    ).reset_index(drop=True)


def extract_selected_trips(
    selection: pd.DataFrame,
    source_dir: Path,
    columns: dict[str, str],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    usecols = list(columns.values())
    for source_file, selected_file in selection.groupby("source_file"):
        source_path = source_dir / source_file
        raw = pd.read_csv(source_path, usecols=usecols)
        for selected in selected_file.itertuples(index=False):
            trip = raw[
                (raw[columns["vehicle"]] == selected.VehId)
                & (raw[columns["trip"]] == selected.Trip)
            ].copy()
            if len(trip) != int(selected.row_count):
                raise RuntimeError(
                    f"Raw cardinality mismatch for {selected.pilot_trip_id}: "
                    f"{len(trip)} != {selected.row_count}."
                )
            trip = trip.reset_index(drop=True)
            trip["point_index"] = np.arange(len(trip), dtype=np.int64)
            output[selected.pilot_trip_id] = trip
    if set(output) != set(selection["pilot_trip_id"]):
        raise RuntimeError("Not every selected pilot trip was extracted.")
    return output


def build_request(
    trip: pd.DataFrame,
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    columns = config["columns"]
    shape = [
        {
            "lat": float(row[columns["latitude"]]),
            "lon": float(row[columns["longitude"]]),
            "time": float(row[columns["timestamp_ms"]]) / 1000.0,
        }
        for _, row in trip.iterrows()
    ]
    return {
        "shape": shape,
        "costing": config["costing"],
        "shape_match": config["shape_match"],
        "trace_options": {
            "gps_accuracy": float(profile["gps_accuracy_m"]),
            "search_radius": float(profile["search_radius_m"]),
            "breakage_distance": float(
                profile.get("breakage_distance_m", config["breakage_distance_m"])
            ),
        },
        "filters": {
            "action": "include",
            "attributes": list(config["response_attributes"]),
        },
    }


def invoke_valhalla(
    service: Path,
    graph_config: Path,
    endpoint: str,
    request_text: str,
    env: dict[str, str],
    timeout: int,
    attempts: int,
) -> tuple[subprocess.CompletedProcess[str], int]:
    last: subprocess.CompletedProcess[str] | None = None
    request_fd, request_name = tempfile.mkstemp(
        prefix="rnee_valhalla_request_",
        suffix=".json",
        text=True,
    )
    os.close(request_fd)
    request_path = Path(request_name)
    request_path.write_text(request_text, encoding="utf-8")
    try:
        for attempt in range(1, attempts + 1):
            last = subprocess.run(
                [str(service), str(graph_config), endpoint, str(request_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
            if last.returncode == 0:
                return last, attempt
            if attempt < attempts:
                time.sleep(0.25 * (2 ** (attempt - 1)))
    finally:
        request_path.unlink(missing_ok=True)
    assert last is not None
    return last, attempts


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.minimum(1, np.sqrt(a)))


def count_silent_fallbacks(point_frame: pd.DataFrame) -> int:
    """Count within inconsistent rows that would hide match uncertainty."""
    allowed = {
        "matched_with_edge",
        "matched_without_edge",
        "unmatched",
        "request_error",
        "error_cardinality",
        "input_break_singleton",
        "input_break_short_chunk",
    }
    status = point_frame["match_status"]
    valid_coordinates = point_frame[
        ["matched_latitude", "matched_longitude"]
    ].notna().all(axis=1)
    valid_edge = point_frame["edge_index_valid"].fillna(False).astype(bool)
    contradiction = (
        status.isna()
        | ~status.isin(allowed)
        | (status.eq("matched_with_edge") & (~valid_coordinates | ~valid_edge))
        | (status.eq("matched_without_edge") & (~valid_coordinates | valid_edge))
    )
    return int(contradiction.sum())


def parse_success(
    selected: Any,
    trip: pd.DataFrame,
    trace_positions: np.ndarray,
    profile_name: str,
    response: dict[str, Any],
    request_id: str,
    response_hash: str,
    attempt_count: int,
    valhalla_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    points = response.get("matched_points")
    edges = response.get("edges", [])
    cardinality_ok = (
        isinstance(points, list) and len(points) == len(trace_positions)
    )
    point_rows: list[dict[str, Any]] = []
    columns = {
        "lat": "Latitude[deg]",
        "lon": "Longitude[deg]",
        "time": "Timestamp(ms)",
    }
    if not cardinality_ok:
        points = [{} for _ in range(len(trace_positions))]
    for index, (_, raw) in enumerate(trip.iterrows()):
        trace_index = int(
            np.searchsorted(trace_positions, index, side="right") - 1
        )
        matched = (
            points[trace_index]
            if isinstance(points[trace_index], dict)
            else {}
        )
        is_coordinate_update = bool(index == trace_positions[trace_index])
        edge_index_raw = matched.get("edge_index")
        edge_index_valid = (
            isinstance(edge_index_raw, int)
            and 0 <= edge_index_raw < len(edges)
            and edge_index_raw != UINT64_MAX
        )
        source_match_type = str(matched.get("type", "unmatched"))
        match_type = (
            source_match_type
            if is_coordinate_update or source_match_type == "unmatched"
            else "sample_hold_propagated"
        )
        if not cardinality_ok:
            status = "error_cardinality"
        elif source_match_type in {
            "unmatched",
            "request_error",
            "input_break_singleton",
            "input_break_short_chunk",
        }:
            status = source_match_type
        elif edge_index_valid:
            status = "matched_with_edge"
        else:
            status = "matched_without_edge"
        point_rows.append(
            {
                "pilot_trip_id": selected.pilot_trip_id,
                "source_file": selected.source_file,
                "VehId": int(selected.VehId),
                "Trip": int(selected.Trip),
                "point_index": index,
                "trace_point_index": trace_index,
                "trace_chunk_index": matched.get("_trace_chunk_index"),
                "input_quality_break_before": bool(
                    matched.get("_input_quality_break_before", False)
                    and is_coordinate_update
                ),
                "is_coordinate_update": is_coordinate_update,
                "sample_hold_propagated": not is_coordinate_update,
                "timestamp_ms_raw": float(raw[columns["time"]]),
                "latitude_raw": float(raw[columns["lat"]]),
                "longitude_raw": float(raw[columns["lon"]]),
                "osm_snapshot_id": selected.osm_snapshot_id,
                "profile": profile_name,
                "request_id": request_id,
                "match_status": status,
                "match_type": match_type,
                "matched_latitude": matched.get("lat"),
                "matched_longitude": matched.get("lon"),
                "matched_edge_index": (
                    int(edge_index_raw) if edge_index_valid else None
                ),
                "edge_index_valid": edge_index_valid,
                "distance_along_edge": matched.get("distance_along_edge"),
                "distance_from_trace_point_m": matched.get(
                    "distance_from_trace_point"
                ),
                "begin_route_discontinuity": bool(
                    is_coordinate_update
                    and matched.get("begin_route_discontinuity", False)
                ),
                "end_route_discontinuity": bool(
                    is_coordinate_update
                    and matched.get("end_route_discontinuity", False)
                ),
                "response_sha256": response_hash,
                "attempt_count": attempt_count,
                "valhalla_version": valhalla_version,
            }
        )
    edge_rows = []
    for edge_index, edge in enumerate(edges):
        edge_rows.append(
            {
                "pilot_trip_id": selected.pilot_trip_id,
                "source_file": selected.source_file,
                "VehId": int(selected.VehId),
                "Trip": int(selected.Trip),
                "osm_snapshot_id": selected.osm_snapshot_id,
                "profile": profile_name,
                "request_id": request_id,
                "edge_index": edge_index,
                "trace_chunk_index": edge.get("_trace_chunk_index"),
                "edge_id": edge.get("id"),
                "way_id": edge.get("way_id"),
                "road_class": edge.get("road_class"),
                "use": edge.get("use"),
                "length_km": edge.get("length"),
                "speed_limit_kmh": edge.get("speed_limit"),
                "lane_count": edge.get("lane_count"),
                "surface": edge.get("surface"),
                "tunnel": edge.get("tunnel"),
                "bridge": edge.get("bridge"),
                "roundabout": edge.get("roundabout"),
                "forward": edge.get("forward"),
                "traffic_signal": edge.get("traffic_signal"),
                "weighted_grade": edge.get("weighted_grade"),
                "max_upward_grade": edge.get("max_upward_grade"),
                "max_downward_grade": edge.get("max_downward_grade"),
                "mean_elevation": edge.get("mean_elevation"),
            }
        )
    point_frame = pd.DataFrame(point_rows)
    matched = point_frame["match_status"].isin(
        ["matched_with_edge", "matched_without_edge"]
    )
    distance = pd.to_numeric(
        point_frame["distance_from_trace_point_m"], errors="coerce"
    )
    valid_coords = point_frame[
        ["matched_latitude", "matched_longitude"]
    ].notna().all(axis=1)
    pair_mask = (
        matched.shift(fill_value=False)
        & matched
        & valid_coords.shift(fill_value=False)
        & valid_coords
    )
    dt = point_frame["timestamp_ms_raw"].diff() / 1000.0
    pair_mask &= dt.gt(0) & dt.le(10)
    jumps = np.zeros(len(point_frame), dtype=bool)
    if pair_mask.any():
        positions = np.flatnonzero(pair_mask.to_numpy())
        jump_distance = haversine_m(
            point_frame.loc[positions - 1, "matched_latitude"].to_numpy(float),
            point_frame.loc[positions - 1, "matched_longitude"].to_numpy(float),
            point_frame.loc[positions, "matched_latitude"].to_numpy(float),
            point_frame.loc[positions, "matched_longitude"].to_numpy(float),
        )
        jumps[positions] = jump_distance > 500
    summary = {
        "pilot_trip_id": selected.pilot_trip_id,
        "source_file": selected.source_file,
        "VehId": int(selected.VehId),
        "Trip": int(selected.Trip),
        "trip_mid_date": selected.trip_mid_date,
        "osm_snapshot_id": selected.osm_snapshot_id,
        "profile": profile_name,
        "request_id": request_id,
        "raw_point_count": len(trip),
        "request_point_count": len(trace_positions),
        "response_point_count": (
            len(response.get("matched_points", []))
            if isinstance(response.get("matched_points"), list)
            else 0
        ),
        "cardinality_ok": cardinality_ok,
        "output_point_count": len(point_rows),
        "row_cardinality_ok": len(point_rows) == len(trip),
        "sample_hold_propagated_count": int(
            point_frame["sample_hold_propagated"].sum()
        ),
        "status_reason_coverage": 1.0,
        "matched_point_count": int(matched.sum()),
        "matched_with_edge_count": int(
            point_frame["match_status"].eq("matched_with_edge").sum()
        ),
        "matched_coverage": float(matched.mean()),
        "edge_association_coverage": float(
            point_frame["match_status"].eq("matched_with_edge").mean()
        ),
        "distance_p50_m": (
            float(distance.quantile(0.50)) if distance.notna().any() else None
        ),
        "distance_p95_m": (
            float(distance.quantile(0.95)) if distance.notna().any() else None
        ),
        "distance_p99_m": (
            float(distance.quantile(0.99)) if distance.notna().any() else None
        ),
        "has_route_discontinuity": bool(
            point_frame[
                ["begin_route_discontinuity", "end_route_discontinuity"]
            ].any(axis=None)
        ),
        "eligible_short_time_pairs": int(pair_mask.sum()),
        "edge_jump_over_500m_within_10s_count": int(jumps.sum()),
        "silent_fallback_count": count_silent_fallbacks(point_frame),
        "edge_count": len(edges),
        "warning_count": len(response.get("warnings", [])),
        "osm_changeset": response.get("osm_changeset"),
        "response_sha256": response_hash,
        "attempt_count": attempt_count,
        "request_succeeded": True,
        "error_code": None,
        "error_message": None,
    }
    return point_rows, edge_rows, summary


def failed_rows(
    selected: Any,
    trip: pd.DataFrame,
    request_point_count: int,
    profile_name: str,
    request_id: str,
    attempt_count: int,
    returncode: int,
    message: str,
    valhalla_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    point_rows = []
    for index, raw in trip.iterrows():
        point_rows.append(
            {
                "pilot_trip_id": selected.pilot_trip_id,
                "source_file": selected.source_file,
                "VehId": int(selected.VehId),
                "Trip": int(selected.Trip),
                "point_index": int(index),
                "trace_point_index": None,
                "is_coordinate_update": None,
                "sample_hold_propagated": None,
                "timestamp_ms_raw": float(raw["Timestamp(ms)"]),
                "latitude_raw": float(raw["Latitude[deg]"]),
                "longitude_raw": float(raw["Longitude[deg]"]),
                "osm_snapshot_id": selected.osm_snapshot_id,
                "profile": profile_name,
                "request_id": request_id,
                "match_status": "request_error",
                "match_type": "unmatched",
                "matched_latitude": None,
                "matched_longitude": None,
                "matched_edge_index": None,
                "edge_index_valid": False,
                "distance_along_edge": None,
                "distance_from_trace_point_m": None,
                "begin_route_discontinuity": False,
                "end_route_discontinuity": False,
                "response_sha256": None,
                "attempt_count": attempt_count,
                "valhalla_version": valhalla_version,
            }
        )
    point_frame = pd.DataFrame(point_rows)
    summary = {
        "pilot_trip_id": selected.pilot_trip_id,
        "source_file": selected.source_file,
        "VehId": int(selected.VehId),
        "Trip": int(selected.Trip),
        "trip_mid_date": selected.trip_mid_date,
        "osm_snapshot_id": selected.osm_snapshot_id,
        "profile": profile_name,
        "request_id": request_id,
        "raw_point_count": len(trip),
        "request_point_count": request_point_count,
        "response_point_count": 0,
        "cardinality_ok": False,
        "output_point_count": len(point_rows),
        "row_cardinality_ok": len(point_rows) == len(trip),
        "sample_hold_propagated_count": 0,
        "status_reason_coverage": 1.0,
        "matched_point_count": 0,
        "matched_with_edge_count": 0,
        "matched_coverage": 0.0,
        "edge_association_coverage": 0.0,
        "distance_p50_m": None,
        "distance_p95_m": None,
        "distance_p99_m": None,
        "has_route_discontinuity": True,
        "eligible_short_time_pairs": 0,
        "edge_jump_over_500m_within_10s_count": 0,
        "silent_fallback_count": count_silent_fallbacks(point_frame),
        "edge_count": 0,
        "warning_count": 0,
        "osm_changeset": None,
        "response_sha256": None,
        "attempt_count": attempt_count,
        "request_succeeded": False,
        "error_code": returncode,
        "error_message": message[-2000:],
    }
    return point_rows, [], summary


def run_one(
    selected: Any,
    trip: pd.DataFrame,
    profile_name: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    base: dict[str, Any],
    pointer: dict[str, Any],
    response_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    columns = config["columns"]
    coordinate_updates = (
        trip[columns["latitude"]].ne(trip[columns["latitude"]].shift())
        | trip[columns["longitude"]].ne(trip[columns["longitude"]].shift())
    ).to_numpy(copy=True)
    coordinate_updates[0] = True
    trace_positions = np.flatnonzero(coordinate_updates)
    trace = trip.iloc[trace_positions].copy()
    request = build_request(trace, config, profile)
    request_text = json.dumps(request, separators=(",", ":"))
    request_hash = sha256_text(request_text)
    request_id = stable_hash(
        "MAP_MATCHING",
        selected.pilot_trip_id,
        selected.osm_snapshot_id,
        profile_name,
        request_hash,
    )[:24]
    process, attempts = invoke_valhalla(
        Path(base["valhalla"]["service_executable"]),
        Path(pointer["valhalla_configs"][selected.osm_snapshot_id]),
        config["endpoint"],
        request_text,
        runtime_environment(base),
        int(config["request_timeout_seconds"]),
        int(config["maximum_attempts"]),
    )
    if process.returncode != 0:
        return failed_rows(
            selected,
            trip,
            len(trace),
            profile_name,
            request_id,
            attempts,
            process.returncode,
            process.stderr,
            base["valhalla"]["version"],
        )
    response_hash = sha256_text(process.stdout)
    response_path = response_dir / f"{request_id}.json.gz"
    with gzip.open(response_path, "wt", encoding="utf-8") as handle:
        handle.write(process.stdout)
    try:
        response = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        return failed_rows(
            selected,
            trip,
            len(trace),
            profile_name,
            request_id,
            attempts,
            -1,
            str(error),
            base["valhalla"]["version"],
        )
    rows = parse_success(
        selected,
        trip,
        trace_positions,
        profile_name,
        response,
        request_id,
        response_hash,
        attempts,
        base["valhalla"]["version"],
    )
    rows[2]["request_sha256"] = request_hash
    rows[2]["response_path"] = str(response_path)
    return rows


def trace_chunk_indices(
    trace: pd.DataFrame,
    config: dict[str, Any],
    remediation: dict[str, Any],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split coordinate updates at frozen raw-GPS quality violations."""
    if len(trace) == 0:
        return [], np.array([], dtype=bool)
    columns = config["columns"]
    latitude = np.radians(trace[columns["latitude"]].to_numpy(float))
    longitude = np.radians(trace[columns["longitude"]].to_numpy(float))
    timestamp = trace[columns["timestamp_ms"]].to_numpy(float) / 1000.0
    break_before = np.zeros(len(trace), dtype=bool)
    break_before[0] = True
    if len(trace) > 1:
        delta_latitude = np.diff(latitude)
        delta_longitude = np.diff(longitude)
        haversine = (
            np.sin(delta_latitude / 2) ** 2
            + np.cos(latitude[:-1])
            * np.cos(latitude[1:])
            * np.sin(delta_longitude / 2) ** 2
        )
        distance = 2 * 6_371_008.8 * np.arcsin(
            np.minimum(1, np.sqrt(haversine))
        )
        dt = np.diff(timestamp)
        speed = np.divide(
            distance * 3.6,
            dt,
            out=np.full_like(distance, np.inf),
            where=dt > 0,
        )
        rules = remediation["raw_gps_break_rules"]
        break_before[1:] = (
            (dt <= 0)
            | (distance > float(rules["maximum_coordinate_update_jump_m"]))
            | (speed > float(rules["maximum_coordinate_update_speed_kmh"]))
        )
    starts = np.flatnonzero(break_before)
    chunks = [
        np.arange(start, end, dtype=int)
        for start, end in zip(starts, list(starts[1:]) + [len(trace)])
    ]
    return chunks, break_before


def valhalla_error_code(process: subprocess.CompletedProcess[str]) -> int | None:
    """Return a structured Valhalla error code when stderr is only warnings."""
    try:
        payload = json.loads(process.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    value = payload.get("error_code")
    return int(value) if isinstance(value, int) else None


def midpoint_fallback_split(
    positions: np.ndarray,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a failed trace deterministically without dropping any point."""
    if len(positions) < 2 * minimum_points:
        return None
    midpoint = len(positions) // 2
    return positions[:midpoint], positions[midpoint:]


def run_segmented_one(
    selected: Any,
    trip: pd.DataFrame,
    profile_name: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    base: dict[str, Any],
    pointer: dict[str, Any],
    response_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    columns = config["columns"]
    coordinate_updates = (
        trip[columns["latitude"]].ne(trip[columns["latitude"]].shift())
        | trip[columns["longitude"]].ne(trip[columns["longitude"]].shift())
    ).to_numpy(copy=True)
    coordinate_updates[0] = True
    trace_positions = np.flatnonzero(coordinate_updates)
    trace = trip.iloc[trace_positions].copy()
    remediation = config["_remediation"]
    chunks, break_before = trace_chunk_indices(trace, config, remediation)
    composite_points: list[dict[str, Any]] = [
        {"type": "request_error"} for _ in range(len(trace))
    ]
    composite_edges: list[dict[str, Any]] = []
    request_texts: list[str] = []
    chunk_records = []
    attempts_total = 0
    chunk_failures = 0
    osm_changesets: list[Any] = []
    fallback = config.get("map_snap_failure_fallback", {})
    fallback_enabled = bool(fallback.get("enabled", False))
    fallback_error_codes = {int(value) for value in fallback.get("error_codes", [444])}
    fallback_minimum_points = int(fallback.get("minimum_points_per_child", 3))
    fallback_max_depth = int(fallback.get("maximum_depth", 4))
    leaf_chunk_count = 0
    for source_chunk_index, source_positions in enumerate(chunks):
        pending: list[tuple[np.ndarray, int]] = [(source_positions, 0)]
        while pending:
            positions, fallback_depth = pending.pop(0)
            chunk_index = leaf_chunk_count
            for trace_index in positions:
                composite_points[trace_index]["_trace_chunk_index"] = chunk_index
                composite_points[trace_index]["_input_quality_break_before"] = bool(
                    break_before[trace_index]
                )
            if len(positions) < 3:
                short_type = "input_break_singleton" if len(positions) == 1 else "input_break_short_chunk"
                for trace_index in positions:
                    composite_points[int(trace_index)] = {
                        "type": short_type,
                        "_trace_chunk_index": chunk_index,
                        "_input_quality_break_before": bool(break_before[trace_index]),
                    }
                chunk_records.append({"chunk_index": chunk_index, "point_count": len(positions), "status": short_type})
                leaf_chunk_count += 1
                continue
            request = build_request(trace.iloc[positions], config, profile)
            request_text = json.dumps(request, separators=(",", ":"))
            request_texts.append(request_text)
            process, attempts = invoke_valhalla(
                Path(base["valhalla"]["service_executable"]),
                Path(pointer["valhalla_configs"][selected.osm_snapshot_id]),
                config["endpoint"], request_text, runtime_environment(base),
                int(config["request_timeout_seconds"]), int(config["maximum_attempts"]),
            )
            attempts_total += attempts
            if process.returncode != 0:
                error_code = valhalla_error_code(process)
                split = midpoint_fallback_split(positions, fallback_minimum_points)
                if fallback_enabled and error_code in fallback_error_codes and fallback_depth < fallback_max_depth and split is not None:
                    left, right = split
                    pending = [(left, fallback_depth + 1), (right, fallback_depth + 1)] + pending
                    chunk_records.append({
                        "source_chunk_index": source_chunk_index, "point_count": len(positions),
                        "status": "map_snap_midpoint_fallback", "error_code": error_code,
                        "fallback_depth": fallback_depth, "child_point_counts": [len(left), len(right)],
                    })
                    continue
                chunk_failures += 1
                chunk_records.append({
                    "chunk_index": chunk_index, "point_count": len(positions), "status": "request_error",
                    "return_code": process.returncode, "error_code": error_code,
                    "stdout": process.stdout[-1000:], "error": process.stderr[-1000:],
                })
                leaf_chunk_count += 1
                continue
            try:
                response = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                chunk_failures += 1
                chunk_records.append({"chunk_index": chunk_index, "point_count": len(positions), "status": "json_error", "error": str(error)})
                leaf_chunk_count += 1
                continue
            matched_points = response.get("matched_points")
            edges = response.get("edges", [])
            if not isinstance(matched_points, list) or len(matched_points) != len(positions):
                chunk_failures += 1
                chunk_records.append({"chunk_index": chunk_index, "point_count": len(positions), "status": "cardinality_error"})
                leaf_chunk_count += 1
                continue
            edge_offset = len(composite_edges)
            for edge in edges:
                edge_copy = dict(edge); edge_copy["_trace_chunk_index"] = chunk_index; composite_edges.append(edge_copy)
            for trace_index, matched_raw in zip(positions, matched_points):
                matched = dict(matched_raw); edge_index = matched.get("edge_index")
                if isinstance(edge_index, int) and edge_index != UINT64_MAX and 0 <= edge_index < len(edges):
                    matched["edge_index"] = edge_index + edge_offset
                matched["_trace_chunk_index"] = chunk_index
                matched["_input_quality_break_before"] = bool(break_before[trace_index])
                composite_points[trace_index] = matched
            osm_changesets.append(response.get("osm_changeset"))
            chunk_records.append({"chunk_index": chunk_index, "point_count": len(positions), "status": "success", "edge_count": len(edges)})
            leaf_chunk_count += 1
    request_hash = sha256_text("\n".join(request_texts))
    request_id = stable_hash(
        "MAP_MATCHING-segmented",
        selected.pilot_trip_id,
        selected.osm_snapshot_id,
        profile_name,
        request_hash,
    )[:24]
    composite_response = {
        "matched_points": composite_points,
        "edges": composite_edges,
        "osm_changeset": osm_changesets[0] if osm_changesets else None,
        "chunks": chunk_records,
    }
    response_text = json.dumps(composite_response, separators=(",", ":"))
    response_hash = sha256_text(response_text)
    response_path = response_dir / f"{request_id}.json.gz"
    with gzip.open(response_path, "wt", encoding="utf-8") as handle:
        handle.write(response_text)
    rows = parse_success(
        selected,
        trip,
        trace_positions,
        profile_name,
        composite_response,
        request_id,
        response_hash,
        attempts_total,
        base["valhalla"]["version"],
    )
    rows[2].update(
        {
            "request_sha256": request_hash,
            "response_path": str(response_path),
            "request_succeeded": chunk_failures == 0,
            "chunk_count": leaf_chunk_count,
            "chunk_failure_count": chunk_failures,
            "input_quality_break_count": int(break_before.sum() - 1),
            "segmented_remediation": True,
        }
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/valhalla.yaml"),
    )
    parser.add_argument("--remediation", type=Path)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(r"configs/rnee_build/map_match_profiles.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    profiles_config = load_yaml(args.profiles)
    remediation = load_yaml(args.remediation) if args.remediation else None
    if remediation:
        config["_remediation"] = remediation
    base = load_yaml(Path(config["base_contract"]))
    pointer = load_json(Path(config["network_pointer"]))
    snapshot_summary = load_json(Path(config["snapshot_summary"]))
    if pointer.get("status") != "PASS_OSM_SNAPSHOT_AND_NETWORK":
        raise RuntimeError("NETWORK_FREEZE network gate has not passed.")
    if snapshot_summary.get("checks", {}).get(
        "no_future_snapshot_assignment"
    ) is not True:
        raise RuntimeError("NETWORK_FREEZE future-snapshot exposure gate has not passed.")
    assignment_ids = {
        item["osm_snapshot_id"]
        for item in snapshot_summary["snapshot_assignment"]
    }
    if assignment_ids != set(pointer["graphs"]):
        raise RuntimeError("Assigned snapshots and available graphs differ.")

    output_root = Path(config["output_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("run_name_prefix") or (
        "map_matching_map_match_segmented_pilot" if remediation else "map_matching_map_match_pilot"
    )
    output_dir = output_root / f"{run_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    response_dir = output_dir / "responses"
    response_dir.mkdir()
    shutil.copy2(args.config, output_dir / "run_config.yaml")
    shutil.copy2(args.profiles, output_dir / "profiles.yaml")
    if args.remediation:
        shutil.copy2(args.remediation, output_dir / "remediation.yaml")

    gps = pd.read_parquet(config["gps_trip_quality"])
    if config.get("all_trips_in_source_files"):
        source_files = list(config["source_files"])
        selection = gps[gps["source_file"].isin(source_files)].copy()
        missing_files = sorted(set(source_files) - set(selection["source_file"]))
        if missing_files:
            raise RuntimeError(f"Configured source files absent from GPS inventory: {missing_files}")
        selection = selection.sort_values(["source_file", "VehId", "Trip"]).reset_index(drop=True)
        selection["pilot_trip_id"] = [
            stable_hash(config.get("experiment", "MAP_MATCHING"), source, vehicle, trip)[:20]
            for source, vehicle, trip in zip(selection["source_file"], selection["VehId"], selection["Trip"])
        ]
    else:
        selection = round_robin_stratified_sample(
            gps,
            int(config["pilot_trip_count"]),
            int(config["pilot_selection_seed"]),
        )
    selection.to_parquet(output_dir / "pilot_selection.parquet", index=False)
    selection.to_csv(output_dir / "pilot_selection.csv", index=False)
    trips = extract_selected_trips(
        selection,
        Path(config["ved_dynamic_dir"]),
        config["columns"],
    )

    point_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    trip_rows: list[dict[str, Any]] = []
    futures = []
    with ThreadPoolExecutor(
        max_workers=int(config["parallel_workers"])
    ) as executor:
        for selected in selection.itertuples(index=False):
            for profile_name, profile in profiles_config["profiles"].items():
                futures.append(
                    executor.submit(
                        run_segmented_one if remediation else run_one,
                        selected,
                        trips[selected.pilot_trip_id],
                        profile_name,
                        profile,
                        config,
                        base,
                        pointer,
                        response_dir,
                    )
                )
        for future in as_completed(futures):
            points, edges, trip_summary = future.result()
            point_rows.extend(points)
            edge_rows.extend(edges)
            trip_rows.append(trip_summary)

    points_frame = pd.DataFrame(point_rows).sort_values(
        ["profile", "pilot_trip_id", "point_index"]
    )
    edges_frame = pd.DataFrame(edge_rows).sort_values(
        ["profile", "pilot_trip_id", "edge_index"]
    )
    trips_frame = pd.DataFrame(trip_rows).sort_values(
        ["profile", "pilot_trip_id"]
    )
    expected_points = int(selection["row_count"].sum()) * len(
        profiles_config["profiles"]
    )
    if len(points_frame) != expected_points:
        raise RuntimeError(
            f"Output point cardinality {len(points_frame)} != {expected_points}."
        )
    points_frame.to_parquet(output_dir / "matched_points.parquet", index=False)
    edges_frame.to_parquet(output_dir / "matched_edges.parquet", index=False)
    trips_frame.to_parquet(output_dir / "trip_match_summary.parquet", index=False)
    trips_frame.to_csv(output_dir / "trip_match_summary.csv", index=False)
    run_summary = {
        "experiment": config.get("experiment", "MAP_MATCHING"),
        "stage": "profile_pilot_raw_results",
        "status": "RAW_RESULTS_READY",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_trip_count": len(selection),
        "profile_count": len(profiles_config["profiles"]),
        "request_count": len(trips_frame),
        "input_point_count_per_profile": int(selection["row_count"].sum()),
        "output_point_count": len(points_frame),
        "expected_output_point_count": expected_points,
        "request_failure_count": int((~trips_frame["request_succeeded"]).sum()),
        "cardinality_failure_count": int((~trips_frame["cardinality_ok"]).sum()),
        "snapshot_ids": sorted(assignment_ids),
        "selection_uses_energy_or_eved_fields": False,
        "segmented_remediation": bool(remediation),
        "remediation_policy": (
            remediation.get("policy_name") if remediation else None
        ),
        "total_input_quality_break_count": int(
            trips_frame.get("input_quality_break_count", pd.Series(dtype=int)).sum()
        ),
        "output_directory": str(output_dir),
    }
    write_json(output_dir / "raw_run_summary.json", run_summary)
    write_json(
        output_root / config.get(
            "latest_raw_pointer_name",
            "map_matching_latest_segmented_raw_pointer.json" if remediation else "map_matching_latest_raw_pointer.json",
        ),
        {
            "status": run_summary["status"],
            "output_directory": str(output_dir),
            "summary": str(output_dir / "raw_run_summary.json"),
            "selection": str(output_dir / "pilot_selection.parquet"),
            "matched_points": str(output_dir / "matched_points.parquet"),
            "matched_edges": str(output_dir / "matched_edges.parquet"),
            "trip_summary": str(output_dir / "trip_match_summary.parquet"),
        },
    )
    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
