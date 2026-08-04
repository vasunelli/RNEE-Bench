#!/usr/bin/env python3
"""SOURCE_INVENTORY: inventory official VED files and materialize immutable row lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - explicit environment gate
    raise SystemExit(
        "pyarrow is required for SOURCE_INVENTORY row_lineage.parquet. "
        "Install it in the isolated RNEE runtime before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def is_read_only(path: Path) -> bool:
    stat = path.stat()
    if os.name == "nt":
        return bool(getattr(stat, "st_file_attributes", 0) & 0x1)
    return not bool(stat.st_mode & 0o200)


def normalized_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        row = next(csv.reader(handle))
    return [str(value).strip().rstrip(";") for value in row]


def data_row_count(path: Path) -> int:
    count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
            last_byte = block[-1:] if block else last_byte
    physical_lines = count + (1 if last_byte and last_byte != b"\n" else 0)
    return max(physical_lines - 1, 0)


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit:
            return hit
    return None


def detect_static_header(path: Path, sheet_name: str) -> int:
    pquality_check = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=30)
    for index, row in pquality_check.iterrows():
        values = {str(value).strip().lower() for value in row if pd.notna(value)}
        if any(value in values for value in {"vehid", "veh id", "vehicle id"}):
            return int(index)
    raise ValueError(f"Could not locate VehId header in {path.name}/{sheet_name}")


def read_static_workbooks(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook_rows: list[dict[str, Any]] = []
    vehicle_frames: list[pd.DataFrame] = []
    for path in paths:
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            header_row = detect_static_header(path, sheet)
            frame = pd.read_excel(path, sheet_name=sheet, header=header_row)
            frame = frame.dropna(how="all").copy()
            frame.columns = [str(column).strip() for column in frame.columns]
            veh_col = find_column(frame.columns, ["VehId", "Veh Id", "Vehicle ID"])
            workbook_rows.append(
                {
                    "source_file": path.name,
                    "sheet_name": sheet,
                    "header_row_0based": header_row,
                    "row_count": int(len(frame)),
                    "column_count": int(len(frame.columns)),
                    "columns_json": json.dumps(list(frame.columns), ensure_ascii=False),
                    "vehid_column": veh_col or "",
                }
            )
            if veh_col:
                frame = frame.rename(columns={veh_col: "VehId"})
                frame["VehId"] = frame["VehId"].astype("string").str.strip()
                engine_col = find_column(
                    frame.columns, ["EngineType", "Vehicle Type", "Engine Type"]
                )
                frame["EngineType_official"] = (
                    frame[engine_col].astype("string").str.strip()
                    if engine_col
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                )
                frame["static_source_file"] = path.name
                frame["static_source_sheet"] = sheet
                vehicle_frames.append(frame)
    if not vehicle_frames:
        raise ValueError("No static metadata sheet with VehId was found")
    static = pd.concat(vehicle_frames, ignore_index=True, sort=False)
    static = static[static["VehId"].notna() & (static["VehId"] != "")]
    return pd.DataFrame(workbook_rows), static


def stable_row_ids(file_sha256: str, start: int, count: int) -> list[str]:
    return [
        hashlib.sha256(f"{file_sha256}:{row_number}".encode("ascii")).hexdigest()
        for row_number in range(start, start + count)
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_7z_archives(paths: list[Path]) -> list[dict[str, Any]]:
    executable = shutil.which("7z")
    if not executable and os.name == "nt":
        candidate = Path(r"C:\Program Files\7-Zip\7z.exe")
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        raise FileNotFoundError("7z executable is required to test VED archives")
    results: list[dict[str, Any]] = []
    for path in paths:
        completed = subprocess.run(
            [executable, "t", str(path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        results.append(
            {
                "file_name": path.name,
                "exit_code": completed.returncode,
                "passed": completed.returncode == 0 and "Everything is Ok" in completed.stdout,
            }
        )
    return results


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_dir = Path(config["source"]["raw_dir"])
    extracted_dir = Path(config["source"]["extracted_dir"])
    official_names = list(config["official_files"])
    official_paths = [raw_dir / name for name in official_names]
    missing = [str(path) for path in official_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official files: {missing}")

    started = datetime.now(timezone.utc)
    archive_inventory: list[dict[str, Any]] = []
    for path in official_paths:
        file_sha = sha256_file(path)
        archive_inventory.append(
            {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha,
                "expected_sha256": config["expected_sha256"][path.name],
                "git_blob_sha1": config["expected_git_blob_sha1"][path.name],
                "hash_matches_contract": file_sha
                == config["expected_sha256"][path.name].lower(),
                "source_commit": config["source"]["commit"],
                "source_url": (
                    "https://raw.githubusercontent.com/gsoh/VED/"
                    f"{config['source']['commit']}/Data/{path.name.replace('&', '%26')}"
                ),
                "is_read_only": is_read_only(path),
            }
        )
    archive_df = pd.DataFrame(archive_inventory)
    archive_df.to_parquet(output_dir / "source_archive_inventory.parquet", index=False)
    archive_df.to_csv(output_dir / "source_archive_inventory.csv", index=False)
    archive_test_df = pd.DataFrame(
        test_7z_archives([path for path in official_paths if path.suffix.lower() == ".7z"])
    )
    archive_test_df.to_csv(output_dir / "archive_test_results.csv", index=False)

    dynamic_paths = sorted(extracted_dir.rglob(config["dynamic_file_pattern"]))
    if not dynamic_paths:
        raise FileNotFoundError(
            f"No dynamic files matching {config['dynamic_file_pattern']} in {extracted_dir}"
        )

    file_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    file_hashes: dict[Path, str] = {}
    for path in dynamic_paths:
        file_sha = sha256_file(path)
        file_hashes[path] = file_sha
        header = normalized_header(path)
        rows = data_row_count(path)
        schema_hash = hashlib.sha256(
            json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        file_rows.append(
            {
                "source_file": path.name,
                "relative_path": path.relative_to(extracted_dir).as_posix(),
                "bytes": path.stat().st_size,
                "data_rows": rows,
                "sha256": file_sha,
                "schema_hash": schema_hash,
                "is_read_only": is_read_only(path),
            }
        )
        schema_rows.append(
            {
                "source_file": path.name,
                "schema_hash": schema_hash,
                "column_count": len(header),
                "columns_json": json.dumps(header, ensure_ascii=False),
            }
        )
    file_df = pd.DataFrame(file_rows)
    schema_df = pd.DataFrame(schema_rows)
    file_df.to_parquet(output_dir / "source_file_inventory.parquet", index=False)
    schema_df.to_parquet(output_dir / "schema_by_file.parquet", index=False)
    file_df.to_csv(output_dir / "source_file_inventory.csv", index=False)
    schema_df.to_csv(output_dir / "schema_by_file.csv", index=False)

    lineage_schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("source_file", pa.string()),
            ("source_file_sha256", pa.string()),
            ("source_row_number", pa.int64()),
            ("VehId", pa.string()),
            ("Trip", pa.string()),
            ("Timestamp_raw", pa.string()),
        ]
    )
    lineage_writer = pq.ParquetWriter(
        output_dir / "row_lineage.parquet", lineage_schema, compression="zstd"
    )
    trip_counter: Counter[tuple[str, str, str]] = Counter()
    dynamic_vehicles: set[str] = set()
    duplicate_key_count = 0
    null_key_rows = 0
    total_processed = 0
    try:
        for path in dynamic_paths:
            header = normalized_header(path)
            veh_col = find_column(header, ["VehId", "Veh Id", "Vehicle ID"])
            trip_col = find_column(header, ["Trip"])
            time_col = find_column(
                header, ["Timestamp(ms)", "Timestamp", "Timestamp [ms]", "TimeStamp"]
            )
            if not veh_col or not trip_col or not time_col:
                raise ValueError(
                    f"Required key columns missing in {path.name}: "
                    f"VehId={veh_col}, Trip={trip_col}, Timestamp={time_col}"
                )
            offset = 0
            prior_keys: set[tuple[str, str, str]] = set()
            for chunk in pd.read_csv(
                path,
                usecols=[veh_col, trip_col, time_col],
                dtype="string",
                chunksize=args.chunk_rows,
                encoding="utf-8-sig",
                low_memory=False,
            ):
                chunk = chunk.rename(
                    columns={veh_col: "VehId", trip_col: "Trip", time_col: "Timestamp_raw"}
                )
                for column in ["VehId", "Trip", "Timestamp_raw"]:
                    chunk[column] = chunk[column].astype("string").str.strip()
                null_key_rows += int(chunk[["VehId", "Trip", "Timestamp_raw"]].isna().any(axis=1).sum())
                keys = list(
                    zip(
                        chunk["VehId"].fillna("").tolist(),
                        chunk["Trip"].fillna("").tolist(),
                        chunk["Timestamp_raw"].fillna("").tolist(),
                    )
                )
                key_counter = Counter(keys)
                duplicate_key_count += sum(value - 1 for value in key_counter.values() if value > 1)
                duplicate_key_count += sum(1 for key in key_counter if key in prior_keys)
                prior_keys.update(key_counter)
                dynamic_vehicles.update(value for value in chunk["VehId"].dropna() if value)
                trip_counter.update(
                    (path.name, veh, trip)
                    for veh, trip in zip(
                        chunk["VehId"].fillna(""),
                        chunk["Trip"].fillna(""),
                    )
                )
                count = len(chunk)
                table = pa.Table.from_pydict(
                    {
                        "row_id": stable_row_ids(file_hashes[path], offset, count),
                        "source_file": [path.name] * count,
                        "source_file_sha256": [file_hashes[path]] * count,
                        "source_row_number": list(range(offset, offset + count)),
                        "VehId": chunk["VehId"].tolist(),
                        "Trip": chunk["Trip"].tolist(),
                        "Timestamp_raw": chunk["Timestamp_raw"].tolist(),
                    },
                    schema=lineage_schema,
                )
                lineage_writer.write_table(table)
                offset += count
                total_processed += count
            expected = int(file_df.loc[file_df["source_file"] == path.name, "data_rows"].iloc[0])
            if offset != expected:
                raise ValueError(f"Row conservation failed for {path.name}: {offset} != {expected}")
    finally:
        lineage_writer.close()

    trip_df = pd.DataFrame(
        [
            {
                "source_file": source_file,
                "VehId": veh,
                "Trip": trip,
                "row_count": row_count,
            }
            for (source_file, veh, trip), row_count in trip_counter.items()
        ]
    )
    trip_df.to_parquet(output_dir / "trip_inventory.parquet", index=False)
    trip_df.to_csv(output_dir / "trip_inventory.csv", index=False)

    static_paths = [path for path in official_paths if path.suffix.lower() == ".xlsx"]
    static_workbook_df, static_df = read_static_workbooks(static_paths)
    for column in static_df.columns:
        if static_df[column].dtype == "object":
            static_df[column] = static_df[column].astype("string")
    static_df.to_parquet(output_dir / "static_vehicle_metadata.parquet", index=False)
    static_workbook_df.to_csv(output_dir / "static_workbook_inventory.csv", index=False)
    static_df.to_csv(output_dir / "static_vehicle_metadata.csv", index=False)

    static_ids = set(static_df["VehId"].astype("string").dropna())
    static_duplicate_ids = int(static_df["VehId"].duplicated().sum())
    powertrain_counts = {
        str(key): int(value)
        for key, value in static_df["EngineType_official"].value_counts(dropna=False).items()
    }
    missing_static = sorted(dynamic_vehicles - static_ids)
    static_not_dynamic = sorted(static_ids - dynamic_vehicles)
    matched = len(dynamic_vehicles & static_ids)
    join_rate = matched / len(dynamic_vehicles) if dynamic_vehicles else 0.0
    join_audit = pd.DataFrame(
        [
            {"audit": "dynamic_vehicle_count", "value": len(dynamic_vehicles)},
            {"audit": "static_vehicle_count", "value": len(static_ids)},
            {"audit": "matched_dynamic_vehicle_count", "value": matched},
            {"audit": "dynamic_static_join_rate", "value": join_rate},
            {"audit": "dynamic_without_static_count", "value": len(missing_static)},
            {"audit": "static_without_dynamic_count", "value": len(static_not_dynamic)},
        ]
    )
    join_audit.to_csv(output_dir / "dynamic_static_join_audit.csv", index=False)
    pd.DataFrame({"VehId": missing_static}).to_csv(
        output_dir / "dynamic_vehicles_without_static.csv", index=False
    )
    pd.DataFrame({"VehId": static_not_dynamic}).to_csv(
        output_dir / "static_vehicles_without_dynamic.csv", index=False
    )

    n_rows = int(file_df["data_rows"].sum())
    schema_variants = int(schema_df["schema_hash"].nunique())
    thresholds = config["quality_gates"]
    checks = {
        "official_file_count": len(official_paths) == thresholds["official_file_count"],
        "official_hashes_match_contract": bool(
            archive_df["hash_matches_contract"].all()
        ),
        "archive_integrity": bool(archive_test_df["passed"].all()),
        "official_files_read_only": bool(archive_df["is_read_only"].all()),
        "dynamic_files_read_only": bool(file_df["is_read_only"].all()),
        "row_conservation": total_processed == n_rows,
        "row_id_uniqueness_by_construction": total_processed == n_rows,
        "schema_variant_count": schema_variants <= thresholds["schema_variant_count_max"],
        "dynamic_static_join_rate": join_rate
        >= thresholds["dynamic_static_vehicle_join_rate_min"],
        "static_vehid_unique": static_duplicate_ids == 0,
    }
    gate = "PASS_RAW_LINEAGE" if all(checks.values()) else "STOP_RAW_LINEAGE"
    summary = {
        "experiment": "SOURCE_INVENTORY",
        "status": gate,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": config["source"]["repository"],
        "source_commit": config["source"]["commit"],
        "n_official_release_files": len(official_paths),
        "n_ved_dynamic_files": len(dynamic_paths),
        "n_ved_rows": n_rows,
        "n_schema_variants": schema_variants,
        "source_file_date_min": min(
            match.group(1)
            for value in file_df["source_file"]
            if (match := re.search(r"VED_(\d{6})_week\.csv$", value))
        ),
        "source_file_date_max": max(
            match.group(1)
            for value in file_df["source_file"]
            if (match := re.search(r"VED_(\d{6})_week\.csv$", value))
        ),
        "n_dynamic_vehicles": len(dynamic_vehicles),
        "n_static_vehicles": len(static_ids),
        "static_duplicate_vehid_count": static_duplicate_ids,
        "powertrain_counts": powertrain_counts,
        "dynamic_static_join_rate": join_rate,
        "dynamic_without_static": missing_static,
        "static_without_dynamic_count": len(static_not_dynamic),
        "duplicate_veh_trip_timestamp_rows": duplicate_key_count,
        "null_key_rows": null_key_rows,
        "row_id_formula": config["lineage"]["row_id_formula"],
        "checks": checks,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    write_json(output_dir / "raw_integrity_summary.json", summary)
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS_RAW_LINEAGE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
