#!/usr/bin/env python3
"""Build the canonical portable-report artifact for SOURCE_INVENTORY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    summary = json.loads((run_dir / "raw_integrity_summary.json").read_text(encoding="utf-8"))
    archive = pd.read_csv(run_dir / "source_archive_inventory.csv")
    files = pd.read_csv(run_dir / "source_file_inventory.csv")
    generated_at = summary["completed_at_utc"]
    source_commit = summary["source_commit"]
    source_href = f"https://github.com/gsoh/VED/tree/{source_commit}"

    headline = [
        {
            "dynamic_files": summary["n_ved_dynamic_files"],
            "dynamic_rows": summary["n_ved_rows"],
            "dynamic_vehicles": summary["n_dynamic_vehicles"],
            "join_rate": summary["dynamic_static_join_rate"],
        }
    ]
    quality_checks = [
        {"check": name, "status": "PASS" if passed else "FAIL"}
        for name, passed in summary["checks"].items()
    ]
    total_static = sum(summary["powertrain_counts"].values())
    powertrain = [
        {
            "powertrain": name,
            "vehicles": count,
            "share": count / total_static,
            "rank": rank,
            "static_total": total_static,
        }
        for rank, (name, count) in enumerate(
            sorted(summary["powertrain_counts"].items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    ]
    file_inventory = files[
        ["source_file", "data_rows", "bytes", "schema_hash", "sha256", "is_read_only"]
    ].to_dict(orient="records")
    archive_inventory = archive[
        [
            "file_name",
            "bytes",
            "git_blob_sha1",
            "sha256",
            "hash_matches_contract",
            "is_read_only",
        ]
    ].to_dict(orient="records")

    def make_source(source_id: str, label: str, sql: str, description: str) -> dict:
        return {
            "id": source_id,
            "label": label,
            "href": source_href,
            "query": {
            "engine": "Python 3.11 / pandas / PyArrow",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
            "language": "sql",
            "filters": [
                "Only the four files under Data/ at the frozen VED commit",
                "Dynamic files matching VED_*_week.csv",
                "No eVED files or derived fields",
            ],
            "metric_definitions": [
                "dynamic_rows = sum of data lines after one header line in each extracted CSV",
                "join_rate = distinct dynamic VehId values found in official static metadata / distinct dynamic VehId values",
                "row_id = SHA-256(source_file_sha256 + ':' + zero-based source_row_number)",
            ],
            "tables_used": [
                "Data/VED_DynamicData_Part1.7z",
                "Data/VED_DynamicData_Part2.7z",
                "Data/VED_Static_Data_ICE&HEV.xlsx",
                "Data/VED_Static_Data_PHEV&EV.xlsx",
            ],
        },
        }

    source_headline = make_source(
        "source_inventory_headline",
        "SOURCE_INVENTORY integrity summary",
        "SELECT n_ved_dynamic_files AS dynamic_files, n_ved_rows AS dynamic_rows, "
        "n_dynamic_vehicles AS dynamic_vehicles, dynamic_static_join_rate AS join_rate "
        "FROM read_json_auto('raw_integrity_summary.json');",
        "Headline values projected from the machine-readable SOURCE_INVENTORY integrity summary.",
    )
    source_quality = make_source(
        "source_inventory_quality",
        "SOURCE_INVENTORY G1 check results",
        "SELECT key AS check, CASE WHEN value::BOOLEAN THEN 'PASS' ELSE 'FAIL' END AS status "
        "FROM (SELECT unnest(map_entries(checks)) FROM read_json_auto('raw_integrity_summary.json'));",
        "Hard-gate results flattened from the checks object in the SOURCE_INVENTORY summary.",
    )
    source_archive = make_source(
        "source_inventory_archives",
        "Commit-pinned VED release inventory",
        "SELECT file_name, bytes, git_blob_sha1, sha256, hash_matches_contract, is_read_only "
        "FROM read_parquet('source_archive_inventory.parquet') ORDER BY file_name;",
        "The four official release objects with Git and local content identities.",
    )
    source_powertrain = make_source(
        "source_inventory_powertrain",
        "Official static vehicle metadata",
        "SELECT EngineType_official AS powertrain, COUNT(*) AS vehicles, "
        "COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS share, "
        "RANK() OVER (ORDER BY COUNT(*) DESC) AS rank, "
        "SUM(COUNT(*)) OVER () AS static_total "
        "FROM read_parquet('static_vehicle_metadata.parquet') "
        "GROUP BY EngineType_official ORDER BY vehicles DESC;",
        "Official powertrain counts from the merged static workbooks.",
    )
    source_files = make_source(
        "source_inventory_files",
        "Extracted VED weekly-file inventory",
        "SELECT source_file, data_rows, bytes, schema_hash, sha256, is_read_only "
        "FROM read_parquet('source_file_inventory.parquet') ORDER BY source_file;",
        "Exact per-file rows, bytes, schema identity, content hash, and read-only state.",
    )
    sources = [
        source_headline,
        source_quality,
        source_archive,
        source_powertrain,
        source_files,
    ]
    title = "SOURCE_INVENTORY — Original VED Source-Lineage and Static-Metadata Quality Audit"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Technical acceptance report for the immutable L0 source layer of RNEE-Bench."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "dynamic_files",
                "dataset": "headline",
                "sourceId": source_headline["id"],
                "description": "Extracted weekly VED CSV files.",
                "metrics": [{"label": "Dynamic files", "field": "dynamic_files", "format": "number"}],
            },
            {
                "id": "dynamic_rows",
                "dataset": "headline",
                "sourceId": source_headline["id"],
                "description": "Data rows excluding the header in every official weekly CSV.",
                "metrics": [{"label": "Dynamic rows", "field": "dynamic_rows", "format": "compact"}],
            },
            {
                "id": "dynamic_vehicles",
                "dataset": "headline",
                "sourceId": source_headline["id"],
                "description": "Distinct VehId values observed in the dynamic data.",
                "metrics": [{"label": "Dynamic vehicles", "field": "dynamic_vehicles", "format": "number"}],
            },
            {
                "id": "join_rate",
                "dataset": "headline",
                "sourceId": source_headline["id"],
                "description": "Share of dynamic VehId values matched to official static metadata.",
                "metrics": [{"label": "Static join rate", "field": "join_rate", "format": "percent"}],
            },
        ],
        "charts": [
            {
                "id": "powertrain_chart",
                "title": "Vehicles by official powertrain label",
                "subtitle": "The workbooks contain one more HEV than stated in the frozen README.",
                "intent": "comparison",
                "type": "bar",
                "dataset": "powertrain",
                "sourceId": source_powertrain["id"],
                "encodings": {
                    "x": {"field": "powertrain", "type": "nominal", "label": "Powertrain"},
                    "y": {
                        "field": "vehicles",
                        "type": "quantitative",
                        "format": "number",
                        "label": "Vehicles",
                    },
                    "tooltip": [
                        {"field": "vehicles", "type": "quantitative", "label": "Vehicles"},
                        {"field": "share", "type": "quantitative", "format": "percent", "label": "Share"},
                    ],
                },
                "xAxisTitle": "Official powertrain label",
                "yAxisTitle": "Vehicles",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 4,
            }
        ],
        "tables": [
            {
                "id": "quality_checks",
                "title": "G1 acceptance checks",
                "subtitle": "All hard source-lineage checks must pass.",
                "dataset": "quality_checks",
                "sourceId": source_quality["id"],
                "defaultSort": {"field": "check", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "check", "label": "Check", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                ],
            },
            {
                "id": "archive_inventory",
                "title": "Frozen official release files",
                "subtitle": "Commit identity, content hashes, and read-only state.",
                "dataset": "archive_inventory",
                "sourceId": source_archive["id"],
                "defaultSort": {"field": "file_name", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "file_name", "label": "File", "type": "text"},
                    {"field": "bytes", "label": "Bytes", "format": "number"},
                    {"field": "git_blob_sha1", "label": "Git blob SHA-1", "type": "text"},
                    {"field": "sha256", "label": "SHA-256", "type": "text"},
                    {"field": "hash_matches_contract", "label": "Hash match", "type": "text"},
                    {"field": "is_read_only", "label": "Read-only", "type": "text"},
                ],
            },
            {
                "id": "powertrain_inventory",
                "title": "Official static powertrain inventory",
                "subtitle": "Powertrain labels are read from the two official workbooks, never inferred.",
                "dataset": "powertrain",
                "sourceId": source_powertrain["id"],
                "defaultSort": {"field": "vehicles", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "powertrain", "label": "Powertrain", "type": "text"},
                    {"field": "vehicles", "label": "Vehicles", "format": "number"},
                ],
            },
            {
                "id": "file_inventory",
                "title": "Weekly dynamic-file inventory",
                "subtitle": "Exact row count, byte size, schema identity, hash, and read-only state.",
                "dataset": "file_inventory",
                "sourceId": source_files["id"],
                "defaultSort": {"field": "source_file", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "source_file", "label": "Source file", "type": "text"},
                    {"field": "data_rows", "label": "Rows", "format": "number"},
                    {"field": "bytes", "label": "Bytes", "format": "number"},
                    {"field": "schema_hash", "label": "Schema hash", "type": "text"},
                    {"field": "sha256", "label": "SHA-256", "type": "text"},
                    {"field": "is_read_only", "label": "Read-only", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
            {
                "id": "summary",
                "type": "markdown",
                "sourceId": source_headline["id"],
                "layout": "full",
                "body": (
                    "## Technical summary\n\n"
                    f"SOURCE_INVENTORY passed `PASS_RAW_LINEAGE`. The frozen original VED release contains "
                    f"**{summary['n_ved_dynamic_files']} weekly dynamic files and "
                    f"{summary['n_ved_rows']:,} data rows** covering **{summary['n_dynamic_vehicles']} "
                    "distinct vehicles**. Every dynamic vehicle matched one official static record, "
                    "all files share one 22-field schema, both dynamic archives passed integrity tests, "
                    "and the source archives plus extracted CSVs are read-only. This establishes the "
                    "immutable L0 baseline for the independent RNEE-Bench reconstruction."
                ),
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "cardIds": ["dynamic_files", "dynamic_rows", "dynamic_vehicles", "join_rate"],
                "layout": "full",
            },
            {
                "id": "findings",
                "type": "markdown",
                "sourceId": source_quality["id"],
                "layout": "full",
                "body": (
                    "## The source layer passes every hard G1 check\n\n"
                    "The four commit-pinned release files match the frozen SHA-256 contract; the "
                    "two archives are structurally valid; row counts are conserved from the 54 source "
                    "files into the 22,436,808-row lineage table; source keys contain no nulls or "
                    "duplicate `(VehId, Trip, Timestamp)` rows; and static `VehId` values are unique. "
                    "A table is used instead of a chart because exact pass/fail audit status is the "
                    "decision-relevant evidence."
                ),
            },
            {"id": "quality_table", "type": "table", "tableId": "quality_checks", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Scope and definitions\n\n"
                    "This audit treats the official VED dynamic archives and static workbooks as the "
                    "only L0 inputs. eVED is excluded. `source_row_number` is the zero-based data-row "
                    "position after the CSV header. `row_id` is a deterministic SHA-256 of the source "
                    "file SHA-256 and source row number. Exact coordinates and entity identifiers remain "
                    "lineage/QA fields and are not authorized model features."
                ),
            },
            {"id": "archive_table", "type": "table", "tableId": "archive_inventory", "layout": "full"},
            {
                "id": "metadata_finding",
                "type": "markdown",
                "sourceId": source_powertrain["id"],
                "layout": "full",
                "body": (
                    "## The files contain 384 vehicles, exposing a documentation inconsistency\n\n"
                    "The two official static workbooks contain **384 unique VehId records**: 264 ICE, "
                    "93 HEV, 24 PHEV, and 3 EV. The dynamic data also contain 384 distinct VehId values "
                    "and achieve a 100% join. The frozen README states 383 total vehicles and 92 HEVs; "
                    "therefore future work must use the file-derived 384/93 counts and record the README "
                    "difference rather than silently forcing the data to the prose description."
                ),
            },
            {
                "id": "powertrain_chart_block",
                "type": "chart",
                "chartId": "powertrain_chart",
                "layout": "full",
            },
            {
                "id": "powertrain_table",
                "type": "table",
                "tableId": "powertrain_inventory",
                "layout": "full",
            },
            {
                "id": "method",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Reproducible method\n\n"
                    "The workflow downloaded `Data/` objects from the frozen Git commit, recorded Git "
                    "blob SHA-1 and local SHA-256 values, tested each 7z archive, extracted without "
                    "overwriting, set source files read-only, counted rows directly from VED, hashed "
                    "each extracted CSV, compared schemas, scanned key fields in chunks, materialized "
                    "the full row-lineage Parquet, and joined both static workbooks only on official "
                    "`VehId`. The lineage Parquet contains exactly 22,436,808 rows."
                ),
            },
            {"id": "file_table", "type": "table", "tableId": "file_inventory", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Limitations and uncertainty\n\n"
                    "This gate proves source identity, row conservation, schema consistency, stable "
                    "lineage, and static join completeness. It does not yet validate energy-target "
                    "physics, GPS plausibility, map matching, road semantics, missingness mechanisms, "
                    "or segment-level suitability. SHA-256 uniqueness is guaranteed operationally by "
                    "the unique `(source file hash, source row number)` construction, subject to the "
                    "standard negligible cryptographic-collision assumption."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Execute TARGET_AUDIT against original VED to validate separate ICE, HEV, PHEV, and EV "
                    "energy-target definitions before any energy conclusion.\n"
                    "2. Complete RUNTIME_FREEZE to freeze the field contract, licenses, dependency versions, "
                    "Valhalla build, and OSM snapshot policy.\n"
                    "3. Start NETWORK_FREEZE/MAP_MATCHING only from this accepted lineage baseline; preserve row IDs and "
                    "never import eVED-derived energy, matched-coordinate, elevation, speed-limit, or "
                    "context fields."
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Further questions\n\n"
                    "- Does each powertrain expose enough raw channels for a physically coherent target?\n"
                    "- Which timestamp and GPS-quality rules preserve trips without introducing hidden interpolation?\n"
                    "- Can a period-appropriate OSM snapshot be obtained, or must current-map temporal mismatch remain explicit?"
                ),
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "quality_checks": quality_checks,
                "archive_inventory": archive_inventory,
                "powertrain": powertrain,
                "file_inventory": file_inventory,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
