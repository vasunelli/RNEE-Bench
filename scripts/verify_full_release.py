#!/usr/bin/env python3
"""Verify the complete RNEE public release without importing model objects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="check structure and sizes without hashing 5.1 GB")
    args = parser.parse_args()

    summary = json.loads((ROOT / "metadata/full_release_summary.json").read_text(encoding="utf-8"))
    with (ROOT / "metadata/full_release_inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["artifact_count"]:
        fail(f"inventory count {len(rows)} != summary {summary['artifact_count']}")

    total = 0
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            fail(f"missing artifact: {row['path']}")
        actual_bytes = path.stat().st_size
        expected_bytes = int(row["bytes"])
        if actual_bytes != expected_bytes:
            fail(f"size mismatch: {row['path']} ({actual_bytes} != {expected_bytes})")
        if not args.quick and sha256(path) != row["sha256"]:
            fail(f"SHA-256 mismatch: {row['path']}")
        total += actual_bytes
    if total != summary["total_bytes"]:
        fail(f"total bytes {total} != summary {summary['total_bytes']}")

    expected_counts = {
        "data/map_matching/matched_points": 54,
        "data/map_matching/matched_edges": 54,
        "data/row_enriched": 54,
    }
    for relative, expected in expected_counts.items():
        actual = len(list((ROOT / relative).glob("*.parquet")))
        if actual != expected:
            fail(f"{relative}: {actual} files != {expected}")
    if len(list((ROOT / "models/theory_validation").rglob("*.joblib"))) != 192:
        fail("THEORY_VALIDATION model count is not 192")

    schemas = json.loads((ROOT / "metadata/parquet_schemas.json").read_text(encoding="utf-8"))["datasets"]
    expected_rows = {
        "map_matched_points": 22436808,
        "row_enriched_trajectories": 22436808,
        "segments_all": 301219,
        "segments_qa_valid": 256419,
        "segments_prediction_usable": 219384,
    }
    for name, expected in expected_rows.items():
        if schemas[name]["total_rows"] != expected:
            fail(f"schema summary {name}: {schemas[name]['total_rows']} != {expected}")

    text_suffixes = {".md", ".txt", ".csv", ".json", ".py", ".yaml", ".yml"}
    hidden_run_id_pattern = re.compile(r"(?i)(?<![A-Za-z0-9])ex[0-9]{3}(?![A-Za-z0-9])")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if hidden_run_id_pattern.search(relative):
            fail(f"technical experiment identifier in path: {relative}")
    secret_patterns = [
        re.compile(r"github_pat_[A-Za-z0-9_]+"),
        re.compile(r"ghp_[A-Za-z0-9]+"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"(?:/mnt/c/Users/|[A-Za-z]:\\Users\\)[^/\\\s]+", re.I),
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if hidden_run_id_pattern.search(text):
            fail(f"technical experiment identifier in {path.relative_to(ROOT)}")
        for pattern in secret_patterns:
            if pattern.search(text):
                fail(f"private path/credential pattern in {path.relative_to(ROOT)}")

    if "did not serialize fitted estimator objects" not in (ROOT / "docs/model_card.md").read_text(encoding="utf-8"):
        fail("RPM-only missing-model boundary is not documented")

    mode = "quick structure/size" if args.quick else "full SHA-256"
    print(f"PASS: {mode} verification; {len(rows)} artifacts; {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
