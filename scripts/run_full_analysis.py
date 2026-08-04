#!/usr/bin/env python3
"""Preflight the checked-in full RNEE reconstruction workflow and raw inputs."""

from __future__ import annotations

import argparse
from pathlib import Path


COMMIT = "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
OSM_FILES = ["michigan-2017-01-01.osm.pbf", "michigan-2018-01-01.osm.pbf"]
REQUIRED_WORKFLOW = [
    "src/rnee_build/00_preflight.py",
    "src/rnee_build/20_run_map_match_partitioned.py",
    "src/rnee_build/24_run_row_assembly_partitioned.py",
    "src/rnee_build/35_build_segments.py",
    "src/rnee_build/36_build_splits.py",
    "src/rnee_build/37_validate_leakage.py",
    "src/rnee_build/45_freeze_model_contract.py",
    "src/rnee_build/49_correct_model_contract_m4_inference.py",
    "scripts/rnee_theory_validation/run_theory_validation.py",
    "scripts/rnee_phase1/run_target_diagnostics.py",
    "configs/rnee_build/requirements-rnee.lock.txt",
    "configs/rnee/theory_validation/theory_validation.yaml",
    "configs/rnee/phase1/phase1_target_diagnostics.yaml",
]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", required=True, type=Path)
    value.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    value.add_argument("--check-only", action="store_true", help="validate without construction or fitting")
    value.add_argument("--list-stages", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.data_root.expanduser().resolve()
    repo = args.repo_root.expanduser().resolve()
    requirements = [
        root / "ved/raw/dynamic",
        root / "ved/raw/static",
        root / "osm" / OSM_FILES[0],
        root / "osm" / OSM_FILES[1],
        root / "ved/SOURCE_COMMIT.txt",
    ]
    missing = [str(path) for path in requirements if not path.exists()]
    missing_workflow = [item for item in REQUIRED_WORKFLOW if not (repo / item).is_file()]
    if missing or missing_workflow:
        print("Preflight failed.")
        for item in missing:
            print(f"- missing input: {item}")
        for item in missing_workflow:
            print(f"- missing workflow file: {item}")
        return 2
    marker = (root / "ved/SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if marker != COMMIT:
        print(f"Preflight failed: SOURCE_COMMIT.txt must contain {COMMIT}; observed {marker!r}")
        return 2
    for directory in [root / "ved/raw/dynamic", root / "ved/raw/static"]:
        if not any(path.is_file() for path in directory.rglob("*")):
            print(f"Preflight failed: no source files found under {directory}")
            return 2
    print("Input data contract: PASS")
    print("Checked-in workflow contract: PASS")
    if args.list_stages:
        for item in REQUIRED_WORKFLOW:
            print(item)
    if args.check_only:
        print("Check-only mode: no construction, fitting, or analysis was executed.")
        return 0
    print("Automatic scientific execution is intentionally disabled.")
    print("Follow docs/full_reproduction.md stage by stage and preserve every frozen gate.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
