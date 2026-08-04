#!/usr/bin/env python3
"""Strict preflight for a full reconstruction using an external complete workflow."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COMMIT = "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
OSM_FILES = ["michigan-2017-01-01.osm.pbf", "michigan-2018-01-01.osm.pbf"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--workflow-root", type=Path, help="Path to the complete construction and modeling workflow")
    p.add_argument("--check-only", action="store_true", help="Validate inputs without executing analysis")
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.data_root.expanduser().resolve()
    requirements = [root / "ved/raw/dynamic", root / "ved/raw/static",
                    root / "osm" / OSM_FILES[0], root / "osm" / OSM_FILES[1],
                    root / "ved/SOURCE_COMMIT.txt"]
    missing = [str(p) for p in requirements if not p.exists()]
    if missing:
        print("Preflight failed; missing required inputs:")
        for item in missing: print(f"- {item}")
        return 2
    marker = (root / "ved/SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if marker != COMMIT:
        print(f"Preflight failed: SOURCE_COMMIT.txt must contain {COMMIT}; observed {marker!r}")
        return 2
    for directory in [root / "ved/raw/dynamic", root / "ved/raw/static"]:
        if not any(p.is_file() for p in directory.rglob("*")):
            print(f"Preflight failed: no source files found under {directory}")
            return 2
    if args.workflow_root is None:
        print("Input data contract: PASS")
        print("Full reconstruction requires --workflow-root pointing to the complete twelve-stage workflow described in docs/full_reproduction.md.")
        return 0 if args.check_only else 3
    workflow = args.workflow_root.expanduser().resolve()
    manifests = [workflow / "WORKFLOW.md", workflow / "workflow_manifest.json"]
    if not workflow.is_dir() or not any(p.is_file() for p in manifests):
        print("Preflight failed: workflow root must contain WORKFLOW.md or workflow_manifest.json.")
        return 2
    print("Input data contract: PASS")
    print("External workflow contract: PASS")
    if args.check_only:
        print("Check-only mode: no construction, fitting, or analysis was executed.")
        return 0
    print("Execution is intentionally not delegated to an unverified external command. Follow the stage commands in the supplied workflow manifest, then run scripts/verify_reported_results.py.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
