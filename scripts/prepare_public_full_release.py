#!/usr/bin/env python3
"""Assemble the frozen full RNEE public release from authoritative artifacts.

This maintainer utility never modifies source artifacts. Large immutable files may
be hard-linked into a local release worktree to avoid duplicating several GB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


def resolve_unique(project: Path, relative_parent: str, pattern: str) -> Path:
    """Resolve one frozen source directory by semantic suffix and timestamp."""
    matches = sorted((project / relative_parent).glob(pattern))
    if len(matches) != 1 or not matches[0].is_dir():
        raise FileNotFoundError(
            f"expected one frozen source under {relative_parent!r} matching {pattern!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def resolve_latest_file(project: Path, pattern: str) -> Path:
    """Resolve the newest source file using a semantic pattern, not a run directory name."""
    matches = sorted(path for path in (project / "results").glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"no source file under results matching {pattern!r}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def resolve_inputs(project: Path) -> dict[str, Path]:
    """Resolve authoritative inputs without exposing technical run identifiers."""
    return {
        "trajectory_build_map": resolve_unique(
            project, "results", "**/*map_match_production_20260717_180329"
        ),
        "trajectory_build_rows": resolve_unique(
            project, "results", "**/*row_production_20260719_053746"
        ),
        "segment_split_segments": resolve_unique(
            project, "results", "**/*segments_60s_20260719_062221"
        ),
        "segment_split_splits": resolve_unique(
            project, "results", "**/*splits_20260719_141853"
        ),
        "theory_validation": resolve_unique(
            project, "results", "**/*theory_validation_20260725_161929"
        ),
        "phase1": resolve_unique(
            project, "results", "**/*20260801_195916"
        ),
    }


def resolve_marker_dir(project: Path, relative_parent: str, marker: str) -> Path:
    """Find one source directory containing a distinctive semantic marker file."""
    candidates = sorted(
        path
        for path in (project / relative_parent).iterdir()
        if path.is_dir() and (path / marker).is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one source directory under {relative_parent!r} containing {marker!r}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size == destination.stat().st_size:
            return
        raise FileExistsError(f"refusing to overwrite different file: {destination}")
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, mode: str, excluded_names: set[str] | None = None) -> None:
    excluded_names = excluded_names or set()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in excluded_names for part in path.relative_to(source).parts):
            continue
        place_file(path, destination / path.relative_to(source), mode)


def partition_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("partition_"):
            return part.split("_")[1]
    raise ValueError(f"partition id not found: {path}")


def component_for(relative: Path) -> str:
    text = relative.as_posix()
    if text.startswith("data/map_matching/"):
        return "TRAJECTORY_BUILD_map_matching"
    if text.startswith("data/row_enriched/"):
        return "TRAJECTORY_BUILD_row_enriched"
    if text.startswith("data/segments/") or text.startswith("data/splits/"):
        return "SEGMENT_SPLIT_segments_and_splits"
    if text.startswith("models/theory_validation/") or text.startswith("predictions/theory_validation/"):
        return "THEORY_VALIDATION_theory_validation"
    if text.startswith("predictions/phase1/"):
        return "Phase1_target_diagnostics"
    return "workflow_or_documentation"


def license_for(relative: Path) -> str:
    text = relative.as_posix()
    if text.startswith("data/") or text.startswith("predictions/") or text.startswith("results/"):
        return "ODbL-1.0 + upstream notices"
    if text.startswith("models/") or text.startswith("src/") or text.startswith("scripts/") or text.startswith("tests/") or text.startswith("configs/"):
        return "MIT"
    return "CC-BY-4.0"


def build_manifest(repo_root: Path) -> None:
    tracked_roots = ["data", "models", "predictions", "results", "src", "scripts", "configs", "tests"]
    rows: list[dict[str, object]] = []
    for root_name in tracked_roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "component": component_for(relative),
                    "license": license_for(relative),
                }
            )
    metadata = repo_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    csv_path = metadata / "full_release_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "bytes", "sha256", "component", "license"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "components": {},
    }
    for row in rows:
        component = str(row["component"])
        item = summary["components"].setdefault(component, {"artifact_count": 0, "total_bytes": 0})
        item["artifact_count"] += 1
        item["total_bytes"] += int(row["bytes"])
    (metadata / "full_release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    repo = args.repo_root.resolve()
    if args.manifest_only:
        build_manifest(repo)
        return 0
    inputs = resolve_inputs(project)
    trajectory_build_map = inputs["trajectory_build_map"]
    trajectory_build_rows = inputs["trajectory_build_rows"]
    segment_split_segments = inputs["segment_split_segments"]
    segment_split_splits = inputs["segment_split_splits"]
    theory_validation = inputs["theory_validation"]
    phase1 = inputs["phase1"]
    theory_validation_scripts = resolve_marker_dir(project, "scripts", "bootstrap.py")
    robustness_scripts = resolve_marker_dir(project, "scripts", "inference.py")
    theory_validation_tests = resolve_marker_dir(project, "tests", "test_noise_contract.py")
    theory_validation_config = resolve_unique(
        project, "configs/rnee", "*/*theory_validation.yaml"
    )
    phase1_config = resolve_unique(project, "configs/rnee/phase1", "*.yaml")

    # TRAJECTORY_BUILD public data products; omit raw Valhalla responses and runtime caches.
    for name, destination_dir in (
        ("matched_points.parquet", repo / "data/map_matching/matched_points"),
        ("matched_edges.parquet", repo / "data/map_matching/matched_edges"),
    ):
        paths = sorted(trajectory_build_map.rglob(name))
        if len(paths) != 54:
            raise RuntimeError(f"expected 54 {name} files, found {len(paths)}")
        for source in paths:
            place_file(source, destination_dir / f"partition_{partition_id(source)}.parquet", args.mode)
    for name in ("partition_manifest.json", "partitioned_map_match_summary.json", "TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_PRODUCTION.md"):
        place_file(trajectory_build_map / name, repo / "data/map_matching" / name, "copy")

    row_paths = sorted(trajectory_build_rows.rglob("enriched_trajectory_rows.parquet"))
    if len(row_paths) != 54:
        raise RuntimeError(f"expected 54 enriched row files, found {len(row_paths)}")
    for source in row_paths:
        place_file(source, repo / "data/row_enriched" / f"partition_{partition_id(source)}.parquet", args.mode)
    for name in ("partition_manifest.json", "partitioned_row_summary.json", "TRAJECTORY_BUILD_PARTITIONED_ROW_PRODUCTION.md"):
        place_file(trajectory_build_rows / name, repo / "data/row_enriched" / name, "copy")

    # SEGMENT_SPLIT contains canonical global segment products, partition products,
    # row-to-segment assignments, split membership, cards, and leakage checks.
    copy_tree(segment_split_segments, repo / "data/segments", args.mode, {"profile_output", "logs"})
    copy_tree(segment_split_splits, repo / "data/splits", args.mode, {"profile_output", "logs"})

    # THEORY_VALIDATION frozen models and predictions, including negative controls.
    copy_tree(theory_validation / "models", repo / "models/theory_validation", args.mode)
    for name in ("paired_test_predictions", "ensemble_test_predictions", "null_controls"):
        copy_tree(theory_validation / name, repo / "predictions/theory_validation" / name, args.mode)
    for name in (
        "THEORY_VALIDATION_REPORT.md", "active_feature_masks.json",
        "bootstrap_diagnostics.json", "bootstrap_results.csv",
        "contract_checks.csv", "effects.csv", "theory_validation_summary.json", "feature_manifest.json",
        "final_decision.json", "release_summary.json",
    ):
        place_file(theory_validation / name, repo / "predictions/theory_validation/metadata" / name, args.mode)

    # Current RPM-only analysis branch. It did not serialize fitted
    # model objects; publish all preserved predictions and fit records as-is.
    copy_tree(phase1 / "rpm_only_test_predictions", repo / "predictions/phase1/rpm_only_test_predictions", args.mode)
    for name in (
        "rpm_only_bootstrap_draws.parquet", "rpm_only_effects.csv", "rpm_only_fit_manifest.json",
        "rpm_only_model_comparison.csv", "rpm_only_null_distribution_summary.csv",
        "rpm_only_original_comparison.csv", "rpm_only_active_feature_masks.json",
        "rpm_only_bootstrap_diagnostics.json", "PHASE1_RPM_ONLY_RERUN_COMPARISON.md",
        "PHASE1_RPM_ONLY_RERUN_RESULTS.csv", "PHASE1_DECISION.json",
        "PHASE1_TARGET_DIAGNOSTICS_REPORT.md", "PUBLIC_SCOPE_SUMMARY.csv",
        "REPRODUCIBILITY_SCOPE_NOTE.md", "run_config.yaml", "runtime_versions.json",
    ):
        place_file(phase1 / name, repo / "predictions/phase1" / name, args.mode)

    # End-to-end construction, model, and rerun sources/configuration/tests.
    copy_tree(project / "src/rnee_build", repo / "src/rnee_build", "copy", {"__pycache__"})
    copy_tree(theory_validation_scripts, repo / "scripts/rnee_theory_validation", "copy", {"__pycache__"})
    copy_tree(robustness_scripts, repo / "scripts/rnee_robustness", "copy", {"__pycache__"})
    copy_tree(project / "scripts/rnee_phase1", repo / "scripts/rnee_phase1", "copy", {"__pycache__"})
    copy_tree(project / "configs/rnee_build", repo / "configs/rnee_build", "copy")
    place_file(theory_validation_config, repo / "configs/rnee/theory_validation/theory_validation.yaml", "copy")
    place_file(phase1_config, repo / "configs/rnee/phase1/phase1_target_diagnostics.yaml", "copy")
    copy_tree(project / "tests/rnee_build", repo / "tests/rnee_build", "copy", {"__pycache__"})
    copy_tree(theory_validation_tests, repo / "tests/rnee_theory_validation", "copy", {"__pycache__"})
    place_file(
        resolve_latest_file(project, "**/network_freeze_latest_gps_trip_quality.parquet"),
        repo / "results/network_freeze_latest_gps_trip_quality.parquet",
        args.mode,
    )
    place_file(
        resolve_latest_file(project, "**/feature_information_contract.json"),
        repo / "results/information_contract/feature_information_contract.json",
        "copy",
    )
    place_file(
        resolve_latest_file(project, "**/feature_contract.json"),
        repo / "results/model_contract/feature_contract.json",
        "copy",
    )

    if not args.skip_manifest:
        build_manifest(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
