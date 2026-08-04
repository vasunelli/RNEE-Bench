#!/usr/bin/env python3
"""Consolidate TRAJECTORY_BUILD partition dry-run evidence into the full-build launch gate."""
from __future__ import annotations

import argparse, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(r"configs/rnee_build/trajectory_build_partition_readiness_gate.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pointers = {name: load(Path(cfg[f"{name}_pointer"])) for name in ("context", "map", "map_equivalence", "edge", "edge_equivalence", "row", "row_equivalence")}
    summaries = {name: load(Path(pointer["summary"])) for name, pointer in pointers.items()}
    end_to_end = load(Path(cfg["end_to_end_summary"]))
    test = subprocess.run([str(Path.cwd()/r".venv-rnee311/Scripts/python.exe"), "-m", "pytest", "tests/rnee_build", "-q"], capture_output=True, text=True)
    free_disk_gb = shutil.disk_usage(Path(cfg["output_root"]).anchor).free / 1024**3
    stage_names = ("map", "edge", "context", "row")
    peaks = {name: float(summaries[name]["peak_working_set_gb"]) for name in stage_names}
    resume = {name: bool(summaries[name].get("resume_noop_verified") or summaries[name].get("resume_noop_elapsed_seconds")) for name in stage_names}
    expected_status = {
        "context": "PASS_TRAJECTORY_BUILD_CONTEXT_GATE", "map": "PASS_TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_DRY_RUN",
        "map_equivalence": "PASS_TRAJECTORY_BUILD_PARTITIONED_MAP_MATCH_EQUIVALENCE", "edge": "PASS_TRAJECTORY_BUILD_PARTITIONED_EDGE_DRY_RUN",
        "edge_equivalence": "PASS_TRAJECTORY_BUILD_PARTITIONED_EDGE_EQUIVALENCE", "row": "PASS_TRAJECTORY_BUILD_PARTITIONED_ROW_DRY_RUN",
        "row_equivalence": "PASS_TRAJECTORY_BUILD_PARTITIONED_ROW_EQUIVALENCE",
    }
    checks = {
        "all_partition_stages_pass": "PASS" if all(str(pointers[n]["status"]).startswith(expected_status[n]) for n in expected_status) else "FAIL",
        "all_exact_equivalence_gates_pass": "PASS" if all(str(pointers[n]["status"]).startswith("PASS_") for n in ("map_equivalence", "edge_equivalence", "row_equivalence")) and summaries["context"]["context_checks"]["exact_equivalence"] == "PASS" else "FAIL",
        "all_resume_noop_verified": "PASS" if all(resume.values()) else "FAIL",
        "all_peak_working_sets_within_ceiling": "PASS" if max(peaks.values()) <= float(cfg["memory_ceiling_gb"]) else "FAIL",
        "free_disk_preflight": "PASS" if free_disk_gb >= float(cfg["minimum_free_disk_gb"]) else "FAIL",
        "rnee_test_suite": "PASS" if test.returncode == 0 and f"{cfg['expected_test_count']} passed" in test.stdout else "FAIL",
        "end_to_end_quality_1_gate": "PASS" if end_to_end["checks"]["quality_1_manual_gate"] == "PASS" and end_to_end["quality_1_incorrect_rate"] <= end_to_end["quality_1_incorrect_rate_pass_max"] else "FAIL",
    }
    gate = "PASS" if "FAIL" not in checks.values() else "FAIL"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["output_root"]) / f"{cfg['run_name_prefix']}_{stamp}"; output.mkdir(parents=True)
    shutil.copy2(args.config, output / "run_config.yaml")
    status = f"{gate}_TRAJECTORY_BUILD_PARTITIONED_PIPELINE_DRY_RUN_AUTHORIZE_FULL_BUILD" if gate == "PASS" else "FAIL_TRAJECTORY_BUILD_PARTITIONED_PIPELINE_READINESS"
    summary = {
        "experiment": cfg["experiment"], "stage": "partitioned_pipeline_full_build_readiness_gate", "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "full_build_authorized": gate == "PASS",
        "execution_started": False, "memory_ceiling_gb": cfg["memory_ceiling_gb"], "peak_working_set_gb_by_stage": peaks,
        "maximum_peak_working_set_gb": max(peaks.values()), "resume_noop_by_stage": resume, "free_disk_gb": free_disk_gb,
        "quality_1_incorrect_rate": end_to_end["quality_1_incorrect_rate"], "quality_1_threshold": end_to_end["quality_1_incorrect_rate_pass_max"],
        "test_command": ".venv-rnee311\\Scripts\\python.exe -m pytest tests\\rnee_build -q", "test_output": test.stdout.strip(),
        "known_runtime_note": "PyOsmium child processes may require controlled post-output reaping on Windows; hashes, manifests, exact equivalence, and clean resume gates must remain mandatory.",
        "launch_boundary": "Authorization covers the partitioned production build only. Model experiments remain blocked until TRAJECTORY_BUILD and SEGMENT_SPLIT quality gates pass.",
        "evidence": {name: pointer["summary"] for name, pointer in pointers.items()}, "output_directory": str(output),
    }
    dump(output / "trajectory_build_partition_readiness_summary.json", summary)
    report = output / "TRAJECTORY_BUILD_PARTITION_READINESS_GATE.md"
    report.write_text("# TRAJECTORY_BUILD partitioned pipeline readiness gate\n\n" + "\n".join([
        f"- Status: `{status}`", f"- Full build authorized: `{summary['full_build_authorized']}`; execution started: `false`.",
        f"- Peak working set (map / edge / context / row): {peaks['map']:.3f} / {peaks['edge']:.3f} / {peaks['context']:.3f} / {peaks['row']:.3f} GB.",
        f"- Free disk: {free_disk_gb:.1f} GB.", f"- Test suite: `{test.stdout.strip()}`.",
        f"- Quality check 1 incorrect rate: {end_to_end['quality_1_incorrect_rate']:.1%} (threshold {end_to_end['quality_1_incorrect_rate_pass_max']:.1%}).",
        "- Windows/PyOsmium controlled post-output reaping remains documented and must stay enabled.",
        "- Next action: freeze the all-file production partition manifest, run immutable-input/disk preflight, then launch TRAJECTORY_BUILD map matching with checkpoint monitoring.",
    ]) + "\n", encoding="utf-8")
    pointer = {"status": status, "full_build_authorized": gate == "PASS", "execution_started": False, "summary": str(output/"trajectory_build_partition_readiness_summary.json"), "report": str(report), "output_directory": str(output)}
    dump(Path(cfg["output_root"]) / f"{cfg['latest_prefix']}_latest_pointer.json", pointer)
    print(json.dumps(pointer, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
