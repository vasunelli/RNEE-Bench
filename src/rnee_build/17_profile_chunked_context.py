#!/usr/bin/env python3
"""Run the chunked context builder under low-overhead Windows memory sampling."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def process_tree_pids(root_pid: int) -> set[int]:
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return {root_pid}
    parents: dict[int, int] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def process_tree_memory(root_pid: int) -> dict[str, int] | None:
    records = [process_memory(pid) for pid in process_tree_pids(root_pid)]
    records = [record for record in records if record is not None]
    if not records:
        return None
    return {
        "process_count": len(records),
        "working_set_bytes": sum(record["working_set_bytes"] for record in records),
        "peak_working_set_bytes": sum(record["working_set_bytes"] for record in records),
        "private_bytes": sum(record["private_bytes"] for record in records),
        "peak_pagefile_bytes": sum(record["PagefileUsage"] if "PagefileUsage" in record else record["private_bytes"] for record in records),
    }


def process_memory(pid: int) -> dict[str, int] | None:
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return {
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage),
            "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
        }
    finally:
        kernel32.CloseHandle(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/trajectory_build_context_chunk_dry_run.yaml"),
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overlay = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(overlay["base_config"]).read_text(encoding="utf-8"))
    config = base | {key: value for key, value in overlay.items() if key != "base_config"}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_root"]) / f"{config['run_name_prefix']}_{stamp}"
    stdout_path = Path(config["output_root"]) / f".{config['run_name_prefix']}_{stamp}.stdout.log"
    interpreter = sys.executable
    command = [
        interpreter,
        str(Path(__file__).resolve().parent / "16_build_context_semantics_chunked.py"),
        "--config",
        str(args.config.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
    ]
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    controlled_post_output_reap = False
    completion_seen_at = None
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2])
        while process.poll() is None:
            memory = process_tree_memory(process.pid)
            if memory:
                samples.append({"elapsed_seconds": time.perf_counter() - started, **memory})
            if (run_dir / "chunked_context_summary.json").exists():
                completion_seen_at = completion_seen_at or time.perf_counter()
                if time.perf_counter() - completion_seen_at >= 1.0:
                    process.terminate()
                    process.wait(timeout=10)
                    controlled_post_output_reap = True
                    break
            time.sleep(max(0.05, args.sample_interval_seconds))
        return_code = int(process.returncode)
    elapsed = time.perf_counter() - started

    profile_dir = run_dir / "profile_output"
    profile_dir.mkdir(parents=True, exist_ok=True)
    stdout_final = profile_dir / "chunked_context_stdout.log"
    stdout_path.replace(stdout_final)
    sample_path = profile_dir / "memory_samples.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "process_count", "working_set_bytes", "peak_working_set_bytes", "private_bytes", "peak_pagefile_bytes"])
        writer.writeheader()
        writer.writerows(samples)
    peak_working_set = max((sample["peak_working_set_bytes"] for sample in samples), default=0)
    peak_private = max((sample["private_bytes"] for sample in samples), default=0)
    ceiling = float(config["memory_ceiling_gb"])
    memory_gate = "PASS" if peak_working_set / (1024**3) <= ceiling else "FAIL"
    child_summary_path = run_dir / "chunked_context_summary.json"
    child_status = None
    if child_summary_path.exists():
        child_status = json.loads(child_summary_path.read_text(encoding="utf-8"))["status"]
    status = "PASS_TRAJECTORY_BUILD_CHUNKED_CONTEXT_PROFILE" if (return_code == 0 or controlled_post_output_reap) and memory_gate == "PASS" and str(child_status).startswith("PASS_") else "FAIL_TRAJECTORY_BUILD_CHUNKED_CONTEXT_PROFILE"
    summary = {
        "experiment": config["experiment"],
        "stage": "chunked_context_resource_profile",
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "controlled_post_output_reap": controlled_post_output_reap,
        "child_status": child_status,
        "elapsed_seconds": elapsed,
        "sample_interval_seconds": args.sample_interval_seconds,
        "sample_count": len(samples),
        "peak_working_set_gb": peak_working_set / (1024**3),
        "peak_private_gb": peak_private / (1024**3),
        "memory_ceiling_gb": ceiling,
        "memory_gate": memory_gate,
        "command": command,
        "run_directory": str(run_dir),
        "profiled_pid": process.pid,
        "interpreter": interpreter,
        "memory_samples": str(sample_path),
        "stdout_log": str(stdout_final),
    }
    summary_path = profile_dir / "profile_summary.json"
    report_path = profile_dir / "PROFILE_REPORT.md"
    write_json(summary_path, summary)
    report_path.write_text(
        "# TRAJECTORY_BUILD chunked context resource profile\n\n"
        f"- Status: `{status}`\n"
        f"- Child status: `{child_status}`\n"
        f"- Wall time: {elapsed:.1f} s\n"
        f"- Peak working set: {summary['peak_working_set_gb']:.2f} GB\n"
        f"- Peak private bytes: {summary['peak_private_gb']:.2f} GB\n"
        f"- Memory ceiling: {ceiling:.2f} GB (`{memory_gate}`)\n"
        f"- Samples: {len(samples)} at {args.sample_interval_seconds:.2f} s intervals\n\n"
        "## Instrumentation changelog\n\n"
        "| File | Change type | What was added/modified |\n"
        "|---|---|---|\n"
        "| `src/rnee_build/17_profile_chunked_context.py` | created | External process wrapper with Windows working-set/private-byte sampling |\n"
        "| `profile_output/memory_samples.csv` | generated | Time-series process memory samples |\n"
        "| `profile_output/profile_summary.json` | generated | Structured resource gate |\n",
        encoding="utf-8",
    )
    latest = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_profile_pointer.json"
    write_json(latest, {
        "status": status,
        "summary": str(summary_path),
        "report": str(report_path),
        "run_directory": str(run_dir),
        "memory_samples": str(sample_path),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
