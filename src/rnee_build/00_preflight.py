#!/usr/bin/env python3
"""Freeze and verify the RUNTIME_FREEZE RNEE-Bench contract and runtime environment."""

from __future__ import annotations

import argparse
import ctypes
import email
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FINGERPRINT_SCHEMA = "rnee_environment_fingerprint_v1"
EXPECTED_GATE = "PASS_CONTRACT_AND_ENVIRONMENT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(data: Any) -> str:
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_read_only(path: Path) -> bool:
    file_stat = path.stat()
    attributes = getattr(file_stat, "st_file_attributes", 0)
    if attributes:
        return bool(attributes & stat.FILE_ATTRIBUTE_READONLY)
    return not bool(file_stat.st_mode & stat.S_IWUSR)


def total_ram_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def parse_pinned_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Requirement is not exactly pinned: {line}")
        requirements[match.group(1)] = match.group(2)
    return requirements


def installed_versions(requirements: dict[str, str]) -> dict[str, dict[str, Any]]:
    versions: dict[str, dict[str, Any]] = {}
    for name, expected in requirements.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return versions


def resolved_lock_status(
    requirements: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], bool]:
    results = installed_versions(requirements)
    return results, bool(results) and all(item["matches"] for item in results.values())


def target_gate_status(
    summary_path: Path, contract_path: Path, accepted_gates: set[str]
) -> dict[str, Any]:
    summary_exists = summary_path.is_file()
    contract_exists = contract_path.is_file()
    gate = None
    if summary_exists:
        summary = load_json(summary_path)
        gate = summary.get("status") or summary.get("overall_gate")
    return {
        "summary_exists": summary_exists,
        "contract_exists": contract_exists,
        "gate": gate,
        "accepted": summary_exists and contract_exists and gate in accepted_gates,
        "contract_sha256": sha256_file(contract_path) if contract_exists else None,
    }


def valhalla_version(config: dict[str, Any]) -> dict[str, Any]:
    executable = Path(config["valhalla"]["build_tiles_executable"])
    dll_directory = Path(config["valhalla"]["dll_directory"])
    wheel_path = Path(config["valhalla"]["wheel_path"])
    wheel_metadata_version = None
    wheel_metadata_name = None
    wheel_tags: list[str] = []
    if wheel_path.is_file():
        with zipfile.ZipFile(wheel_path) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            wheel_info_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
            )
            metadata = email.message_from_bytes(archive.read(metadata_name))
            wheel_info = email.message_from_bytes(archive.read(wheel_info_name))
            wheel_metadata_version = metadata.get("Version")
            wheel_metadata_name = metadata.get("Name")
            wheel_tags = wheel_info.get_all("Tag", [])

    result: dict[str, Any] = {
        "executable": str(executable),
        "exists": executable.is_file(),
        "expected": str(config["valhalla"]["version"]),
        "actual": None,
        "matches": False,
        "returncode": None,
        "wheel_metadata_name": wheel_metadata_name,
        "wheel_metadata_version": wheel_metadata_version,
        "wheel_tags": wheel_tags,
        "wheel_metadata_matches": (
            wheel_metadata_name == "pyvalhalla"
            and wheel_metadata_version == str(config["valhalla"]["version"])
        ),
    }
    if not executable.is_file() or not dll_directory.is_dir():
        return result
    env = os.environ.copy()
    env["PATH"] = str(dll_directory) + os.pathsep + env.get("PATH", "")
    process = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    output = (process.stdout + "\n" + process.stderr).strip()
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    actual = match.group(1) if match else None
    tag_process = subprocess.run(
        [
            "git",
            "ls-remote",
            "--tags",
            config["valhalla"]["repository"],
            config["valhalla"]["source_tag_ref"],
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    tag_line = tag_process.stdout.strip().splitlines()
    tag_commit = tag_line[0].split()[0] if tag_line else None
    tag_commit_matches = (
        tag_process.returncode == 0
        and tag_commit == config["valhalla"]["commit"]
    )
    result.update(
        {
            "actual": actual,
            "matches": process.returncode == 0
            and actual == str(config["valhalla"]["version"])
            and result["wheel_metadata_matches"]
            and tag_commit_matches,
            "returncode": process.returncode,
            "raw_output": output,
            "source_tag_ref": config["valhalla"]["source_tag_ref"],
            "source_tag_commit": tag_commit,
            "source_tag_commit_matches": tag_commit_matches,
            "source_tag_check_returncode": tag_process.returncode,
        }
    )
    return result


def validate_field_contract(contract: dict[str, Any]) -> dict[str, Any]:
    forbidden_mappings = contract.get("explicitly_forbidden_mappings", [])
    legacy_fields = set(contract.get("legacy_reference_only_fields", []))
    semantic = contract.get("semantic_namespaces", {})
    road_fields = set(semantic.get("road_function", {}).get("fields", {}).keys())
    speed_fields = set(
        semantic.get("speed_management", {}).get("fields", {}).keys()
    )
    prohibited = set(contract.get("prohibited_model_features", []))

    class_mapping_explicitly_forbidden = any(
        item.get("source_field") == "Class of Speed Limit"
        and item.get("forbidden_target") in {"road_class", "functional_road_class"}
        for item in forbidden_mappings
    )
    namespaces_disjoint = road_fields.isdisjoint(speed_fields)
    required_separation = {
        "osm_highway",
        "valhalla_road_class",
        "functional_road_class",
    }.issubset(road_fields) and {
        "speed_limit_source",
        "speed_limit_kmh_observed",
        "speed_limit_kmh_inferred",
        "speed_limit_kmh_used",
    }.issubset(speed_fields)
    exact_coordinates_prohibited = {
        "latitude_raw",
        "longitude_raw",
        "matched_latitude",
        "matched_longitude",
        "exact_latitude",
        "exact_longitude",
    }.issubset(prohibited)
    identifiers_prohibited = {
        "VehId",
        "Trip",
        "row_id",
        "trip_uid",
        "matched_edge_id",
        "matched_way_id",
        "road_id",
        "route_id",
    }.issubset(prohibited)
    eved_energy_prohibited = (
        "Energy_Consumption" in legacy_fields
        and "Energy_Consumption" in prohibited
    )
    checks = {
        "class_of_speed_limit_mapping_forbidden": class_mapping_explicitly_forbidden,
        "road_and_speed_namespaces_disjoint": namespaces_disjoint,
        "required_road_speed_separation_present": required_separation,
        "exact_coordinates_prohibited": exact_coordinates_prohibited,
        "entity_identifiers_prohibited": identifiers_prohibited,
        "legacy_eved_energy_prohibited": eved_energy_prohibited,
    }
    return {
        "contract_version": contract.get("contract_version"),
        "road_function_fields": sorted(road_fields),
        "speed_management_fields": sorted(speed_fields),
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_sources(config: dict[str, Any]) -> dict[str, Any]:
    ved_results: list[dict[str, Any]] = []
    ved_directory = Path(config["ved"]["raw_directory"])
    for item in config["ved"]["release_files"]:
        path = ved_directory / item["file_name"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        read_only = is_read_only(path) if exists else False
        ved_results.append(
            {
                **item,
                "path": str(path),
                "exists": exists,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_hash,
                "read_only": read_only,
                "passed": exists
                and actual_bytes == int(item["bytes"])
                and actual_hash == item["sha256"].lower()
                and read_only,
            }
        )

    osm_results: list[dict[str, Any]] = []
    for item in config["osm"]["snapshots"]:
        path = Path(item["path"])
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        read_only = is_read_only(path) if exists else False
        historical_url = bool(
            re.search(r"-\d{6}\.osm\.pbf$", str(item["url"]))
        )
        osm_results.append(
            {
                **item,
                "exists": exists,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_hash,
                "read_only": read_only,
                "historical_url": historical_url,
                "passed": exists
                and actual_bytes == int(item["bytes"])
                and actual_hash == item["sha256"].lower()
                and read_only
                and historical_url,
            }
        )

    wheel_path = Path(config["valhalla"]["wheel_path"])
    wheel_exists = wheel_path.is_file()
    wheel_hash = sha256_file(wheel_path) if wheel_exists else None
    wheel_url = str(config["valhalla"]["wheel_url"])
    official_wheel_url = (
        wheel_url.startswith("https://files.pythonhosted.org/")
        and wheel_url.endswith("/" + config["valhalla"]["wheel_name"])
    )
    read_only = is_read_only(wheel_path) if wheel_exists else False
    wheel = {
        "path": str(wheel_path),
        "url": wheel_url,
        "exists": wheel_exists,
        "read_only": read_only,
        "official_versioned_url": official_wheel_url,
        "expected_sha256": config["valhalla"]["wheel_sha256"],
        "actual_sha256": wheel_hash,
        "passed": wheel_exists
        and wheel_hash == config["valhalla"]["wheel_sha256"].lower(),
        "immutable_passed": wheel_exists
        and wheel_hash == config["valhalla"]["wheel_sha256"].lower()
        and read_only
        and official_wheel_url,
    }
    return {
        "ved_release_files": ved_results,
        "osm_snapshots": osm_results,
        "valhalla_wheel": wheel,
        "passed": all(item["passed"] for item in ved_results)
        and all(item["passed"] for item in osm_results)
        and wheel["immutable_passed"],
    }


def license_manifest(
    config: dict[str, Any], lock_requirements: dict[str, str]
) -> dict[str, Any]:
    records = [
        {
            "component": "VED",
            "source": config["ved"]["repository"],
            "version_or_commit": config["ved"]["commit"],
            "license": config["ved"]["license"],
            "license_url": config["ved"]["license_url"],
            "attribution": config["ved"]["attribution"],
        },
        {
            "component": "OpenStreetMap/Geofabrik historical data",
            "source": "Geofabrik historical Michigan extracts",
            "version_or_commit": ", ".join(
                item["snapshot_date"] for item in config["osm"]["snapshots"]
            ),
            "license": config["osm"]["data_license"],
            "license_url": config["osm"]["license_url"],
            "attribution": config["osm"]["attribution"],
        },
        {
            "component": "Valhalla",
            "source": config["valhalla"]["repository"],
            "version_or_commit": (
                f"{config['valhalla']['version']} / "
                f"{config['valhalla']['commit']}"
            ),
            "license": config["valhalla"]["license"],
            "license_url": config["valhalla"]["license_url"],
            "attribution": "Retain the MIT copyright and permission notice.",
        },
    ]
    required = {
        "component",
        "source",
        "version_or_commit",
        "license",
        "license_url",
        "attribution",
    }
    complete = all(
        required.issubset(record)
        and all(str(record[key]).strip() for key in required)
        for record in records
    )
    dependency_records: list[dict[str, Any]] = []
    for name, version in sorted(lock_requirements.items()):
        try:
            metadata = importlib.metadata.metadata(name)
            expression = metadata.get("License-Expression")
            license_value = metadata.get("License")
            classifiers = [
                value
                for value in metadata.get_all("Classifier", [])
                if value.startswith("License ::")
            ]
            if license_value and len(license_value) > 200:
                license_value = "embedded_license_text"
            dependency_records.append(
                {
                    "name": name,
                    "version": version,
                    "license_expression": expression,
                    "license_metadata": license_value,
                    "license_classifiers": classifiers,
                    "project_url": metadata.get("Project-URL"),
                    "installed": True,
                }
            )
        except importlib.metadata.PackageNotFoundError:
            dependency_records.append(
                {"name": name, "version": version, "installed": False}
            )
    dependency_inventory_complete = (
        len(dependency_records) == len(lock_requirements)
        and all(item["installed"] for item in dependency_records)
    )
    return {
        "records": records,
        "python_dependency_records": dependency_records,
        "python_dependency_inventory_complete": dependency_inventory_complete,
        "license_quality_check_note": (
            "Package metadata is an inventory aid; redistribution obligations "
            "must be inspected before public release."
        ),
        "complete": complete and dependency_inventory_complete,
    }


def build_fingerprint(
    config: dict[str, Any],
    package_results: dict[str, dict[str, Any]],
    lock_results: dict[str, dict[str, Any]],
    valhalla_result: dict[str, Any],
    contract_hashes: dict[str, str | None],
) -> dict[str, Any]:
    stable = {
        "schema": FINGERPRINT_SCHEMA,
        "project_name": config["project"]["working_name"],
        "release_prefix": config["project"]["release_prefix"],
        "ved_commit": config["ved"]["commit"],
        "ved_release_sha256": sorted(
            item["sha256"] for item in config["ved"]["release_files"]
        ),
        "osm_snapshots": [
            {
                "snapshot_id": item["snapshot_id"],
                "snapshot_date": item["snapshot_date"],
                "sha256": item["sha256"],
            }
            for item in config["osm"]["snapshots"]
        ],
        "valhalla": {
            "version": valhalla_result["actual"],
            "commit": config["valhalla"]["commit"],
            "wheel_sha256": config["valhalla"]["wheel_sha256"],
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "direct_packages": {
            name: result["actual"]
            for name, result in sorted(package_results.items())
        },
        "resolved_lock_packages": {
            name: result["actual"] for name, result in sorted(lock_results.items())
        },
        "contract_hashes": contract_hashes,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "total_ram_bytes": total_ram_bytes(),
        },
        "spatial_reference": config["spatial_reference"],
    }
    return {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "stable": stable,
        "stable_sha256": canonical_sha256(stable),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in summary["checks"].items()
    )
    return f"""# RUNTIME_FREEZE RNEE-Bench contract and environment gate

Run completed: {summary['completed_at_utc']}

## Decision

`{summary['status']}`

Stable environment fingerprint:
`{summary['environment_fingerprint_sha256']}`

## Checks

| Check | Result |
| --- | --- |
{rows}

## Frozen runtime

- Python: `{summary['runtime']['python_version']}`
- Python executable: `{summary['runtime']['python_executable']}`
- Valhalla: `{summary['runtime']['valhalla_version']}`
- Valhalla commit: `{summary['runtime']['valhalla_commit']}`
- Free space on build drive: `{summary['runtime']['free_disk_gb']:.2f} GB`
- Full-build stop threshold: `{summary['runtime']['minimum_free_disk_gb']:.2f} GB`

## Interpretation

The G0 gate passes only when every source, license, field-separation, dependency,
runtime, target-contract, disk, and repeat-fingerprint check passes. This gate
does not validate map matching or road-semantic accuracy; those remain NETWORK_FREEZE
through END_TO_END gates.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/base.yaml"),
    )
    parser.add_argument(
        "--field-contract",
        type=Path,
        default=Path(r"configs/rnee_build/field_contract.yaml"),
    )
    parser.add_argument(
        "--qa-thresholds",
        type=Path,
        default=Path(r"configs/rnee_build/qa_thresholds.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"results/rnee_build"),
    )
    parser.add_argument(
        "--latest-prefix",
        default="runtime_freeze_latest",
    )
    args = parser.parse_args()

    started = utc_now()
    config = load_yaml(args.config)
    field_contract = load_yaml(args.field_contract)
    qa_thresholds = load_yaml(args.qa_thresholds)
    direct_requirements_path = Path(config["runtime"]["direct_requirements"])
    lock_path = Path(config["runtime"]["dependency_lock"])
    direct_requirements = parse_pinned_requirements(direct_requirements_path)
    package_results = installed_versions(direct_requirements)
    lock_requirements = parse_pinned_requirements(lock_path)
    lock_results, lock_is_complete = resolved_lock_status(lock_requirements)

    sources = verify_sources(config)
    licenses = license_manifest(config, lock_requirements)
    field_validation = validate_field_contract(field_contract)
    valhalla_result = valhalla_version(config)

    required_python = str(config["runtime"]["required_python_major_minor"])
    python_matches = platform.python_version().startswith(required_python + ".")
    packages_match = all(item["matches"] for item in package_results.values())

    target_contract_path = Path(config["contracts"]["target_contract"])
    target_summary_path = Path(config["contracts"]["target_gate_summary"])
    accepted_target_gates = set(qa_thresholds["target"]["accepted_gate"])
    target_status = target_gate_status(
        target_summary_path, target_contract_path, accepted_target_gates
    )

    drive_root = Path(config["runtime"]["disk"]["build_drive"])
    disk = shutil.disk_usage(drive_root)
    free_disk_gb = disk.free / (1024**3)
    minimum_free_disk_gb = float(
        config["runtime"]["disk"]["stop_full_build_below_free_gb"]
    )

    contract_hashes = {
        "base_yaml_sha256": sha256_file(args.config),
        "field_contract_yaml_sha256": sha256_file(args.field_contract),
        "qa_thresholds_yaml_sha256": sha256_file(args.qa_thresholds),
        "direct_requirements_sha256": sha256_file(direct_requirements_path),
        "dependency_lock_sha256": sha256_file(lock_path),
        "target_contract_sha256": target_status["contract_sha256"],
        "preflight_script_sha256": sha256_file(Path(__file__)),
    }
    fingerprint = build_fingerprint(
        config,
        package_results,
        lock_results,
        valhalla_result,
        contract_hashes,
    )
    latest_fingerprint_path = (
        args.output_root / f"{args.latest_prefix}_environment_fingerprint.json"
    )
    prior_fingerprint_sha = None
    repeated_fingerprint_equal = False
    if latest_fingerprint_path.is_file():
        prior = load_json(latest_fingerprint_path)
        prior_fingerprint_sha = prior.get("stable_sha256")
        repeated_fingerprint_equal = (
            prior_fingerprint_sha == fingerprint["stable_sha256"]
        )

    source_contract = {
        "project": config["project"],
        "source_policy": config["source_policy"],
        "ved": config["ved"],
        "osm": config["osm"],
        "valhalla": config["valhalla"],
        "spatial_reference": config["spatial_reference"],
        "verified_sources": sources,
        "target_contract_path": str(target_contract_path),
        "target_contract_sha256": target_status["contract_sha256"],
        "target_gate": target_status,
    }

    checks = {
        "working_name_and_prefix_frozen": (
            config["project"]["naming_status"] == "frozen_working_name"
            and bool(config["project"]["release_prefix"])
            and not config["project"]["naming_check"]["exact_collision_found"]
        ),
        "source_policy_is_ved_only": (
            config["source_policy"]["l0_inputs"]
            == ["official_VED_dynamic_data", "official_VED_static_data"]
            and config["source_policy"]["eved_role"] == "differential_audit_only"
        ),
        "source_hash_size_readonly_checks": sources["passed"],
        "historical_osm_only": (
            config["osm"]["temporal_policy"] == "historical_snapshots_only"
            and all(item["historical_url"] for item in sources["osm_snapshots"])
        ),
        "licenses_complete": licenses["complete"],
        "field_contract_passes": field_validation["passed"],
        "python_3_11_frozen": python_matches,
        "direct_dependency_versions_match": packages_match,
        "resolved_dependency_lock_matches": lock_is_complete,
        "valhalla_version_matches": valhalla_result["matches"],
        "target_contract_exists": target_status["contract_exists"],
        "target_gate_accepted": target_status["accepted"],
        "full_build_disk_budget_passes": free_disk_gb >= minimum_free_disk_gb,
        "repeated_environment_fingerprint_equal": repeated_fingerprint_equal,
    }
    passed = all(checks.values())
    status = EXPECTED_GATE if passed else "FAIL_CONTRACT_AND_ENVIRONMENT"

    completed = utc_now()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_root / f"runtime_freeze_preflight_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "experiment": "RUNTIME_FREEZE",
        "status": status,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "checks": checks,
        "environment_fingerprint_sha256": fingerprint["stable_sha256"],
        "prior_environment_fingerprint_sha256": prior_fingerprint_sha,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "valhalla_version": valhalla_result["actual"],
            "valhalla_commit": config["valhalla"]["commit"],
            "free_disk_gb": free_disk_gb,
            "minimum_free_disk_gb": minimum_free_disk_gb,
        },
        "target_gate": target_status["gate"],
        "output_directory": str(output_dir),
    }

    artifacts: dict[str, Any] = {
        "preflight_summary.json": summary,
        "environment_fingerprint.json": fingerprint,
        "source_contract.json": source_contract,
        "license_manifest.json": licenses,
        "field_contract_validation.json": field_validation,
        "dependency_validation.json": {
            "direct_requirements": package_results,
            "resolved_lock_requirements": lock_results,
            "resolved_lock_count": len(lock_requirements),
            "resolved_lock_matches": lock_is_complete,
        },
        "valhalla_validation.json": valhalla_result,
    }
    for filename, payload in artifacts.items():
        write_json(output_dir / filename, payload)
    shutil.copy2(args.config, output_dir / "run_config.yaml")
    shutil.copy2(args.field_contract, output_dir / "field_contract.yaml")
    shutil.copy2(args.qa_thresholds, output_dir / "qa_thresholds.yaml")
    shutil.copy2(lock_path, output_dir / "requirements-rnee.lock.txt")
    report = markdown_report(summary)
    (output_dir / "RUNTIME_FREEZE_CONTRACT_AND_ENVIRONMENT_GATE.md").write_text(
        report, encoding="utf-8"
    )

    latest_map = {
        "preflight_summary.json": f"{args.latest_prefix}_summary.json",
        "environment_fingerprint.json": (
            f"{args.latest_prefix}_environment_fingerprint.json"
        ),
        "source_contract.json": f"{args.latest_prefix}_source_contract.json",
        "license_manifest.json": f"{args.latest_prefix}_license_manifest.json",
        "field_contract_validation.json": (
            f"{args.latest_prefix}_field_contract_validation.json"
        ),
        "dependency_validation.json": (
            f"{args.latest_prefix}_dependency_validation.json"
        ),
        "valhalla_validation.json": (
            f"{args.latest_prefix}_valhalla_validation.json"
        ),
        "RUNTIME_FREEZE_CONTRACT_AND_ENVIRONMENT_GATE.md": (
            f"{args.latest_prefix}_report.md"
        ),
    }
    for source_name, latest_name in latest_map.items():
        shutil.copy2(output_dir / source_name, args.output_root / latest_name)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
