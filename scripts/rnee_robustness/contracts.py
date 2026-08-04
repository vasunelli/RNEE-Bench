from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_json, sha256_file, verify_manifest


EXPECTED_ROLES = {"train", "validation", "calibration", "test"}


def forbidden_hits(features: list[str], contract: dict[str, Any]) -> list[str]:
    forbidden = contract["forbidden_feature_contract"]
    exact = {str(value).lower() for value in forbidden["exact"]}
    tokens = [str(value).lower() for value in forbidden["tokens"]]
    hits: list[str] = []
    for feature in features:
        lowered = feature.lower()
        if lowered in exact or any(token in lowered for token in tokens):
            hits.append(feature)
    return sorted(set(hits))


def train_active_columns(frame: pd.DataFrame) -> list[str]:
    active: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.nunique(dropna=True) >= 2:
            active.append(column)
    return active


def validate_membership(membership: pd.DataFrame, cell_key: str) -> dict[str, Any]:
    required = {"segment_id", "split_role"}
    missing = sorted(required - set(membership.columns))
    if missing:
        raise RuntimeError(f"{cell_key}: membership missing columns {missing}")
    if membership["segment_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{cell_key}: duplicate segment membership.")
    observed_roles = set(membership["split_role"].astype(str).unique())
    if observed_roles != EXPECTED_ROLES:
        raise RuntimeError(
            f"{cell_key}: roles {sorted(observed_roles)} do not match "
            f"{sorted(EXPECTED_ROLES)}."
        )
    role_counts = membership["split_role"].value_counts().to_dict()
    if any(int(role_counts.get(role, 0)) == 0 for role in EXPECTED_ROLES):
        raise RuntimeError(f"{cell_key}: at least one split role is empty.")
    return {
        "cell_key": cell_key,
        "rows": int(len(membership)),
        "segment_ids_unique": True,
        "role_counts": {role: int(role_counts[role]) for role in sorted(EXPECTED_ROLES)},
    }


def validate_upstream_artifact_manifests(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        manifest = load_json(path)
        if "artifacts" in manifest:
            verified, mismatches = verify_manifest(manifest, path.parent)
            artifact_count = len(manifest["artifacts"])
        elif "artifact_manifest_sha256" in manifest:
            # INFORMATION_FREEZE uses a compact release-control record rather than a second
            # artifact list. Verify its explicit pointer to the sibling
            # primary manifest without weakening the upstream gate.
            from .artifacts import sha256_file

            primary = path.parent / "artifact_manifest.json"
            verified = (
                primary.exists()
                and sha256_file(primary) == manifest["artifact_manifest_sha256"]
            )
            mismatches = [] if verified else [str(primary)]
            artifact_count = int(manifest.get("primary_artifact_count", 0))
        else:
            raise RuntimeError(f"Not an artifact or release manifest: {path}")
        records.append(
            {
                "path": str(path),
                "verified": verified,
                "artifact_count": artifact_count,
                "mismatches": mismatches,
            }
        )
    if not all(record["verified"] for record in records):
        raise RuntimeError(f"Upstream artifact verification failed: {records}")
    return {"all_verified": True, "manifests": records}


def validate_split_checks(path: Path) -> dict[str, Any]:
    checks = pd.read_csv(path)
    if "status" not in checks.columns:
        raise RuntimeError("SEGMENT_SPLIT leakage-check artifact lacks a status column.")
    failed = checks[~checks["status"].astype(str).str.upper().eq("PASS")]
    if len(failed):
        raise RuntimeError(
            "The frozen SEGMENT_SPLIT split contract contains non-PASS checks: "
            + "|".join(failed["check"].astype(str).head(20))
        )
    return {
        "path": str(path),
        "check_count": int(len(checks)),
        "pass_count": int(len(checks)),
        "fail_count": 0,
    }


def _verify_recorded_hash(path_value: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_value)
    observed = sha256_file(path) if path.exists() else None
    return {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "verified": observed == expected_sha256,
    }


def validate_segment_split_release_manifests(
    segment_manifest_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    """Verify every hash-bearing SEGMENT_SPLIT segment and split release record."""
    segment_manifest = load_json(segment_manifest_path)
    partitions = list(segment_manifest.get("partitions", []))
    if not partitions:
        raise RuntimeError("SEGMENT_SPLIT segment manifest contains no partitions.")
    segment_records: list[dict[str, Any]] = []
    segment_contracts_pass = True
    for partition in partitions:
        checks = partition.get("checks", {})
        contract_pass = (
            str(partition.get("status", "")).upper() == "PASS"
            and bool(checks)
            and all(str(value).upper() == "PASS" for value in checks.values())
        )
        segment_contracts_pass = segment_contracts_pass and contract_pass
        segment_records.extend(
            [
                _verify_recorded_hash(
                    partition["segments"], partition["segments_sha256"]
                ),
                _verify_recorded_hash(
                    partition["row_assignments"],
                    partition["row_assignments_sha256"],
                ),
            ]
        )

    split_manifest = load_json(split_manifest_path)
    families = list(split_manifest.get("families", []))
    if not families:
        raise RuntimeError("SEGMENT_SPLIT split manifest contains no families.")
    split_records = [
        _verify_recorded_hash(
            split_manifest["membership_contract"],
            split_manifest["membership_contract_sha256"],
        )
    ]
    for family in families:
        split_records.extend(
            [
                _verify_recorded_hash(
                    family["assignment_path"], family["assignment_sha256"]
                ),
                _verify_recorded_hash(
                    family["membership_path"], family["membership_sha256"]
                ),
            ]
        )

    all_hashes_verified = all(
        record["verified"] for record in segment_records + split_records
    )
    split_contract_pass = (
        str(split_manifest.get("status", "")).upper()
        == "PASS_SEGMENT_SPLIT_SPLIT_BUILD"
    )
    result = {
        "segment_manifest": str(segment_manifest_path),
        "segment_manifest_sha256": sha256_file(segment_manifest_path),
        "segment_partition_count": len(partitions),
        "segment_hash_record_count": len(segment_records),
        "segment_contracts_pass": segment_contracts_pass,
        "segment_records": segment_records,
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_family_count": len(families),
        "split_hash_record_count": len(split_records),
        "split_contract_pass": split_contract_pass,
        "split_records": split_records,
        "all_hashes_verified": all_hashes_verified,
    }
    if not (
        result["segment_contracts_pass"]
        and result["split_contract_pass"]
        and result["all_hashes_verified"]
    ):
        raise RuntimeError(f"SEGMENT_SPLIT release-manifest verification failed: {result}")
    return result


def validate_numeric_frame(
    frame: pd.DataFrame, expected_columns: list[str], context: str
) -> dict[str, Any]:
    if list(frame.columns) != list(expected_columns):
        raise RuntimeError(f"{context}: feature order or columns changed.")
    non_numeric = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise RuntimeError(f"{context}: non-numeric features {non_numeric}")
    finite_or_missing = np.isfinite(
        frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ) | frame.isna().to_numpy()
    if not bool(finite_or_missing.all()):
        raise RuntimeError(f"{context}: infinite feature values found.")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
    }


def null_control_contract_pass(metadata: dict[str, Any]) -> bool:
    contracts = metadata["contracts"]
    return bool(
        contracts["same_rows"]
        and contracts["same_columns"]
        and contracts["same_dtypes"]
        and contracts["noise_recipient_missing_mask_exact"]
        and contracts["permutation_role_block_multiset_exact"]
        and int(contracts["permutation_cross_role_moves"]) == 0
        and int(contracts["permutation_fixed_points"]) == 0
    )
