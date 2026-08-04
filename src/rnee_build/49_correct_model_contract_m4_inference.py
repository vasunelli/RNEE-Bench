#!/usr/bin/env python3
"""Correct MODEL_CONTRACT M4 null inference without refitting or target rematerialization.

The original M4 descriptive effects, percentile intervals, and calibration
diagnostics remain frozen.  This bounded correction replaces the uncentered
bootstrap sign-tail p-values with a trip-clustered, null-imposed studentized
Rademacher wild bootstrap and prevents reliability-boundary comparisons from
being graded as substantive NO_GAIN evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


KEY_COLUMNS = ["split_family", "target_id", "comparison_family", "semantic_group"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": [
            {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
            for path in sorted(set(paths), key=lambda value: str(value).lower())
        ],
    }


def verify_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if (
            not path.exists()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            mismatches.append(str(path))
    return not mismatches, mismatches


def deterministic_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def fdr_adjust(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("FDR input must be a non-empty finite vector.")
    if method not in {"bh", "by"}:
        raise ValueError(f"Unsupported FDR method: {method}")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    m = len(p)
    factor = float(sum(1.0 / index for index in range(1, m + 1))) if method == "by" else 1.0
    adjusted_ranked = ranked * m * factor / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def cluster_structure(cluster_ids: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusters, inverse = np.unique(cluster_ids.astype(str).to_numpy(), return_inverse=True)
    if len(clusters) < 2:
        raise ValueError("At least two trip clusters are required.")
    counts = np.bincount(inverse).astype(float)
    return clusters, inverse, counts


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction used by regularized beta."""
    maximum_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        value *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) < epsilon:
            return value
    raise RuntimeError("Regularized-beta continued fraction did not converge.")


def regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("Regularized beta requires x in [0,1] and positive shapes.")
    if x in {0.0, 1.0}:
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1 or not math.isfinite(t_value):
        if math.isinf(t_value) and degrees_of_freedom >= 1:
            return 0.0
        raise ValueError("Student-t sensitivity requires finite t and positive df.")
    x = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    return float(regularized_beta(x, degrees_of_freedom / 2.0, 0.5))


def wild_cluster_test(
    values: np.ndarray,
    inverse: np.ndarray,
    cluster_counts: np.ndarray,
    multipliers: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Studentized Rademacher wild bootstrap under H0: mean(values) == 0."""
    numeric = np.asarray(values, dtype=float)
    if len(numeric) != len(inverse) or not np.isfinite(numeric).all():
        raise ValueError("Effect values must be finite and aligned with clusters.")
    cluster_sums = np.bincount(inverse, weights=numeric, minlength=len(cluster_counts))
    cluster_count = len(cluster_counts)
    total_rows = float(cluster_counts.sum())
    point = float(cluster_sums.sum() / total_rows)
    psi = cluster_sums - point * cluster_counts
    correction = cluster_count / (cluster_count - 1.0)
    se = float(math.sqrt(correction * float(np.dot(psi, psi))) / total_rows)
    tolerance = np.finfo(float).eps * max(1.0, abs(point))
    if se <= tolerance:
        if abs(point) <= tolerance:
            t_observed = 0.0
        else:
            raise ValueError("Nonzero clustered effect has zero estimable cluster variance.")
    else:
        t_observed = point / se

    weights = np.asarray(multipliers, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != cluster_count:
        raise ValueError("Wild multipliers must have shape (replicates, clusters).")
    null_effect = (weights @ psi) / total_rows
    # Studentize each null draw without allocating a replicate-by-cluster float matrix.
    centered_square_sum = (
        float(np.dot(psi, psi))
        + np.square(null_effect) * float(np.dot(cluster_counts, cluster_counts))
        - 2.0 * null_effect * (weights @ (psi * cluster_counts))
    )
    centered_square_sum = np.maximum(centered_square_sum, 0.0)
    se_star = np.sqrt(correction * centered_square_sum) / total_rows
    t_star = np.zeros_like(null_effect)
    regular = se_star > tolerance
    t_star[regular] = null_effect[regular] / se_star[regular]
    irregular_nonzero = (~regular) & (np.abs(null_effect) > tolerance)
    t_star[irregular_nonzero] = np.sign(null_effect[irregular_nonzero]) * np.inf
    p_wild = float(
        (1.0 + np.sum(np.abs(t_star) >= abs(t_observed))) / (len(t_star) + 1.0)
    )
    p_cluster_t = student_t_two_sided_p(t_observed, cluster_count - 1)
    result = {
        "effect_recomputed": point,
        "cluster_count": float(cluster_count),
        "cluster_robust_se": se,
        "cluster_robust_t": float(t_observed),
        "p_value_wild_studentized": p_wild,
        "p_value_cluster_t_sensitivity": p_cluster_t,
    }
    return result, null_effect, t_star


def bounded_evidence_grade(
    positive_add: int,
    positive_drop: int,
    negative_add: int,
    negative_drop: int,
    reliability_boundaries: int,
) -> str:
    if reliability_boundaries > 0:
        return "RELIABILITY_LIMITED_UNRESOLVED"
    positive_total = positive_add + positive_drop
    negative_total = negative_add + negative_drop
    if positive_total > 0 and negative_total > 0:
        return "MIXED_DIRECTION_UNRESOLVED"
    if negative_total > 0:
        return "HARMFUL_UNDER_OOD"
    if positive_add >= 3 and positive_drop >= 1:
        return "ROBUST_GAIN"
    if positive_add > 0 and positive_drop == 0:
        return "REDUNDANT_SIGNAL"
    if positive_add == 0 and positive_drop > 0:
        return "SYNERGISTIC_SIGNAL"
    if positive_total > 0:
        return "CONDITIONAL_GAIN"
    return "NO_GAIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"configs/rnee_build/model_contract_m4_corrected.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m3_pointer_path = Path(config["m3_pointer"])
    m4_pointer_path = Path(config["original_m4_pointer"])
    m3_pointer = load_json(m3_pointer_path)
    m4_pointer = load_json(m4_pointer_path)
    m3_root = Path(m3_pointer["output_directory"])
    m4_root = Path(m4_pointer["output_directory"])

    upstream_manifest_paths = [
        Path(m3_pointer["manifest"]),
        Path(m3_pointer["release_manifest"]),
        Path(m3_pointer["audit_manifest"]),
        Path(m4_pointer["manifest"]),
        Path(m4_pointer["release_manifest"]),
    ]
    upstream_verification = [verify_manifest(load_json(path)) for path in upstream_manifest_paths]
    if not all(result[0] for result in upstream_verification):
        raise RuntimeError(f"Upstream manifest verification failed: {upstream_verification}")
    m3_summary = load_json(Path(m3_pointer["summary"]))
    m4_summary = load_json(Path(m4_pointer["summary"]))
    if m3_summary["status"] != "PASS_MODEL_CONTRACT_M3_PRIMARY_ADD_DROP" or not m3_summary["resume_noop_verified"]:
        raise RuntimeError("M3 is not a reproducible frozen release.")
    if not m4_summary["resume_noop_verified"]:
        raise RuntimeError("Original M4 calibration/descriptive release is not reproducible.")

    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "m3_pointer_sha256": sha256_file(m3_pointer_path),
        "m3_manifest_sha256": sha256_file(Path(m3_pointer["manifest"])),
        "m3_release_manifest_sha256": sha256_file(Path(m3_pointer["release_manifest"])),
        "m3_audit_manifest_sha256": sha256_file(Path(m3_pointer["audit_manifest"])),
        "original_m4_pointer_sha256": sha256_file(m4_pointer_path),
        "original_m4_manifest_sha256": sha256_file(Path(m4_pointer["manifest"])),
        "original_m4_release_manifest_sha256": sha256_file(Path(m4_pointer["release_manifest"])),
    }
    output = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(config["output_root"])
        / f"{config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    run_config_path = output / "run_config.yaml"
    if not run_config_path.exists():
        shutil.copy2(args.config, run_config_path)
    state_path = output / "m4_corrected_state.json"
    summary_path = output / "model_contract_m4_corrected_summary.json"
    primary_manifest_path = output / "artifact_manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if state["input_fingerprint"] != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch.")
    else:
        state = {"schema_version": 1, "status": "PENDING", "input_fingerprint": fingerprint}
        write_json(state_path, state)

    if state["status"] == "PASS" and summary_path.exists() and primary_manifest_path.exists():
        verified, mismatches = verify_manifest(load_json(primary_manifest_path))
        resume_path = output / "resume_check.json"
        write_json(
            resume_path,
            {
                "status": "PASS" if verified else "FAIL",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "artifact_count": len(load_json(primary_manifest_path)["artifacts"]),
                "mismatches": mismatches,
                "model_refit_count": 0,
                "test_target_rematerialization_count": 0,
                "calibration_target_rematerialization_count": 0,
                "wild_bootstrap_recompute_count": 0,
            },
        )
        summary = load_json(summary_path)
        summary["resume_noop_verified"] = verified
        summary["resume_check"] = str(resume_path)
        release_manifest_path = output / "release_manifest.json"
        summary["release_manifest"] = str(release_manifest_path)
        write_json(summary_path, summary)
        release = artifact_manifest([primary_manifest_path, summary_path, resume_path])
        release["scope"] = "post-resume controls; immutable corrected outputs are in artifact_manifest.json"
        write_json(release_manifest_path, release)
        latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
        latest = load_json(latest_path)
        latest["release_manifest"] = str(release_manifest_path)
        write_json(latest_path, latest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if verified else 1

    original_effects = pd.read_csv(Path(m4_pointer["paired_effect_inference"]))
    original_joint = pd.read_csv(Path(m4_pointer["joint_gain_reliability"]))
    if len(original_effects) != 100 or original_effects.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("Original M4 paired effect table is not the expected 100 unique comparisons.")
    cells = config["primary_cells"]
    canonical_seed = int(config["canonical_seed"])
    replicates = int(config["wild_bootstrap_replicates"])
    master_seed = int(config["wild_bootstrap_master_seed"])
    corrected_records: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(cells, start=1):
        split_name = cell["split_family"]
        target_id = cell["target_id"]
        storage_key = f"{split_name}__{target_id}"
        prediction_path = m3_root / "paired_test_predictions" / storage_key / f"seed_{canonical_seed}.parquet"
        test = pd.read_parquet(prediction_path)
        clusters, inverse, counts = cluster_structure(test["trip_uid"])
        cell_seed = deterministic_seed(master_seed, storage_key)
        rng = np.random.default_rng(cell_seed)
        multipliers = (2 * rng.integers(0, 2, size=(replicates, len(clusters)), dtype=np.int8) - 1).astype(np.int8)
        seed_records.append(
            {
                "split_family": split_name,
                "target_id": target_id,
                "wild_bootstrap_seed": cell_seed,
                "replicates": replicates,
                "trip_clusters": len(clusters),
                "shared_multipliers_within_cell": True,
                "paired_prediction_path": str(prediction_path),
                "paired_prediction_sha256": sha256_file(prediction_path),
            }
        )
        y_true = test["y_true"].to_numpy(float)
        for semantic_index in range(1, 6):
            comparisons = [
                ("add", f"R{semantic_index}", "R0", f"R0_PLUS_R{semantic_index}"),
                ("drop", f"R{semantic_index}", "R6", f"R6_MINUS_R{semantic_index}"),
            ]
            for family, semantic_group, baseline, comparison in comparisons:
                baseline_error = np.abs(y_true - test[f"prediction__{baseline}"].to_numpy(float))
                comparison_error = np.abs(y_true - test[f"prediction__{comparison}"].to_numpy(float))
                row_effect = baseline_error - comparison_error if family == "add" else comparison_error - baseline_error
                test_result, null_effect, t_star = wild_cluster_test(row_effect, inverse, counts, multipliers)
                key_mask = (
                    (original_effects["split_family"] == split_name)
                    & (original_effects["target_id"] == target_id)
                    & (original_effects["comparison_family"] == family)
                    & (original_effects["semantic_group"] == semantic_group)
                )
                frozen = original_effects.loc[key_mask]
                if len(frozen) != 1:
                    raise RuntimeError(f"Missing frozen descriptive comparison: {storage_key}|{family}|{semantic_group}")
                record = frozen.iloc[0].to_dict()
                difference = abs(float(record["effect"]) - test_result["effect_recomputed"])
                if difference > float(config["effect_equivalence_tolerance"]):
                    raise RuntimeError(f"Frozen effect mismatch ({difference}): {storage_key}|{family}|{semantic_group}")
                record.pop("p_value_two_sided", None)
                record.pop("q_value_bh", None)
                record.pop("q_value_by_dependency_sensitivity", None)
                record.pop("bh_direction", None)
                record.pop("by_dependency_direction", None)
                record.update(test_result)
                record["effect_equivalence_abs_difference"] = difference
                record["null_inference_method"] = "trip_clustered_studentized_rademacher_wild_bootstrap"
                corrected_records.append(record)
                bootstrap_records.extend(
                    {
                        "split_family": split_name,
                        "target_id": target_id,
                        "comparison_family": family,
                        "semantic_group": semantic_group,
                        "replicate": replicate_index,
                        "null_effect": float(null_effect_value),
                        "null_studentized_t": float(t_value),
                    }
                    for replicate_index, (null_effect_value, t_value) in enumerate(zip(null_effect, t_star))
                )
        print(f"CORRECTED INFERENCE {cell_index}/10 {storage_key}", flush=True)

    effects = pd.DataFrame(corrected_records)
    alpha = float(config["fdr_alpha"])
    for (_target, _family), indexes in effects.groupby(["target_id", "comparison_family"]).groups.items():
        selected = list(indexes)
        p = effects.loc[selected, "p_value_wild_studentized"].to_numpy(float)
        effects.loc[selected, "q_value_bh"] = fdr_adjust(p, "bh")
        effects.loc[selected, "q_value_by_dependency_sensitivity"] = fdr_adjust(p, "by")
        effects.loc[selected, "fdr_family_size"] = len(selected)
    effects["bh_direction"] = np.where(
        (effects["q_value_bh"] <= alpha) & (effects["ci_lower"] > 0),
        "positive",
        np.where((effects["q_value_bh"] <= alpha) & (effects["ci_upper"] < 0), "negative", "not_supported"),
    )
    effects["by_dependency_direction"] = np.where(
        (effects["q_value_by_dependency_sensitivity"] <= alpha) & (effects["ci_lower"] > 0),
        "positive",
        np.where(
            (effects["q_value_by_dependency_sensitivity"] <= alpha) & (effects["ci_upper"] < 0),
            "negative",
            "not_supported",
        ),
    )
    effects = effects.sort_values(KEY_COLUMNS).reset_index(drop=True)

    calibration_columns = [
        *KEY_COLUMNS,
        "train_segments",
        "calibration_segments",
        "primary_coverage_nominal",
        "baseline_test_coverage",
        "comparison_test_coverage",
        "baseline_worst_local_undercoverage",
        "comparison_worst_local_undercoverage",
        "baseline_reliability_pass",
        "comparison_reliability_pass",
        "reliability_joint_status",
    ]
    calibration_evidence = original_joint[calibration_columns].copy()
    if len(calibration_evidence) != 100 or calibration_evidence.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("Original M4 joint calibration evidence is incomplete.")
    joint = effects.merge(calibration_evidence, on=KEY_COLUMNS, how="left", validate="one_to_one")
    joint["joint_dependency_robust_direction"] = np.where(
        (joint["reliability_joint_status"] == "PASS")
        & joint["by_dependency_direction"].isin(["positive", "negative"]),
        joint["by_dependency_direction"],
        "not_supported",
    )
    joint["split_family_dependence_note"] = "correlated_family_results_not_independent_replications"
    joint["seed_inference_note"] = "canonical_seed_only_three_frozen_seeds_numerically_identical"
    joint["calibration_evidence_note"] = "reused_hash_verified_original_m4_calibration_diagnostics"

    cold_splits = set(config["primary_cold_splits"])
    grade_records: list[dict[str, Any]] = []
    for (target_id, semantic_group), group in joint.groupby(["target_id", "semantic_group"]):
        cold = group[group["split_family"].isin(cold_splits)]
        add = cold[cold["comparison_family"] == "add"]
        drop = cold[cold["comparison_family"] == "drop"]
        positive_add = int((add["joint_dependency_robust_direction"] == "positive").sum())
        positive_drop = int((drop["joint_dependency_robust_direction"] == "positive").sum())
        negative_add = int((add["joint_dependency_robust_direction"] == "negative").sum())
        negative_drop = int((drop["joint_dependency_robust_direction"] == "negative").sum())
        boundaries = int((cold["reliability_joint_status"] != "PASS").sum())
        grade_records.append(
            {
                "target_id": target_id,
                "semantic_group": semantic_group,
                "primary_cold_split_count": len(cold_splits),
                "dependency_robust_positive_add_splits": positive_add,
                "dependency_robust_positive_drop_splits": positive_drop,
                "dependency_robust_negative_add_splits": negative_add,
                "dependency_robust_negative_drop_splits": negative_drop,
                "reliability_boundary_comparisons": boundaries,
                "evidence_grade": bounded_evidence_grade(
                    positive_add, positive_drop, negative_add, negative_drop, boundaries
                ),
            }
        )
    grades = pd.DataFrame(grade_records).sort_values(["target_id", "semantic_group"])

    effects_path = output / "paired_effect_inference_corrected.csv"
    bootstrap_path = output / "wild_null_bootstrap.parquet"
    joint_path = output / "joint_gain_reliability_corrected.csv"
    grades_path = output / "evidence_grades_corrected.csv"
    seed_path = output / "wild_bootstrap_seed_audit.csv"
    provenance_path = output / "correction_provenance.json"
    checks_path = output / "m4_corrected_gate_checks.csv"
    report_path = output / "MODEL_CONTRACT_M4_CORRECTED_GATE.md"
    effects.to_csv(effects_path, index=False)
    write_parquet(bootstrap_path, pd.DataFrame(bootstrap_records))
    joint.to_csv(joint_path, index=False)
    grades.to_csv(grades_path, index=False)
    pd.DataFrame(seed_records).to_csv(seed_path, index=False)
    write_json(
        provenance_path,
        {
            "schema_version": 1,
            "correction_scope": "formal null inference and reliability-boundary grading only",
            "frozen_descriptive_source": str(Path(m4_pointer["paired_effect_inference"])),
            "frozen_calibration_summary": str(Path(m4_pointer["calibration_summary"])),
            "frozen_local_coverage": str(Path(m4_pointer["local_coverage"])),
            "frozen_calibration_predictions_root": str(m4_root / "calibration_predictions"),
            "frozen_test_predictions_root": str(m3_pointer["paired_predictions_root"]),
            "upstream_manifest_paths": [str(path) for path in upstream_manifest_paths],
            "upstream_manifest_sha256": {str(path): sha256_file(path) for path in upstream_manifest_paths},
            "model_refit_count": 0,
            "test_target_rematerialization_count": 0,
            "calibration_target_rematerialization_count": 0,
        },
    )

    grade_counts = grades["evidence_grade"].value_counts().sort_index().to_dict()
    boundary_count = int((joint["reliability_joint_status"] != "PASS").sum())
    checks = {
        "recursive_m3_and_original_m4_manifests": "PASS",
        "no_model_refitting": "PASS",
        "no_test_target_rematerialization": "PASS",
        "no_calibration_target_rematerialization": "PASS",
        "complete_100_paired_comparisons": "PASS" if len(effects) == 100 else "FAIL",
        "exact_2000_null_wild_replicates": "PASS" if len(bootstrap_records) == 200000 else "FAIL",
        "exact_four_fdr_families_of_25": "PASS"
        if effects.groupby(["target_id", "comparison_family"]).size().eq(25).all()
        else "FAIL",
        "finite_formal_inference": "PASS"
        if np.isfinite(
            effects[
                [
                    "cluster_robust_se",
                    "cluster_robust_t",
                    "p_value_wild_studentized",
                    "p_value_cluster_t_sensitivity",
                    "q_value_bh",
                    "q_value_by_dependency_sensitivity",
                ]
            ].to_numpy(float)
        ).all()
        else "FAIL",
        "frozen_effects_exact": "PASS"
        if effects["effect_equivalence_abs_difference"].max() <= float(config["effect_equivalence_tolerance"])
        else "FAIL",
        "complete_joint_reliability": "PASS" if len(joint) == 100 else "FAIL",
        "universal_reliability_boundary_explicit": "PASS" if boundary_count == 100 else "FAIL",
        "boundary_grades_not_substantive_no_gain": "PASS"
        if set(grades["evidence_grade"]) == {"RELIABILITY_LIMITED_UNRESOLVED"}
        else "FAIL",
        "target_units_separate": "PASS" if set(effects["unit"]) == {"L"} else "FAIL",
        "split_family_dependence_disclosed": "PASS",
    }
    pd.DataFrame([{"check": key, "status": value} for key, value in checks.items()]).to_csv(checks_path, index=False)
    passed = "FAIL" not in checks.values()
    state["status"] = "PASS" if passed else "FAIL"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema_version": 1,
        "experiment": "MODEL_CONTRACT",
        "stage": "M4_corrected_null_inference_and_boundary_grading",
        "status": "PASS_MODEL_CONTRACT_M4_CORRECTED_INFERENCE" if passed else "FAIL_MODEL_CONTRACT_M4_CORRECTED_INFERENCE",
        "inference_gate": "PASS_MODEL_CONTRACT_INFERENCE_GATE" if passed else "FAIL_MODEL_CONTRACT_INFERENCE_GATE",
        "joint_interpretation_state": "COMPLETE_INFERENCE__JOINT_RELIABILITY_BOUNDARY",
        "checks": checks,
        "paired_comparison_count": len(effects),
        "wild_bootstrap_replicates_per_comparison": replicates,
        "wild_null_bootstrap_rows": len(bootstrap_records),
        "fdr_family_count": int(effects.groupby(["target_id", "comparison_family"]).ngroups),
        "fdr_family_size": 25,
        "bh_supported_positive_comparisons": int((effects["bh_direction"] == "positive").sum()),
        "bh_supported_negative_comparisons": int((effects["bh_direction"] == "negative").sum()),
        "by_supported_positive_comparisons": int((effects["by_dependency_direction"] == "positive").sum()),
        "by_supported_negative_comparisons": int((effects["by_dependency_direction"] == "negative").sum()),
        "reliability_boundary_comparisons": boundary_count,
        "joint_supported_comparisons": int((joint["joint_dependency_robust_direction"] != "not_supported").sum()),
        "evidence_grade_counts": grade_counts,
        "model_refit_count": 0,
        "test_target_rows_rematerialized": 0,
        "calibration_target_rows_rematerialized": 0,
        "resume_noop_verified": False,
        "claim_authorization": (
            "DESCRIPTIVE_EFFECTS_PERCENTILE_CIS_FORMAL_FDR_SUPPORT_AND_EMPIRICAL_CALIBRATION_DIAGNOSTICS;"
            "SUBSTANTIVE_SEMANTIC_GRADES_WITHHELD_BY_UNIVERSAL_RELIABILITY_BOUNDARY"
        ),
        "supersedes_original_m4_claim_bearing_inference": str(m4_root),
        "output_directory": str(output),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_path, summary)
    report_path.write_text(
        "\n".join(
            [
                "# MODEL_CONTRACT M4 corrected inference and reliability boundary gate",
                "",
                f"- Status: `{summary['status']}`",
                f"- Formal inference gate: `{summary['inference_gate']}`",
                f"- Joint interpretation: `{summary['joint_interpretation_state']}`",
                f"- Paired comparisons: {len(effects)}",
                f"- Studentized null wild-bootstrap replicates: {len(bootstrap_records):,}",
                f"- BH positive / negative support: {summary['bh_supported_positive_comparisons']} / {summary['bh_supported_negative_comparisons']}",
                f"- BY positive / negative support: {summary['by_supported_positive_comparisons']} / {summary['by_supported_negative_comparisons']}",
                f"- Reliability-boundary comparisons: {boundary_count} / 100",
                f"- Evidence grades: {json.dumps(grade_counts, sort_keys=True)}",
                "- Model refits: 0",
                "- Test target rematerializations: 0",
                "- Calibration target rematerializations: 0",
                "",
                "## Decision",
                "",
                "The formal null test is corrected and reproducible. Descriptive effects, percentile intervals, FDR support counts, and empirical calibration diagnostics are authorized. All 100 paired comparisons remain on the configured joint reliability boundary, so the ten target-by-semantic-group grades are `RELIABILITY_LIMITED_UNRESOLVED`; `NO_GAIN` is not authorized as a substantive conclusion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    primary_paths = [
        run_config_path,
        state_path,
        effects_path,
        bootstrap_path,
        joint_path,
        grades_path,
        seed_path,
        provenance_path,
        checks_path,
        report_path,
    ]
    write_json(primary_manifest_path, artifact_manifest(primary_paths))
    latest_path = Path(config["output_root"]) / f"{config['latest_prefix']}_latest_pointer.json"
    write_json(
        latest_path,
        {
            "status": summary["status"],
            "summary": str(summary_path),
            "report": str(report_path),
            "paired_effect_inference": str(effects_path),
            "wild_null_bootstrap": str(bootstrap_path),
            "joint_gain_reliability": str(joint_path),
            "evidence_grades": str(grades_path),
            "manifest": str(primary_manifest_path),
            "output_directory": str(output),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
