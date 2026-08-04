from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "49_correct_model_contract_m4_inference.py"
SPEC = importlib.util.spec_from_file_location("model_contract_m4_corrected", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_null_wild_bootstrap_is_deterministic_and_returns_one_for_exact_zero() -> None:
    clusters = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"])
    _, inverse, counts = MODULE.cluster_structure(clusters)
    rng = np.random.default_rng(42)
    multipliers = 2 * rng.integers(0, 2, size=(2000, 4), dtype=np.int8) - 1
    result, null_effect, t_star = MODULE.wild_cluster_test(
        np.zeros(len(clusters)), inverse, counts, multipliers
    )
    assert result["p_value_wild_studentized"] == 1.0
    assert result["p_value_cluster_t_sensitivity"] == 1.0
    assert np.array_equal(null_effect, np.zeros(2000))
    assert np.array_equal(t_star, np.zeros(2000))


def test_strong_clustered_effect_is_detected() -> None:
    rng = np.random.default_rng(7)
    cluster_count = 80
    per_cluster = 5
    clusters = pd.Series(np.repeat(np.arange(cluster_count), per_cluster).astype(str))
    _, inverse, counts = MODULE.cluster_structure(clusters)
    values = np.repeat(0.5 + rng.normal(0.0, 0.15, cluster_count), per_cluster)
    multipliers = 2 * rng.integers(0, 2, size=(2000, cluster_count), dtype=np.int8) - 1
    result, _, _ = MODULE.wild_cluster_test(values, inverse, counts, multipliers)
    assert result["effect_recomputed"] > 0.4
    assert result["p_value_wild_studentized"] < 0.01
    assert result["p_value_cluster_t_sensitivity"] < 0.01


def test_fdr_family_of_25_and_by_conservatism() -> None:
    p = np.linspace(0.001, 0.5, 25)
    bh = MODULE.fdr_adjust(p, "bh")
    by = MODULE.fdr_adjust(p, "by")
    assert len(bh) == 25
    assert np.all(by >= bh)
    assert np.all(np.diff(bh) >= 0)


def test_student_t_reference_matches_known_values() -> None:
    assert MODULE.student_t_two_sided_p(0.0, 10) == 1.0
    assert abs(MODULE.student_t_two_sided_p(2.2281388519649385, 10) - 0.05) < 1.0e-10


def test_reliability_boundary_withholds_substantive_grade() -> None:
    assert MODULE.bounded_evidence_grade(4, 4, 0, 0, 1) == "RELIABILITY_LIMITED_UNRESOLVED"
    assert MODULE.bounded_evidence_grade(0, 0, 0, 0, 8) == "RELIABILITY_LIMITED_UNRESOLVED"


def test_mixed_directions_are_not_misclassified_as_add_or_drop_signal() -> None:
    assert MODULE.bounded_evidence_grade(1, 0, 1, 0, 0) == "MIXED_DIRECTION_UNRESOLVED"
    assert MODULE.bounded_evidence_grade(0, 1, 0, 1, 0) == "MIXED_DIRECTION_UNRESOLVED"


def test_unbounded_grade_rules_remain_explicit() -> None:
    assert MODULE.bounded_evidence_grade(3, 1, 0, 0, 0) == "ROBUST_GAIN"
    assert MODULE.bounded_evidence_grade(1, 0, 0, 0, 0) == "REDUNDANT_SIGNAL"
    assert MODULE.bounded_evidence_grade(0, 1, 0, 0, 0) == "SYNERGISTIC_SIGNAL"
    assert MODULE.bounded_evidence_grade(0, 0, 1, 0, 0) == "HARMFUL_UNDER_OOD"
    assert MODULE.bounded_evidence_grade(0, 0, 0, 0, 0) == "NO_GAIN"
