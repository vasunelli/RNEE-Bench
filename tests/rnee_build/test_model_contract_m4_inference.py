from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "48_run_model_contract_m4_inference.py"
SPEC = importlib.util.spec_from_file_location("model_contract_m4", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bh_and_by_are_monotone_and_by_is_conservative() -> None:
    p = np.array([0.001, 0.02, 0.03, 0.5])
    bh = MODULE.fdr_adjust(p, "bh")
    by = MODULE.fdr_adjust(p, "by")
    assert np.all(by >= bh)
    order = np.argsort(p)
    assert np.all(np.diff(bh[order]) >= 0)
    assert np.all((0 <= bh) & (bh <= 1))


def test_conformal_qhat_uses_finite_sample_order_statistic() -> None:
    residuals = np.arange(1.0, 11.0)
    assert MODULE.conformal_qhat(residuals, 0.8) == 9.0
    assert MODULE.conformal_qhat(residuals, 0.95) == 10.0


def test_cluster_bootstrap_is_deterministic_and_constant_preserving() -> None:
    clusters = pd.Series(["a", "a", "b", "c", "c", "c"])
    first = MODULE.bootstrap_weights(clusters, 50, 42)
    second = MODULE.bootstrap_weights(clusters, 50, 42)
    assert np.array_equal(first[0], second[0])
    values = np.full(len(clusters), 3.5)
    means = MODULE.cluster_bootstrap_means(values, first[0], first[1], first[2])
    assert np.allclose(means, 3.5)


def test_bootstrap_summary_reports_direction_and_two_sided_p() -> None:
    replicates = np.linspace(0.1, 1.0, 2000)
    result = MODULE.bootstrap_summary(replicates, 0.5)
    assert result["ci_lower"] > 0
    assert 0 < result["p_value_two_sided"] < 0.01


def test_evidence_grade_rules_are_bounded() -> None:
    assert MODULE.evidence_grade(3, 1, 0, 0) == "ROBUST_GAIN"
    assert MODULE.evidence_grade(1, 0, 0, 0) == "REDUNDANT_SIGNAL"
    assert MODULE.evidence_grade(0, 1, 0, 0) == "SYNERGISTIC_SIGNAL"
    assert MODULE.evidence_grade(0, 0, 1, 0) == "HARMFUL_UNDER_OOD"
    assert MODULE.evidence_grade(0, 0, 0, 0) == "NO_GAIN"
