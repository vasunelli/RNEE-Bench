import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "44_audit_eved_rnee_differential.py"
SPEC = importlib.util.spec_from_file_location("differential_audit_differential", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalize_source_name_only_rewrites_legacy_prefix():
    assert MODULE.normalize_source_name("eVED_171101_week.csv") == "VED_171101_week.csv"
    assert MODULE.normalize_source_name("VED_171101_week.csv") == "VED_171101_week.csv"


def test_binary_confusion_conserves_rows():
    old = np.array([False, False, True, True])
    new = np.array([False, True, False, True])
    result = MODULE.binary_confusion(old, new)
    assert result == {
        "old_absent_new_absent": 1,
        "old_absent_new_present": 1,
        "old_present_new_absent": 1,
        "old_present_new_present": 1,
    }


def test_g9_decision_separates_source_and_benchmark_failures():
    assert MODULE.decide_g9([], []) == "RNEE_RESULTS_VALID"
    assert MODULE.decide_g9([], ["support"]) == "RNEE_DATASET_ONLY"
    assert MODULE.decide_g9(["alignment"], []) == "RNEE_NO_GO"


def test_haversine_is_zero_for_identical_points():
    distance = MODULE.haversine_m(
        np.array([42.0]), np.array([-83.0]), np.array([42.0]), np.array([-83.0])
    )
    assert distance[0] == 0.0


def test_strict_integer_id_rejects_null_and_fractional_keys():
    with pytest.raises(ValueError):
        MODULE.strict_integer_id(pd.Series([1, None]), "key")
    with pytest.raises(ValueError):
        MODULE.strict_integer_id(pd.Series([1.25, 2.0]), "key")


def test_observation_alignment_is_key_based_not_position_based():
    old = pd.DataFrame({"VehId": [1, 2], "Trip": [10, 20], "Timestamp(ms)": [0, 0], "legacy": [5, 6]})
    new = pd.DataFrame({"VehId": [2, 1], "Trip": [20, 10], "Timestamp(ms)": [0, 0], "rnee": [60, 50]})
    aligned, unmatched = MODULE.align_observation_rows(old, new, ["VehId", "Trip", "Timestamp(ms)"])
    assert unmatched.empty
    assert len(aligned) == 2
    assert dict(zip(aligned["legacy"], aligned["rnee"])) == {5: 50, 6: 60}


def test_observation_alignment_rejects_duplicate_keys():
    old = pd.DataFrame({"VehId": [1, 1], "Trip": [10, 10], "Timestamp(ms)": [0, 0]})
    new = pd.DataFrame({"VehId": [1], "Trip": [10], "Timestamp(ms)": [0]})
    with pytest.raises(ValueError):
        MODULE.align_observation_rows(old, new, ["VehId", "Trip", "Timestamp(ms)"])


def test_exact_split_gate_rejects_extra_or_failed_checks():
    valid = {
        "status": "PASS_BENCHMARK_SPLITS",
        "checks_total": 135,
        "checks_passed": 135,
        "checks_failed": 0,
        "failed_checks": [],
        "family_gates": [{"leakage_status": "PASS", "failed_checks": []} for _ in range(6)],
    }
    assert MODULE.exact_split_gate(valid, "PASS_BENCHMARK_SPLITS")
    invalid = {**valid, "checks_total": 136}
    assert not MODULE.exact_split_gate(invalid, "PASS_BENCHMARK_SPLITS")


def test_exact_segment_gate_requires_noop_and_all_pass():
    valid = {"status": "PASS_EXPECTED", "resume_noop_verified": True, "checks": {"a": "PASS"}}
    assert MODULE.exact_segment_gate(valid, "PASS_EXPECTED")
    assert not MODULE.exact_segment_gate({**valid, "resume_noop_verified": False}, "PASS_EXPECTED")
