from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rnee_build" / "46_select_model_contract_backbone.py"
SPEC = importlib.util.spec_from_file_location("model_contract_m2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_deterministic_sample_is_stable_and_bounded() -> None:
    frame = pd.DataFrame({"segment_id": [f"s{i}" for i in range(100)], "value": np.arange(100)})
    first = MODULE.deterministic_sample(frame, 17, 42, "train")
    second = MODULE.deterministic_sample(frame.sample(frac=1, random_state=7), 17, 42, "train")
    assert len(first) == 17
    assert first["segment_id"].tolist() == second["segment_id"].tolist()


def test_regression_metrics_match_known_values() -> None:
    metrics = MODULE.regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0]))
    assert np.isclose(metrics["mae"], 2 / 3)
    assert np.isclose(metrics["rmse"], np.sqrt(2 / 3))
    assert np.isclose(metrics["medae"], 1.0)


def test_forbidden_hits_uses_frozen_contract() -> None:
    contract = {
        "forbidden_feature_contract": {
            "exact": ["segment_id"],
            "tokens": ["latitude", "split_role"],
        }
    }
    assert MODULE.forbidden_hits(["speed_mean", "segment_id", "matched_latitude"], contract) == [
        "segment_id",
        "matched_latitude",
    ]


def test_candidate_factory_rejects_unregistered_kind_without_importing_ml_backends() -> None:
    with pytest.raises(ValueError, match="Unsupported candidate kind"):
        MODULE.build_candidate("bad", {"kind": "unknown", "params": {}}, 1)


def test_train_active_columns_drops_constant_and_all_missing() -> None:
    frame = pd.DataFrame(
        {
            "varying": [0.0, 1.0, 2.0],
            "constant": [1.0, 1.0, 1.0],
            "missing": [np.nan, np.nan, np.nan],
            "mixed": [np.nan, 2.0, 3.0],
        }
    )
    assert MODULE.train_active_columns(frame) == ["varying", "mixed"]


def test_global_protection_has_zero_residual_overlap() -> None:
    development = pd.DataFrame(
        {
            "segment_id": ["train-safe", "cross-split-test", "validation-safe"],
            "split_role": ["train", "train", "validation"],
        }
    )
    selected, residual_overlap = MODULE.exclude_globally_protected_ids(
        development, {"cross-split-test", "other-protected"}
    )
    assert selected["segment_id"].tolist() == ["train-safe", "validation-safe"]
    assert residual_overlap == 0
    assert not selected["segment_id"].isin({"cross-split-test", "other-protected"}).any()
