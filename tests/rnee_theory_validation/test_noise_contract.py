from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import null_control_contract_pass
from null_controls import generate_noise_road


def test_noise_preserves_rows_columns_dtypes_and_recipient_missing_mask() -> None:
    road = pd.DataFrame(
        {
            "a": pd.Series([1.0, np.nan, 3.0, 4.0], dtype="float64"),
            "b": pd.Series([2.0, 3.0, np.nan, 5.0], dtype="float32"),
        }
    )
    train = road.iloc[:3].copy()
    generated, metadata = generate_noise_road(
        road, train_reference=train, seed=189, zero_variance_scale=1.0
    )
    assert generated.shape == road.shape
    assert list(generated.columns) == list(road.columns)
    assert [str(value) for value in generated.dtypes] == [
        str(value) for value in road.dtypes
    ]
    assert generated.isna().equals(road.isna())
    assert metadata["seed"] == 189
    assert metadata["recipient_missing_mask_exact"] is True


def test_noise_is_reproducible_and_seed_sensitive() -> None:
    road = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 20, dtype=np.float64),
            "b": np.linspace(1, 2, 20, dtype=np.float64),
        }
    )
    first, _ = generate_noise_road(road, road.iloc[:10], seed=1)
    second, _ = generate_noise_road(road, road.iloc[:10], seed=1)
    third, _ = generate_noise_road(road, road.iloc[:10], seed=2)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(third)


def test_zero_violation_counts_are_success_not_falsey_failures() -> None:
    metadata = {
        "contracts": {
            "same_rows": True,
            "same_columns": True,
            "same_dtypes": True,
            "noise_recipient_missing_mask_exact": True,
            "permutation_role_block_multiset_exact": True,
            "permutation_cross_role_moves": 0,
            "permutation_fixed_points": 0,
        }
    }
    assert null_control_contract_pass(metadata)
