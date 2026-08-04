from __future__ import annotations

import numpy as np
import pandas as pd

from null_controls import generate_permuted_road


def test_whole_block_permutation_preserves_role_multiset_and_has_no_fixed_points() -> None:
    road = pd.DataFrame(
        {
            "a": np.arange(12, dtype=np.float64),
            "b": np.arange(12, dtype=np.float64) * 100 + 7,
            "c": [np.nan if index % 3 == 0 else float(index) for index in range(12)],
        }
    )
    groups = pd.Series(["train"] * 6 + ["test"] * 6)
    permuted, mapping, metadata = generate_permuted_road(road, groups, seed=189)
    assert list(permuted.columns) == list(road.columns)
    assert [str(value) for value in permuted.dtypes] == [
        str(value) for value in road.dtypes
    ]
    assert (mapping["recipient_position"] != mapping["source_position"]).all()
    assert (mapping["permutation_group"].to_numpy() == groups.to_numpy()).all()
    assert metadata["independent_column_shuffle"] is False
    assert metadata["cross_group_moves"] == 0
    for row in mapping.itertuples(index=False):
        pd.testing.assert_series_equal(
            permuted.iloc[row.recipient_position],
            road.iloc[row.source_position],
            check_names=False,
        )
    for group in ("train", "test"):
        mask = groups.eq(group)
        original_hash = np.sort(
            pd.util.hash_pandas_object(
                road.loc[mask].reset_index(drop=True), index=False
            ).to_numpy()
        )
        permuted_hash = np.sort(
            pd.util.hash_pandas_object(
                permuted.loc[mask].reset_index(drop=True), index=False
            ).to_numpy()
        )
        assert np.array_equal(original_hash, permuted_hash)


def test_permutation_is_reproducible() -> None:
    road = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2})
    groups = pd.Series(["train"] * 10 + ["test"] * 10)
    first, first_mapping, _ = generate_permuted_road(road, groups, seed=7)
    second, second_mapping, _ = generate_permuted_road(road, groups, seed=7)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_mapping, second_mapping)
