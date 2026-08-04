from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _cast_like(values: np.ndarray, dtype: Any) -> pd.Series:
    if pd.api.types.is_bool_dtype(dtype):
        return pd.Series(values > 0.5, dtype=dtype)
    if pd.api.types.is_integer_dtype(dtype):
        return pd.Series(np.rint(values), dtype=dtype)
    return pd.Series(values, dtype=dtype)


def generate_noise_road(
    road: pd.DataFrame,
    train_reference: pd.DataFrame | None = None,
    seed: int = 0,
    zero_variance_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate equal-dimensional noise using train-only column moments.

    The output preserves row order, column order, dtypes, and each recipient
    cell's missingness mask. No target values are accepted by this API.
    """
    reference = road if train_reference is None else train_reference
    if list(reference.columns) != list(road.columns):
        raise ValueError("Noise reference and recipient road columns differ.")
    rng = np.random.default_rng(seed)
    output = road.copy(deep=True)
    statistics: list[dict[str, Any]] = []
    for column in road.columns:
        train_values = pd.to_numeric(reference[column], errors="coerce").to_numpy(
            dtype=float
        )
        finite = train_values[np.isfinite(train_values)]
        mean = float(finite.mean()) if finite.size else 0.0
        observed_std = float(finite.std(ddof=0)) if finite.size else 0.0
        scale = observed_std if observed_std > 0 else float(zero_variance_scale)
        generated = rng.normal(mean, scale, size=len(road))
        missing_mask = road[column].isna().to_numpy()
        generated[missing_mask] = np.nan
        output[column] = _cast_like(generated, road[column].dtype).to_numpy()
        statistics.append(
            {
                "column": column,
                "train_observed_count": int(finite.size),
                "train_mean": mean,
                "train_std": observed_std,
                "generation_scale": scale,
                "recipient_missing_count": int(missing_mask.sum()),
                "dtype": str(road[column].dtype),
            }
        )
    if not output.isna().equals(road.isna()):
        raise RuntimeError("Noise control failed recipient missing-mask preservation.")
    if [str(dtype) for dtype in output.dtypes] != [
        str(dtype) for dtype in road.dtypes
    ]:
        raise RuntimeError("Noise control failed dtype preservation.")
    metadata = {
        "method": "independent_gaussian_from_train_column_moments",
        "seed": int(seed),
        "rows": int(len(road)),
        "columns": int(road.shape[1]),
        "recipient_missing_mask_exact": True,
        "statistics": statistics,
    }
    return output, metadata


def _sattolo_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("Whole-block derangement requires at least two rows.")
    values = np.arange(size)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        values[index], values[other] = values[other], values[index]
    if np.any(values == np.arange(size)):
        raise RuntimeError("Sattolo permutation unexpectedly contains a fixed point.")
    return values


def generate_permuted_road(
    road: pd.DataFrame,
    groups: pd.Series,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Permute the complete R6 row block within each pre-frozen split role."""
    if len(groups) != len(road):
        raise ValueError("Permutation groups must align one-to-one with road rows.")
    group_values = groups.astype(str).reset_index(drop=True)
    source = road.reset_index(drop=True)
    output = source.copy(deep=True)
    rng = np.random.default_rng(seed)
    mapping_rows: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    for group in sorted(group_values.unique()):
        recipient_positions = np.flatnonzero(group_values.to_numpy() == group)
        local_permutation = _sattolo_indices(len(recipient_positions), rng)
        source_positions = recipient_positions[local_permutation]
        output.iloc[recipient_positions] = source.iloc[source_positions].to_numpy()
        mapping_rows.extend(
            {
                "recipient_position": int(recipient),
                "source_position": int(source_position),
                "permutation_group": group,
            }
            for recipient, source_position in zip(
                recipient_positions, source_positions, strict=True
            )
        )
        group_records.append(
            {
                "group": group,
                "rows": int(len(recipient_positions)),
                "fixed_points": int(
                    np.sum(recipient_positions == source_positions)
                ),
            }
        )
    mapping = pd.DataFrame(mapping_rows).sort_values(
        "recipient_position"
    ).reset_index(drop=True)
    if [str(dtype) for dtype in output.dtypes] != [
        str(dtype) for dtype in source.dtypes
    ]:
        raise RuntimeError("Permutation failed dtype preservation.")
    if any(record["fixed_points"] for record in group_records):
        raise RuntimeError("Permutation control contains fixed points.")
    metadata = {
        "method": "whole_R6_block_sattolo_derangement_within_split_role",
        "seed": int(seed),
        "rows": int(len(source)),
        "columns": int(source.shape[1]),
        "independent_column_shuffle": False,
        "cross_group_moves": 0,
        "fixed_points": 0,
        "groups": group_records,
    }
    return output, mapping, metadata
