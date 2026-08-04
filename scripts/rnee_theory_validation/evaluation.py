from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


SENSORS = ("low", "high")
ROAD_VARIANTS = ("real", "noise", "permutation")
SYSTEM_TO_PREDICTION = {
    "L-R0": "prediction_l_r0",
    "L-R-real": "prediction_l_r_real",
    "L-R-noise": "prediction_l_r_noise",
    "L-R-perm": "prediction_l_r_perm",
    "H-R0": "prediction_h_r0",
    "H-R-real": "prediction_h_r_real",
    "H-R-noise": "prediction_h_r_noise",
    "H-R-perm": "prediction_h_r_perm",
}


def regression_metrics(
    y_true: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    target = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    residual = target - predicted
    absolute = np.abs(residual)
    denominator = float(np.square(target - np.mean(target)).sum())
    return {
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(np.square(residual).mean())),
        "medae": float(np.median(absolute)),
        "r2": (
            float(1.0 - np.square(residual).sum() / denominator)
            if denominator > 0
            else math.nan
        ),
    }


def evaluate_system(
    frame: pd.DataFrame,
    prediction_column: str,
    target_column: str = "target_l",
) -> dict[str, Any]:
    result = regression_metrics(
        frame[target_column].to_numpy(float),
        frame[prediction_column].to_numpy(float),
    )
    result.update(
        {
            "sample_count": int(len(frame)),
            "vehicle_count": int(frame["VehId"].astype(str).nunique()),
            "trip_count": int(frame["trip_uid"].astype(str).nunique()),
        }
    )
    return result
