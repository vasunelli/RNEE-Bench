from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def train_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> HistGradientBoostingRegressor:
    params = dict(config["params"])
    if params.get("loss") != "absolute_error":
        raise ValueError("THEORY_VALIDATION requires absolute-error loss.")
    model = HistGradientBoostingRegressor(random_state=int(seed), **params)
    model.fit(x_train, np.asarray(y_train, dtype=float))
    return model


def dump_model(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump({"model": model, "metadata": metadata}, temporary, compress=3)
    temporary.replace(path)


def load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)
