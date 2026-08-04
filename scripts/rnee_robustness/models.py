from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    models_config: dict[str, Any],
    seed: int,
) -> Any:
    target = np.asarray(y_train, dtype=float)
    if model_name == "hgb":
        params = dict(models_config["hgb"]["params"])
        if params.get("loss") != "absolute_error":
            raise ValueError("ROBUSTNESS HGB requires absolute-error loss.")
        model = HistGradientBoostingRegressor(random_state=int(seed), **params)
    elif model_name == "ridge":
        params = dict(models_config["ridge"]["params"])
        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=bool(
                            models_config["ridge"]["preprocessing"][
                                "keep_empty_features"
                            ]
                        ),
                    ),
                ),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(**params)),
            ]
        )
    elif model_name == "catboost":
        params = dict(models_config["catboost"]["params"])
        params["random_seed"] = int(seed)
        model = CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unknown ROBUSTNESS model: {model_name}")
    model.fit(x_train, target)
    return model


def predict_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=float)


def dump_model(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump({"model": model, "metadata": metadata}, temporary, compress=3)
    temporary.replace(path)


def load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)
