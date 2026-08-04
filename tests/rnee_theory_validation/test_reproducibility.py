from __future__ import annotations

import numpy as np
import pandas as pd

from bootstrap import simultaneous_max_t_intervals
from config import stable_seed
from models import train_model


def test_model_predictions_are_reproducible_for_fixed_seed() -> None:
    rng = np.random.default_rng(189)
    x = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = x["a"].to_numpy() - 0.5 * x["b"].to_numpy()
    config = {
        "params": {
            "early_stopping": False,
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "loss": "absolute_error",
            "max_iter": 25,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
        }
    }
    first = train_model(x, y, config, seed=0).predict(x)
    second = train_model(x, y, config, seed=0).predict(x)
    assert np.array_equal(first, second)


def test_stable_seed_is_repeatable_and_context_specific() -> None:
    assert stable_seed(1, "cold_trip", "ICE") == stable_seed(
        1, "cold_trip", "ICE"
    )
    assert stable_seed(1, "cold_trip", "ICE") != stable_seed(
        1, "cold_month", "ICE"
    )


def test_simultaneous_interval_contains_point_estimates() -> None:
    point = np.array([0.03, 0.07, -0.02])
    bootstrap = np.array(
        [
            [0.02, 0.08, -0.01],
            [0.04, 0.06, -0.03],
            [0.03, 0.075, -0.025],
            [0.025, 0.065, -0.015],
        ]
    )
    lower, upper, critical = simultaneous_max_t_intervals(
        point, bootstrap, alpha=0.05
    )
    assert critical >= 0
    assert np.all(lower <= point)
    assert np.all(upper >= point)
