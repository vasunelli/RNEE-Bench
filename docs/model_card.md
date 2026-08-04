# Frozen model and prediction card

## THEORY_VALIDATION model family

`models/theory_validation/` contains 192 frozen `HistGradientBoostingRegressor` Joblib objects covering four OOD environments, ICE/HEV operational targets, low/high sensor regimes, road-free/real-road/noise-road/permuted-road systems, and three fixed seeds where specified by the THEORY_VALIDATION protocol.

The experiment uses an L1 loss, fixed hyperparameters, train-only variance filtering and a shared Real-Road train-active mask across Real, Noise and Permutation Road controls within a sensor regime. The THEORY_VALIDATION branch is technical theory-validation evidence, not independent confirmation.

## Predictions

- `predictions/theory_validation/paired_test_predictions/`: 24 per-seed evaluation Parquet files, 201,318 total rows.
- `predictions/theory_validation/ensemble_test_predictions/`: eight frozen ensemble Parquet files, 67,106 total rows.
- `predictions/theory_validation/null_controls/`: full negative-control predictions plus mapping/metadata products; eight main files and 601,722 main prediction rows.
- `predictions/phase1/rpm_only_test_predictions/`: four current RPM-only evaluation files, 43,143 total rows.
- `predictions/phase1/rpm_only_bootstrap_draws.parquet`: frozen current bootstrap draws.

## Current RPM-only archival boundary

The current RPM-only analysis serialized its predictions, active masks, fit manifest, effects and inference draws, but its runner did not serialize fitted estimator objects. Those model binaries therefore do not exist in the frozen run. They are not reconstructed here because refitting would create new evidence rather than recover the original object.

## Safe loading

Joblib uses Python object serialization and may execute code during loading. Verify `metadata/full_release_inventory.csv`, use the pinned environment and load only trusted artifacts from this repository. Example:

```python
from pathlib import Path
import hashlib
import joblib

path = Path("models/theory_validation/cold_trip__ICE_fuel_L/H_R_real/seed_0.joblib")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == "<value from metadata/full_release_inventory.csv>"
model = joblib.load(path)
```

## Intended use and limitations

The models are reproducibility artifacts, not deployment-ready estimators. They do not accept identifiers, exact coordinates or route/edge IDs as predictors. Performance and semantic-specificity statements are limited to the verified VED-derived populations, operational targets and frozen OOD assignments. This release does not support causal, external-fleet or safety-critical claims.
