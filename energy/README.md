# Data and code for the Energy manuscript

**Historical Road Attributes for Real-World Vehicle Fuel-Use Estimation: Dependence on Telemetry, Deployment Conditions, and Accounting Scale**

This directory contains the reported result tables, manuscript figures, saved predictions, data partitions, estimator-selection inputs, kinematic variables and bootstrap resampling records. The analysis includes 112,787 nominal 60-s segments from 197 ICE vehicles and 14,647 trips.

## Use the resources

The Git repository contains tables, figures and the calculation script. Download the complete data archive from [v0.2.0-energy](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.2.0-energy) and extract it at the repository root to supply the remaining files under `energy/`.

```bash
7z x /path/to/RNEE-Bench-energy-v0.2.0.7z
python -m pip install -r energy/requirements.txt
python energy/reproduce_results.py
```

The script recalculates prediction metrics, estimator-selection scores, predictive-overlap contrasts, fuel-accounting quantities and intervals from saved predictions and resampling records. It does not retrain models. Full reconstruction and fitting additionally require the VED/OSM sources and the earlier repository's construction environment. Some historical intervals have saved endpoints but do not retain every resample matrix; these endpoints are supplied as reported results.

## Contents

| Directory | Contents |
|---|---|
| `tables/` | Tables 1–4, full-precision results and displayed values |
| `figures/` | The eleven manuscript figures and captions |
| `figure_data/` | Numerical data for the predictive-gain, overlap and accounting figures |
| `data/primary/` | Compact-telemetry and RPM-augmented fuel predictions |
| `data/membership/` | Evaluation partitions |
| `data/selection/` | Independent estimator-selection inputs, scores, selected models and predictions |
| `data/overlap/` | Kinematic variables, fuel, kinetic-energy and conditional-RPM predictions |
| `data/accounting/` | Common-cohort block definitions, prefix/tail memberships and accounting results |
| `data/bootstrap/` | Saved resampling weights and replicate statistics |
| `data/diagnostics/` | Loss sensitivity, spatial, target-conditioned and speed-conditioned results |
| `results/` | Road-pairing controls, model sensitivity and distribution diagnostics |

## Interpretation

Table 3 reports relative MAE reductions. The fixed-HGB unseen-vehicle result of 5.07% and the separate independent-selection result of 5.09% remain distinct. Table 4 instead reports percentage-point gains normalised by observed fuel volume. The one-, two-, four- and eight-segment and full-prefix scales share observations; complete trips restore tails.

The target is OBD-derived fuel volume, predominantly reconstructed from MAF. It was not independently validated against physical fuel-flow measurements. Findings describe predictive associations within the evaluated data. They do not establish causal infrastructure effects or external-fleet transfer. Vehicle/trip/segment identifiers support joins and clustering and are excluded from model predictors.

Software uses MIT; figures, documentation and aggregated results use CC BY 4.0; derived database and prediction products use ODbL 1.0. See the licence files and [upstream attribution](NOTICE.md).
