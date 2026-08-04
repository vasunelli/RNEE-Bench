# Full reconstruction from public inputs

The compact package verifies every reported number and recreates the reported table and figure structures without raw data. End-to-end model refitting requires the original construction workflow plus the official inputs below; that workflow is too large to duplicate here without weakening the compact package. `scripts/run_full_analysis.py` is therefore a strict preflight entry point, not a hidden substitute for refitting.

## Required inputs

Use the layout in `data_access.md`. Required vehicle inputs are official VED Dynamic and Static Data at commit `6baa4963782d515a67d32a5490bd5d11f5d9bf0d`. Required road inputs are Michigan OSM PBF snapshots dated 2017-01-01 and 2018-01-01. Record SHA-256 hashes for every raw file.

Dynamic data must supply vehicle and trip identifiers, timestamps, GPS position, vehicle speed, ambient temperature, engine RPM, MAF, Fuel Rate, and available short- and long-term fuel-trim channels. Static data must supply vehicle identifier and EngineType. Missing channels remain missing; do not synthesize target inputs. Exact latitude/longitude, vehicle identifiers, trip identifiers, edge IDs, and route IDs are prohibited predictors.

## Preflight

```bash
python scripts/run_full_analysis.py --data-root /path/to/DATA_ROOT --check-only
```

The preflight checks the VED commit marker, required directories, OSM snapshot names, and availability of an external workflow root. When the complete workflow is available:

```bash
python scripts/run_full_analysis.py \
  --data-root /path/to/DATA_ROOT \
  --workflow-root /path/to/complete-rnee-workflow \
  --check-only
```

The external workflow must expose the twelve stages below through documented commands or a workflow manifest. This compact repository deliberately does not claim to execute stages it does not contain.

## Required stages and gates

1. Inventory and hash all official VED dynamic/static inputs.
2. Reconstruct the operational target from direct Fuel Rate or MAF plus available trim correction.
3. Audit target units, sign, missingness, source exclusivity, and prediction-feature exclusions.
4. Select the dated OSM snapshot with no future-map fallback.
5. Map-match with Valhalla 3.7.0, auto costing, 90-m search, and 1,000-m trace breaks.
6. Build the 142 candidate road attributes while excluding exact entity and route identifiers.
7. Construct non-overlapping nominal 60-s segments before any split.
8. Apply the four transfer assignments in `config/transfer_populations.json`; verify zero segment/trip overlap across roles.
9. Retain road attributes using training data only and build the two telemetry regimes in `config/feature_sets.json`.
10. Fit HGB, Ridge, and CatBoost under `config/main_analysis.json`; keep calibration and evaluation data out of fitting and selection.
11. Apply 20 matched-noise and 20 reassigned-road replicates per primary comparison; save evaluation predictions.
12. Recompute vehicle-cluster bootstrap max-t intervals, tables, diagnostics, and figures; then run `scripts/verify_reported_results.py` against the exported compact files.

## Completion criteria

Full reconstruction is complete only when the source inventory hashes are frozen, all construction and leakage gates pass, saved evaluation predictions reproduce the public effects within numeric tolerance, and the compact verifier exits with status 0. No new feature selection, refitting choice, post-evaluation tuning, or scientific interpretation is authorized by these instructions.
