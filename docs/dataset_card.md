# RNEE-Bench dataset card

## Purpose

RNEE-Bench supports audited research on road-network-semantic vehicle energy estimation under leakage-controlled out-of-distribution splits. It is not a fleet-monitoring product, a causal road-infrastructure study, or evidence of transfer to unobserved fleets or vehicle classes.

## Lineage

The only L0 vehicle inputs are the official VED Dynamic and Static Data at frozen commit `6baa4963782d515a67d32a5490bd5d11f5d9bf0d`. Historical Michigan OSM snapshots dated 2017-01-01 and 2018-01-01 supply road-network context. Valhalla 3.7.0 performs trace matching. eVED fields are not used as new-dataset source columns.

The public release does not contain the upstream archives, raw OSM PBFs, Valhalla tiles or response cache. Their filenames, URLs and SHA-256 contracts are in `configs/rnee_build/base.yaml`.

## Public layers

### Map matching

`data/map_matching/matched_points/partition_0000.parquet` through `partition_0053.parquet` contain the full 22,436,808-point matched trajectory product. `matched_edges/` contains 2,069,055 edge records. Point fields include raw and matched coordinates, trace/chunk indices, match state and Valhalla response identity.

### Row-enriched trajectories

`data/row_enriched/` contains 54 weekly Parquet partitions totaling 22,436,808 rows and 121 columns. It combines source telemetry, target inputs, match state, matched coordinates, edge semantics, historical speed-limit provenance, node/network measures, and distance-based context-object features.

### Segments

`data/segments/segments_all.parquet` is the canonical 301,219-row non-overlapping nominal 60-s dataset. `segments_qa_valid.parquet` contains 256,419 QA-valid segments and `segments_prediction_usable.parquet` contains 219,384 prediction-usable segments. Partitioned segment files and row-to-segment assignments are also included to make row-level reconstruction auditable.

### Splits

`data/splits/` contains random-trip-blocked, cold-vehicle, cold-trip, cold-month, cold-spatial and cold-functional-road-class assignments. All 135 frozen construction/leakage checks passed. Split cards state exclusions and stress-test boundaries.

## Target

The released segment target is an operational fuel-volume target: valid direct Fuel Rate where available, otherwise MAF with available trim correction under the frozen contract. It is not a laboratory fuel-meter reference. Fuel and battery targets are not pooled.

## Identifiers and location fields

Vehicle/trip identifiers, exact raw/matched coordinates, edge identity and spatial grid fields are retained for construction lineage, grouping, leakage checks and geographic verification. They are forbidden model features. Do not use the release for re-identification or tracking of individuals.

## Quality boundaries

- Map matching: 32,552 trips/requests, three failed requests; the production gate passed with explicit abstention handling.
- Row assembly: 40 abstained trips and 40,999 abstention rows are preserved rather than fabricated.
- Segmentation: zero overlapping segments and zero duplicate segment IDs in the frozen gate.
- Splits: test/calibration/validation/train roles are separated; preprocessing and feature activity must be fit using training data only.
- `cold_spatial` is a strong stress test and should not be interpreted as a generic external transfer estimate.
- `cold_functional_road_class` is a frozen functional-road-class transfer definition; consult its split card before comparison.

## Schemas and integrity

`metadata/parquet_schemas.json` records file counts, total rows, representative schemas and schema hashes. `metadata/full_release_inventory.csv` records every released artifact's byte count and SHA-256.

## License

Database products are distributed under the component notice in `LICENSE-DATA.md`. Preserve both VED and OpenStreetMap attribution when redistributing or publishing adapted databases.
