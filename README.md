# RNEE-Bench

RNEE-Bench is a VED-native, quality-controlled, leakage-controlled road-network-semantic benchmark for vehicle energy estimation. This repository now contains both the compact reported-results package and the complete frozen public data/model release used to verify and reproduce the RNEE construction and theory-validation workflow.

The accompanying study is **“When Do Historical Road Attributes Improve Fuel-Use Estimation in Ann Arbor, Michigan?”** Its conclusions are predictive, not causal. The operational fuel-volume target is predominantly MAF-derived and was not independently validated against physical fuel-flow measurements.

## Download

The repository itself is lightweight. The complete **5.1 GB** frozen data/model payload is distributed as three split 7-Zip assets in the [`v0.1.0-full` GitHub Release](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.1.0-full):

```text
RNEE-Bench-full-artifacts-v0.1.0.7z.001  1,992,294,400 bytes
RNEE-Bench-full-artifacts-v0.1.0.7z.002  1,992,294,400 bytes
RNEE-Bench-full-artifacts-v0.1.0.7z.003  1,118,597,247 bytes
```

Download all three volumes and the checksum file into one directory, then extract the `.001` volume at the repository root:

```bash
git clone https://github.com/vasunelli/RNEE-Bench.git
cd /path/to/downloaded/volumes
sha256sum -c /path/to/RNEE-Bench/release/RNEE-Bench-full-artifacts-v0.1.0-SHA256SUMS.txt
cd /path/to/RNEE-Bench
7z x /path/to/RNEE-Bench-full-artifacts-v0.1.0.7z.001
python scripts/verify_full_release.py
```

See [`release/RNEE-Bench-full-artifacts-v0.1.0-README.md`](release/RNEE-Bench-full-artifacts-v0.1.0-README.md) for Windows extraction instructions. Users who only need reported tables, figures, code, configs and documentation do not need the Release assets.

## Included data and artifacts

| Component | Public contents | Frozen scale |
|---|---|---:|
| TRAJECTORY_BUILD map matching | 54 matched-point and 54 matched-edge Parquet partitions | 22,436,808 points; 2,069,055 edge records |
| TRAJECTORY_BUILD row-enriched trajectories | 54 Parquet partitions, 121 columns | 22,436,808 rows |
| SEGMENT_SPLIT segments | global and partitioned 60-s segment products plus row-to-segment assignments | 301,219 segments; 256,419 QA-valid; 219,384 prediction-usable |
| SEGMENT_SPLIT splits | six assignment families, membership files, split cards and leakage checks | 135/135 checks passed |
| THEORY_VALIDATION models | frozen HistGradientBoosting model objects and JSON sidecars | 192 `.joblib` models |
| THEORY_VALIDATION predictions | paired-seed, ensemble and complete negative-control Parquet products | 201,318 paired rows; 67,106 ensemble rows; 601,722 control rows |
| Current RPM-only rerun | test predictions, bootstrap draws, effects and fit manifest | 43,143 test-prediction rows |
| Pipeline | construction, segmentation, split, model, inference, control and rerun code/config/tests | RUNTIME_FREEZE–ROBUSTNESS and current RPM-only branch |

Machine-readable per-artifact sizes, SHA-256 values and license groups are in [`metadata/full_release_inventory.csv`](metadata/full_release_inventory.csv). Aggregate Parquet schemas and row counts are in [`metadata/parquet_schemas.json`](metadata/parquet_schemas.json). The three browser-upload volumes have separate checksums under `release/`.

## Repository map

- `data/map_matching/`: matched points and matched edges by VED weekly partition.
- `data/row_enriched/`: full row-level telemetry joined to historical road-network context.
- `data/segments/`: complete SEGMENT_SPLIT 60-s segment release and row-to-segment assignments.
- `data/splits/`: leakage-controlled split assignments, memberships, cards and checks.
- `models/theory_validation/`: frozen THEORY_VALIDATION trained model objects and model metadata.
- `predictions/theory_validation/`: paired predictions, ensemble predictions and negative controls.
- `predictions/phase1/`: current RPM-only prediction branch.
- `src/rnee_build/`: end-to-end VED/OSM/Valhalla construction and benchmark builders.
- `scripts/rnee_theory_validation/`, `scripts/rnee_robustness/`, `scripts/rnee_phase1/`: training, inference, controls and reruns.
- `configs/` and `tests/`: frozen workflow configuration and contract tests.
- `results/`, `figure_data/`, `figures/`, `config/`, `expected/`: compact reported-results package.

See the [dataset card](docs/dataset_card.md), [model card](docs/model_card.md), and [full pipeline guide](docs/full_reproduction.md) before using the release.

## Verification

Compact reported-result verification requires only Python 3.10+:

```bash
python scripts/verify_reported_results.py
```

After extracting all Release volumes at the repository root, full release integrity verification hashes approximately 5.1 GB:

```bash
python scripts/verify_full_release.py
```

Schema regeneration additionally requires PyArrow:

```bash
python scripts/export_parquet_schemas.py
```

## Rebuild and rerun

The frozen environment contract is in `configs/rnee_build/requirements-rnee.lock.txt`. The full workflow is stage-oriented rather than a one-command scientific rerun: acquire and hash VED/OSM inputs, build the Valhalla network, map-match, enrich rows, segment, split, validate leakage, train, infer, run negative controls, and verify the frozen release. Exact commands and quality criteria are documented in [`docs/full_reproduction.md`](docs/full_reproduction.md).

The checked-in configs preserve the historical source layout as provenance. Copy them to a working directory and change root paths for a new machine; do not edit the frozen release in place. A new execution is a reproduction attempt, not the already-frozen evidence.

## Important boundaries

- `VehId`, `Trip`, road/route/entity IDs, exact latitude/longitude and spatial grid identifiers are retained only for construction, grouping, leakage checks and reproducibility. They are forbidden model features.
- The release contains de-identified public VED trajectories and derived OSM products, but still exposes precise historical coordinates and timestamps. Treat them as research data and do not attempt re-identification.
- Road-network attributes are pipeline-assigned historical context. They do not establish causal effects or external-fleet transfer.
- THEORY_VALIDATION `.joblib` files use Python object serialization. Load only artifacts from this repository after verifying SHA-256; never load untrusted Joblib/Pickle files.
- The current RPM-only branch preserved predictions and a complete fit manifest but did **not** serialize fitted model binaries. No model was refit merely to fill that archival gap. The available frozen trained binaries are the THEORY_VALIDATION models.
- Raw VED archives, raw OSM PBFs, Valhalla tiles/responses, software wheels and local execution caches are intentionally excluded. Their exact upstream contracts and hashes are documented in the configs and provenance files.

## Licensing and attribution

- Software (`scripts/`, `src/`, `tests/`): [MIT](LICENSE).
- Original documentation, compact result tables, figure data and figures: [CC BY 4.0](LICENSE-CONTENT.md).
- RNEE database products in `data/` and prediction tables in `predictions/`: [ODbL 1.0 release notice](LICENSE-DATA.md), with required attribution to © OpenStreetMap contributors and the upstream VED project.
- Upstream VED material remains under Apache-2.0. See [`third_party/NOTICE.md`](third_party/NOTICE.md) for source commit and notices.

Licenses apply by component; no third-party rights are relicensed by the MIT or CC BY notices.
