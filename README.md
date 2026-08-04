# RNEE-Bench reported-results package

This repository accompanies **“When Does Road Context Help Vehicle Fuel-Use Estimation? Transfer Boundaries Under Sparse Onboard Sensing.”** It provides a compact, machine-verifiable record of the reported numerical results, source data for Figures 3–6, publication figure PDFs, configuration contracts, and a documented route to full reconstruction from the public VED inputs.

The study evaluates whether pipeline-assigned historical road attributes improve prediction of an operational fuel-volume target under four transfer populations and two sparse telemetry regimes. The target is predominantly MAF-derived and was not independently validated against physical fuel-flow measurements. Results are predictive, not causal.

## Quick verification

From the repository root, use Python 3.10 or newer:

```bash
python scripts/verify_reported_results.py
```

No package installation is required: the verifier uses only the Python standard library. The included `requirements.txt` is intentionally dependency-free. The script checks the study counts; the eight primary effects and simultaneous intervals; all 320 correspondence-control draws and their summaries; model sensitivity; RPM and predictor-overlap diagnostics; map-matching counts; inference-family sizes; figure-data crosswalks; PDF presence; and path/privacy portability.

## Reproduce reported tables and figure structures

```bash
python scripts/reproduce_tables_and_figures.py
```

This creates `generated/tables/` and `generated/figures/` from the public CSV files. The generated SVGs reproduce the numerical content and panel structure; they are not intended to be pixel-identical to the publication PDFs in `figures/`.

## Repository contents

- `results/`: reported effects, controls, diagnostics, and construction summaries.
- `figure_data/`: machine-readable source data for Figures 3–6.
- `figures/`: publication PDFs for Figures 3–6.
- `config/`: feature, model, transfer-population, control, and inference contracts.
- `metadata/`: variable dictionary, training-retained road-attribute flags, and data contract.
- `docs/`: provenance, data-access, and full-reconstruction instructions.
- `expected/`: frozen values used by the independent verifier.

## Scope and claim boundary

The package reports 112,787 nominal 60-s ICE segments from 14,647 trips and 197 vehicles. Of these, 112,766 segments were MAF-only and 21 were direct-Fuel-Rate-only; MAF supplied 99.9814% of effective target duration. Absolute Load is excluded from the primary sensing comparison and is only an exploratory target-proxy sensitivity in the study.

No raw VED or OpenStreetMap data are redistributed. See [data access](docs/data_access.md), [provenance](docs/data_provenance.md), and [full reconstruction](docs/full_reproduction.md).

## License

- Software in `scripts/` and `requirements.txt`: [MIT License](LICENSE).
- Original result tables, figure data, figures, metadata, configuration and expected-result contracts, and documentation: [CC BY 4.0](LICENSE-CONTENT.md).
- Third-party source datasets and marks are not redistributed or relicensed; see [third-party notices](third_party/NOTICE.md).
