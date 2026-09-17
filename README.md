# RNEE-Bench

Data and code accompanying **Historical Road Attributes for Real-World Vehicle Fuel-Use Estimation: Dependence on Telemetry, Deployment Conditions, and Accounting Scale**.

The study evaluates the added predictive value of historical road attributes using 112,787 nominal 60-s segments from 197 ICE vehicles and 14,647 naturalistic trips. Comparisons hold vehicle inputs and evaluation observations fixed while adding road attributes, across five evaluation settings and several fuel-accounting scales.

## Paper resources

| Resource | Location |
|---|---|
| Tables 1–4 | [tables/](tables/README.md) |
| Figures 1–11 | [figures/](figures/README.md) |
| Full-precision results | [results/](results/README.md) |
| Figure source data | [figure_data/](figure_data/README.md) |
| Predictor sets, evaluation settings and analysis specifications | [config/](config/) |
| Data and reproduction instructions | [docs/reproduction.md](docs/reproduction.md) |

## Quick start

Use Python 3.11 or later. Install [7-Zip](https://www.7-zip.org/) or a compatible `7zz` command for archive extraction.

```bash
python -m pip install -r requirements.txt
python scripts/verify_resources.py
python scripts/download_data.py
python scripts/reproduce_results.py
```

The download command verifies the fixed [v0.2.0-energy data archive](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.2.0-energy) and installs its unchanged contents under `data/energy/`. The result script recalculates metrics and intervals from saved predictions and resampling records. It does not retrain models. Full raw-data reconstruction and current-study training are separate from this saved-result workflow; see [source and code availability](docs/sources.md).

The target is OBD-derived fuel volume, predominantly MAF-based, without independent fuel-flow validation. Findings describe prediction within the evaluated data, not causal infrastructure effects or external-fleet transfer. The fixed-HGB primary unseen-vehicle gain of 5.07% remains separate from the 5.09% independent-selection sensitivity. See [the analysis protocol](docs/protocol.md) for input sets, partitioning and uncertainty definitions.

Software uses [MIT](LICENSE); original figures, documentation and aggregated results use [CC BY 4.0](LICENSE-CONTENT.md); derived database and prediction products use [ODbL 1.0](LICENSE-DATA.md). Attribute RNEE-Bench, the Michigan Vehicle Energy Dataset and © OpenStreetMap contributors; see [upstream notices](NOTICE.md).
