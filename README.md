# RNEE-Bench

Data and code for **Historical Road Attributes for Real-World Vehicle Fuel-Use Estimation: Dependence on Telemetry, Deployment Conditions, and Accounting Scale**.

The study evaluates how the predictive contribution of historical road attributes changes with telemetry, deployment conditions and fuel-accounting scale. It uses 112,787 nominal 60-s segments from 197 ICE vehicles and 14,647 trips in VED.

## Paper resources

- [Data and code guide](energy/README.md)
- [Tables 1–4](energy/tables/)
- [Figures 1–11](energy/figures/README.md)
- [Reported-result calculation script](energy/reproduce_results.py)
- [Complete data download](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.2.0-energy)

Download the archive and its SHA-256 checksum, extract it at the repository root, and run:

```bash
python -m pip install -r energy/requirements.txt
python energy/reproduce_results.py
```

The current paper's figures and tables are under `energy/`. Root-level `figures/`, `figure_data/` and `results/` retain the numbering of the earlier study. The [v0.1.0-full release](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.1.0-full) supplies the earlier 5.1 GB construction, prediction and model resources. Its data scale is broader than the ICE analysis cohort in this manuscript. See the [construction guide](docs/full_reproduction.md) for upstream data and environment requirements.

The fuel target is OBD-derived and predominantly MAF-based; it was not independently validated against physical fuel-flow measurements. Results concern prediction within the evaluated populations, not causal effects or external-fleet transfer.

Software uses [MIT](LICENSE); original figures, documentation and aggregated results use [CC BY 4.0](LICENSE-CONTENT.md); derived database and prediction products use [ODbL 1.0](energy/LICENSE-DATA.md). Attribute RNEE-Bench, the Michigan Vehicle Energy Dataset and © OpenStreetMap contributors; see [upstream notices](third_party/NOTICE.md).
