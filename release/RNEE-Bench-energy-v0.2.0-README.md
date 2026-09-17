# Energy manuscript data and code

Supporting resources for **Historical Road Attributes for Real-World Vehicle Fuel-Use Estimation: Dependence on Telemetry, Deployment Conditions, and Accounting Scale**.

Download the 7-Zip archive and SHA256SUMS file from this release. Verify the checksum before extraction.

Linux:

```bash
sha256sum -c RNEE-Bench-energy-v0.2.0-SHA256SUMS.txt
```

Windows PowerShell:

```powershell
Get-FileHash RNEE-Bench-energy-v0.2.0.7z -Algorithm SHA256
```

Compare the Windows result with the complete SHA-256 in the checksum file. At the repository root, extract and run:

```bash
7z x /path/to/RNEE-Bench-energy-v0.2.0.7z
python -m pip install -r energy/requirements.txt
python energy/reproduce_results.py
```

Python 3.11 or later is required. The archive supplies the complete `energy/` directory. The script calculates reported results from saved predictions and resampling records. Model fitting and raw VED/OSM reconstruction require the additional inputs and environments described in the repository documentation.

The archive includes result tables, figures, data partitions, estimator-selection inputs, predictions, kinematic variables and resampling records. Component licences and attribution are included.
