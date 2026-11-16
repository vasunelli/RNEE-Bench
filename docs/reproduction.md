# Reproduce the reported results

Use the scripts below to check the paper's tables and figures and recalculate results. The v0.2.0-energy archive contains predictions, estimator-selection inputs, memberships and resampling arrays.

## 1. Environment and compact resources

Use Python 3.11 or later and install the pinned packages:

```bash
python -m pip install -r requirements.txt
python scripts/verify_resources.py
```

The resource check uses Python's standard library. It checks file hashes, all five primary evaluation settings, the displayed values in Tables 3 and 4, figure coverage and analysis specifications. You can run it before downloading the archive.

## 2. Download the scientific data

```bash
python scripts/download_data.py
```

The command downloads the archive identified in `data/release.json`, checks its checksum and file size, extracts it with 7-Zip, and checks every packaged file. It installs 287 files in `data/energy/`. The script reuses a verified installation and refuses to overwrite a directory that fails verification.

An archive already downloaded from the release can be used offline:

```bash
python scripts/download_data.py --archive /path/to/RNEE-Bench-energy-v0.2.0.7z
```

Use `--data-dir /path/to/energy` to install the data elsewhere. Pass the same location to the result script. Paths with spaces should be quoted. The archive retains its published `energy/` layout internally; it should not be extracted over the repository root.

## 3. Recalculate reported quantities

```bash
python scripts/reproduce_results.py
```

For a separate data location:

```bash
python scripts/reproduce_results.py --data-dir /path/to/energy
```

The script checks the archived files before calculating primary prediction errors, estimator-selection criteria, predictive-overlap contrasts, aggregation identities, scale-specific gains and saved-replicate intervals. It uses the packaged memberships and resampling weights, with no network access, fitted-model loading, model training, new predictions or new resampling.

Some earlier primary, paired-RPM, target-conditioned and speed-conditioned interval endpoints are retained as reported outputs without every original resample matrix. They are checked as supplied results; missing resample matrices are not reconstructed. The script prints its results and leaves the scientific input files unchanged.

## Rebuilding and fitting

Rebuilding the dataset and refitting the models require the upstream data, complete feature cohorts and the corresponding model environments. The packages in `requirements.txt` are for recalculating results from saved predictions. [Source and code availability](sources.md) describes the separately preserved construction resources.
