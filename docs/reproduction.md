# Reproduce the reported results

The repository contains the current manuscript's tables, figures, analysis specifications and result-calculation code. Detailed predictions, selection inputs, memberships and resampling arrays are distributed in the immutable v0.2.0-energy archive.

## 1. Environment and compact resources

Use Python 3.11 or later and install the pinned packages:

```bash
python -m pip install -r requirements.txt
python scripts/verify_resources.py
```

The compact check uses only the standard library. It checks public-file hashes, all five primary evaluation settings, the displayed Table 3 and Table 4 values, figure coverage and analysis specifications. It does not require the large archive.

## 2. Download the scientific data

```bash
python scripts/download_data.py
```

The command downloads the archive identified in `data/release.json`, checks its complete SHA-256 and byte count, extracts it with 7-Zip, and checks every packaged file. It installs the 287-file payload under `data/energy/`. An existing valid installation is reused; an existing incomplete or modified installation is not overwritten.

An archive already downloaded from the release can be used offline:

```bash
python scripts/download_data.py --archive /path/to/RNEE-Bench-energy-v0.2.0.7z
```

Use `--data-dir /path/to/energy` to install the payload elsewhere. Pass the same location to the result script. Paths with spaces should be quoted. The archive retains its published `energy/` layout internally; it should not be extracted over the repository root.

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

The above commands reproduce calculations from saved evidence. They are not a complete raw-VED-to-all-current-models training pipeline. Raw reconstruction and fitting additionally require upstream data, complete feature cohorts and the corresponding model environments. The three pinned packages in `requirements.txt` cover saved-result calculations only. [Source and code availability](sources.md) describes the separately preserved construction resources.
