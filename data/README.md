# Scientific data

Run `python scripts/download_data.py` from the repository root. The command verifies the archive described in [release.json](release.json) and installs the data in `data/energy/`. Large data files are not tracked in Git.

The archive contains predictions, partition and trip/block memberships, estimator-selection inputs, kinematic variables, resampling arrays and figure data. The scripts verify the extracted files before calculating results. The installer reuses a verified copy and refuses to overwrite a directory that fails verification.

[checksums.json](checksums.json) lists the published checksums for paths relative to `data/energy/`.
