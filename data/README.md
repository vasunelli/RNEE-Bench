# Scientific data

Run `python scripts/download_data.py` from the repository root. The command verifies the archive described in [release.json](release.json) and installs the unchanged scientific payload in `data/energy/`. Large data files are not tracked in Git.

The payload contains predictions, partition and trip/block memberships, estimator-selection inputs, kinematic variables, resampling arrays and complete figure data. Its own checksums are verified before result calculations. Existing valid data are reused and existing different data are not overwritten.

[checksums.json](checksums.json) lists the payload files and their published SHA-256 values; its paths are relative to the installed `data/energy/` directory.
