# RNEE-Bench full artifact volumes v0.1.0

These split 7-Zip volumes contain the complete frozen RNEE data, model, and prediction payload that accompanies `vasunelli/RNEE-Bench`.

## Required files

Download every file named:

```text
RNEE-Bench-full-artifacts-v0.1.0.7z.001
RNEE-Bench-full-artifacts-v0.1.0.7z.002
RNEE-Bench-full-artifacts-v0.1.0.7z.003
```

Keep all volumes in the same directory. Verify each SHA-256 against `RNEE-Bench-full-artifacts-v0.1.0-SHA256SUMS.txt` before extraction.

## Extraction

Clone the lightweight repository and extract the first volume at its root:

```bash
git clone https://github.com/vasunelli/RNEE-Bench.git
cd RNEE-Bench
7z x /path/to/RNEE-Bench-full-artifacts-v0.1.0.7z.001
python scripts/verify_full_release.py
```

On Windows, install 7-Zip, right-click the `.7z.001` file, choose **7-Zip > Extract files...**, and select the cloned `RNEE-Bench` directory. Do not extract `.002` or `.003` separately.

## Payload

- `data/`: complete TRAJECTORY_BUILD map-matching and row-enriched trajectory partitions plus SEGMENT_SPLIT segments and splits.
- `models/`: 192 frozen THEORY_VALIDATION Joblib models and sidecars.
- `predictions/`: THEORY_VALIDATION paired/ensemble/control predictions and current RPM-only predictions/inference artifacts.
- `results/`: compact reported-result tables and public contract fixtures used by the checked-in verification scripts.

The repository supplies the source code, configs, tests, schemas, inventory, dataset/model cards, and component licenses. Joblib files are executable Python serialization artifacts; load only after checksum verification and only from a trusted release.
