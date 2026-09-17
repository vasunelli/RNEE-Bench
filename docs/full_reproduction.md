# Full construction, training and rerun pipeline

For the current manuscript's data, tables and figures, see the [paper resource guide](../energy/README.md). This page describes the earlier construction release.

This guide documents the complete checked-in workflow used to create the released data and frozen model evidence. The commands preserve the published data, feature, and evaluation contracts.

## 1. Environment

Use Python 3.11 in an isolated environment. The recorded dependency contract is:

```bash
python3.11 -m venv .venv-rnee
source .venv-rnee/bin/activate
python -m pip install -r configs/rnee_build/requirements-rnee.lock.txt
```

Valhalla is separately frozen at version 3.7.0 and commit `72f459fc5661fb906ad424be5378c4e32d9a5b3b`. The historical run used the portable Windows executables in the official PyPI wheel; the source URLs and hashes are in `configs/rnee_build/base.yaml`.

The checked-in configs record the source-build layout. For a new machine, first copy `configs/` to a run-specific configuration directory and replace the input and output roots. Do not edit the checked-in configs or released artifacts in place.

After arranging the raw inputs, validate both the source and checked-in workflow surfaces:

```bash
python scripts/run_full_analysis.py --data-root /path/to/DATA_ROOT --check-only --list-stages
```

## 2. Source acquisition

Acquire the four official VED files at commit `6baa4963782d515a67d32a5490bd5d11f5d9bf0d` and the Michigan OSM snapshots dated 2017-01-01 and 2018-01-01. Verify every byte count and SHA-256 in `configs/rnee_build/base.yaml`. Raw inputs must remain immutable.

Run the inventory and target gates with explicit configs/output locations:

```bash
python src/rnee_build/00_preflight.py --config RUN_CONFIG/base.yaml
python src/rnee_build/01_index_raw.py --config RUN_CONFIG/raw_schema.yaml
python src/rnee_build/02_audit_energy_target.py --config RUN_CONFIG/target_audit.yaml
```

Each script supports `--help`; use explicit configuration and output arguments rather than relying on historical defaults.

## 3. Historical road network and map matching

Prepare the dated snapshot, build Valhalla tiles, run the partitioned matcher, validate equivalence and finalize the production gate:

```bash
python src/rnee_build/03_prepare_osm_snapshot.py --config RUN_CONFIG/osm_snapshot.yaml
python src/rnee_build/04_build_network.py --config RUN_CONFIG/network_build.yaml
python src/rnee_build/20_run_map_match_partitioned.py --config RUN_CONFIG/trajectory_build_production.yaml
python src/rnee_build/21_validate_partitioned_map_match_equivalence.py --config RUN_CONFIG/trajectory_build_map_match_gate.yaml
python src/rnee_build/28_finalize_trajectory_build_map_match_production.py --config RUN_CONFIG/trajectory_build_map_match_gate.yaml
```

The detailed QA/remediation utilities are `06`–`09` and `29`–`32`. Preserve explicit abstentions; never fabricate a match for failed/off-network traces.

## 4. Edge/context semantics and row enrichment

The checked-in stages are:

1. `10_extract_edge_semantics.py` and `11_parse_osm_tags.py` for edge semantics.
2. `12_build_context_semantics.py` or the chunked `16`–`19` route for context objects.
3. `22_run_edge_semantics_partitioned.py` and `23_validate_partitioned_edge_equivalence.py` for production edge partitions.
4. `24_run_row_assembly_partitioned.py` and `25_validate_partitioned_row_equivalence.py` for the 54 row-enriched partitions.
5. `26_finalize_trajectory_build_partition_gate.py` and `34_finalize_trajectory_build_release_gate.py` for production/release gates.

Run each with the corresponding copied `RUN_CONFIG/trajectory_build_*.yaml`. A valid rebuild must conserve all 22,436,808 VED rows, retain the frozen abstention set and pass the equivalence/release gates.

## 5. Segmentation and leakage-controlled splits

```bash
python src/rnee_build/35_build_segments.py --config RUN_CONFIG/segment_split_segments_60s.yaml
python src/rnee_build/36_build_splits.py --config RUN_CONFIG/segment_split_splits.yaml
python src/rnee_build/37_validate_leakage.py --config RUN_CONFIG/segment_split_splits.yaml
```

The primary gate expects 301,219 non-overlapping segments, 256,419 QA-valid segments and 219,384 prediction-usable segments. The frozen split gate has 135 passing and zero failing checks. The 30-s, 500-m and sensitivity builders (`38`–`43`) are separate robustness branches and must not replace the primary 60-s contract.

## 6. Feature/model backbone

Stages `45`–`49` freeze the feature contract, select the backbone, execute add/drop comparisons and run/correct inference. Stage `68` runs the sensor factorial. Use their same-numbered YAML configs. Training-only feature selection and preprocessing are mandatory; validation, calibration and test roles must remain separate.

## 7. THEORY_VALIDATION training and controls

The complete THEORY_VALIDATION implementation is in `scripts/rnee_theory_validation/`:

```bash
python scripts/rnee_theory_validation/run_theory_validation.py \
  --config configs/rnee/theory_validation/theory_validation.yaml
```

The run fixes systems, seeds, model hyperparameters, shared active-mask policy, control generation and max-t inference before evaluation. It writes models, paired/ensemble predictions and control predictions. To reproduce the published manifests, keep the 5.5% practical-effect threshold, controls, active-mask policy and evaluation settings unchanged. A changed setting produces a separate run and will not be byte-identical to this release.

## 8. Current RPM-only rerun

The current runner depends on the checked-in ROBUSTNESS utilities:

```bash
python scripts/rnee_phase1/run_target_diagnostics.py \
  --config configs/rnee/phase1/phase1_target_diagnostics.yaml \
  --output-dir /path/to/new-output
```

The runner records the source inventory and reports missing inputs rather than inferring them. The frozen current run did not serialize estimator objects; a rerun will not recover model binaries byte-for-byte.

## 9. Verification gates

```bash
pytest -q tests/rnee_build tests/rnee_theory_validation
python scripts/verify_reported_results.py
python scripts/verify_full_release.py
```

Full reconstruction is complete only when source hashes are frozen, row/segment counts match, all construction/equivalence/leakage checks pass, predictions reproduce the published effects within numeric tolerance and the public artifact hashes validate. Interpret any rerun within the published scope and source contracts.
