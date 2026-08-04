#!/usr/bin/env python3
"""Export aggregate Parquet row counts and representative schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = {
    "map_matched_points": "data/map_matching/matched_points/*.parquet",
    "map_matched_edges": "data/map_matching/matched_edges/*.parquet",
    "row_enriched_trajectories": "data/row_enriched/*.parquet",
    "segments_all": "data/segments/segments_all.parquet",
    "segments_qa_valid": "data/segments/segments_qa_valid.parquet",
    "segments_prediction_usable": "data/segments/segments_prediction_usable.parquet",
    "theory_validation_paired_test_predictions": "predictions/theory_validation/paired_test_predictions/**/*.parquet",
    "theory_validation_ensemble_test_predictions": "predictions/theory_validation/ensemble_test_predictions/*.parquet",
    "theory_validation_null_controls": "predictions/theory_validation/null_controls/*.parquet",
    "phase1_rpm_only_test_predictions": "predictions/phase1/rpm_only_test_predictions/*.parquet",
}


def schema_record(path: Path) -> dict[str, object]:
    schema = pq.read_schema(path)
    schema_text = schema.to_string(show_field_metadata=True)
    return {
        "representative_file": path.as_posix(),
        "column_count": len(schema),
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("metadata/parquet_schemas.json"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output: dict[str, object] = {"schema_version": 1, "datasets": {}}
    for name, pattern in DATASETS.items():
        files = sorted(root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no files for {name}: {pattern}")
        metadata = [pq.ParquetFile(path).metadata for path in files]
        record = schema_record(files[0])
        record.update(
            {
                "glob": pattern,
                "file_count": len(files),
                "total_rows": sum(item.num_rows for item in metadata),
                "total_row_groups": sum(item.num_row_groups for item in metadata),
            }
        )
        output["datasets"][name] = record
    destination = args.output if args.output.is_absolute() else root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
