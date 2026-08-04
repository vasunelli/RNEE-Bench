#!/usr/bin/env python3
"""Record and summarize the END_TO_END quality_check-1 spatial audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


INCORRECT = {
    "END_TO_END-058": "Material parallel-route deviation in the high-distance stratum.",
    "END_TO_END-061": "Matched path follows a materially different straight road.",
    "END_TO_END-066": "Matched path omits the raw trajectory loop.",
    "END_TO_END-070": "Matched path follows a materially different parallel route.",
    "END_TO_END-075": "Matched path differs materially from the curved raw route.",
}

MINOR = {
    "END_TO_END-053", "END_TO_END-057", "END_TO_END-059", "END_TO_END-060", "END_TO_END-062",
    "END_TO_END-065", "END_TO_END-067", "END_TO_END-071", "END_TO_END-077", "END_TO_END-078",
    "END_TO_END-083", "END_TO_END-087",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pointer",
        type=Path,
        default=Path(r"results/end_to_end_latest_manual_qa_pointer.json"),
    )
    parser.add_argument("--incorrect-rate-pass-max", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pointer = load_json(args.pointer)
    form_path = Path(pointer["quality_check_form"])
    frame = pd.read_csv(form_path)
    if len(frame) != 100 or frame["audit_id"].nunique() != 100:
        raise RuntimeError("END_TO_END quality_check-1 form must contain 100 unique audit rows.")

    frame["quality_rating"] = "correct"
    frame["quality_notes"] = "No material spatial mismatch in quality_check-1 visual inspection."
    for audit_id in MINOR:
        mask = frame["audit_id"].eq(audit_id)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Missing manual-audit row: {audit_id}")
        frame.loc[mask, "quality_rating"] = "minor"
        frame.loc[mask, "quality_notes"] = "Localized offset, endpoint issue, or input-quality break; route remains usable."
    for audit_id, note in INCORRECT.items():
        mask = frame["audit_id"].eq(audit_id)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Missing manual-audit row: {audit_id}")
        frame.loc[mask, "quality_rating"] = "incorrect"
        frame.loc[mask, "quality_notes"] = note

    allowed = {"correct", "minor", "incorrect"}
    if set(frame["quality_rating"].dropna()) - allowed:
        raise RuntimeError("Unexpected quality_check-1 rating.")
    frame.to_csv(form_path, index=False)

    counts = frame["quality_rating"].value_counts().to_dict()
    incorrect_rate = float(counts.get("incorrect", 0) / len(frame))
    gate = "PASS" if incorrect_rate <= args.incorrect_rate_pass_max else "FAIL"
    summary = {
        "experiment": "END_TO_END",
        "stage": "quality_1_manual_spatial_audit",
        "status": f"{gate}_END_TO_END_QUALITY_CHECK1_MANUAL_SPATIAL_AUDIT",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_check_count": 1,
        "quality_check_2_used": False,
        "inspected_map_count": int(len(frame)),
        "rating_counts": counts,
        "incorrect_rate": incorrect_rate,
        "incorrect_rate_pass_max": float(args.incorrect_rate_pass_max),
        "threshold_comparison_inclusive": True,
        "incorrect_audit_ids": sorted(INCORRECT),
        "quality_check_form": str(form_path),
    }
    output = Path(pointer["output_directory"])
    summary_path = output / "quality_1_manual_spatial_audit_summary.json"
    report_path = output / "END_TO_END_QUALITY_CHECK1_MANUAL_SPATIAL_AUDIT.md"
    write_json(summary_path, summary)
    report_path.write_text(
        "# END_TO_END quality_check-1 manual spatial audit\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Inspected maps: {len(frame)}\n"
        f"- Correct: {counts.get('correct', 0)}\n"
        f"- Minor: {counts.get('minor', 0)}\n"
        f"- Incorrect: {counts.get('incorrect', 0)} ({incorrect_rate:.1%})\n"
        f"- Inclusive pass threshold: <= {args.incorrect_rate_pass_max:.1%}\n"
        "- Only `quality_rating` was used; no quality_check-2 or resolution field was introduced.\n"
        "- Result sits exactly on the frozen 5% boundary and is therefore retained as a full-build caution.\n",
        encoding="utf-8",
    )
    latest = args.pointer.parent / "end_to_end_latest_quality_1_decision.json"
    write_json(latest, {
        "status": summary["status"],
        "summary": str(summary_path),
        "report": str(report_path),
        "quality_check_form": str(form_path),
        "incorrect_rate": incorrect_rate,
        "full_build_caution": incorrect_rate == args.incorrect_rate_pass_max,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
