#!/usr/bin/env python3
"""Compare partitioned edge semantics with accepted END_TO_END outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rnee_equivalence", HERE / "18_validate_chunked_context_equivalence.py")
EQUIV = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(EQUIV)


def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, value: Any) -> None: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(); parser.add_argument("--partition-pointer",type=Path,default=Path(r"results/trajectory_build_edge_partition_dry_run_latest_pointer.json")); parser.add_argument("--baseline-pointer",type=Path,default=Path(r"results/end_to_end_edge_latest_pointer.json")); return parser.parse_args()


def main() -> int:
    args=parse_args(); pointer=load_json(args.partition_pointer); baseline_pointer=load_json(args.baseline_pointer); manifest=load_json(Path(pointer["manifest"])); partition_summary=load_json(Path(pointer["summary"]))
    baseline_edges=pd.read_parquet(baseline_pointer["edge_semantics"]); baseline_points=pd.read_parquet(baseline_pointer["point_edge_semantics"]); baseline_tags=pd.read_parquet(baseline_pointer["osm_way_tags"])
    results=[]
    for partition in manifest["partitions"]:
        actual_edges=pd.read_parquet(partition["pointer"]["edge_semantics"]); actual_points=pd.read_parquet(partition["pointer"]["point_edge_semantics"]); actual_tags=pd.read_parquet(partition["pointer"]["osm_way_tags"])
        trip_ids=set(actual_edges["pilot_trip_id"]); expected_edges=baseline_edges[baseline_edges["pilot_trip_id"].isin(trip_ids)].copy(); expected_points=baseline_points[baseline_points["pilot_trip_id"].isin(trip_ids)].copy()
        tag_keys=actual_tags[["osm_snapshot_id","way_id"]].drop_duplicates(); expected_tags=baseline_tags.merge(tag_keys,on=["osm_snapshot_id","way_id"],how="inner",validate="one_to_one")
        comparisons={}
        for name,expected,actual,keys in [("edge_semantics",expected_edges,actual_edges,["profile","pilot_trip_id","edge_index"]),("point_edge_semantics",expected_points,actual_points,["pilot_trip_id","point_index"]),("osm_way_tags",expected_tags,actual_tags,["osm_snapshot_id","way_id"])]:
            ok,error,expected_hash,actual_hash=EQUIV.compare_frame(expected,actual,keys); comparisons[name]={"equivalent":ok,"error":error,"expected_hash":expected_hash,"actual_hash":actual_hash,"row_count":len(actual)}
        results.append({"partition_id":partition["partition_id"],"comparisons":comparisons})
    checks={"all_partition_frames_exact":"PASS" if all(c["equivalent"] for result in results for c in result["comparisons"].values()) else "FAIL","global_edge_count":"PASS" if sum(r["comparisons"]["edge_semantics"]["row_count"] for r in results)==len(baseline_edges) else "FAIL","global_point_count":"PASS" if sum(r["comparisons"]["point_edge_semantics"]["row_count"] for r in results)==len(baseline_points) else "FAIL","resume_noop_verified":"PASS" if partition_summary.get("resume_noop_verified") is True else "FAIL"}; gate="PASS" if "FAIL" not in checks.values() else "FAIL"
    output=Path(pointer["output_directory"])/"equivalence_audit"; output.mkdir(exist_ok=True); summary={"experiment":"TRAJECTORY_BUILD-DRYRUN","stage":"partitioned_edge_equivalence","status":f"{gate}_TRAJECTORY_BUILD_PARTITIONED_EDGE_EQUIVALENCE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"partition_count":len(results),"edge_count":sum(r["comparisons"]["edge_semantics"]["row_count"] for r in results),"point_count":sum(r["comparisons"]["point_edge_semantics"]["row_count"] for r in results),"partition_results":results,"output_directory":str(output)}
    summary_path=output/"equivalence_summary.json"; report_path=output/"EQUIVALENCE_REPORT.md"; write_json(summary_path,summary); report_path.write_text(f"# TRAJECTORY_BUILD partitioned edge equivalence\n\n- Status: `{summary['status']}`\n- Partitions / edges / points: {len(results)} / {summary['edge_count']:,} / {summary['point_count']:,}\n- Edge, point-edge and OSM-tag frames exact: `{checks['all_partition_frames_exact']}`\n- Resume no-op: `{checks['resume_noop_verified']}`\n",encoding="utf-8")
    write_json(args.partition_pointer.parent/"trajectory_build_edge_partition_dry_run_latest_equivalence_pointer.json",{"status":summary["status"],"summary":str(summary_path),"report":str(report_path),"output_directory":str(output)}); print(json.dumps({k:v for k,v in summary.items() if k!="partition_results"},ensure_ascii=False,indent=2)); return 0 if gate=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
