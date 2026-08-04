#!/usr/bin/env python3
"""Compare partitioned enriched rows with the accepted END_TO_END assembly."""
from __future__ import annotations
import argparse, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
HERE=Path(__file__).resolve().parent;SPEC=importlib.util.spec_from_file_location("rnee_equivalence",HERE/"18_validate_chunked_context_equivalence.py");EQUIV=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(EQUIV)
def load_json(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def write_json(path:Path,value:Any)->None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser();p.add_argument("--partition-pointer",type=Path,default=Path(r"results/trajectory_build_row_partition_dry_run_latest_pointer.json"));p.add_argument("--baseline-pointer",type=Path,default=Path(r"results/end_to_end_rows_latest_pointer.json"));p.add_argument("--schema-config",type=Path,default=Path(r"configs/rnee_build/row_assembly.yaml"));return p.parse_args()
def main()->int:
    args=parse_args();pointer=load_json(args.partition_pointer);baseline_pointer=load_json(args.baseline_pointer);manifest=load_json(Path(pointer["manifest"]));partition_summary=load_json(Path(pointer["summary"]));baseline=pd.read_parquet(baseline_pointer["rows"]);baseline_roles=load_json(Path(baseline_pointer["feature_roles"]));dtype_contract=yaml.safe_load(args.schema_config.read_text(encoding="utf-8")).get("output_dtype_contract", {});results=[]
    for part in manifest["partitions"]:
        actual=pd.read_parquet(part["pointer"]["rows"]);expected=baseline[baseline["source_file"].eq(part["source_file"])].copy()
        for column,dtype in dtype_contract.items(): expected[column]=expected[column].astype(dtype);actual[column]=actual[column].astype(dtype)
        ok,error,expected_hash,actual_hash=EQUIV.compare_frame(expected,actual,["row_id"]);roles=load_json(Path(part["pointer"]["feature_roles"]));roles_ok=roles["model_allowlist"]==baseline_roles["model_allowlist"] and roles["forbidden_hits_in_model_allowlist"]==[] and roles["missing_explicit_allowlist_columns"]==[];results.append({"partition_id":part["partition_id"],"source_file":part["source_file"],"row_equivalent":ok,"roles_equivalent":roles_ok,"error":error,"expected_hash":expected_hash,"actual_hash":actual_hash,"row_count":len(actual)})
    checks={"all_row_partitions_exact":"PASS" if all(r["row_equivalent"] for r in results) else "FAIL","all_feature_roles_safe":"PASS" if all(r["roles_equivalent"] for r in results) else "FAIL","global_row_count":"PASS" if sum(r["row_count"] for r in results)==len(baseline) else "FAIL","resume_noop_verified":"PASS" if partition_summary.get("resume_noop_verified") is True else "FAIL"};gate="PASS" if "FAIL" not in checks.values() else "FAIL";output=Path(pointer["output_directory"])/"equivalence_audit";output.mkdir(exist_ok=True);summary={"experiment":"TRAJECTORY_BUILD-DRYRUN","stage":"partitioned_row_equivalence","status":f"{gate}_TRAJECTORY_BUILD_PARTITIONED_ROW_EQUIVALENCE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"partition_count":len(results),"baseline_row_count":len(baseline),"partition_row_count":sum(r["row_count"] for r in results),"partition_results":results,"output_directory":str(output)};summary_path=output/"equivalence_summary.json";report=output/"EQUIVALENCE_REPORT.md";write_json(summary_path,summary);report.write_text(f"# TRAJECTORY_BUILD partitioned row equivalence\n\n- Status: `{summary['status']}`\n- Baseline/partition rows: {len(baseline):,} / {summary['partition_row_count']:,}\n- All row fields exact: `{checks['all_row_partitions_exact']}`\n- Feature roles safe: `{checks['all_feature_roles_safe']}`\n- Resume no-op: `{checks['resume_noop_verified']}`\n",encoding="utf-8");write_json(args.partition_pointer.parent/"trajectory_build_row_partition_dry_run_latest_equivalence_pointer.json",{"status":summary["status"],"summary":str(summary_path),"report":str(report),"output_directory":str(output)});print(json.dumps({k:v for k,v in summary.items() if k!="partition_results"},ensure_ascii=False,indent=2));return 0 if gate=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
