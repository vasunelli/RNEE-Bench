#!/usr/bin/env python3
"""Aggregate full TRAJECTORY_BUILD partitioned map matching and apply the frozen QA bands."""
from __future__ import annotations
import argparse, collections, gzip, json, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def dump(path:Path,value:Any)->None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def band_min(value:float,rule:dict[str,float])->str:
    return "PASS" if value>=rule["pass_min"] else ("CAUTION" if value>=rule["quality_check_min"] else "FAIL")
def band_max(value:float,pass_max:float,quality_check_max:float)->str:
    return "PASS" if value<=pass_max else ("CAUTION" if value<=quality_check_max else "FAIL")
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=Path(r"configs/rnee_build/trajectory_build_map_match_gate.yaml"));args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"));qa=yaml.safe_load(Path(cfg["qa_thresholds"]).read_text(encoding="utf-8"))["map_matching"]
    pointer=load(Path(cfg["map_pointer"]));manifest=load(Path(pointer["manifest"]));run_summary=load(Path(pointer["summary"]));status_counts=collections.Counter();distances=[];trip_frames=[]
    for item in manifest["partitions"]:
        points=pd.read_parquet(item["pointer"]["matched_points"],columns=["match_status","distance_from_trace_point_m"]);status_counts.update(points["match_status"].fillna("<missing>").value_counts().to_dict());values=points["distance_from_trace_point_m"].dropna().to_numpy(float)
        if len(values):distances.append(values)
        trip_frames.append(pd.read_parquet(item["pointer"]["trip_summary"]))
    trips=pd.concat(trip_frames,ignore_index=True);distance=np.concatenate(distances) if distances else np.array([],dtype=float);total_points=sum(status_counts.values());matched=status_counts["matched_with_edge"]+status_counts["matched_without_edge"];edge=status_counts["matched_with_edge"]
    quantiles=np.quantile(distance,[.5,.95,.99]) if len(distance) else [np.nan]*3;discontinuity=float(trips["has_route_discontinuity"].mean());eligible=int(trips["eligible_short_time_pairs"].sum());jumps=int(trips["edge_jump_over_500m_within_10s_count"].sum());silent=int(trips["silent_fallback_count"].sum());failed=trips[~trips["request_succeeded"]].copy();fallback_rows=[]
    candidates=trips[trips["attempt_count"]>trips["chunk_count"]]
    for row in candidates.itertuples(index=False):
        with gzip.open(row.response_path,"rt",encoding="utf-8") as handle:response=json.load(handle)
        records=[x for x in response.get("chunks",[]) if x.get("status")=="map_snap_midpoint_fallback"]
        if records:fallback_rows.append({"source_file":row.source_file,"VehId":int(row.VehId),"Trip":int(row.Trip),"pilot_trip_id":row.pilot_trip_id,"fallback_split_count":len(records),"raw_point_count":int(row.raw_point_count),"matched_coverage":float(row.matched_coverage)})
    checks={
        "partition_completion":"PASS" if len(manifest["partitions"])==cfg["expected_source_file_count"] and all(x["status"]=="PASS" for x in manifest["partitions"]) else "FAIL",
        "row_conservation":"PASS" if total_points==cfg["expected_row_count"]==run_summary["point_count"] else "FAIL",
        "trip_conservation":"PASS" if len(trips)==cfg["expected_trip_count"]==run_summary["trip_count"] else "FAIL",
        "status_reason_coverage":"PASS" if status_counts["<missing>"]==0 else "FAIL",
        "request_failure_bound":"PASS" if len(failed)/len(trips)<=cfg["maximum_request_failure_rate_global"] else "FAIL",
        "matched_coverage":band_min(matched/total_points,qa["matched_coverage"]),
        "distance_p50":band_max(float(quantiles[0]),qa["distance_from_trace_point_m"]["p50_pass_max"],qa["distance_from_trace_point_m"]["p50_quality_check_max"]),
        "distance_p95":band_max(float(quantiles[1]),qa["distance_from_trace_point_m"]["p95_pass_max"],qa["distance_from_trace_point_m"]["p95_quality_check_max"]),
        "distance_p99":band_max(float(quantiles[2]),qa["distance_from_trace_point_m"]["p99_pass_max"],qa["distance_from_trace_point_m"]["p99_quality_check_max"]),
        "trip_route_discontinuity":band_max(discontinuity,qa["trip_route_discontinuity_rate"]["pass_max"],qa["trip_route_discontinuity_rate"]["quality_check_max"]),
        "silent_fallbacks":"PASS" if silent<=qa["silent_fallback_count"]["pass_max"] else "FAIL",
        "edge_jump_rate":band_max(jumps/max(eligible,1),qa["matched_edge_jump_over_500m_within_10s_rate"]["pass_max"],qa["matched_edge_jump_over_500m_within_10s_rate"]["quality_check_max"]),
        "resume_noop":"PASS" if run_summary.get("resume_noop_verified") is True else "FAIL",
    }
    gate="FAIL" if "FAIL" in checks.values() else ("CAUTION" if "CAUTION" in checks.values() else "PASS");stamp=datetime.now().strftime("%Y%m%d_%H%M%S");out=Path(cfg["output_root"])/f"{cfg['run_name_prefix']}_{stamp}";out.mkdir(parents=True);shutil.copy2(args.config,out/"run_config.yaml");failed_cols=["source_file","VehId","Trip","pilot_trip_id","raw_point_count","request_point_count","chunk_count","chunk_failure_count","error_code","error_message","response_path"];failed[failed_cols].to_csv(out/"terminal_request_failures.csv",index=False);pd.DataFrame(fallback_rows).to_csv(out/"map_snap_fallback_recoveries.csv",index=False)
    summary={"experiment":"TRAJECTORY_BUILD","stage":"full_map_match_quality_gate","status":f"{gate}_TRAJECTORY_BUILD_MAP_MATCH_PRODUCTION_GATE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"source_file_count":len(manifest["partitions"]),"row_count":total_points,"trip_count":len(trips),"status_counts":dict(status_counts),"matched_coverage":matched/total_points,"edge_association_coverage":edge/total_points,"distance_p50_m":float(quantiles[0]),"distance_p95_m":float(quantiles[1]),"distance_p99_m":float(quantiles[2]),"trip_route_discontinuity_rate":discontinuity,"edge_jump_rate":jumps/max(eligible,1),"silent_fallback_count":silent,"request_failure_count":len(failed),"request_failure_rate":len(failed)/len(trips),"map_snap_fallback_recovery_count":len(fallback_rows),"manual_quality_1_audit_required":gate=="CAUTION","secondary_quality_check_used":False,"output_directory":str(out)};dump(out/"trajectory_build_map_match_gate_summary.json",summary)
    report=out/"TRAJECTORY_BUILD_MAP_MATCH_PRODUCTION_GATE.md";report.write_text("# TRAJECTORY_BUILD full map-match quality gate\n\n"+"\n".join([f"- Status: `{summary['status']}`",f"- Files / rows / trips: {summary['source_file_count']} / {total_points:,} / {len(trips):,}",f"- Matched / edge-associated coverage: {summary['matched_coverage']:.4%} / {summary['edge_association_coverage']:.4%}",f"- Distance P50/P95/P99: {summary['distance_p50_m']:.2f} / {summary['distance_p95_m']:.2f} / {summary['distance_p99_m']:.2f} m",f"- Route discontinuity: {discontinuity:.4%}",f"- Terminal request failures: {len(failed)} ({summary['request_failure_rate']:.5%}); recovered 444 trips: {len(fallback_rows)}",f"- Quality check 1 manual audit required: `{summary['manual_quality_1_audit_required']}`; quality_check 2 not used."])+"\n",encoding="utf-8")
    latest={"status":summary["status"],"summary":str(out/"trajectory_build_map_match_gate_summary.json"),"report":str(report),"terminal_failures":str(out/"terminal_request_failures.csv"),"fallback_recoveries":str(out/"map_snap_fallback_recoveries.csv"),"output_directory":str(out)};dump(Path(cfg["output_root"])/f"{cfg['latest_prefix']}_latest_pointer.json",latest);print(json.dumps(summary,ensure_ascii=False,indent=2));return 1 if gate=="FAIL" else 0
if __name__=="__main__":raise SystemExit(main())
