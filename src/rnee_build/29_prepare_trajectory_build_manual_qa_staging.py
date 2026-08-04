#!/usr/bin/env python3
"""Materialize only the full-build trips needed by the TRAJECTORY_BUILD manual map audit."""
from __future__ import annotations
import argparse, importlib.util, json
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("manual_qa",HERE/"08_build_map_match_manual_qa.py")
MOD=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MOD)
def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def dump(path:Path,value:Any)->None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def synthetic_edges(complexity:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for row in complexity.itertuples(index=False):
        count=max(int(row.road_class_count),int(row.edge_use_count),1)
        for index in range(count):
            rows.append({"pilot_trip_id":row.pilot_trip_id,"profile":"balanced_final","road_class":f"rc_{index}" if index<int(row.road_class_count) else None,"use":f"use_{index}" if index<int(row.edge_use_count) else None})
    return pd.DataFrame(rows)
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--map-pointer",type=Path,default=Path(r"results/trajectory_build_map_match_production_latest_pointer.json"));ap.add_argument("--thresholds",type=Path,default=Path(r"configs/rnee_build/qa_thresholds.yaml"));ap.add_argument("--output-root",type=Path,default=Path(r"results/rnee_build"));ap.add_argument("--seed",type=int,default=181);args=ap.parse_args();pointer=load(args.map_pointer);manifest=load(Path(pointer["manifest"]));trip_frames=[];complexity_frames=[]
    for item in manifest["partitions"]:
        trip_frames.append(pd.read_parquet(item["pointer"]["trip_summary"]))
        edges=pd.read_parquet(item["pointer"]["matched_edges"],columns=["pilot_trip_id","road_class","use"])
        complexity_frames.append(edges.groupby("pilot_trip_id").agg(road_class_count=("road_class","nunique"),edge_use_count=("use","nunique")).reset_index())
    trips=pd.concat(trip_frames,ignore_index=True);complexity=pd.concat(complexity_frames,ignore_index=True);edges=synthetic_edges(complexity);thresholds=yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))["map_matching"]["manual_spatial_audit"];selected=MOD.select_audit_trips(trips,edges,thresholds,args.seed);point_frames=[];selected_by_source=selected.groupby("source_file")["pilot_trip_id"].apply(set).to_dict()
    for item in manifest["partitions"]:
        ids=selected_by_source.get(item["source_file"])
        if ids:point_frames.append(pd.read_parquet(item["pointer"]["matched_points"],filters=[("pilot_trip_id","in",sorted(ids))]))
    points=pd.concat(point_frames,ignore_index=True);stamp=datetime.now().strftime("%Y%m%d_%H%M%S");out=args.output_root/f"trajectory_build_manual_qa_staging_{stamp}";out.mkdir(parents=True);trips.to_parquet(out/"trip_summary.parquet",index=False);edges.to_parquet(out/"synthetic_edges.parquet",index=False);points.to_parquet(out/"selected_points.parquet",index=False)
    raw={"trip_summary":str(out/"trip_summary.parquet"),"matched_edges":str(out/"synthetic_edges.parquet"),"matched_points":str(out/"selected_points.parquet")};dump(out/"selected_raw_pointer.json",raw);qa={"status":"CAUTION_TRAJECTORY_BUILD_MAP_MATCH_PRODUCTION_GATE","selected_profile":"balanced_final","raw_pointer":str(out/"selected_raw_pointer.json")};dump(out/"qa_pointer.json",qa);summary={"status":"PASS_TRAJECTORY_BUILD_MANUAL_QA_STAGING","trip_count":len(trips),"selected_trip_count":len(selected),"selected_point_count":len(points),"qa_pointer":str(out/"qa_pointer.json"),"output_directory":str(out)};dump(out/"staging_summary.json",summary);dump(args.output_root/"trajectory_build_manual_qa_staging_latest_pointer.json",{"status":summary["status"],"summary":str(out/"staging_summary.json"),"qa_pointer":str(out/"qa_pointer.json"),"output_directory":str(out)});print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
