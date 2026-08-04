#!/usr/bin/env python3
"""Assemble enriched rows in resumable source-file partitions."""

from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("rnee_profile",HERE/"17_profile_chunked_context.py"); PROFILE=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(PROFILE)
def load_json(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def write_json(path:Path,value:Any)->None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_atomic(path:Path,value:Any)->None: temporary=path.with_suffix(path.suffix+".tmp");write_json(temporary,value);temporary.replace(path)
def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        while block:=handle.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()
def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path(r"configs/rnee_build/trajectory_build_row_partition_dry_run.yaml"));p.add_argument("--run-dir",type=Path);return p.parse_args()

def select_map_partitions(map_manifest:dict[str,Any],overlay:dict[str,Any])->list[dict[str,Any]]:
    partitions=list(map_manifest["partitions"]);selected=overlay.get("selected_source_files")
    if not selected:return partitions
    requested=list(dict.fromkeys(str(value) for value in selected));by_source={str(item["source_file"]):item for item in partitions};missing=sorted(set(requested)-set(by_source))
    if missing:raise RuntimeError(f"Selected row-assembly source files are absent from the production map manifest: {missing}")
    return [by_source[source_file] for source_file in requested]

def bool_false_mask(series:pd.Series)->pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"false","0","no"})

def main()->int:
    main_started=time.perf_counter();args=parse_args();overlay=yaml.safe_load(args.config.read_text(encoding="utf-8"));base=yaml.safe_load(Path(overlay["base_config"]).read_text(encoding="utf-8"))
    map_pointer=load_json(Path(overlay["map_partition_pointer"]));edge_pointer=load_json(Path(overlay["edge_partition_pointer"]));context_pointer=load_json(Path(overlay["context_chunk_pointer"]));map_manifest=load_json(Path(map_pointer["manifest"]));edge_manifest=load_json(Path(edge_pointer["manifest"]));context_manifest=load_json(Path(context_pointer["manifest"]))
    map_partitions=select_map_partitions(map_manifest,overlay);abstention_pointer_path=Path(base["road_semantics_abstention_pointer"]);abstention_pointer=load_json(abstention_pointer_path);trip_policy_path=Path(abstention_pointer["trip_policy"]);trip_policy=pd.read_csv(trip_policy_path);abstain_policy=trip_policy.loc[bool_false_mask(trip_policy["road_semantics_available"])].copy()
    fingerprint={"config_sha256":sha256_file(args.config),"base_config_sha256":sha256_file(Path(overlay["base_config"])),"map_manifest_sha256":sha256_file(Path(map_pointer["manifest"])),"edge_manifest_sha256":sha256_file(Path(edge_pointer["manifest"])),"context_manifest_sha256":sha256_file(Path(context_pointer["manifest"])),"abstention_pointer_sha256":sha256_file(abstention_pointer_path),"abstention_trip_policy_sha256":sha256_file(trip_policy_path)}
    output=args.run_dir.resolve() if args.run_dir else Path(overlay["output_root"])/f"{overlay['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}";output.mkdir(parents=True,exist_ok=True);parts_root=output/"partitions";parts_root.mkdir(exist_ok=True);profile_root=output/"profile_output";profile_root.mkdir(exist_ok=True);manifest_path=output/"partition_manifest.json";summary_path=output/"partitioned_row_summary.json"
    if not (output/"run_config.yaml").exists():shutil.copy2(args.config,output/"run_config.yaml")
    if manifest_path.exists():
        manifest=load_json(manifest_path)
        if manifest["input_fingerprint"]!=fingerprint:raise RuntimeError("Resume fingerprint mismatch.")
    else:
        manifest={"schema_version":1,"experiment":overlay["experiment"],"input_fingerprint":fingerprint,"partitions":[{"partition_id":item["partition_id"],"source_file":item["source_file"],"status":"PENDING"} for item in map_partitions]};write_atomic(manifest_path,manifest)
    if all(item["status"]=="PASS" for item in manifest["partitions"]) and summary_path.exists():
        summary=load_json(summary_path);summary["resume_noop_verified"]=True;summary["resume_noop_elapsed_seconds"]=time.perf_counter()-main_started;summary["last_resume_check_at_utc"]=datetime.now(timezone.utc).isoformat();write_atomic(summary_path,summary);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
    map_by={item["partition_id"]:item for item in map_manifest["partitions"]};edge_by={item["partition_id"]:item for item in edge_manifest["partitions"]}
    for item in manifest["partitions"]:
        if item["status"]=="PASS":continue
        part_root=parts_root/item["partition_id"];part_root.mkdir(exist_ok=True);child_root=part_root/"runs";child_root.mkdir(exist_ok=True)
        context_path=part_root/"row_context_semantics.parquet"
        if not context_path.exists():
            frames=[]
            for chunk in context_manifest["chunks"]:
                frame=pd.read_parquet(chunk["outputs"]["row_context"],filters=[("source_file","==",item["source_file"])] )
                if not frame.empty:frames.append(frame)
            context=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();context.to_parquet(context_path,index=False)
        map_pointer_path=part_root/"map_pointer.json";edge_pointer_path=part_root/"edge_pointer.json";context_pointer_path=part_root/"context_pointer.json";write_json(map_pointer_path,map_by[item["partition_id"]]["pointer"]);write_json(edge_pointer_path,edge_by[item["partition_id"]]["pointer"]);write_json(context_pointer_path,{"status":"PASS_CONTEXT_SEMANTICS_PARTITION","row_context":str(context_path)})
        child_config=dict(base);child_config.update({"experiment":overlay["experiment"],"stage":"partitioned_row_assembly","status_suffix":"ROW_ASSEMBLY_PARTITION","run_name_prefix":item["partition_id"],"latest_prefix":f"{item['partition_id']}_rows","derivation_version":"TRAJECTORY_BUILD-partition-v1","map_match_pointer":str(map_pointer_path),"edge_semantics_pointer":str(edge_pointer_path),"context_semantics_pointer":str(context_pointer_path),"output_root":str(child_root),"raw_hash_sample_count":min(5000,map_by[item["partition_id"]]["point_count"]),"raw_hash_seed":181})
        config_path=part_root/"partition_config.yaml";config_path.write_text(yaml.safe_dump(child_config,sort_keys=False,allow_unicode=True),encoding="utf-8");stdout_path=part_root/"child_stdout.log";command=[sys.executable,str(HERE/"13_assemble_rows.py"),"--config",str(config_path)];item["status"]="RUNNING";write_atomic(manifest_path,manifest);samples=[];started=time.perf_counter()
        with stdout_path.open("w",encoding="utf-8") as stdout:
            process=subprocess.Popen(command,stdout=stdout,stderr=subprocess.STDOUT,cwd=HERE.parents[1])
            while process.poll() is None:
                memory=PROFILE.process_tree_memory(process.pid)
                if memory:samples.append({"elapsed_seconds":time.perf_counter()-started,**memory})
                time.sleep(max(.05,float(overlay["sample_interval_seconds"])))
            return_code=int(process.returncode)
        pointer_path=child_root/f"{item['partition_id']}_rows_latest_pointer.json"
        if return_code!=0 or not pointer_path.exists():item.update({"status":"FAIL","return_code":return_code});write_atomic(manifest_path,manifest);raise RuntimeError(f"Row partition failed: {item['partition_id']}")
        pointer=load_json(pointer_path);child=load_json(Path(pointer["summary"]));peak_ws=max((s["working_set_bytes"] for s in samples),default=0)/(1024**3);peak_private=max((s["private_bytes"] for s in samples),default=0)/(1024**3);sample_path=profile_root/f"{item['partition_id']}_memory_samples.csv"
        with sample_path.open("w",newline="",encoding="utf-8") as handle:
            fields=["elapsed_seconds","process_count","working_set_bytes","peak_working_set_bytes","private_bytes","peak_pagefile_bytes"];w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(samples)
        field_catalog=pd.read_csv(pointer["field_catalog"]);schema_signature=hashlib.sha256(field_catalog.loc[:,["field","dtype"]].to_csv(index=False).encode("utf-8")).hexdigest();partition_policy=abstain_policy.loc[abstain_policy["source_file"].eq(item["source_file"])];abstention_pass=True;abstention_row_count=0;observed_abstain_trip_count=0
        if not partition_policy.empty:
            semantic_columns=field_catalog.loc[(field_catalog["role"].eq("candidate_model_feature")) & (field_catalog["source"].isin(["historical_osm_or_valhalla_edge_semantics","historical_osm_context_semantics"])),"field"].tolist();audit_columns=["VehId","Trip","road_semantics_available","road_semantics_exclusion_reason",*semantic_columns];rows=pd.read_parquet(pointer["rows"],columns=audit_columns);expected_pairs={(int(vehicle),int(trip)) for vehicle,trip in partition_policy.loc[:,["VehId","Trip"]].itertuples(index=False,name=None)};row_pairs=list(zip(pd.to_numeric(rows["VehId"],errors="coerce").fillna(-1).astype("int64"),pd.to_numeric(rows["Trip"],errors="coerce").fillna(-1).astype("int64")));mask=pd.Series(row_pairs,index=rows.index).isin(expected_pairs);abstained=rows.loc[mask];abstention_row_count=int(mask.sum());observed_abstain_trip_count=len(set(row_pairs[index] for index in rows.index[mask]));abstention_pass=abstention_row_count>0 and observed_abstain_trip_count==len(expected_pairs) and abstained["road_semantics_available"].eq(False).all() and abstained["road_semantics_exclusion_reason"].eq("off_network_open_lot_quality_1_abstention").all() and abstained[semantic_columns].isna().all().all()
        checks={"status":"PASS" if str(child["status"]).startswith("PASS_ROW_ASSEMBLY") else "FAIL","row_conservation":"PASS" if child["output_input_row_ratio"]==1.0 and child["duplicate_row_id_count"]==0 else "FAIL","raw_hash":"PASS" if child["raw_hash_match_rate"]==1.0 else "FAIL","output_dtype_contract":"PASS" if child["checks"].get("output_dtype_contract")=="PASS" else "FAIL","forbidden_features":"PASS" if child["forbidden_feature_hit_count"]==0 else "FAIL","quality_1_abstention":"PASS" if abstention_pass else "FAIL","memory_ceiling":"PASS" if peak_ws<=float(overlay["memory_ceiling_gb"]) else "FAIL"}
        item.update({"status":"PASS" if "FAIL" not in checks.values() else "FAIL","checks":checks,"row_count":child["output_row_count"],"expected_abstain_trip_count":len(partition_policy),"observed_abstain_trip_count":observed_abstain_trip_count,"abstention_row_count":abstention_row_count,"schema_signature":schema_signature,"peak_working_set_gb":peak_ws,"peak_private_gb":peak_private,"sample_count":len(samples),"pointer":pointer,"memory_samples":str(sample_path),"sha256":{key:sha256_file(Path(pointer[key])) for key in ["rows","field_catalog","feature_roles"]}});write_atomic(manifest_path,manifest)
        if item["status"]!="PASS":raise RuntimeError(f"Row partition QA failed: {checks}")
    manifest=load_json(manifest_path);checks={"all_partitions_pass":"PASS" if all(i["status"]=="PASS" for i in manifest["partitions"]) else "FAIL","manifest_hashes_complete":"PASS" if all("sha256" in i for i in manifest["partitions"]) else "FAIL","stable_output_dtypes":"PASS" if len({i["schema_signature"] for i in manifest["partitions"]})==1 else "FAIL","quality_1_abstention":"PASS" if all(i["checks"]["quality_1_abstention"]=="PASS" for i in manifest["partitions"]) else "FAIL","memory_ceiling":"PASS" if max(i["peak_working_set_gb"] for i in manifest["partitions"])<=float(overlay["memory_ceiling_gb"]) else "FAIL"};gate="PASS" if "FAIL" not in checks.values() else "FAIL";status_suffix=overlay.get("status_suffix","TRAJECTORY_BUILD_PARTITIONED_ROW_DRY_RUN");stage=overlay.get("stage","partitioned_row_assembly_dry_run");summary={"experiment":overlay["experiment"],"stage":stage,"status":f"{gate}_{status_suffix}","completed_at_utc":datetime.now(timezone.utc).isoformat(),"partition_count":len(manifest["partitions"]),"row_count":sum(i["row_count"] for i in manifest["partitions"]),"expected_abstain_trip_count":sum(i["expected_abstain_trip_count"] for i in manifest["partitions"]),"observed_abstain_trip_count":sum(i["observed_abstain_trip_count"] for i in manifest["partitions"]),"abstention_row_count":sum(i["abstention_row_count"] for i in manifest["partitions"]),"peak_working_set_gb":max(i["peak_working_set_gb"] for i in manifest["partitions"]),"peak_private_gb":max(i["peak_private_gb"] for i in manifest["partitions"]),"checks":checks,"resume_noop_verified":False,"partition_manifest":str(manifest_path),"output_directory":str(output)};write_atomic(summary_path,summary);report=output/overlay.get("report_filename","TRAJECTORY_BUILD_PARTITIONED_ROW_DRY_RUN.md");report.write_text(f"# {overlay.get('report_title','TRAJECTORY_BUILD partitioned row assembly')}\n\n- Status: `{summary['status']}`\n- Partitions / rows: {summary['partition_count']} / {summary['row_count']:,}\n- Quality check 1 abstained trips / rows: {summary['observed_abstain_trip_count']} / {summary['abstention_row_count']:,}\n- Peak working set/private: {summary['peak_working_set_gb']:.2f} / {summary['peak_private_gb']:.2f} GB\n",encoding="utf-8");write_json(Path(overlay["output_root"])/f"{overlay['latest_prefix']}_latest_pointer.json",{"status":summary["status"],"summary":str(summary_path),"report":str(report),"manifest":str(manifest_path),"output_directory":str(output)});print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if gate=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
