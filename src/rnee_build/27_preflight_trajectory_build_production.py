#!/usr/bin/env python3
"""Freeze and validate the immutable TRAJECTORY_BUILD all-file production partition plan."""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, stat, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def dump(path: Path, value: Any) -> None: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()
def valhalla_probe(executable: Path, config: Path, dll_dir: Path) -> dict[str, Any]:
    request={"locations":[{"lat":42.2808,"lon":-83.7430}],"costing":"auto"}
    with tempfile.NamedTemporaryFile("w",suffix=".json",delete=False,encoding="utf-8") as f:
        json.dump(request,f); request_path=Path(f.name)
    env=os.environ.copy(); env["PATH"]=str(dll_dir)+os.pathsep+env.get("PATH","")
    try:
        p=subprocess.run([str(executable),str(config),"locate",str(request_path)],capture_output=True,text=True,timeout=60,env=env)
        parsed=False
        if p.returncode==0:
            try: json.loads(p.stdout); parsed=True
            except json.JSONDecodeError: pass
        return {"config":str(config),"return_code":p.returncode,"json_parsed":parsed,"stderr_tail":p.stderr[-500:]}
    finally: request_path.unlink(missing_ok=True)
def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=Path(r"configs/rnee_build/trajectory_build_production.yaml"));args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"));base=yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    authorization=load(Path(cfg["authorization_pointer"]));runtime_freeze=load(Path(cfg["runtime_freeze_summary"]));network=load(Path(cfg["network_pointer"]));inventory_path=Path(cfg["source_inventory"]);inventory_hash=sha256(inventory_path);inventory=pd.read_parquet(inventory_path).sort_values("source_file").reset_index(drop=True);ved_dir=Path(base["ved_dynamic_dir"])
    file_results=[]
    for row in inventory.itertuples(index=False):
        path=ved_dir/row.source_file; actual=sha256(path) if path.exists() else None
        file_results.append({"source_file":row.source_file,"path":str(path),"data_rows":int(row.data_rows),"bytes":int(row.bytes),"sha256":row.sha256,"actual_sha256":actual,"exists":path.exists(),"hash_matches":actual==row.sha256,"read_only":path.exists() and not bool(path.stat().st_mode & stat.S_IWRITE)})
    executable=Path(base["base_contract"] if False else yaml.safe_load(Path(base["base_contract"]).read_text(encoding="utf-8"))["valhalla"]["service_executable"])
    contract=yaml.safe_load(Path(base["base_contract"]).read_text(encoding="utf-8"));dll_dir=Path(contract["valhalla"]["dll_directory"])
    probes=[valhalla_probe(executable,Path(path),dll_dir) for path in network["valhalla_configs"].values()]
    test=subprocess.run([str(Path.cwd()/r".venv-rnee311/Scripts/python.exe"),"-m","pytest","tests/rnee_build","-q"],capture_output=True,text=True)
    free_disk_gb=shutil.disk_usage(Path(cfg["output_root"]).anchor).free/1024**3
    checks={
        "dry_run_authorization":"PASS" if authorization.get("full_build_authorized") is True else "FAIL",
        "runtime_freeze_contract":"PASS" if runtime_freeze["status"]=="PASS_CONTRACT_AND_ENVIRONMENT" and all(runtime_freeze["checks"].values()) else "FAIL",
        "inventory_hash":"PASS" if inventory_hash==cfg["source_inventory_sha256"] else "FAIL",
        "inventory_cardinality":"PASS" if len(inventory)==cfg["expected_source_file_count"] and int(inventory.data_rows.sum())==cfg["expected_source_row_count"] else "FAIL",
        "all_source_hashes":"PASS" if all(x["hash_matches"] for x in file_results) else "FAIL",
        "all_sources_read_only":"PASS" if all(x["read_only"] for x in file_results) else "FAIL",
        "historical_network":"PASS" if network["status"]=="PASS_OSM_SNAPSHOT_AND_NETWORK" and all(Path(x).exists() for x in network["graphs"].values()) else "FAIL",
        "valhalla_executable_probes":"PASS" if executable.exists() and all(x["return_code"]==0 and x["json_parsed"] for x in probes) else "FAIL",
        "free_disk":"PASS" if free_disk_gb>=cfg["minimum_free_disk_gb"] else "FAIL",
        "test_suite":"PASS" if test.returncode==0 and f"{cfg['expected_test_count']} passed" in test.stdout else "FAIL",
        "ved_only_source_policy":"PASS" if "eved" not in str(ved_dir).lower() and "eved" not in str(inventory_path).lower() else "FAIL",
    }
    gate="PASS" if "FAIL" not in checks.values() else "FAIL";stamp=datetime.now().strftime("%Y%m%d_%H%M%S");out=Path(cfg["output_root"])/f"{cfg['preflight_run_name_prefix']}_{stamp}";out.mkdir(parents=True);shutil.copy2(args.config,out/"run_config.yaml")
    partitions=[{"partition_id":f"partition_{i:04d}","source_file":x["source_file"],"data_rows":x["data_rows"],"bytes":x["bytes"],"sha256":x["sha256"]} for i,x in enumerate(file_results)]
    plan={"schema_version":1,"experiment":"TRAJECTORY_BUILD","created_at_utc":datetime.now(timezone.utc).isoformat(),"source_inventory":str(inventory_path),"source_inventory_sha256":inventory_hash,"source_file_count":len(partitions),"source_row_count":sum(x["data_rows"] for x in partitions),"partitions":partitions};dump(out/"production_partition_plan.json",plan)
    pd.DataFrame(file_results).to_csv(out/"source_hash_preflight.csv",index=False)
    status=f"{gate}_TRAJECTORY_BUILD_PRODUCTION_PREFLIGHT";summary={"experiment":"TRAJECTORY_BUILD","stage":"production_preflight","status":status,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"source_file_count":len(partitions),"source_row_count":plan["source_row_count"],"free_disk_gb":free_disk_gb,"valhalla_probes":probes,"test_output":test.stdout.strip(),"partition_plan":str(out/"production_partition_plan.json"),"output_directory":str(out)};dump(out/"trajectory_build_production_preflight_summary.json",summary)
    report=out/"TRAJECTORY_BUILD_PRODUCTION_PREFLIGHT.md";report.write_text(f"# TRAJECTORY_BUILD production preflight\n\n- Status: `{status}`\n- Frozen files / rows: {len(partitions)} / {plan['source_row_count']:,}\n- Source hashes and read-only flags: `{checks['all_source_hashes']}` / `{checks['all_sources_read_only']}`\n- Historical Valhalla probes: `{checks['valhalla_executable_probes']}`\n- Free disk: {free_disk_gb:.1f} GB\n- Tests: `{test.stdout.strip()}`\n",encoding="utf-8")
    pointer={"status":status,"summary":str(out/"trajectory_build_production_preflight_summary.json"),"report":str(report),"partition_plan":str(out/"production_partition_plan.json"),"output_directory":str(out)};dump(Path(cfg["output_root"])/f"{cfg['preflight_latest_prefix']}_latest_pointer.json",pointer);print(json.dumps(pointer,ensure_ascii=False,indent=2));return 0 if gate=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
