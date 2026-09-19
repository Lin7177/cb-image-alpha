from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from paper_replication.p3_supervision import supervise

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts/paper-v1/p4-reval-20260919-r2"; CFG=ROOT/"configs/paper/p4_reval_v1.toml"; WORKER=Path(__file__).with_name("paper_p4_reval_worker.py")
def sha(p): return hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
if __name__=="__main__":
 cfg=tomllib.loads(CFG.read_text())
 if cfg["status"]!="FROZEN" or OUT.exists(): raise SystemExit("run gate")
 if shutil.disk_usage(ROOT).free < cfg["resources"]["minimum_free_disk_bytes"]: raise SystemExit("free disk gate")
 OUT.mkdir(); (OUT/"binding.json").write_text(json.dumps({"run_id":cfg["run_id"],"protocol_sha256":sha(CFG),"wrapper_sha256":sha(Path(__file__)),"worker_sha256":sha(WORKER),"input_fingerprints":{k:cfg[k] for k in ("archive_manifest_sha256","inventory_sha256","p4_protocol_sha256","p3_audit_sha256","model_source_sha256","environment_lock_sha256","archive_sha256")},"oos_seen":True},indent=2),encoding="utf-8")
 os.environ["P4_REVAL_OUTPUT"]=str(OUT)
 result=supervise([sys.executable,str(WORKER)],OUT/"supervision",OUT,cfg["resources"]["total_wall_seconds"],rss_limit=cfg["resources"]["rss_tree_limit_bytes"],disk_limit=cfg["resources"]["artifact_limit_bytes"])
 if result["status"]!="passed" or not (OUT/"results.json").exists(): raise SystemExit(1)
 ledger=json.loads((OUT/"supervision"/"ledger.jsonl").read_text().splitlines()[-1])
 if ledger.get("event")!="finished" or ledger.get("status")!="passed": raise SystemExit(1)
 files=[]
 for path in sorted(OUT.glob("member-*-predictions.npz")):
  files.append({"path":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
 files.append({"path":"results.json","bytes":(OUT/"results.json").stat().st_size,"sha256":sha(OUT/"results.json")})
 (OUT/"seal.json.tmp").write_text(json.dumps({"status":"SEALED_AWAITING_AUDIT","run_id":cfg["run_id"],"protocol_sha256":sha(CFG),"wrapper_sha256":sha(Path(__file__)),"worker_sha256":sha(WORKER),"input_fingerprints":{k:cfg[k] for k in ("archive_manifest_sha256","inventory_sha256","p4_protocol_sha256","p3_audit_sha256","model_source_sha256","environment_lock_sha256","archive_sha256")},"supervision":result,"files":files,"test_qualification":"none","oos_seen":True},indent=2),encoding="utf-8")
 os.replace(OUT/"seal.json.tmp",OUT/"seal.json")