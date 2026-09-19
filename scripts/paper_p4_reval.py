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

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts/paper-v1/p4-reval-20260919-r1"; CFG=ROOT/"configs/paper/p4_reval_v1.toml"; WORKER=Path(__file__).with_name("paper_p4_reval_worker.py")
def sha(p): return hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
if __name__=="__main__":
 cfg=tomllib.loads(CFG.read_text())
 if cfg["status"]!="FROZEN" or OUT.exists(): raise SystemExit("run gate")
 if shutil.disk_usage(ROOT).free < cfg["resources"]["minimum_free_disk_bytes"]: raise SystemExit("free disk gate")
 OUT.mkdir(); (OUT/"binding.json").write_text(json.dumps({"run_id":cfg["run_id"],"protocol_sha256":sha(CFG),"worker_sha256":sha(WORKER),"oos_seen":True},indent=2),encoding="utf-8")
 os.environ["P4_REVAL_OUTPUT"]=str(OUT)
 result=supervise([sys.executable,str(WORKER)],OUT/"supervision",OUT,cfg["resources"]["total_wall_seconds"],rss_limit=cfg["resources"]["rss_tree_limit_bytes"],disk_limit=cfg["resources"]["artifact_limit_bytes"])
 if result["status"]!="passed": raise SystemExit(1)