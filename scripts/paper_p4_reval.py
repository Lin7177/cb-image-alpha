from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paper_replication.p3_supervision import supervise

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/paper-v1/p4-reval-20260919-r1'; CFG=ROOT/'configs/paper/p4_reval_v1.toml'
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
if __name__=='__main__':
 OUT.mkdir(exist_ok=False); (OUT/'binding.json').write_text(json.dumps({'run_id':'p4-reval-20260919-r1','protocol_sha256':sha(CFG),'code_sha256':sha(Path(__file__).with_name('paper_p4_reval_worker.py')),'oos_seen':True},indent=2))
 r=supervise([sys.executable,str(Path(__file__).with_name('paper_p4_reval_worker.py'))],OUT/'supervision',OUT,14400,rss_limit=8*1024**3,disk_limit=8*1024**3)
 if r['status']!='passed': raise SystemExit(1)