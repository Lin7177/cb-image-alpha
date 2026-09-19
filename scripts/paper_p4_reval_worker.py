"""Fast, single-pass seen-OOS reproducibility rebuild; no training or selection."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from pyarrow import feather

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cb_image_alpha.models.paper_cnn import PaperCNN

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/paper/p4_reval_v1.toml'; OUT=ROOT/'artifacts/paper-v1/p4-reval-20260919-r2'
ZIP=ROOT/'artifacts/paper-v1/acquire-20260917T010443459965Z/recovery-1/img_data.zip'; INV=ROOT/'artifacts/paper-v1/p4-oos-inventory.json'
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def auc(y,p):
 pos=p[y==1]; neg=p[y==0]
 if len(pos)==0 or len(neg)==0:return None
 order=np.argsort(p,kind='mergesort'); ranks=np.empty(len(p)); ranks[order]=np.arange(1,len(p)+1); return float((ranks[y==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def metrics(y,p):
 pred=(p>=.5).astype(np.int64); cm=np.bincount(y*2+pred,minlength=4); tn,fp,fn,tp=cm
 return {'rows':len(y),'cross_entropy':float(-np.mean(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))),'roc_auc':auc(y,p),'accuracy':float((pred==y).mean()),'balanced_accuracy':float(((tp/(tp+fn)) if tp+fn else 0)+((tn/(tn+fp)) if tn+fp else 0))/2,'brier':float(np.mean((p-y)**2)),'positive_rate':float(y.mean())}
def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    cfg = tomllib.loads(CFG.read_text()); inv = json.loads(INV.read_text())
    if cfg["status"] != "FROZEN" or sha(INV) != cfg["inventory_sha256"] or inv["data_member_count"] != 38 or inv["metadata_member_count"] != 38:
        raise RuntimeError("reval gate")
    binds = [(ROOT / cfg["archive_manifest"], cfg["archive_manifest_sha256"]), (ROOT / cfg["p4_protocol"], cfg["p4_protocol_sha256"]), (ROOT / cfg["p3_audit"], cfg["p3_audit_sha256"]), (ROOT / "src/cb_image_alpha/models/paper_cnn.py", cfg["model_source_sha256"]), (ROOT / "uv.lock", cfg["environment_lock_sha256"])]
    if any(sha(p) != h for p, h in binds) or inv["archive_sha256"] != cfg["archive_sha256"] or sha(ZIP) != cfg["archive_sha256"]:
        raise RuntimeError("input hash mismatch")
    expected = {f"monthly_20d/20d_month_has_vb_[20]_ma_{y}_{s}" for y in range(2001, 2020) for s in ("images.dat", "labels_w_delay.feather")}
    with zipfile.ZipFile(ZIP) as z:
        infos = {x.filename: x for x in z.infolist()}; data = {x["name"]: x for x in inv["data_members"]}
        if set(data) != expected or len(inv["metadata_members"]) != 38:
            raise RuntimeError("allowlist mismatch")
        metadata_items = {x["name"]: x for x in inv["metadata_members"]}
        metadata = set(metadata_items)
        if set(infos) != expected | metadata:
            raise RuntimeError("archive contains unexpected or unallowed members")
        for name, item in {**data, **metadata_items}.items():
            x = infos.get(name)
            if x is None or x.file_size != item["bytes"] or f"{x.CRC:08x}" != item["crc32"]:
                raise RuntimeError("member inventory mismatch")
        seal = json.loads((ROOT / "artifacts/paper-v1/p3-paper-training-v1/seal/manifest.json").read_text())
        for member in seal["members"]:
            for name, expected_hash in member["files"].items():
                path = ROOT / "artifacts/paper-v1/p3-paper-training-v1/seal" / f"member-{member['member']}" / name
                if sha(path) != expected_hash:
                    raise RuntimeError("sealed file hash mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.set_num_threads(cfg["resources"]["cpu_threads"])
        torch.cuda.set_per_process_memory_fraction(cfg["resources"]["cuda_reserved_limit_bytes"] / torch.cuda.get_device_properties(0).total_memory)
        models = []; norms = []
        for m in range(5):
            model = PaperCNN().cuda().eval(); state = torch.load(ROOT / f"artifacts/paper-v1/p3-paper-training-v1/seal/member-{m}/state.pt", map_location="cuda", weights_only=False); model.load_state_dict(state["model"]); models.append(model); norms.append(json.loads((ROOT / f"artifacts/paper-v1/p3-paper-training-v1/seal/member-{m}/normalization.json").read_text()))
        ids_parts = []; labels_parts = []; preds = [[] for _ in range(5)]; exclusions = []; started = time.monotonic()
        for year in range(2001, 2020):
            ip = f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_images.dat"; lp = f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"; raw = z.read(ip); tab = feather.read_table(BytesIO(z.read(lp))); n = tab.num_rows
            if len(raw) != n * 3840: raise RuntimeError("image row mismatch")
            required = {"Date", "StockID", "Ret_month", "EWMA_vol"}
            if not required.issubset(tab.column_names):
                raise RuntimeError(f"schema mismatch: {sorted(required - set(tab.column_names))}")
            dates = np.asarray(tab["Date"].to_pylist(), dtype="datetime64[D]")
            if np.isnat(dates).any() or np.any(dates.astype("datetime64[Y]").astype(int) + 1970 != year):
                raise RuntimeError("date/year mismatch")
            stocks = tab["StockID"].to_pylist()
            if any(stock is None for stock in stocks):
                raise RuntimeError("null stock identifier")
            keys = list(zip(dates.astype(str).tolist(), stocks, strict=True))
            if len(set(keys)) != n:
                raise RuntimeError("duplicate date/stock key")
            ret = np.asarray(tab["Ret_month"].to_pylist(), dtype=np.float64); ew = np.asarray(tab["EWMA_vol"].to_pylist(), dtype=np.float64); keep = np.isfinite(ret) & (ret != -99) & np.isfinite(ew) & (ew > 0)
            if not keep.any():
                raise RuntimeError("no valid rows")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 64, 60)[keep]
            if not np.all((arr == 0) | (arr == 255)) or np.any(np.count_nonzero(arr, axis=(1, 2)) == 0): raise RuntimeError("invalid image")
            rows = np.flatnonzero(keep); ids_parts.append(year * 1_000_000_000 + rows); labels_parts.append((ret[keep] > 0).astype(np.int64)); exclusions.append({"year": year, "raw": n, "kept": int(keep.sum()), "excluded": int((~keep).sum())})
            for begin in range(0, len(arr), cfg["performance"]["batch_size"]):
                raw_batch = torch.from_numpy(arr[begin:begin + cfg["performance"]["batch_size"]].copy()).cuda().unsqueeze(1)
                base = raw_batch.float().div_(255.)
                for m, (model, norm) in enumerate(zip(models, norms, strict=True)):
                    x = base.sub(float(norm["mean"])).div(float(norm["std"]))
                    with torch.inference_mode(): preds[m].append(model(x).softmax(1)[:, 1].cpu().numpy())
    ids = np.concatenate(ids_parts); y = np.concatenate(labels_parts); prob = [np.concatenate(x) for x in preds]
    for m, p in enumerate(prob):
        if len(p) != len(ids) or not np.isfinite(p).all(): raise RuntimeError("prediction integrity")
        np.savez_compressed(OUT / f"member-{m}-predictions.npz", sample_id=ids, label=y, probability=p)
    ensemble = np.mean(np.stack(prob), axis=0); result = {"status": "COMPLETED_AWAITING_AUDIT", "run_id": cfg["run_id"], "read_class": "already_seen_OOS_reassessment", "rows": len(y), "common_rows": len(y), "exclusions": exclusions, "member_metrics": [metrics(y, p) for p in prob], "ensemble_metrics": metrics(y, ensemble), "elapsed_seconds": time.monotonic() - started, "protocol_sha256": sha(CFG), "worker_sha256": sha(Path(__file__)), "inventory_sha256": sha(INV), "archive_sha256": cfg["archive_sha256"], "input_fingerprints": {k: cfg[k] for k in ("archive_manifest_sha256", "inventory_sha256", "p4_protocol_sha256", "p3_audit_sha256", "model_source_sha256", "environment_lock_sha256", "archive_sha256")}, "p3_seal_manifest_sha256": sha(ROOT / "artifacts/paper-v1/p3-paper-training-v1/seal/manifest.json"), "endpoint_verified": False, "strict_leakage_free_claim": False, "test_qualification": "none", "old_results_guidance": False, "oos_reread_for_reassessment": True, "no_training": True, "no_tuning": True, "no_backtest": True, "device": "cuda", "cuda_device": torch.cuda.get_device_name(0)}
    atomic_json(OUT / "results.json", result)


if __name__ == "__main__":
    main()

