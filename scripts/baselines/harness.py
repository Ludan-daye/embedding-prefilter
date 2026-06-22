#!/usr/bin/env python3
"""统一基线评测：复用 v8 evaluate.py 的 11 数据集协议，对任意 predict_batch 算 DR/FPR/延迟。"""
import os, sys, json, time, importlib.util
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.parent.parent
_spec = importlib.util.spec_from_file_location("v8eval", str(BASE / "scripts/v8_cs_supcon/evaluate.py"))
v8eval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v8eval)
DATASETS = v8eval.DATASETS          # 11 数据集
load_texts = v8eval.load_texts      # seed=42 在 v8eval 模块顶部已 set


def _to01(p):
    if isinstance(p, str):
        return 1 if p.strip().lower() == "harmful" else 0
    return int(p)


def run_baseline(name, predict_batch, params="?", dim="-", note=""):
    """predict_batch: list[str]->list[int|str]; 写 results/baselines/<name>/metrics.json"""
    out = {"method": name, "params": params, "dim": dim, "note": note, "datasets": {}}
    t_total, n_total = 0.0, 0
    for ds, cfg in DATASETS.items():
        texts = load_texts(cfg)
        t0 = time.perf_counter()
        preds = [_to01(p) for p in predict_batch(texts)]
        dt = time.perf_counter() - t0
        t_total += dt; n_total += len(texts)
        N = len(texts); pos = int(sum(preds))
        if cfg["expected"] == "harmful":
            out["datasets"][ds] = {"detection_rate": round(pos / N, 4), "total": N}
        else:
            out["datasets"][ds] = {"fpr": round(pos / N, 4), "total": N}
    out["latency_ms_per_sample"] = round(1000 * t_total / max(n_total, 1), 3)
    d = BASE / "results/baselines" / name; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "metrics.json", "w"), indent=2, ensure_ascii=False)
    print(f"[{name}] latency={out['latency_ms_per_sample']}ms/sample")
    for ds in DATASETS:
        m = out["datasets"][ds]
        print(f"  {ds:20s} {m.get('detection_rate', m.get('fpr'))}")
    return out
