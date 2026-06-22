#!/usr/bin/env python3
"""
V8 (32d_1c) 延迟基准。

分两段计时：
  full      = 编码器(BGE) + 投影 + 双质心检测（端到端单条）
  proj_only = 仅 投影 + 检测器（嵌入已预先算好）
另报批量端到端吞吐。

用法: python scripts/bench_latency.py
"""
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa: E402
from train import load_v7_encoder, extract_embeddings  # noqa: E402

DIM = 32
CONFIG = "32d_1c"
N = 200


def stats(arr):
    a = np.asarray(arr, dtype=float)
    return {"mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3)}


def main():
    device = get_device()
    encoder, tokenizer = load_v7_encoder(device)
    proj = LearnedCSProjection(768, DIM)
    proj.load_state_dict(torch.load(
        str(BASE_DIR / f"models/v8_cs_supcon/cs_projection_{DIM}d.pt"), map_location="cpu"))
    proj.eval()
    det = DualMultiCentroidDetector.load(
        str(BASE_DIR / f"models/v8_cs_supcon/detector_{CONFIG}.npz"))
    tr_path = BASE_DIR / "models/v8_cs_supcon/training_results.json"
    thr = json.load(open(tr_path)).get(CONFIG, {}).get("threshold", 0.0) if tr_path.exists() else 0.0

    texts = [f"sample user query number {i} about various everyday topics and questions" for i in range(N)]

    # warmup
    _ = extract_embeddings(encoder, tokenizer, texts[:8], device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # full pipeline, per-sample
    t_full = []
    for t in texts:
        t0 = time.perf_counter()
        emb = extract_embeddings(encoder, tokenizer, [t], device)
        with torch.no_grad():
            z = proj(torch.tensor(emb, dtype=torch.float32)).numpy()
        det.predict(z, threshold=thr)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_full.append((time.perf_counter() - t0) * 1000)

    # proj + detector only (embeddings precomputed)
    emb_all = extract_embeddings(encoder, tokenizer, texts, device).astype(np.float32)
    t_pd = []
    for i in range(N):
        t0 = time.perf_counter()
        with torch.no_grad():
            z = proj(torch.tensor(emb_all[i:i + 1], dtype=torch.float32)).numpy()
        det.predict(z, threshold=thr)
        t_pd.append((time.perf_counter() - t0) * 1000)

    # batch end-to-end
    t0 = time.perf_counter()
    emb = extract_embeddings(encoder, tokenizer, texts, device)
    with torch.no_grad():
        z = proj(torch.tensor(emb, dtype=torch.float32)).numpy()
    det.predict(z, threshold=thr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - t0) * 1000

    print(f"device: {device}  N={N}  config={CONFIG}")
    print(f"full  (encoder+proj+detector) per-sample ms: {stats(t_full)}")
    print(f"proj+detector only            per-sample ms: {stats(t_pd)}")
    print(f"batch {N} end-to-end total: {batch_ms:.1f} ms  => {batch_ms / N:.3f} ms/sample")


if __name__ == "__main__":
    main()
