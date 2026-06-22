#!/usr/bin/env python3
"""
V8 CS-SupCon 复现评测：在 11 个数据集上计算 DR/FPR。

- 方法学对齐 scripts/test_comprehensive_v7.py（同款 DATASETS / 采样 / 指标公式）。
- 模型换成 v8：LearnedCSProjection(768->dim) + DualMultiCentroidDetector。
- 编码器复用 train.py 的 load_v7_encoder / extract_embeddings（保证与训练一致的 768D 嵌入）。
- 每个数据集只嵌入一次，6 个配置复用，省时。

用法: python scripts/v8_cs_supcon/evaluate.py  -> 写 models/v8_cs_supcon/eval_results.json
"""
import os
import sys
import gc
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
V8_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models" / "v8_cs_supcon"

sys.path.insert(0, str(V8_DIR))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device
from train import load_v7_encoder, extract_embeddings

random.seed(42)

# ── 数据集配置（与 test_comprehensive_v7.py 完全一致）──
DATASETS = {
    "GCG": {"path": DATASETS_DIR / "jailbreakbench" / "jbb_gcg_all.csv",
            "col": "prompt", "expected": "harmful", "n": 100},
    "PAIR": {"path": DATASETS_DIR / "jailbreakbench" / "jbb_pair_all.csv",
             "col": "prompt", "expected": "harmful", "n": 86},
    "JailbreakHub": {"path": DATASETS_DIR / "jailbreakhub" / "jailbreakhub.csv",
                     "col": "prompt", "expected": "harmful", "n": 79},
    "AdvBench": {"path": DATASETS_DIR / "advbench" / "advbench_harmful_behaviors.csv",
                 "col": "goal", "expected": "harmful", "n": 200},
    "HarmBench": {"path": DATASETS_DIR / "harmbench" / "harmbench_behaviors.csv",
                  "col": "Behavior", "expected": "harmful", "n": 200},
    "ToxicChat_harmful": {"path": DATASETS_DIR / "gcg_attacks" / "toxic_chat_full.csv",
                          "col": "user_input", "expected": "harmful", "n": 300,
                          "label_col": "human_annotation", "label_val": "True"},
    "BeaverTails_harmful": {"path": DATASETS_DIR / "beavertails" / "beavertails_test.csv",
                            "col": "prompt", "expected": "harmful", "n": 300,
                            "label_col": "is_safe", "label_val": False},
    "JBB_Benign": {"path": DATASETS_DIR / "gcg_attacks" / "jbb_benign_behaviors.csv",
                   "col": "Goal", "expected": "benign", "n": 100},
    "Alpaca": {"path": DATASETS_DIR / "normal" / "alpaca.jsonl",
               "col": "text", "expected": "benign", "n": 200},
    "ToxicChat_benign": {"path": DATASETS_DIR / "gcg_attacks" / "toxic_chat_full.csv",
                         "col": "user_input", "expected": "benign", "n": 300,
                         "label_col": "human_annotation", "label_val": "False"},
    "BeaverTails_benign": {"path": DATASETS_DIR / "beavertails" / "beavertails_test.csv",
                           "col": "prompt", "expected": "benign", "n": 300,
                           "label_col": "is_safe", "label_val": True},
}

# 评测的 6 个配置（与 committed eval_results.json 对齐）
CONFIGS = ["8d_1c", "16d_1c", "32d_1c", "32d_3c", "64d_1c", "128d_1c"]


def load_texts(cfg: dict) -> list:
    """与 test_comprehensive_v7.py 的 load_texts 完全一致。"""
    path = cfg["path"]
    if not path.exists():
        return []
    col = cfg["col"]
    n = cfg.get("n")
    label_col = cfg.get("label_col")
    label_val = cfg.get("label_val")

    if str(path).endswith(".jsonl"):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        texts = [r.get(col, "") for r in records if r.get(col, "").strip()]
        if n and len(texts) > n:
            texts = random.sample(texts, n)
        return texts

    df = pd.read_csv(path)
    if label_col and label_col in df.columns:
        col_vals = df[label_col].astype(str).str.strip().str.lower()
        filter_val = str(label_val).strip().lower()
        df = df[col_vals == filter_val]
    if col not in df.columns:
        for c in ["prompt", "goal", "Goal", "text", "Behavior", "user_input"]:
            if c in df.columns:
                col = c
                break
    texts = df[col].dropna().astype(str).tolist()
    texts = [t.strip() for t in texts if t.strip()]
    if n and len(texts) > n:
        texts = random.sample(texts, n)
    return texts


def parse_cfg(key: str):
    dpart, cpart = key.split("_")
    return int(dpart[:-1]), int(cpart[:-1])  # (dim, n_centroids)


def main():
    device = get_device()
    print(f"device: {device}")

    # 1. 加载 V7 微调编码器（与训练一致）
    print("[1/3] 加载 V7 编码器...")
    encoder, tokenizer = load_v7_encoder(device)

    # 2. 每个数据集嵌入一次（random 顺序与 test_comprehensive 一致）
    print("[2/3] 嵌入各数据集...")
    ds_emb, ds_expected = {}, {}
    for name, cfg in DATASETS.items():
        texts = load_texts(cfg)
        if not texts:
            print(f"  ✗ {name} 不可用，跳过")
            continue
        emb = extract_embeddings(encoder, tokenizer, texts, device)
        ds_emb[name] = emb.astype(np.float32)
        ds_expected[name] = cfg["expected"]
        print(f"  {name}: {len(texts)} 条已嵌入 -> {emb.shape}")
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # training_results.json 提供每个配置的阈值
    tr = {}
    tr_path = MODEL_DIR / "training_results.json"
    if tr_path.exists():
        tr = json.load(open(tr_path))

    # 3. 逐配置评测
    print("[3/3] 逐配置评测...")
    results = {}
    for key in CONFIGS:
        dim, nc = parse_cfg(key)
        proj_path = MODEL_DIR / f"cs_projection_{dim}d.pt"
        det_path = MODEL_DIR / f"detector_{key}.npz"
        if not proj_path.exists() or not det_path.exists():
            print(f"  ✗ {key} 缺权重，跳过 ({proj_path.exists()=}, {det_path.exists()=})")
            continue
        proj = LearnedCSProjection(768, dim)
        proj.load_state_dict(torch.load(str(proj_path), map_location="cpu"))
        proj.eval()
        det = DualMultiCentroidDetector.load(str(det_path))
        thr = tr.get(key, {}).get("threshold", 0.0)

        res = {}
        for name, emb in ds_emb.items():
            with torch.no_grad():
                z = proj(torch.tensor(emb, dtype=torch.float32)).numpy()
            preds, _ = det.predict(z, threshold=thr)
            N = len(preds)
            if ds_expected[name] == "harmful":
                asr = float((preds == 0).sum()) / N
                res[name] = {"detection_rate": round(1 - asr, 4),
                             "asr": round(asr, 4), "total": N}
            else:
                fpr = float((preds == 1).sum()) / N
                res[name] = {"fpr": round(fpr, 4), "total": N}
        results[key] = res
        adv = res.get("AdvBench", {}).get("detection_rate")
        pair = res.get("PAIR", {}).get("detection_rate")
        jbb = res.get("JBB_Benign", {}).get("fpr")
        bt = res.get("BeaverTails_benign", {}).get("fpr")
        print(f"  [{key}] thr={thr} | AdvBench DR={adv} PAIR DR={pair} "
              f"| JBB_Benign FPR={jbb} BeaverTails_benign FPR={bt}")

    out_path = MODEL_DIR / "eval_results.json"
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
