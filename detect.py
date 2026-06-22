#!/usr/bin/env python3
"""
交互式恶意输入检测接口 — V8 CS-SupCon (32d_1c)

加载真实的 v8 模型：
  V7 微调编码器 (BAAI/bge-base-en-v1.5, 768D)
  → LearnedCSProjection(768 -> 32)
  → DualMultiCentroidDetector (双侧多质心, 余弦打分)

用法:
    python detect.py            # 逐行从 stdin 读入文本检测
    echo "..." | python detect.py
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa: E402
from train import load_v7_encoder, extract_embeddings  # noqa: E402

CONFIG = "32d_1c"
DIM = 32


def load_detector():
    device = get_device()
    print("加载 V7 编码器 + V8 投影/检测器 ...", file=sys.stderr)
    encoder, tokenizer = load_v7_encoder(device)
    proj = LearnedCSProjection(768, DIM)
    proj.load_state_dict(torch.load(
        str(BASE_DIR / f"models/v8_cs_supcon/cs_projection_{DIM}d.pt"),
        map_location="cpu"))
    proj.eval()
    det = DualMultiCentroidDetector.load(
        str(BASE_DIR / f"models/v8_cs_supcon/detector_{CONFIG}.npz"))
    tr_path = BASE_DIR / "models/v8_cs_supcon/training_results.json"
    thr = json.load(open(tr_path)).get(CONFIG, {}).get("threshold", 0.0) if tr_path.exists() else 0.0
    return encoder, tokenizer, proj, det, thr, device


def detect_text(text, encoder, tokenizer, proj, det, thr, device):
    emb = extract_embeddings(encoder, tokenizer, [text], device)  # [1,768]
    with torch.no_grad():
        z = proj(torch.tensor(emb, dtype=torch.float32)).numpy()  # [1,32]
    preds, score = det.predict(z, threshold=thr)
    return int(preds[0]), float(score[0]), z[0]


def main():
    encoder, tokenizer, proj, det, thr, device = load_detector()
    print(f"就绪 (device={device}, config={CONFIG}, threshold={thr}). "
          f"逐行输入文本检测，Ctrl-D 退出。", file=sys.stderr)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        pred, score, z = detect_text(text, encoder, tokenizer, proj, det, thr, device)
        verdict = "恶意/有害 (harmful)" if pred == 1 else "正常 (benign)"
        print(f"  text: {text[:80]}")
        print(f"  压缩向量[:5]: {np.round(z[:5], 4).tolist()}")
        print(f"  score(max_harmful_sim - max_benign_sim) = {score:.4f}  (threshold={thr})")
        print(f"  => {verdict}\n")


if __name__ == "__main__":
    main()
