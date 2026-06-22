#!/usr/bin/env python3
"""扫阈值画 DR-FPR 帕累托曲线，对比多个 v8/v9 模型目录。

攻击集只用句法/语义攻击（剔除标注语义不匹配的 ToxicChat/BeaverTails）。
对每个模型目录，加载 cs_projection_{DIM}d.pt + detector_{DIM}d_{NC}c.npz，
扫阈值算 (平均攻击 DR, JBB_Benign FPR, Alpaca FPR)。
用法: python scripts/v9_boundary/pareto_eval.py [--models name=dir,name=dir,...]
"""
import os, sys, json, random, argparse
import numpy as np, torch
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa: E402
from train import load_v7_encoder, extract_embeddings  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location("v8eval", str(BASE / "scripts/v8_cs_supcon/evaluate.py"))
v8eval = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8eval)  # 复用 DATASETS/load_texts

random.seed(42)
ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench"]
BENIGN = ["JBB_Benign", "Alpaca"]
DIM, NC = 32, 1
THRESHOLDS = [round(float(t), 3) for t in np.arange(-0.5, 0.51, 0.02)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8=models/v8_cs_supcon,v9_a1=models/v9_boundary")
    ap.add_argument("--out", default="results/v9_boundary/pareto.json")
    args = ap.parse_args()
    models = {}
    for pair in args.models.split(","):
        name, d = pair.split("=", 1)
        models[name] = BASE / d

    device = get_device()
    encoder, tok = load_v7_encoder(device)
    emb = {}
    for name in ATTACK + BENIGN:
        texts = v8eval.load_texts(v8eval.DATASETS[name])
        emb[name] = extract_embeddings(encoder, tok, texts, device).astype(np.float32)
        print(f"{name}: {emb[name].shape}")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for mname, mdir in models.items():
        pj = mdir / f"cs_projection_{DIM}d.pt"
        dj = mdir / f"detector_{DIM}d_{NC}c.npz"
        if not pj.exists() or not dj.exists():
            print(f"SKIP {mname}: missing {pj.exists()=} {dj.exists()=}"); continue
        proj = LearnedCSProjection(768, DIM)
        proj.load_state_dict(torch.load(str(pj), map_location="cpu")); proj.eval()
        det = DualMultiCentroidDetector.load(str(dj))
        with torch.no_grad():
            z = {n: proj(torch.tensor(e)).numpy() for n, e in emb.items()}
        curve = []
        for t in THRESHOLDS:
            drs = [float((det.predict(z[n], t)[0] == 1).mean()) for n in ATTACK]
            jbb = float((det.predict(z["JBB_Benign"], t)[0] == 1).mean())
            alp = float((det.predict(z["Alpaca"], t)[0] == 1).mean())
            curve.append({"thr": t, "avg_attack_DR": round(sum(drs) / len(drs), 4),
                          "JBB_Benign_FPR": round(jbb, 4), "Alpaca_FPR": round(alp, 4)})
        out[mname] = curve
        print(f"{mname}: done")
    (BASE / args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BASE / args.out, "w"), indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
