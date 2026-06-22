#!/usr/bin/env python3
"""边界判别上限 ~0.86 到底是 encoder 的极限,还是 mean-pooling 的瓶颈?
v7/v8/v9 全部跑在 mean-pooling 的单个 768D 向量上。边界例(如'kill a process' vs 'kill a person')
的判别 token 很可能被平均稀释。本实验在同一 encoder 上换 4 种池化,只测边界 AUC,不训练:
  mean(现状) / cls / max / last。若 max/cls 显著 > mean -> 边界 FP 是'池化瓶颈',不是几何/压缩问题。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder  # noqa
from model import get_device  # noqa
np.random.seed(42)


def load():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:300]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:300]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:600]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    return adv + harm, alp, jbb


def pools(model, tok, texts, dev, bs=32):
    out = {"mean": [], "cls": [], "max": [], "last": []}
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]
        x = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.no_grad():
            H = model.encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        mean = (H * m).sum(1) / m.sum(1).clamp(min=1e-9)
        cls = H[:, 0]
        Hm = H.masked_fill(m == 0, -1e9); mx = Hm.max(1).values
        idx = x["attention_mask"].sum(1) - 1
        last = H[torch.arange(H.size(0), device=dev), idx]
        for k, v in zip(out, (mean, cls, mx, last)):
            out[k].append(v.cpu().numpy())
    return {k: np.vstack(v).astype(np.float64) for k, v in out.items()}


def auc(X, y):
    return cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                           StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean()


def main():
    dev = get_device()
    model, tok = load_v7_encoder(dev)
    harm_t, easy_t, bound_t = load()
    Ph = pools(model, tok, harm_t, dev)
    Pe = pools(model, tok, easy_t, dev)
    Pb = pools(model, tok, bound_t, dev)
    yb = np.array([1] * len(harm_t) + [0] * len(bound_t))   # boundary
    ye = np.array([1] * len(harm_t) + [0] * len(easy_t))    # easy ref
    print(f"harmful={len(harm_t)} easy={len(easy_t)} boundary={len(bound_t)}")
    print(f"\n{'池化':<8}{'easy-AUC':>10}{'boundary-AUC':>14}")
    for k in ("mean", "cls", "max", "last"):
        Xe = np.vstack([Ph[k], Pe[k]]); Xb = np.vstack([Ph[k], Pb[k]])
        print(f"{k:<8}{auc(Xe, ye):>10.4f}{auc(Xb, yb):>14.4f}")
    print("POOL_DONE")


if __name__ == "__main__":
    main()
