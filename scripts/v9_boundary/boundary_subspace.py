#!/usr/bin/env python3
"""确证实验：边界判别信息是否活在'主安全轴'的正交补里。

负面结果链已证明: (1) easy-case 几乎 rank-1、任何投影都行(JL);
(2) hard-case 上 v9 压缩的 32D 最差、丢弃的补空间最好。
本实验直接给出几何解释 —— 把 v9 的"学到的压缩"换成可解释的'主安全轴 PCA':
  d_bulk  = mean(harmful) - mean(easy_benign=Alpaca)   # 主安全轴(易例)
  d_bound = mean(harmful) - mean(bound_benign=JBB)      # 边界轴(难例)
预测:
  - cos(d_bulk, d_bound) 偏小  -> 两条判别轴并不同向
  - bulk-PCA-top32 子空间:  bulk-AUC 高,  bound-AUC 低   (主轴对边界没用)
  - 其正交补(736D):         bound-AUC 高                 (边界信息在补里)
若成立 => 真正的贡献不是"压缩",而是"边界信息正交于主安全轴,压到主轴上恰好把它丢掉"。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder, extract_embeddings  # noqa
from model import get_device  # noqa
np.random.seed(42)


def load():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:300]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:300]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:600]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    return adv + harm, alp, jbb


def auc(X, y):
    return cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                           StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean()


def main():
    dev = get_device()
    enc, tok = load_v7_encoder(dev)
    harm_t, easy_t, bound_t = load()
    H = extract_embeddings(enc, tok, harm_t, dev).astype(np.float64)
    Eb = extract_embeddings(enc, tok, easy_t, dev).astype(np.float64)
    Bb = extract_embeddings(enc, tok, bound_t, dev).astype(np.float64)
    print(f"harmful={len(H)} easy_benign={len(Eb)} bound_benign={len(Bb)}")

    d_bulk = H.mean(0) - Eb.mean(0); d_bulk /= np.linalg.norm(d_bulk)
    d_bound = H.mean(0) - Bb.mean(0); d_bound /= np.linalg.norm(d_bound)
    print(f"\ncos(主安全轴 d_bulk, 边界轴 d_bound) = {float(d_bulk @ d_bound):.4f}")

    # 主安全轴 PCA-top-32 子空间 (在 harmful+easy 上拟合方差主轴)
    pca = PCA(n_components=32, random_state=42).fit(np.vstack([H, Eb]))
    B = pca.components_.T                      # (768,32) 正交基
    def proj(X): return X @ B                  # 投到 bulk-top32
    def comp(X): return X - (X @ B) @ B.T       # 投到补空间(仍 768 维但秩 736)

    yb = np.array([1] * len(H) + [0] * len(Eb))       # bulk 标签
    Xb = np.vstack([H, Eb])
    yd = np.array([1] * len(H) + [0] * len(Bb))       # boundary 标签
    Xd = np.vstack([H, Bb])

    print("\n=== AUC: 行=任务, 列=所在子空间 ===")
    print(f"{'':<16}{'full768':>10}{'bulk-top32':>12}{'补空间736':>12}")
    print(f"{'bulk(易例)':<14}{auc(Xb, yb):>10.4f}{auc(proj(Xb), yb):>12.4f}{auc(comp(Xb), yb):>12.4f}")
    print(f"{'boundary(难例)':<12}{auc(Xd, yd):>10.4f}{auc(proj(Xd), yd):>12.4f}{auc(comp(Xd), yd):>12.4f}")

    # 把主安全轴(rank-1)从 embedding 里抠掉后,边界判别还剩多少
    def kill1(X): return X - np.outer(X @ d_bulk, d_bulk)
    print(f"\n抠掉主安全轴(rank-1)后 boundary-AUC = {auc(kill1(Xd), yd):.4f}  (full={auc(Xd, yd):.4f})")
    print("BSUB_DONE")


if __name__ == "__main__":
    main()
