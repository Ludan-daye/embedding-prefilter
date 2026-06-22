#!/usr/bin/env python3
"""难-case 子空间实验：有害 vs 边界正常(JBB-Benign, 训练时严格留出)。
测 v9 学到的 32D 是否比随机 32D / 全 768D / 丢弃补空间 更能分开"难样本"。
若 v9-32D 显著 > 随机32D → 学到的投影专门重塑了边界几何（v9 的真贡献）。"""
import sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder, extract_embeddings  # noqa: E402
from model import get_device  # noqa: E402

random.seed(42); np.random.seed(42)
MDIR = BASE / "models/v9_a2_l1.0_m0.5"


def orth_np(A):
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())
    return U[:, :r]


def load_hard():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:200]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:200]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()  # 边界正常(留出)
    harmful = adv + harm
    texts = harmful + jbb
    y = np.array([1] * len(harmful) + [0] * len(jbb))
    return texts, y


def auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                           Xs, y, cv=5, scoring="roc_auc").mean()


def main():
    device = get_device()
    enc, tok = load_v7_encoder(device)
    texts, y = load_hard()
    E = extract_embeddings(enc, tok, texts, device).astype(np.float64)
    print(f"难case: 有害={int(y.sum())}  边界正常(JBB)={int((y == 0).sum())}  emb={E.shape}")

    W = torch.load(MDIR / "cs_projection_32d.pt", map_location="cpu")["proj.weight"].numpy().astype(np.float64)
    B = orth_np(W.T); r = B.shape[1]
    z_keep = E @ B
    E_drop = E - z_keep @ B.T
    R = orth_np(np.random.randn(768, r))
    z_rand = E @ R

    print("\n=== 有害 vs 边界正常 的判别 AUC (5-fold) ===")
    print(f"  full 768D            : {auc(E, y):.4f}")
    print(f"  v9 的 {r}D 子空间      : {auc(z_keep, y):.4f}")
    print(f"  丢弃的 {768 - r}D 补空间 : {auc(E_drop, y):.4f}")
    print(f"  随机 {r}D (对照)      : {auc(z_rand, y):.4f}")
    print("HARD_DONE")


if __name__ == "__main__":
    main()
