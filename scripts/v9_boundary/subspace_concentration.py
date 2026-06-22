#!/usr/bin/env python3
"""验证 (路线B 理论地基)：安全判别信号是否富集在 v9 学到的 32D 子空间，而非散布在 768D。
对比 logistic 探针 5-fold AUC：全 768D / v9-32D 子空间 / 丢弃的 736D 补空间 / 随机 32D 对照。
若 v9-32D ≈ full ≫ 补空间 ≈ 0.5 且 ≫ 随机32D → 安全信号确实富集在学到的低维子空间。"""
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
    """A: [d, k] -> 列正交基 of range(A), [d, r]"""
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())
    return U[:, :r]


def load_labeled():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:300]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:300]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:600]
    texts = adv + harm + alp
    y = np.array([1] * (len(adv) + len(harm)) + [0] * len(alp))
    return texts, y


def auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=3000), Xs, y, cv=5, scoring="roc_auc").mean()


def main():
    device = get_device()
    enc, tok = load_v7_encoder(device)
    texts, y = load_labeled()
    E = extract_embeddings(enc, tok, texts, device).astype(np.float64)
    print(f"embeddings {E.shape}  harmful={int(y.sum())} benign={int((y == 0).sum())}")

    W = torch.load(MDIR / "cs_projection_32d.pt", map_location="cpu")["proj.weight"].numpy().astype(np.float64)  # 32x768
    B = orth_np(W.T)            # 768 x r  正交基(保留子空间)
    r = B.shape[1]
    print(f"v9 保留子空间维度 r={r}")
    z_keep = E @ B             # N x r
    E_keep = z_keep @ B.T      # 重建到 768
    E_drop = E - E_keep        # 丢弃补空间 (768-r 维)
    R = orth_np(np.random.randn(768, r))   # 随机 r 维子空间
    z_rand = E @ R

    def varfrac(M, basis):
        Mc = M - M.mean(0)
        return float(((Mc @ basis) ** 2).sum() / (Mc ** 2).sum())

    print("\n=== 判别探针 AUC (5-fold, logistic) ===")
    print(f"  full 768D            : {auc(E, y):.4f}")
    print(f"  v9 的 {r}D 子空间      : {auc(z_keep, y):.4f}")
    print(f"  丢弃的 {768 - r}D 补空间 : {auc(E_drop, y):.4f}")
    print(f"  随机 {r}D (对照)      : {auc(z_rand, y):.4f}")
    print("\n=== 子空间捕获的样本方差占比 ===")
    print(f"  harmful 在 v9-{r}D : {varfrac(E[y == 1], B):.4f}    benign 在 v9-{r}D : {varfrac(E[y == 0], B):.4f}")
    print(f"  harmful 在 随机{r}D: {varfrac(E[y == 1], R):.4f}    benign 在 随机{r}D: {varfrac(E[y == 0], R):.4f}")
    print("SUBSPACE_DONE")


if __name__ == "__main__":
    main()
