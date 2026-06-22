#!/usr/bin/env python3
"""在同一个 v9-32D 压缩空间里，比较三种决策在"有害 vs 边界正常(JBB)"上的判别力：
  (a) v9 双质心余弦决策 (max_harm_sim - max_benign_sim)
  (b) 同 32D 上 logistic
  (c) 同 32D 上 SVM
若 (a) > (b)(c) → "几何决策"本身是贡献；若 ≈/更低 → 价值只在训练数据。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder, extract_embeddings  # noqa
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa
np.random.seed(42)
MDIR = BASE / "models/v9_a2_l1.0_m0.5"


def load_hard():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:200]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:200]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    return adv + harm + jbb, np.array([1] * (len(adv) + len(harm)) + [0] * len(jbb))


def main():
    dev = get_device()
    enc, tok = load_v7_encoder(dev)
    texts, y = load_hard()
    E = extract_embeddings(enc, tok, texts, dev).astype(np.float32)
    proj = LearnedCSProjection(768, 32)
    proj.load_state_dict(torch.load(MDIR / "cs_projection_32d.pt", map_location="cpu")); proj.eval()
    with torch.no_grad():
        z = proj(torch.tensor(E)).numpy()   # [N,32] 已 L2 归一化
    det = DualMultiCentroidDetector.load(str(MDIR / "detector_32d_1c.npz"))
    # (a) 几何决策分数 = max 有害相似 - max 正常相似
    geo = (z @ det.harmful_centroids.T).max(1) - (z @ det.benign_centroids.T).max(1)
    auc_geo = roc_auc_score(y, geo)
    # (b)(c) 同 32D 上线性/SVM
    zs = StandardScaler().fit_transform(z)
    auc_lin = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"), zs, y, cv=5, scoring="roc_auc").mean()
    auc_svm = cross_val_score(SVC(kernel="rbf", class_weight="balanced"), zs, y, cv=5, scoring="roc_auc").mean()
    print(f"有害={int(y.sum())} 边界正常={int((y==0).sum())}")
    print("=== 同 v9-32D 空间内，三种决策 AUC (有害 vs 边界正常) ===")
    print(f"  (a) v9 双质心余弦决策 : {auc_geo:.4f}")
    print(f"  (b) logistic (32D)    : {auc_lin:.4f}")
    print(f"  (c) SVM-rbf  (32D)    : {auc_svm:.4f}")
    print("DECISION_DONE")


if __name__ == "__main__":
    main()
