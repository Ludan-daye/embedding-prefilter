#!/usr/bin/env python3
"""直接检验 CS 前置条件：'安全判别信息是否稀疏/低维分布'。
1) 维度扫描: AUC vs k (PCA-top-k / 随机-k) —— 任务的有效维度。
2) 判别方向稀疏度: d = mean(harmful)-mean(benign)，能量是否集中在少数坐标(稀疏=CS成立) vs 摊匀(稠密=CS不成立)。
3) 判别信号的有效秩(between-class participation ratio)。"""
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


def load_data():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:300]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:300]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:600]
    return adv + harm + alp, np.array([1] * 600 + [0] * 600)


def auc_on(X, y):
    return cross_val_score(LogisticRegression(max_iter=2000), StandardScaler().fit_transform(X),
                           y, cv=5, scoring="roc_auc").mean()


def main():
    dev = get_device()
    enc, tok = load_v7_encoder(dev)
    texts, y = load_data()
    E = extract_embeddings(enc, tok, texts, dev).astype(np.float64)
    print(f"emb {E.shape} harmful={int(y.sum())} benign={int((y==0).sum())}")

    # --- 1. 维度扫描 ---
    pca = PCA(n_components=256, random_state=42).fit(E)
    Ep = pca.transform(E)
    print("\n=== 维度扫描 AUC ===")
    print(f"{'k':>5} {'PCA-top-k':>10} {'随机-k':>10}")
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        rk = E @ np.linalg.svd(np.random.randn(768, k), full_matrices=False)[0]
        print(f"{k:>5} {auc_on(Ep[:, :k], y):>10.4f} {auc_on(rk, y):>10.4f}")
    print(f" 768  {auc_on(E, y):>10.4f}  (full)")

    # --- 2. 判别方向稀疏度 ---
    d = E[y == 1].mean(0) - E[y == 0].mean(0)        # 均值差方向 (768,)
    e = d ** 2; e = e / e.sum()                       # 每坐标能量占比
    order = np.sort(e)[::-1]
    cum = np.cumsum(order)
    print("\n=== 判别方向(均值差)能量集中度 ===")
    for m in [8, 16, 32, 64, 128, 256]:
        print(f"  top-{m:>3} 坐标占能量: {cum[m-1]*100:5.1f}%   (均匀分布应为 {m/768*100:4.1f}%)")
    # 参与比率(判别方向的有效坐标数)
    pr = 1.0 / np.sum(e ** 2)
    print(f"  判别方向参与比率(有效坐标数) = {pr:.1f} / 768")

    # --- 3. between-class 有效秩 (用 LDA 方向其实是 1 维; 这里看类均值差的有效维) ---
    print("\n=== 结论判读 ===")
    sparse = cum[31] > 0.5
    print(f"  top-32 坐标能量 {cum[31]*100:.1f}% -> CS稀疏前提 {'成立(>50%)' if sparse else '不成立(信号稠密)'}")
    print("PRECOND_DONE")


if __name__ == "__main__":
    main()
