#!/usr/bin/env python3
"""干净对照(回应'学习型不可能比随机差'的质疑):
用 v9 的【真实 forward 输出】(权重 + L2 归一化), 对比【20 个随机投影】的分布,
在 边界任务 和 易例任务 上各测 5 折 logistic AUC, 报均值±标准差。
若 v9 落在随机分布下方且超出抖动 -> '学习压缩在 off-objective 上劣于随机' 成立, 非 bug/噪声。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder, extract_embeddings  # noqa
from model import LearnedCSProjection, get_device  # noqa
np.random.seed(0)
MDIR = BASE / "models/v9_a2_l1.0_m0.5"


def load():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:200]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:200]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:400]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    return adv + harm, alp, jbb


def auc_folds(X, y):
    s = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                        StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc")
    return s.mean(), s.std()


def main():
    dev = get_device()
    enc, tok = load_v7_encoder(dev)
    harm_t, easy_t, bound_t = load()
    H = extract_embeddings(enc, tok, harm_t, dev).astype(np.float64)
    Eb = extract_embeddings(enc, tok, easy_t, dev).astype(np.float64)
    Bb = extract_embeddings(enc, tok, bound_t, dev).astype(np.float64)
    yb = np.array([1] * len(H) + [0] * len(Bb))
    ye = np.array([1] * len(H) + [0] * len(Eb))

    # v9 真实 forward 输出 (Linear -> L2 normalize)
    proj = LearnedCSProjection(768, 32)
    proj.load_state_dict(torch.load(MDIR / "cs_projection_32d.pt", map_location="cpu")); proj.eval()
    def v9out(X):
        with torch.no_grad():
            return proj(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
    zb_v9 = np.vstack([v9out(H), v9out(Bb)]); ze_v9 = np.vstack([v9out(H), v9out(Eb)])

    print(f"有害={len(H)} 易例={len(Eb)} 边界={len(Bb)}  (随机投影 20 个种子)")
    fb, _ = auc_folds(np.vstack([H, Bb]), yb)
    fe, _ = auc_folds(np.vstack([H, Eb]), ye)
    print(f"\n{'任务':<8}{'full-768':>10}{'v9实际32D(±折标准差)':>24}{'随机32D 20种子(均值±标准差, [min,max])':>40}")

    for name, (Xa, ya), (zv, yv) in [("边界", (np.vstack([H, Bb]), yb), (zb_v9, yb)),
                                     ("易例", (np.vstack([H, Eb]), ye), (ze_v9, ye))]:
        full_m, _ = auc_folds(Xa, ya)
        v9_m, v9_s = auc_folds(zv, yv)
        E_all = Xa  # 768D 原始
        rand_aucs = []
        for seed in range(20):
            rng = np.random.RandomState(100 + seed)
            R = np.linalg.svd(rng.randn(768, 32), full_matrices=False)[0]  # 正交基
            rm, _ = auc_folds(E_all @ R, ya)
            rand_aucs.append(rm)
        rand = np.array(rand_aucs)
        print(f"{name:<8}{full_m:>10.4f}{v9_m:>16.4f}±{v9_s:.3f}"
              f"{rand.mean():>22.4f}±{rand.std():.3f}  [{rand.min():.3f},{rand.max():.3f}]")
        # v9 相对随机分布的位置
        z = (v9_m - rand.mean()) / (rand.std() + 1e-9)
        below = int((rand > v9_m).sum())
        print(f"        -> v9 在随机分布中的 z = {z:+.2f};  20 个随机里有 {below} 个 > v9")
    print("RV_DONE")


if __name__ == "__main__":
    main()
