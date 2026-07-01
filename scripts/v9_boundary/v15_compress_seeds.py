#!/usr/bin/env python3
"""v15 = 带种子的压缩 sweep,定论"压缩 vs 不压缩"。

在 BGE-base(v7ft) × 判别头(=v13/v14 口径)上,对压缩比 {1×, 3×=256, 5×=153, 6×=128, 12×=64, 24×=32}
的【随机正交投影】与【PCA】,各跑 5 个种子(42–46),每个种子重做 70/30 划分 + 重新抽随机矩阵 +
重拟合 PCA + 重训判别头 + 重选阈值(DR≥0.90 on val)。报 attack_DR5 / JBB / XSTest / Alpaca 的
均值±std,以及【每个压缩变体相对不压缩(full-768)的 ΔJ 配对差】—— 直接回答"压缩是否严格更优"。

高效:BGE 编码只做一次(与种子无关),种子循环只在 embedding 数组上做线性代数 + logistic。
J = attack_DR5 − mean(JBB_FPR, XSTest_FPR)。"""
import os, sys, json, gc
import random as _r
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M          # noqa: E402
import train as v8train    # noqa: E402
import evaluate as EV      # noqa: E402

ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]
SEEDS = [42, 43, 44, 45, 46]
RATIOS = [(256, "3x"), (153, "5x"), (128, "6x"), (64, "12x"), (32, "24x")]


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def thr_target_dr(scores, ybin, target=0.90):
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bf, ok = 0.0, 2.0, False
    for t in np.unique(scores):
        p = scores > t
        if (p & (ybin == 1)).sum() / P >= target:
            f = (p & (ybin == 0)).sum() / N
            if f < bf:
                bf, bt, ok = f, float(t), True
    return (bt, True) if ok else (float(np.min(scores)) - 1e-6, False)


def split_idx(n, f, seed):
    r = np.random.RandomState(seed)
    idx = r.permutation(n)
    k = int(n * f)
    return idx[:k], idx[k:]


def main():
    dev = M.get_device()

    def load_ds(name):
        _r.seed(42)
        return EV.load_texts(EV.DATASETS[name])

    atk_txt = {k: load_ds(k) for k in ATTACKS}
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:1700]
    orb = [json.loads(l)["text"] for l in open(BASE / "datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()

    # ---- BGE 编码一次(与种子无关) ----
    enc, tok = v8train.load_v7_encoder(dev)

    def E(texts):
        return v8train.extract_embeddings(enc, tok, texts, dev).astype(np.float64)

    A = {k: E(v) for k, v in atk_txt.items()}
    ALP, ORB, XS, JBB = E(alp), E(orb), E(xs), E(jbb)
    del enc; gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"编码完成. 攻击 {[len(A[k]) for k in ATTACKS]} Alpaca{len(ALP)} ORB{len(ORB)} XS{len(XS)} JBB{len(JBB)}", flush=True)

    variants = ["full-768"] + [f"rand-{d}({r})" for d, r in RATIOS] + [f"PCA-{d}({r})" for d, r in RATIOS]
    per_seed = {v: [] for v in variants}   # v -> list of dict(adr,jbb,xs,alp,J)

    for seed in SEEDS:
        atk_tr_i = {k: split_idx(len(A[k]), 0.7, seed)[0] for k in ATTACKS}
        atk_ev_i = {k: split_idx(len(A[k]), 0.7, seed)[1] for k in ATTACKS}
        alp_tr_i, alp_ev_i = split_idx(200, 0.7, seed)             # 评测池 = 前 200
        xs_tr_i, xs_ev_i = split_idx(len(XS), 0.7, seed)
        jbb_tr_i, jbb_ev_i = split_idx(len(JBB), 0.7, seed)

        Htr_all = np.vstack([A[k][atk_tr_i[k]] for k in ATTACKS])
        benign_train = np.vstack([ALP[alp_tr_i], ALP[200:1700]])   # 与评测 60 不相交
        gray = np.vstack([ORB, XS[xs_tr_i], JBB[jbb_tr_i]])
        hti, hvi = split_idx(len(Htr_all), 0.85, seed)
        bti, bvi = split_idx(len(benign_train), 0.85, seed)
        Htr, Hval = Htr_all[hti], Htr_all[hvi]
        Btr, Bval = benign_train[bti], benign_train[bvi]
        atk_ev = {k: A[k][atk_ev_i[k]] for k in ATTACKS}
        jbb_ev, xs_ev, alp_ev = JBB[jbb_ev_i], XS[xs_ev_i], ALP[alp_ev_i]

        Xtr_raw = np.vstack([Htr, Btr, gray])   # 原始(未 L2),供 PCA 拟合 / 随机基
        comps = {"full-768": (lambda X: X)}
        for d, ratio in RATIOS:
            R = np.linalg.svd(np.random.RandomState(seed * 1000 + d).randn(768, d), full_matrices=False)[0]
            comps[f"rand-{d}({ratio})"] = (lambda X, R=R: X @ R)
            pca = PCA(n_components=d, random_state=seed).fit(Xtr_raw)
            comps[f"PCA-{d}({ratio})"] = (lambda X, p=pca: p.transform(X))

        for cname, proj in comps.items():
            def C(X):
                return l2(proj(X))
            Xtr = np.vstack([C(Htr), C(Btr), C(gray)])
            ytr = np.array([1] * len(Htr) + [0] * (len(Btr) + len(gray)))
            sca = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
            sc = lambda X: clf.decision_function(sca.transform(C(X)))
            yval = np.array([1] * len(Hval) + [0] * len(Bval))
            thr, _ = thr_target_dr(np.concatenate([sc(Hval), sc(Bval)]), yval)
            adr = float(np.mean([(sc(atk_ev[k]) > thr).mean() for k in ATTACKS]))
            jbb_f = float((sc(jbb_ev) > thr).mean())
            xs_f = float((sc(xs_ev) > thr).mean())
            alp_f = float((sc(alp_ev) > thr).mean())
            J = adr - (jbb_f + xs_f) / 2
            per_seed[cname].append({"adr": adr, "jbb": jbb_f, "xs": xs_f, "alp": alp_f, "J": J})
        print(f"seed {seed} 完成", flush=True)

    # ---- 汇总 mean±std + ΔJ vs full ----
    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    fullJ = np.array([r["J"] for r in per_seed["full-768"]])
    summary = {}
    for v in variants:
        rows = per_seed[v]
        adr_m, adr_s = ms([r["adr"] for r in rows])
        jbb_m, jbb_s = ms([r["jbb"] for r in rows])
        xs_m, xs_s = ms([r["xs"] for r in rows])
        alp_m, alp_s = ms([r["alp"] for r in rows])
        J_arr = np.array([r["J"] for r in rows])
        J_m, J_s = ms(J_arr)
        dJ = J_arr - fullJ            # 配对差(同种子)
        dJ_m, dJ_s = ms(dJ)
        summary[v] = {"adr": [round(adr_m, 4), round(adr_s, 4)], "jbb": [round(jbb_m, 4), round(jbb_s, 4)],
                      "xs": [round(xs_m, 4), round(xs_s, 4)], "alp": [round(alp_m, 4), round(alp_s, 4)],
                      "J": [round(J_m, 4), round(J_s, 4)], "dJ_vs_full": [round(dJ_m, 4), round(dJ_s, 4)]}

    print("\n" + "=" * 110, flush=True)
    print(f"=== v15 压缩 sweep({len(SEEDS)} 种子, mean±std). J=攻击DR5−mean(JBB,XSTest). ΔJ=相对不压缩配对差 ===", flush=True)
    print(f"{'变体':16s}{'攻击DR5':>16}{'JBB':>14}{'XSTest':>14}{'J':>14}{'ΔJ vs full':>16}{'判定':>10}", flush=True)
    for v in variants:
        s = summary[v]
        dJm, dJs = s["dJ_vs_full"]
        if v == "full-768":
            verdict = "基准"
        elif dJm - dJs > 0:
            verdict = "更优★"
        elif dJm + dJs < 0:
            verdict = "更差"
        else:
            verdict = "打平(噪声内)"
        print(f"{v:16s}{s['adr'][0]:>8.3f}±{s['adr'][1]:<6.3f}{s['jbb'][0]:>6.3f}±{s['jbb'][1]:<5.3f}"
              f"{s['xs'][0]:>6.3f}±{s['xs'][1]:<5.3f}{s['J'][0]:>7.3f}±{s['J'][1]:<5.3f}"
              f"{dJm:>+8.3f}±{dJs:<5.3f}{verdict:>10}", flush=True)

    out = {"seeds": SEEDS, "note": "J=attack_DR5-mean(JBB,XSTest); dJ_vs_full=paired J(variant)-J(full) per seed",
           "summary": summary, "per_seed": per_seed}
    json.dump(out, open(BASE / "results/v9_boundary/v15_compress_seeds_results.json", "w"), indent=2, ensure_ascii=False)
    print("\nV15_DONE", flush=True)


if __name__ == "__main__":
    main()
