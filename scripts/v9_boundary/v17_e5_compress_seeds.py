#!/usr/bin/env python3
"""v17 = E5-large × 判别头 的压缩 sweep(带种子)。回答"最强变体(E5,1024D)能压到几倍还保 J≈0.84"。

结构与 v15 相同,只把编码器换成 E5-large(mean-pool + 'query: ' 前缀, 1024D)。压缩比
1024→{256(4×),153(6.7×),128(8×),64(16×),32(32×)},随机正交 与 PCA 各一套,5 种子(42–46)。
报 attack_DR5 / JBB / XSTest / Alpaca 的 mean±std + 相对不压缩 E5(full-1024)的配对 ΔJ。
J = 攻击DR5 − mean(JBB, XSTest, Alpaca)  (与 pareto_seeds 口径一致, 便于直接对照主表)。"""
import os, sys, json, gc
import random as _r
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
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
DIN = 1024
RATIOS = [(256, "4x"), (153, "6.7x"), (128, "8x"), (64, "16x"), (32, "32x")]


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

    # ---- E5 编码一次 ----
    tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    mdl = AutoModel.from_pretrained("intfloat/e5-large-v2").to(dev).eval()

    def E(texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            x = tok(["query: " + t for t in texts[i:i + bs]], padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            x = {k: v.to(dev) for k, v in x.items()}
            with torch.no_grad():
                H = mdl(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            out.append(((H * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
        return np.vstack(out).astype(np.float64)

    A = {k: E(v) for k, v in atk_txt.items()}
    ALP, ORB, XS, JBB = E(alp), E(orb), E(xs), E(jbb)
    del mdl, tok; gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"E5 编码完成. 攻击 {[len(A[k]) for k in ATTACKS]} Alpaca{len(ALP)} ORB{len(ORB)} XS{len(XS)} JBB{len(JBB)}", flush=True)

    variants = ["full-1024"] + [f"rand-{d}({r})" for d, r in RATIOS] + [f"PCA-{d}({r})" for d, r in RATIOS]
    per_seed = {v: [] for v in variants}

    for seed in SEEDS:
        atk_tr_i = {k: split_idx(len(A[k]), 0.7, seed)[0] for k in ATTACKS}
        atk_ev_i = {k: split_idx(len(A[k]), 0.7, seed)[1] for k in ATTACKS}
        alp_tr_i, alp_ev_i = split_idx(200, 0.7, seed)
        xs_tr_i, xs_ev_i = split_idx(len(XS), 0.7, seed)
        jbb_tr_i, jbb_ev_i = split_idx(len(JBB), 0.7, seed)

        Htr_all = np.vstack([A[k][atk_tr_i[k]] for k in ATTACKS])
        benign_train = np.vstack([ALP[alp_tr_i], ALP[200:1700]])
        gray = np.vstack([ORB, XS[xs_tr_i], JBB[jbb_tr_i]])
        hti, hvi = split_idx(len(Htr_all), 0.85, seed)
        bti, bvi = split_idx(len(benign_train), 0.85, seed)
        Htr, Hval = Htr_all[hti], Htr_all[hvi]
        Btr, Bval = benign_train[bti], benign_train[bvi]
        atk_ev = {k: A[k][atk_ev_i[k]] for k in ATTACKS}
        jbb_ev, xs_ev, alp_ev = JBB[jbb_ev_i], XS[xs_ev_i], ALP[alp_ev_i]

        Xtr_raw = np.vstack([Htr, Btr, gray])
        comps = {"full-1024": (lambda X: X)}
        for d, ratio in RATIOS:
            R = np.linalg.svd(np.random.RandomState(seed * 1000 + d).randn(DIN, d), full_matrices=False)[0]
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
            J = adr - (jbb_f + xs_f + alp_f) / 3
            per_seed[cname].append({"adr": adr, "jbb": jbb_f, "xs": xs_f, "alp": alp_f, "J": J})
        print(f"seed {seed} 完成", flush=True)

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    fullJ = np.array([r["J"] for r in per_seed["full-1024"]])
    summary = {}
    for v in variants:
        rows = per_seed[v]
        adr_m, adr_s = ms([r["adr"] for r in rows]); jbb_m, jbb_s = ms([r["jbb"] for r in rows])
        xs_m, xs_s = ms([r["xs"] for r in rows]); alp_m, alp_s = ms([r["alp"] for r in rows])
        J_arr = np.array([r["J"] for r in rows]); J_m, J_s = ms(J_arr)
        dJ_m, dJ_s = ms(J_arr - fullJ)
        summary[v] = {"adr": [round(adr_m, 4), round(adr_s, 4)], "jbb": [round(jbb_m, 4), round(jbb_s, 4)],
                      "xs": [round(xs_m, 4), round(xs_s, 4)], "alp": [round(alp_m, 4), round(alp_s, 4)],
                      "J": [round(J_m, 4), round(J_s, 4)], "dJ_vs_full": [round(dJ_m, 4), round(dJ_s, 4)]}

    print("\n" + "=" * 118, flush=True)
    print(f"=== v17 E5 压缩 sweep({len(SEEDS)} 种子). J=攻击DR5−mean(JBB,XSTest,Alpaca). ΔJ=相对不压缩E5配对差 ===", flush=True)
    print(f"{'变体':16s}{'攻击DR5':>15}{'JBB':>13}{'XSTest':>13}{'Alpaca':>13}{'J':>14}{'ΔJ vs full':>15}{'判定':>10}", flush=True)
    order = ["full-1024"] + sorted([v for v in variants if v != "full-1024"], key=lambda v: -summary[v]["J"][0])
    for v in order:
        s = summary[v]; dJm, dJs = s["dJ_vs_full"]
        verdict = "基准" if v == "full-1024" else ("更优★" if dJm - dJs > 0 else ("更差" if dJm + dJs < 0 else "打平"))
        print(f"{v:16s}{s['adr'][0]:>8.3f}±{s['adr'][1]:<5.3f}{s['jbb'][0]:>6.3f}±{s['jbb'][1]:<4.3f}"
              f"{s['xs'][0]:>6.3f}±{s['xs'][1]:<4.3f}{s['alp'][0]:>6.3f}±{s['alp'][1]:<4.3f}"
              f"{s['J'][0]:>7.3f}±{s['J'][1]:<4.3f}{dJm:>+8.3f}±{dJs:<4.3f}{verdict:>10}", flush=True)

    json.dump({"seeds": SEEDS, "din": DIN, "summary": summary, "per_seed": per_seed},
              open(BASE / "results/v9_boundary/v17_e5_compress_seeds_results.json", "w"), indent=2, ensure_ascii=False)
    print("\nV17_DONE", flush=True)


if __name__ == "__main__":
    main()
