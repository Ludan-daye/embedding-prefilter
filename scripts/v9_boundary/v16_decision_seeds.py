#!/usr/bin/env python3
"""v16 = 表1(判别头 2×2)的种子确认。编码器{BGE-v7ft, E5-large} × 决策{双质心, 判别logistic}。

口径与 v13 完全一致(有害=5类攻击各70%; 判别头负类=Alpaca+边界; 双质心负类=干净Alpaca;
阈值 DR≥0.90 on val; 留出30%评测)。5 种子(42–46),报 attack_DR5 / JBB / XSTest 的 mean±std。
编码器各编码一次(与种子无关),种子只在 embedding 数组上重划分+重训。"""
import os, sys, json, gc
import random as _r
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M          # noqa: E402
import train as v8train    # noqa: E402
import evaluate as EV      # noqa: E402

ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]
SEEDS = [42, 43, 44, 45, 46]


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
    benign = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:1500]
    orb = [json.loads(l)["text"] for l in open(BASE / "datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()

    def embed_bge():
        enc, tok = v8train.load_v7_encoder(dev)
        f = lambda t: l2(v8train.extract_embeddings(enc, tok, t, dev).astype(np.float64))
        d = {"benign": f(benign), "orb": f(orb), "xs": f(xs), "jbb": f(jbb), **{k: f(v) for k, v in atk_txt.items()}}
        del enc; gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return d

    def embed_e5():
        tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        mdl = AutoModel.from_pretrained("intfloat/e5-large-v2").to(dev).eval()

        def f(texts, bs=32):
            out = []
            for i in range(0, len(texts), bs):
                x = tok(["query: " + t for t in texts[i:i + bs]], padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
                x = {k: v.to(dev) for k, v in x.items()}
                with torch.no_grad():
                    H = mdl(**x).last_hidden_state
                m = x["attention_mask"].unsqueeze(-1).float()
                out.append(((H * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
            return l2(np.vstack(out).astype(np.float64))

        d = {"benign": f(benign), "orb": f(orb), "xs": f(xs), "jbb": f(jbb), **{k: f(v) for k, v in atk_txt.items()}}
        del mdl, tok; gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return d

    ENC = {"BGE-base(v7ft)": embed_bge(), "E5-large": embed_e5()}
    print("编码完成(BGE + E5)", flush=True)

    per_seed = {}   # (enc,dec) -> list of dict
    for seed in SEEDS:
        atk_tr_i = {k: split_idx(len(atk_txt[k]), 0.7, seed)[0] for k in ATTACKS}
        atk_ev_i = {k: split_idx(len(atk_txt[k]), 0.7, seed)[1] for k in ATTACKS}
        xs_tr_i, xs_ev_i = split_idx(len(xs), 0.7, seed)
        jbb_tr_i, jbb_ev_i = split_idx(len(jbb), 0.7, seed)

        for ename, E in ENC.items():
            harm_tr_all = np.vstack([E[k][atk_tr_i[k]] for k in ATTACKS])
            hti, hvi = split_idx(len(harm_tr_all), 0.85, seed)
            bti, bvi = split_idx(len(E["benign"]), 0.85, seed)
            Htr, Hval = harm_tr_all[hti], harm_tr_all[hvi]
            Btr, Bval = E["benign"][bti], E["benign"][bvi]
            gray = np.vstack([E["orb"], E["xs"][xs_tr_i], E["jbb"][jbb_tr_i]])
            atk_ev = {k: E[k][atk_ev_i[k]] for k in ATTACKS}
            jbb_ev, xs_ev = E["jbb"][jbb_ev_i], E["xs"][xs_ev_i]
            yval = np.array([1] * len(Hval) + [0] * len(Bval))

            # 双质心(干净: 正常侧只 Alpaca)
            det = M.DualMultiCentroidDetector(1, 1); det.fit(Htr, Btr)
            sc_dc = lambda X: det.predict(X, 0.0)[1]
            # 判别头(正常侧 Alpaca+边界)
            Xtr = np.vstack([Htr, Btr, gray]); ytr = np.array([1] * len(Htr) + [0] * (len(Btr) + len(gray)))
            sca = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
            sc_lg = lambda X: clf.decision_function(sca.transform(X))

            for dname, sc in [("双质心", sc_dc), ("判别logistic", sc_lg)]:
                thr, _ = thr_target_dr(np.concatenate([sc(Hval), sc(Bval)]), yval)
                adr = float(np.mean([(sc(atk_ev[k]) > thr).mean() for k in ATTACKS]))
                jbb_f = float((sc(jbb_ev) > thr).mean())
                xs_f = float((sc(xs_ev) > thr).mean())
                key = f"{ename} × {dname}"
                per_seed.setdefault(key, []).append({"adr": adr, "jbb": jbb_f, "xs": xs_f})
        print(f"seed {seed} 完成", flush=True)

    def ms(vals):
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    summary = {}
    for key, rows in per_seed.items():
        summary[key] = {"adr": ms([r["adr"] for r in rows]), "jbb": ms([r["jbb"] for r in rows]),
                        "xs": ms([r["xs"] for r in rows])}

    print("\n" + "=" * 90, flush=True)
    print(f"=== v16 表1 判别头2×2({len(SEEDS)} 种子 mean±std) ===", flush=True)
    print(f"{'编码器 × 决策':30s}{'攻击DR5':>16}{'JBB↓':>14}{'XSTest↓':>14}", flush=True)
    for key in per_seed:
        s = summary[key]
        print(f"{key:30s}{s['adr'][0]:>8.3f}±{s['adr'][1]:<6.3f}{s['jbb'][0]:>6.3f}±{s['jbb'][1]:<5.3f}"
              f"{s['xs'][0]:>6.3f}±{s['xs'][1]:<5.3f}", flush=True)

    json.dump({"seeds": SEEDS, "summary": summary, "per_seed": per_seed},
              open(BASE / "results/v9_boundary/v16_decision_seeds_results.json", "w"), indent=2, ensure_ascii=False)
    print("\nV16_DONE", flush=True)


if __name__ == "__main__":
    main()
