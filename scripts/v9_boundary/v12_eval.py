#!/usr/bin/env python3
"""v12 = E5-large 编码器 + 判别式决策头, 攻克过度拒绝。

Phase1 扫描揭示: (1) XSTest 在任何编码器上 AUC≈1.0 → 过度拒绝是【决策规则】问题(双质心余弦
漏掉可分方向); (2) JBB 0.86 是 bge-base 特有, E5-large 抬到 0.91 → 换编码器能破。
本实验做 2×2 消融: 编码器{BGE-base, E5-large} × 决策{双质心(v9), 判别式logistic} —— 干净分离两个杠杆。

训练: 有害(AdvBench+HarmBench)=1 vs 正常(Alpaca)+边界(OR-Bench+XSTest训+JBB训)=0。
留出 XSTest30%+JBB30% 只评测; 攻击DR用 GCG/PAIR/JailbreakHub。阈值在 val 选 DR≥0.90。"""
import os, sys, json, gc, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M
import train as v8train
import evaluate as EV
ATTACK_EVAL = ["GCG", "PAIR", "JailbreakHub"]
ENCODERS = [("BGE-base(v7ft)", None, ""), ("E5-large", "intfloat/e5-large-v2", "query: ")]


def embed_hf(hf_id, prefix, texts, dev, bs=32):
    tok = AutoTokenizer.from_pretrained(hf_id); mdl = AutoModel.from_pretrained(hf_id).to(dev).eval()
    out = []
    for i in range(0, len(texts), bs):
        x = tok([prefix + t for t in texts[i:i+bs]], padding=True, truncation=True, max_length=256, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.no_grad():
            H = mdl(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        out.append(((H * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


def thr_target_dr(scores, ybin, target=0.90):
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bf, ok = 0.0, 2.0, False
    for t in np.unique(scores):
        p = scores > t
        if (p & (ybin == 1)).sum() / P >= target:
            f = (p & (ybin == 0)).sum() / N
            if f < bf: bf, bt, ok = f, float(t), True
    return (bt, True) if ok else (float(np.min(scores)) - 1e-6, False)


def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main():
    dev = M.get_device(); rng = np.random.RandomState(42)
    adv = pd.read_csv(BASE/"datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()
    harm = pd.read_csv(BASE/"datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()
    harmful = adv + harm
    benign = [json.loads(l)["text"] for l in open(BASE/"datasets/normal/alpaca.jsonl")][:1500]
    orb = [json.loads(l)["text"] for l in open(BASE/"datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE/"datasets/overrefusal/xstest.csv"); xs = xs[xs["label"]=="safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE/"datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()

    def split(lst, f=0.7):
        idx = rng.permutation(len(lst)); k = int(len(lst)*f); return [lst[i] for i in idx[:k]], [lst[i] for i in idx[k:]]
    xs_tr, xs_ev = split(xs); jbb_tr, jbb_ev = split(jbb)
    h_tr, h_val = split(harmful, 0.85); b_tr, b_val = split(benign, 0.85)
    gray_tr = orb + xs_tr + jbb_tr
    atk = {k: EV.load_texts(EV.DATASETS[k]) for k in ATTACK_EVAL if EV.load_texts(EV.DATASETS[k])}
    sets = {"h_tr": h_tr, "b_tr": b_tr, "gray_tr": gray_tr, "h_val": h_val, "b_val": b_val,
            "jbb_ev": jbb_ev, "xs_ev": xs_ev, **{f"atk_{k}": v for k, v in atk.items()}}
    print(f"有害{len(harmful)} 正常{len(benign)} 灰(训){len(gray_tr)} | 留出 JBB{len(jbb_ev)} XSTest{len(xs_ev)} | 攻击{list(atk)}", flush=True)

    v7enc = None; rows = []
    for ename, hf_id, pref in ENCODERS:
        try:
            if hf_id is None:
                if v7enc is None: v7enc = v8train.load_v7_encoder(dev)
                m, t = v7enc; E = {k: l2(v8train.extract_embeddings(m, t, v, dev).astype(np.float64)) for k, v in sets.items()}
            else:
                E = {k: l2(embed_hf(hf_id, pref, v, dev)) for k, v in sets.items()}
        except Exception as e:
            print(f"{ename} SKIP: {type(e).__name__}: {str(e)[:60]}", flush=True); continue

        Xtr = np.vstack([E["h_tr"], E["b_tr"], E["gray_tr"]])
        ytr = np.array([1]*len(E["h_tr"]) + [0]*(len(E["b_tr"]) + len(E["gray_tr"])))
        Xval = np.vstack([E["h_val"], E["b_val"]]); yval = np.array([1]*len(E["h_val"]) + [0]*len(E["b_val"]))

        # 两种决策
        # (a) 双质心(v9): 有害质心 vs 正常+边界质心
        det = M.DualMultiCentroidDetector(1, 1); det.fit(np.vstack([E["h_tr"]]), np.vstack([E["b_tr"], E["gray_tr"]]))
        sc_dc = lambda X: det.predict(X, 0.0)[1]
        # (b) 判别式 logistic
        sca = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
        sc_lg = lambda X: clf.decision_function(sca.transform(X))

        for dname, sc in [("双质心(v9)", sc_dc), ("判别logistic", sc_lg)]:
            thr, ok = thr_target_dr(sc(Xval), yval)
            jbb_fpr = round(float((sc(E["jbb_ev"]) > thr).sum())/len(E["jbb_ev"]), 4)
            xs_fpr = round(float((sc(E["xs_ev"]) > thr).sum())/len(E["xs_ev"]), 4)
            adr = round(float(np.mean([1 - float((sc(E[f"atk_{k}"]) <= thr).sum())/len(E[f"atk_{k}"]) for k in atk])), 4)
            rows.append({"enc": ename, "dec": dname, "JBB_FPR": jbb_fpr, "XSTest_FPR": xs_fpr, "attack_DR": adr, "DR_met": ok})
            print(f"  [{ename:14s} × {dname:12s}] JBB={jbb_fpr} XSTest={xs_fpr} 攻击DR={adr} DR达成={ok}", flush=True)

    print("\n=== v12 2×2 消融(边界FPR越低越好, 攻击DR守住) ===", flush=True)
    print(f"{'编码器':16s}{'决策':14s}{'JBB_FPR':>9}{'XSTest_FPR':>11}{'攻击DR':>8}", flush=True)
    for r in rows:
        print(f"{r['enc']:16s}{r['dec']:14s}{r['JBB_FPR']:>9}{r['XSTest_FPR']:>11}{r['attack_DR']:>8}", flush=True)
    json.dump(rows, open(BASE/"results/v9_boundary/v12_eval_results.json", "w"), indent=2, ensure_ascii=False)
    print("V12_DONE", flush=True)


if __name__ == "__main__":
    main()
