#!/usr/bin/env python3
"""v13 = E5-large + 判别头(多样攻击训练 + 干净eval), 把 v12 两个杠杆合成可用系统。

v12 教训: 判别头修了 XSTest, E5 修了 JBB, 但判别头只在 AdvBench/HarmBench 上训 → 不泛化到
GCG/PAIR/JailbreakHub, 攻击DR崩; 且双质心正常侧混了边界污染了攻击DR。
v13 修复:
  - 有害训练用【5类攻击】(AdvBench/HarmBench/GCG/PAIR/JailbreakHub)各70%, 留30%评测 → 判别头见过各类攻击
  - 双质心正常侧只用 Alpaca(干净), 判别头正常侧用 Alpaca+边界
  - 阈值在【含多样攻击的 val】上选 DR≥0.90
2×2: 编码器{BGE,E5-large} × 决策{双质心,判别logistic}。报留出攻击DR(+每类) / 留出JBB,XSTest FPR。"""
import os, sys, json, gc
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
ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]
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

    def split(lst, f=0.7):
        idx = rng.permutation(len(lst)); k = int(len(lst)*f); return [lst[i] for i in idx[:k]], [lst[i] for i in idx[k:]]

    # 5类攻击各 70/30
    atk_tr, atk_ev = {}, {}
    for k in ATTACKS:
        txt = EV.load_texts(EV.DATASETS[k])
        atk_tr[k], atk_ev[k] = split(txt)
    harm_tr_all = sum(atk_tr.values(), [])
    benign = [json.loads(l)["text"] for l in open(BASE/"datasets/normal/alpaca.jsonl")][:1500]
    orb = [json.loads(l)["text"] for l in open(BASE/"datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE/"datasets/overrefusal/xstest.csv"); xs = xs[xs["label"]=="safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE/"datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    xs_tr, xs_ev = split(xs); jbb_tr, jbb_ev = split(jbb)
    gray_tr = orb + xs_tr + jbb_tr
    h_tr, h_val = split(harm_tr_all, 0.85); b_tr, b_val = split(benign, 0.85)

    sets = {"h_tr": h_tr, "b_tr": b_tr, "gray_tr": gray_tr, "h_val": h_val, "b_val": b_val,
            "jbb_ev": jbb_ev, "xs_ev": xs_ev, **{f"atk_{k}": atk_ev[k] for k in ATTACKS}}
    print(f"有害训练{len(h_tr)}(5类攻击) 正常{len(b_tr)} 灰{len(gray_tr)} | 留出 JBB{len(jbb_ev)} XSTest{len(xs_ev)} "
          f"攻击留出{ {k: len(atk_ev[k]) for k in ATTACKS} }", flush=True)

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

        Xval = np.vstack([E["h_val"], E["b_val"]]); yval = np.array([1]*len(E["h_val"]) + [0]*len(E["b_val"]))
        # 双质心(v9, 干净: 正常侧只 Alpaca)
        det = M.DualMultiCentroidDetector(1, 1); det.fit(E["h_tr"], E["b_tr"])
        sc_dc = lambda X: det.predict(X, 0.0)[1]
        # 判别头(正常侧 Alpaca+边界, 有害侧多样攻击)
        Xtr = np.vstack([E["h_tr"], E["b_tr"], E["gray_tr"]]); ytr = np.array([1]*len(E["h_tr"]) + [0]*(len(E["b_tr"])+len(E["gray_tr"])))
        sca = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
        sc_lg = lambda X: clf.decision_function(sca.transform(X))

        for dname, sc in [("双质心(v9)", sc_dc), ("判别logistic", sc_lg)]:
            thr, ok = thr_target_dr(sc(Xval), yval)
            per_atk = {k: round(float((sc(E[f"atk_{k}"]) > thr).sum())/len(E[f"atk_{k}"]), 3) for k in ATTACKS}
            adr = round(float(np.mean(list(per_atk.values()))), 4)
            jbb_fpr = round(float((sc(E["jbb_ev"]) > thr).sum())/len(E["jbb_ev"]), 4)
            xs_fpr = round(float((sc(E["xs_ev"]) > thr).sum())/len(E["xs_ev"]), 4)
            rows.append({"enc": ename, "dec": dname, "attack_DR": adr, "JBB_FPR": jbb_fpr, "XSTest_FPR": xs_fpr,
                         "per_attack": per_atk, "DR_met": ok})
            print(f"  [{ename:14s} × {dname:12s}] 攻击DR={adr} JBB={jbb_fpr} XSTest={xs_fpr} | 每类={per_atk}", flush=True)

    print("\n=== v13 2×2(攻击DR守住 + 边界FPR越低越好) ===", flush=True)
    print(f"{'编码器':16s}{'决策':14s}{'攻击DR':>8}{'JBB_FPR':>9}{'XSTest_FPR':>11}", flush=True)
    for r in rows:
        print(f"{r['enc']:16s}{r['dec']:14s}{r['attack_DR']:>8}{r['JBB_FPR']:>9}{r['XSTest_FPR']:>11}", flush=True)
    # 找守住攻击DR(≥0.85)里边界FPR(JBB+XSTest)最低的
    good = [r for r in rows if r["attack_DR"] >= 0.85]
    if good:
        best = min(good, key=lambda r: r["JBB_FPR"] + r["XSTest_FPR"])
        print(f"\n最优(攻击DR≥0.85内边界最低): {best['enc']} × {best['dec']}  攻击DR={best['attack_DR']} JBB={best['JBB_FPR']} XSTest={best['XSTest_FPR']}", flush=True)
    else:
        print("\n⚠️ 没有配置攻击DR≥0.85 —— 阈值/数据还需调", flush=True)
    json.dump(rows, open(BASE/"results/v9_boundary/v13_eval_results.json", "w"), indent=2, ensure_ascii=False)
    print("V13_DONE", flush=True)


if __name__ == "__main__":
    main()
