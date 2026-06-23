#!/usr/bin/env python3
"""v14 = v13(判别头) + 压缩。把用户的压缩加回到 v13 上,看判别头在低维还撑不撑得住。

v13 用完整 768D。本实验在 BGE-base 上,判别头前先压缩到 {完整, 随机256/128/64/32, PCA-32},
看攻击DR / 边界FPR(JBB,XSTest) 随压缩维度怎么变。若 32D 仍接近 768D → 压缩(小/省)+ 过度拒绝
修复 + 防御 三赢。训练/评测口径与 v13 完全一致(多样攻击训练, 留出评测, 阈值 DR≥0.90 on val)。"""
import os, sys, json, gc
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M
import train as v8train
import evaluate as EV
ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]


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

    atk_tr, atk_ev = {}, {}
    for k in ATTACKS:
        atk_tr[k], atk_ev[k] = split(EV.load_texts(EV.DATASETS[k]))
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

    enc, tok = v8train.load_v7_encoder(dev)
    E = {k: v8train.extract_embeddings(enc, tok, v, dev).astype(np.float64) for k, v in sets.items()}
    del enc; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    print(f"BGE 768D 提取完毕. 有害训练{len(h_tr)} 灰{len(gray_tr)} 留出JBB{len(jbb_ev)} XSTest{len(xs_ev)}", flush=True)

    # 压缩器: 名字 -> 把 dict-of-emb 压缩(在 train 上 fit, 全部 transform)
    def make_compressors():
        comps = {"full-768": (lambda X: X)}
        Xtr = np.vstack([E["h_tr"], E["b_tr"], E["gray_tr"]])
        for k in [256, 128, 64, 32]:
            R = np.linalg.svd(np.random.RandomState(42).randn(768, k), full_matrices=False)[0]
            comps[f"rand-{k}"] = (lambda X, R=R: X @ R)
        pca = PCA(n_components=32, random_state=42).fit(Xtr)
        comps["PCA-32"] = (lambda X, p=pca: p.transform(X))
        return comps

    comps = make_compressors()
    rows = []
    for cname, proj in comps.items():
        Ec = {k: l2(proj(v)) for k, v in E.items()}
        Xtr = np.vstack([Ec["h_tr"], Ec["b_tr"], Ec["gray_tr"]])
        ytr = np.array([1]*len(Ec["h_tr"]) + [0]*(len(Ec["b_tr"])+len(Ec["gray_tr"])))
        Xval = np.vstack([Ec["h_val"], Ec["b_val"]]); yval = np.array([1]*len(Ec["h_val"]) + [0]*len(Ec["b_val"]))
        sca = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
        sc = lambda X: clf.decision_function(sca.transform(X))
        thr, ok = thr_target_dr(sc(Xval), yval)
        per = {k: round(float((sc(Ec[f"atk_{k}"]) > thr).sum())/len(Ec[f"atk_{k}"]), 3) for k in ATTACKS}
        adr = round(float(np.mean(list(per.values()))), 4)
        jbb_fpr = round(float((sc(Ec["jbb_ev"]) > thr).sum())/len(Ec["jbb_ev"]), 4)
        xs_fpr = round(float((sc(Ec["xs_ev"]) > thr).sum())/len(Ec["xs_ev"]), 4)
        dim = Ec["h_tr"].shape[1]
        rows.append({"compress": cname, "dim": dim, "attack_DR": adr, "JBB_FPR": jbb_fpr, "XSTest_FPR": xs_fpr, "per_attack": per})
        print(f"  [{cname:9s} {dim:>4}D] 攻击DR={adr} JBB={jbb_fpr} XSTest={xs_fpr}", flush=True)

    print("\n=== v14 压缩×判别头(BGE; 压缩比看 dim, 越低=越省) ===", flush=True)
    print(f"{'压缩':12s}{'维度':>6}{'攻击DR':>8}{'JBB_FPR':>9}{'XSTest_FPR':>11}", flush=True)
    for r in rows:
        print(f"{r['compress']:12s}{r['dim']:>6}{r['attack_DR']:>8}{r['JBB_FPR']:>9}{r['XSTest_FPR']:>11}", flush=True)
    json.dump(rows, open(BASE/"results/v9_boundary/v14_compress_results.json", "w"), indent=2, ensure_ascii=False)
    print("V14_DONE", flush=True)


if __name__ == "__main__":
    main()
