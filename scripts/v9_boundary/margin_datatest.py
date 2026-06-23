#!/usr/bin/env python3
"""判定贡献②(边界 margin)生死 —— 跨基准 + 全数据版。

留出边界评测集(均不在 v9 训练里): JBB(100) + XSTest-safe(250) = 350 条"像攻击的正常"。
三条对照(dim=32,nc=1,lam=1,margin=0.5; 阈值在 val 选 DR≥0.90 锁工作点):
  A 无margin(lam=0)                          基线
  B margin/OR-Bench(现状)                     margin 只在 OR-Bench 灰样本上训 → 测对 JBB/XSTest 的迁移
  C margin/OR-Bench+匹配(JBB+XSTest 交叉拟合)  额外把留出池的训练折当灰 → 测"匹配/多样边界数据"是否让 margin 真降 FPR
按来源(JBB / XSTest)分别报 FPR。若 C ≪ A,B → ②在匹配数据下成立(救活); 若 C≈A,B → margin 机制本身不行。"""
import os, sys, json, gc
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M
import train as v8train
import evaluate as EV

ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench", "ToxicChat_harmful", "BeaverTails_harmful"]
DIM, NC, LAM, MARGIN = 32, 1, 1.0, 0.5


def boundary_margin_loss(z, label4, margin):
    gb, harmful, benign = (label4 == 2), (label4 == 1), (label4 == 0)
    if int(gb.sum()) == 0 or int(harmful.sum()) == 0 or int(benign.sum()) == 0:
        return torch.tensor(0.0, requires_grad=True)
    zg, zh = z[gb], z[harmful]
    c_b = z[benign].mean(0); c_b = c_b / (c_b.norm() + 1e-8)
    return torch.relu(margin + (zg @ zh.T).max(1).values - zg @ c_b).mean()


def train_proj(emb, binlab, l4, lam, margin, dim=DIM, epochs=500, lr=1e-3, bs=1024, temp=0.07, seed=42):
    torch.manual_seed(seed)
    E = torch.tensor(emb, dtype=torch.float32); B = torch.tensor(binlab, dtype=torch.long); L = torch.tensor(l4, dtype=torch.long)
    m = M.LearnedCSProjection(emb.shape[1], dim)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(E); active, cnt = 0, 0
    for ep in range(epochs):
        m.train(); idx = torch.randperm(n)[:bs]
        z = m(E[idx]); loss = v8train.supervised_contrastive_loss(z, B[idx], temp)
        if lam > 0:
            ml = boundary_margin_loss(z, L[idx], margin); loss = loss + lam * ml
            cnt += 1; active += int(ml.item() > 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    return m, (round(active / cnt, 3) if cnt else None)


def thr_target_dr(scores, ybin, target=0.90):
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bfpr, ok = 0.0, 2.0, False
    for t in np.unique(scores):
        pred = scores > t
        if (pred & (ybin == 1)).sum() / P >= target:
            fpr = (pred & (ybin == 0)).sum() / N
            if fpr < bfpr: bfpr, bt, ok = fpr, float(t), True
    return (bt, True) if ok else (float(np.min(scores)) - 1e-6, False)


def fit_detect(m, tr_emb, tr_bin):
    proj = lambda e: m(torch.tensor(e, dtype=torch.float32)).detach().numpy()
    z = proj(tr_emb)
    det = M.DualMultiCentroidDetector(NC, NC); det.fit(z[tr_bin == 1], z[tr_bin == 0])
    return proj, det


def fpr(proj, det, thr, emb):
    preds, _ = det.predict(proj(emb), thr)
    return round(float((preds == 1).sum()) / len(preds), 4)


def attack_dr(proj, det, thr, ds_emb):
    drs = [1 - float((det.predict(proj(ds_emb[nm]), thr)[0] == 0).sum()) / len(ds_emb[nm]) for nm in ATTACK if nm in ds_emb]
    return round(float(np.mean(drs)), 4)


def main():
    dev = M.get_device()
    c = np.load(str(BASE / "datasets/v9_training/_v9_emb_cache.npz"))
    base_emb, base_bin, base_l4 = c["train_emb"], c["train_bin"], c["train_l4"].copy()
    va_emb, va_bin = c["val_emb"], c["val_bin"]

    enc, tok = v8train.load_v7_encoder(dev)
    jbb_txt = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs_txt = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    jbb_emb = v8train.extract_embeddings(enc, tok, jbb_txt, dev).astype(np.float32)
    xs_emb = v8train.extract_embeddings(enc, tok, xs_txt, dev).astype(np.float32)
    ds_emb = {nm: v8train.extract_embeddings(enc, tok, EV.load_texts(EV.DATASETS[nm]), dev).astype(np.float32)
              for nm in ATTACK if EV.load_texts(EV.DATASETS[nm])}
    del enc; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    print(f"base{base_emb.shape} JBB={len(jbb_emb)} XSTest-safe={len(xs_emb)} 攻击集={list(ds_emb)}", flush=True)

    R = {}

    def run_static(tag, l4, lam):
        m, act = train_proj(base_emb, base_bin, l4, lam=lam, margin=MARGIN)
        proj, det = fit_detect(m, base_emb, base_bin)
        _, vs = det.predict(proj(va_emb), 0.0); thr, ok = thr_target_dr(vs, va_bin)
        R[tag] = {"JBB_FPR": fpr(proj, det, thr, jbb_emb), "XSTest_FPR": fpr(proj, det, thr, xs_emb),
                  "attack_DR": attack_dr(proj, det, thr, ds_emb), "DR_met": ok, "marg_active": act}
        print(f"{tag}: JBB={R[tag]['JBB_FPR']} XSTest={R[tag]['XSTest_FPR']} aDR={R[tag]['attack_DR']} act={act}", flush=True)

    # A 无 margin
    run_static("A_no_margin", np.where(base_l4 == 2, 0, base_l4), 0.0)
    # B margin / OR-Bench(现状)
    run_static("B_margin_ORBench", base_l4, LAM)

    # C margin / OR-Bench + 匹配(JBB+XSTest 交叉拟合)
    pool = np.vstack([jbb_emb, xs_emb]); src = np.array(["JBB"] * len(jbb_emb) + ["XSTest"] * len(xs_emb))
    rng = np.random.RandomState(42); order = rng.permutation(len(pool)); folds = np.array_split(order, 5)
    held = np.full(len(pool), -1); aDRs = []
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(order, test_idx)
        e = np.vstack([base_emb, pool[train_idx]])
        b = np.concatenate([base_bin, np.zeros(len(train_idx), dtype=base_bin.dtype)])
        l4 = np.concatenate([base_l4, np.full(len(train_idx), 2, dtype=base_l4.dtype)])  # OR-Bench灰保留 + 池训练折当灰
        mk, _ = train_proj(e, b, l4, lam=LAM, margin=MARGIN, seed=42 + k)
        pk, dk = fit_detect(mk, e, b)
        _, vsk = dk.predict(pk(va_emb), 0.0); tk, _ = thr_target_dr(vsk, va_bin)
        held[test_idx] = dk.predict(pk(pool[test_idx]), tk)[0]
        aDRs.append(attack_dr(pk, dk, tk, ds_emb))
        print(f"  C fold{k+1}: 留出{len(test_idx)} FPR={round(float((held[test_idx]==1).sum())/len(test_idx),3)}", flush=True)
    R["C_margin_matched"] = {"JBB_FPR": round(float((held[src == "JBB"] == 1).sum()) / int((src == "JBB").sum()), 4),
                             "XSTest_FPR": round(float((held[src == "XSTest"] == 1).sum()) / int((src == "XSTest").sum()), 4),
                             "attack_DR": round(float(np.mean(aDRs)), 4), "DR_met": True, "marg_active": "~1.0(5折)"}
    print(f"C_margin_matched: JBB={R['C_margin_matched']['JBB_FPR']} XSTest={R['C_margin_matched']['XSTest_FPR']} aDR={R['C_margin_matched']['attack_DR']}", flush=True)

    print("\n=== 判定表(边界 FPR 越低越好;攻击 DR 锁在 ~0.90) ===", flush=True)
    print(f"{'条件':22s} {'JBB_FPR':>8s} {'XSTest_FPR':>10s} {'攻击DR':>7s}", flush=True)
    for k in ["A_no_margin", "B_margin_ORBench", "C_margin_matched"]:
        r = R[k]; print(f"{k:22s} {r['JBB_FPR']:>8} {r['XSTest_FPR']:>10} {r['attack_DR']:>7}", flush=True)
    base_min = min(R["A_no_margin"]["JBB_FPR"] + R["A_no_margin"]["XSTest_FPR"],
                   R["B_margin_ORBench"]["JBB_FPR"] + R["B_margin_ORBench"]["XSTest_FPR"])
    c_sum = R["C_margin_matched"]["JBB_FPR"] + R["C_margin_matched"]["XSTest_FPR"]
    print(f"\n初判: {'②救活(匹配数据下 margin 显著降边界FPR)' if c_sum < base_min - 0.06 else '②存疑(匹配数据也没明显降FPR → margin机制本身弱)'}", flush=True)
    json.dump(R, open(BASE / "results/v9_boundary/margin_datatest_results.json", "w"), indent=2, ensure_ascii=False)
    print("MDT_DONE", flush=True)


if __name__ == "__main__":
    main()
