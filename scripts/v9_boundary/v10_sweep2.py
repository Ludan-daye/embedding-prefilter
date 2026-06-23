#!/usr/bin/env python3
"""v10 niche 对齐扫描：直接优化"砍过度拒绝"= 固定高攻击 DR 下的边界 FPR。

相对 v10_sweep.py 的三处改动:
1. 阈值规则: 在 val 上选"DR≥target(默认0.90)且 FPR 最小"的阈值(而非 max-J)。
2. 模型选择: 用收敛后的最终模型(去掉 val-SupCon-loss 选 epoch, 那会洗掉 margin 的作用)。
3. margin 诊断 + lam=0 基线: 打印 margin 激活比例; 加无-margin 对照, 量出边界 margin 是否真帮到边界 FPR。
按 JBB 边界 FPR 升序排名(越低越好), 附攻击 DR / DR目标是否达成 / margin诊断。"""
import os, sys, json, argparse, gc, time, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import model as M
import train as v8train
import evaluate as EV

ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench", "ToxicChat_harmful", "BeaverTails_harmful"]
BENIGN = ["JBB_Benign", "Alpaca", "ToxicChat_benign", "BeaverTails_benign"]
# (lam, margin): 无margin基线 + margin两个强度 + 高lam
LAM_MARGIN = [(0.0, 0.0), (1.0, 0.5), (1.0, 1.5), (2.0, 0.5)]


def boundary_margin_loss(z, label4, margin):
    gb, harmful, benign = (label4 == 2), (label4 == 1), (label4 == 0)
    if int(gb.sum()) == 0 or int(harmful.sum()) == 0 or int(benign.sum()) == 0:
        return torch.tensor(0.0, requires_grad=True)
    zg, zh = z[gb], z[harmful]
    c_b = z[benign].mean(0); c_b = c_b / (c_b.norm() + 1e-8)
    sim_h = (zg @ zh.T).max(1).values
    sim_cb = zg @ c_b
    return torch.relu(margin + sim_h - sim_cb).mean()


def train_proj(tr_emb, tr_bin, tr_l4, dim, lam, margin, epochs=500, lr=1e-3, bs=1024, temp=0.07, seed=42):
    """收敛后的最终模型(不按 val 选 epoch),并诊断 margin 激活情况。"""
    torch.manual_seed(seed)
    E = torch.tensor(tr_emb, dtype=torch.float32); B = torch.tensor(tr_bin, dtype=torch.long); L = torch.tensor(tr_l4, dtype=torch.long)
    m = M.LearnedCSProjection(tr_emb.shape[1], dim)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(E); active, msum, mcnt = 0, 0.0, 0
    for ep in range(epochs):
        m.train(); idx = torch.randperm(n)[:bs]
        z = m(E[idx]); loss = v8train.supervised_contrastive_loss(z, B[idx], temp)
        if lam > 0:
            ml = boundary_margin_loss(z, L[idx], margin)
            loss = loss + lam * ml
            v = float(ml); msum += v; mcnt += 1; active += int(v > 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    diag = {"marg_active_frac": round(active / max(mcnt, 1), 3), "marg_mean": round(msum / max(mcnt, 1), 4)} if lam > 0 else {"marg_active_frac": None, "marg_mean": None}
    return m, diag


def thr_target_dr(scores, ybin, target=0.90):
    """val 上选 DR≥target 且 FPR 最小的阈值；达不到则取 DR 最高的阈值。返回 (thr, 是否达成)。"""
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bfpr, ok = None, 2.0, False
    for t in np.unique(scores):
        pred = scores > t
        dr = (pred & (ybin == 1)).sum() / P
        fpr = (pred & (ybin == 0)).sum() / N
        if dr >= target and fpr < bfpr:
            bfpr, bt, ok = fpr, float(t), True
    if ok:
        return bt, True
    bt2, bdr = 0.0, -1.0
    for t in np.unique(scores):
        dr = ((scores > t) & (ybin == 1)).sum() / P
        if dr > bdr: bdr, bt2 = dr, float(t)
    return bt2, False


def eval_rows(project, det, thr, ds_emb, name):
    res = {}
    for nm, emb in ds_emb.items():
        preds, _ = det.predict(project(emb), threshold=thr)
        if EV.DATASETS[nm]["expected"] == "harmful":
            res[nm] = round(1 - float((preds == 0).sum()) / len(preds), 4)
        else:
            res[nm] = round(float((preds == 1).sum()) / len(preds), 4)
    aDR = float(np.mean([res[d] for d in ATTACK if d in res]))
    bFPR = float(np.mean([res[d] for d in BENIGN if d in res]))
    return {"config": name, "AdvBench_DR": res.get("AdvBench"), "HarmBench_DR": res.get("HarmBench"),
            "mean_attack_DR": round(aDR, 4), "JBB_Benign_FPR": res.get("JBB_Benign"),
            "Alpaca_FPR": res.get("Alpaca"), "mean_benign_FPR": round(bFPR, 4), "Youden_J": round(aDR - bFPR, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="32,48,64")
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--target_dr", type=float, default=0.90)
    ap.add_argument("--data", default=str(BASE / "datasets/v9_training"))
    ap.add_argument("--out", default=str(BASE / "results/v9_boundary/v10_sweep2_results.json"))
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    dev = M.get_device()
    grid = list(itertools.product(dims, LAM_MARGIN))
    print(f"device={dev}  grid={len(grid)}  dims={dims}  lam_margin={LAM_MARGIN}  target_dr={args.target_dr}", flush=True)

    c = np.load(str(Path(args.data) / "_v9_emb_cache.npz"))
    tr_emb, tr_bin, tr_l4, va_emb, va_bin = c["train_emb"], c["train_bin"], c["train_l4"], c["val_emb"], c["val_bin"]
    print(f"训练嵌入(缓存) {tr_emb.shape}", flush=True)

    enc, tok = v8train.load_v7_encoder(dev)
    ds_emb = {}
    for nm, cfg in EV.DATASETS.items():
        txt = EV.load_texts(cfg)
        if txt:
            ds_emb[nm] = v8train.extract_embeddings(enc, tok, txt, dev).astype(np.float32)
    del enc; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()

    rows = []
    # v9 参照(同 niche 阈值口径)
    ld = BASE / "models/v9_a2_l1.0_m0.5"
    p9m = M.LearnedCSProjection(768, 32)
    p9m.load_state_dict(torch.load(str(ld / "cs_projection_32d.pt"), map_location="cpu")); p9m.eval()
    det9 = M.DualMultiCentroidDetector.load(str(ld / "detector_32d_1c.npz"))
    p9 = lambda e: p9m(torch.tensor(e, dtype=torch.float32)).detach().numpy()
    _, vs9 = det9.predict(p9(va_emb), 0.0)
    t9, ok9 = thr_target_dr(vs9, va_bin, args.target_dr)
    r9 = eval_rows(p9, det9, t9, ds_emb, "v9 现状 learned-32D"); r9["dr_target_met"], r9["marg"] = ok9, "-"
    rows.append(r9)
    print(f"ref v9: JBB_FPR={r9['JBB_Benign_FPR']} aDR={r9['mean_attack_DR']} (DR目标达成={ok9})", flush=True)

    for i, (dim, (lam, margin)) in enumerate(grid):
        t0 = time.time()
        m, diag = train_proj(tr_emb, tr_bin, tr_l4, dim, lam, margin)
        proj = lambda e, m=m: m(torch.tensor(e, dtype=torch.float32)).detach().numpy()
        zc_tr, zc_va = proj(tr_emb), proj(va_emb)
        det = M.DualMultiCentroidDetector(args.nc, args.nc)
        det.fit(zc_tr[tr_bin == 1], zc_tr[tr_bin == 0])
        _, vs = det.predict(zc_va, 0.0)
        thr, ok = thr_target_dr(vs, va_bin, args.target_dr)
        nm = f"v10 d{dim}_lam{lam}_m{margin}_nc{args.nc}"
        row = eval_rows(proj, det, thr, ds_emb, nm); row["dr_target_met"], row["marg"] = ok, diag
        rows.append(row)
        print(f"[{i+1}/{len(grid)}] {nm}: JBB_FPR={row['JBB_Benign_FPR']} aDR={row['mean_attack_DR']} "
              f"DR达成={ok} margin激活={diag['marg_active_frac']} ({time.time()-t0:.0f}s)", flush=True)

    # 按边界 FPR 升序(达成 DR 目标的优先)
    rank = sorted(rows, key=lambda r: (not r.get("dr_target_met", False), r["JBB_Benign_FPR"]))
    cols = ["config", "mean_attack_DR", "JBB_Benign_FPR", "Alpaca_FPR", "mean_benign_FPR", "Youden_J", "dr_target_met"]
    print("\n=== 按边界 FPR 升序(DR≥目标优先) ===", flush=True)
    print(" | ".join(cols))
    for r in rank:
        print(" | ".join(str(r[c]) for c in cols) + f" | margin={r['marg']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "ranked": rank}, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"WROTE {args.out}\nV10B_DONE", flush=True)


if __name__ == "__main__":
    main()
