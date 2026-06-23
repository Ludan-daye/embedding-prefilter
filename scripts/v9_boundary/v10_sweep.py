#!/usr/bin/env python3
"""v10 调参扫描：学习投影 + 边界感知训练，网格搜最优，统一口径(max Youden's J 选阈值)评测。
网格: dim{32,48,64} × lam{0.5,1,2} × margin{0.3,0.5,0.7} × nc{1,3}  (默认 54 组)
参照行: 现有 v9(learned-32D a2_l1.0_m0.5) + 随机-64D+proto 消融。
所有行同一评测口径 → "v10 比 v9 强多少" 干净可比。报 attack-DR / JBB边界FPR / Youden's J，按 J 排名。
投影训练在 CPU(~25K 参数)；仅嵌入提取用 GPU。"""
import os, sys, json, argparse, gc, time, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import KMeans

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))   # 只挂 v8，避免 train.py 名字冲突
import model as M
import train as v8train          # load_v7_encoder, extract_embeddings, supervised_contrastive_loss
import evaluate as EV            # DATASETS, load_texts (seed=42)

ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench", "ToxicChat_harmful", "BeaverTails_harmful"]
BENIGN = ["JBB_Benign", "Alpaca", "ToxicChat_benign", "BeaverTails_benign"]


def boundary_margin_loss(z, label4, margin):
    """把 gray_benign(==2) 从最难有害负样本推回 benign 原型一侧。"""
    gb, harmful, benign = (label4 == 2), (label4 == 1), (label4 == 0)
    if int(gb.sum()) == 0 or int(harmful.sum()) == 0 or int(benign.sum()) == 0:
        return torch.tensor(0.0, requires_grad=True)
    zg, zh = z[gb], z[harmful]
    c_b = z[benign].mean(0); c_b = c_b / (c_b.norm() + 1e-8)
    sim_h = (zg @ zh.T).max(1).values
    sim_cb = zg @ c_b
    return torch.relu(margin + sim_h - sim_cb).mean()


def train_proj(tr_emb, tr_bin, tr_l4, va_emb, va_bin, dim, lam, margin,
               epochs=500, lr=1e-3, bs=1024, temp=0.07, seed=42):
    torch.manual_seed(seed)
    E = torch.tensor(tr_emb, dtype=torch.float32); B = torch.tensor(tr_bin, dtype=torch.long); L = torch.tensor(tr_l4, dtype=torch.long)
    Ev = torch.tensor(va_emb, dtype=torch.float32); Bv = torch.tensor(va_bin, dtype=torch.long)
    m = M.LearnedCSProjection(tr_emb.shape[1], dim)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, n = 1e9, None, len(E)
    for ep in range(epochs):
        m.train(); idx = torch.randperm(n)[:bs]
        z = m(E[idx]); loss = v8train.supervised_contrastive_loss(z, B[idx], temp)
        if lam > 0:
            loss = loss + lam * boundary_margin_loss(z, L[idx], margin)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 50 == 0:
            m.eval()
            with torch.no_grad():
                vl = v8train.supervised_contrastive_loss(m(Ev), Bv, temp).item()
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    if best_state: m.load_state_dict(best_state)
    m.eval(); return m


def best_thr_J(scores, ybin):
    """在 val 上按 max Youden's J(DR-FPR) 选阈值。"""
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    cand = np.unique(scores)
    bt, bj = 0.0, -2.0
    for t in cand:
        pred = scores > t
        j = (pred & (ybin == 1)).sum() / P - (pred & (ybin == 0)).sum() / N
        if j > bj: bj, bt = j, float(t)
    return bt


def eval_rows(project, det, thr, ds_emb, name):
    res = {}
    for nm, emb in ds_emb.items():
        z = project(emb)
        preds, _ = det.predict(z, threshold=thr)
        if EV.DATASETS[nm]["expected"] == "harmful":
            res[nm] = round(1 - float((preds == 0).sum()) / len(preds), 4)
        else:
            res[nm] = round(float((preds == 1).sum()) / len(preds), 4)
    aDR = float(np.mean([res[d] for d in ATTACK if d in res]))
    bFPR = float(np.mean([res[d] for d in BENIGN if d in res]))
    return {"config": name, "AdvBench_DR": res.get("AdvBench"), "HarmBench_DR": res.get("HarmBench"),
            "mean_attack_DR": round(aDR, 4), "JBB_Benign_FPR": res.get("JBB_Benign"),
            "Alpaca_FPR": res.get("Alpaca"), "mean_benign_FPR": round(bFPR, 4),
            "Youden_J": round(aDR - bFPR, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="32,48,64")
    ap.add_argument("--lams", default="0.5,1.0,2.0")
    ap.add_argument("--margins", default="0.3,0.5,0.7")
    ap.add_argument("--ncs", default="1,3")
    ap.add_argument("--data", default=str(BASE / "datasets/v9_training"))
    ap.add_argument("--out", default=str(BASE / "results/v9_boundary/v10_sweep_results.json"))
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    margins = [float(x) for x in args.margins.split(",")]
    ncs = [int(x) for x in args.ncs.split(",")]
    dev = M.get_device()
    grid = list(itertools.product(dims, lams, margins, ncs))
    print(f"device={dev}  grid={len(grid)} 组  dims={dims} lams={lams} margins={margins} ncs={ncs}", flush=True)

    # 训练嵌入(复用 v9 缓存)
    cache = Path(args.data) / "_v9_emb_cache.npz"
    c = np.load(str(cache))
    tr_emb, tr_bin, tr_l4, va_emb, va_bin = c["train_emb"], c["train_bin"], c["train_l4"], c["val_emb"], c["val_bin"]
    print(f"训练嵌入(缓存) {tr_emb.shape}", flush=True)

    # 评测集嵌入(一次)
    enc, tok = v8train.load_v7_encoder(dev)
    ds_emb = {}
    for nm, cfg in EV.DATASETS.items():
        txt = EV.load_texts(cfg)
        if txt:
            ds_emb[nm] = v8train.extract_embeddings(enc, tok, txt, dev).astype(np.float32)
    del enc; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()

    rows = []
    # 参照行: 现有 v9 (统一 max-J 阈值口径)
    ld = BASE / "models/v9_a2_l1.0_m0.5"
    proj9 = M.LearnedCSProjection(768, 32)
    proj9.load_state_dict(torch.load(str(ld / "cs_projection_32d.pt"), map_location="cpu")); proj9.eval()
    det9 = M.DualMultiCentroidDetector.load(str(ld / "detector_32d_1c.npz"))
    p9 = lambda e: proj9(torch.tensor(e, dtype=torch.float32)).detach().numpy()
    with torch.no_grad():
        _, vs9 = det9.predict(p9(va_emb), 0.0)
    rows.append(eval_rows(p9, det9, best_thr_J(vs9, va_bin), ds_emb, "v9 现状 learned-32D(统一口径)"))
    print(f"ref done: v9  J={rows[-1]['Youden_J']}", flush=True)

    # v10 网格
    for i, (dim, lam, margin, nc) in enumerate(grid):
        t0 = time.time()
        m = train_proj(tr_emb, tr_bin, tr_l4, va_emb, va_bin, dim, lam, margin)
        proj = lambda e, m=m: m(torch.tensor(e, dtype=torch.float32)).detach().numpy()
        zc_tr, zc_va = proj(tr_emb), proj(va_emb)
        det = M.DualMultiCentroidDetector(nc, nc)
        det.fit(zc_tr[tr_bin == 1], zc_tr[tr_bin == 0])
        _, vs = det.predict(zc_va, 0.0)
        thr = best_thr_J(vs, va_bin)
        name = f"v10 d{dim}_l{lam}_m{margin}_nc{nc}"
        rows.append(eval_rows(proj, det, thr, ds_emb, name))
        print(f"[{i+1}/{len(grid)}] {name}: J={rows[-1]['Youden_J']} "
              f"JBB_FPR={rows[-1]['JBB_Benign_FPR']} aDR={rows[-1]['mean_attack_DR']} ({time.time()-t0:.0f}s)", flush=True)

    # 排名输出
    rows_sorted = sorted(rows, key=lambda r: r["Youden_J"], reverse=True)
    cols = ["config", "AdvBench_DR", "HarmBench_DR", "mean_attack_DR", "JBB_Benign_FPR", "Alpaca_FPR", "mean_benign_FPR", "Youden_J"]
    print("\n=== 按 Youden's J 排名(Top 15) ===", flush=True)
    print(" | ".join(cols))
    for r in rows_sorted[:15]:
        print(" | ".join(str(r[c]) for c in cols))
    v9J = next(r["Youden_J"] for r in rows if r["config"].startswith("v9"))
    best = rows_sorted[0]
    print(f"\n最优: {best['config']}  J={best['Youden_J']}  (v9={v9J}, Δ={round(best['Youden_J']-v9J,4)})  "
          f"JBB_FPR={best['JBB_Benign_FPR']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "v9_J": v9J, "best": best}, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"WROTE {args.out}\nV10_DONE", flush=True)


if __name__ == "__main__":
    main()
