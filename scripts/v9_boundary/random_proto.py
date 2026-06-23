#!/usr/bin/env python3
"""v9 重设计：压缩=固定随机正交投影，决策=可学习双侧原型 + 边界 margin。

动机：实验证明安全信号在 embedding 中稠密冗余 → 随机投影(JL)即可保住信号(EXP1)，
且在边界上随机 32D(0.82) > 学习 32D(0.76)(EXP2)。于是把"压缩"交给随机(便宜、不过拟合、
保边界)，把唯一真贡献"边界感知"搬到决策层：可学习双原型 + boundary margin 训练原型位置。

同一评测口径(11 数据集, seed=42, 对齐 evaluate.py)下并排跑 3 行：
  learned        : 现有 v9 学习投影(参考)            models/v9_a2_l1.0_m0.5
  random+kmeans  : 随机投影 + KMeans 质心(无学习, 无 margin)
  random+proto   : 随机投影 + 可学习双原型 + 边界 margin (重设计)
随机投影维度扫 --dims (默认 32,64,128)。报 attack-DR / 边界 JBB-FPR / 普通 FPR / Youden's J。

注意: 本脚本只训练 ~(n_proto×dim) 个原型参数, CPU 足够; 仅嵌入提取用 GPU。
"""
import os, sys, json, argparse, gc, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import KMeans

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))   # 只挂 v8, 避免 train.py 名字冲突
import model as M                       # LearnedCSProjection, DualMultiCentroidDetector, get_device
import train as v8train                 # load_v7_encoder, extract_embeddings, optimize_threshold
import evaluate as EV                   # DATASETS, load_texts (seed=42)

ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench", "ToxicChat_harmful", "BeaverTails_harmful"]
BENIGN = ["JBB_Benign", "Alpaca", "ToxicChat_benign", "BeaverTails_benign"]


# ───────────────────────── 组件 ─────────────────────────
def random_proj(in_dim, out_dim, seed):
    """固定随机正交投影矩阵 [in_dim, out_dim] (列正交)。"""
    rng = np.random.RandomState(seed)
    R = np.linalg.svd(rng.randn(in_dim, out_dim), full_matrices=False)[0]
    return R.astype(np.float32)


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


class ProtoDetector:
    """与 DualMultiCentroidDetector.predict 同接口, 用(可为学习得到的)归一化原型打分。"""
    def __init__(self, harmful_centroids, benign_centroids):
        self.harmful_centroids = harmful_centroids
        self.benign_centroids = benign_centroids

    def predict(self, emb, threshold=0.0):
        score = (emb @ self.harmful_centroids.T).max(1) - (emb @ self.benign_centroids.T).max(1)
        return (score > threshold).astype(int), score


def train_prototypes(zc, ybin, l4, dim, n_proto, lam, margin,
                     epochs=400, lr=0.02, temp=0.1, seed=42):
    """随机投影后的归一化特征 zc 上, 学习双侧原型。
    损失 = 原型对比(拉向同类原型/推离异类) + lam·边界margin(把 gray_benign 推离最近有害原型)。"""
    torch.manual_seed(seed)
    z = torch.tensor(zc, dtype=torch.float32)
    y = torch.tensor(ybin, dtype=torch.long)
    l4t = torch.tensor(l4, dtype=torch.long)
    # KMeans 初始化原型(好起点)
    km_h = KMeans(n_proto, random_state=seed, n_init=10).fit(zc[ybin == 1])
    km_b = KMeans(n_proto, random_state=seed, n_init=10).fit(zc[ybin == 0])
    Ph = nn.Parameter(torch.tensor(l2(km_h.cluster_centers_), dtype=torch.float32))
    Pb = nn.Parameter(torch.tensor(l2(km_b.cluster_centers_), dtype=torch.float32))
    opt = torch.optim.AdamW([Ph, Pb], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gb = (l4t == 2)
    for ep in range(epochs):
        ph, pb = F.normalize(Ph, dim=-1), F.normalize(Pb, dim=-1)
        sim_h, sim_b = z @ ph.T / temp, z @ pb.T / temp          # [N,n_proto]
        logZ = torch.logsumexp(torch.cat([sim_h, sim_b], 1), 1)  # 分母=全部原型
        pos = torch.where(y == 1, torch.logsumexp(sim_h, 1), torch.logsumexp(sim_b, 1))
        loss = (logZ - pos).mean()
        if lam > 0 and int(gb.sum()) > 0:                        # 边界 margin(原型版)
            zg = z[gb]
            mh = (zg @ ph.T).max(1).values                       # 最近有害原型
            mb = (zg @ pb.T).max(1).values                       # 最近正常原型
            loss = loss + lam * torch.relu(margin + mh - mb).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return (F.normalize(Ph, dim=-1).numpy().astype(np.float32),
                F.normalize(Pb, dim=-1).numpy().astype(np.float32))


# ───────────────────────── 评测 ─────────────────────────
def eval_detector(det, thr, ds_emb, project):
    """project: emb(768)->压缩归一化特征。返回 {dataset: DR or FPR}。"""
    out = {}
    for name, emb in ds_emb.items():
        z = project(emb)
        preds, _ = det.predict(z, threshold=thr)
        if EV.DATASETS[name]["expected"] == "harmful":
            out[name] = round(1 - float((preds == 0).sum()) / len(preds), 4)   # DR
        else:
            out[name] = round(float((preds == 1).sum()) / len(preds), 4)       # FPR
    return out


def summarize(name, res):
    aDR = np.mean([res[d] for d in ATTACK if d in res])
    bFPR = np.mean([res[d] for d in BENIGN if d in res])
    J = aDR - bFPR
    return {"config": name, "AdvBench_DR": res.get("AdvBench"), "HarmBench_DR": res.get("HarmBench"),
            "mean_attack_DR": round(float(aDR), 4), "JBB_Benign_FPR": res.get("JBB_Benign"),
            "Alpaca_FPR": res.get("Alpaca"), "mean_benign_FPR": round(float(bFPR), 4),
            "Youden_J": round(float(J), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="32,64,128")
    ap.add_argument("--nproto", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--learned_dir", default=str(BASE / "models/v9_a2_l1.0_m0.5"))
    ap.add_argument("--data", default=str(BASE / "datasets/v9_training"))
    ap.add_argument("--out", default=str(BASE / "results/v9_boundary/random_proto_results.json"))
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    dev = M.get_device()
    print(f"device={dev}  dims={dims}  nproto={args.nproto}  lam={args.lam}  margin={args.margin}")

    # 训练嵌入(复用 v9 缓存)
    cache = Path(args.data) / "_v9_emb_cache.npz"
    if cache.exists():
        c = np.load(str(cache))
        tr_emb, tr_bin, tr_l4, va_emb, va_bin = c["train_emb"], c["train_bin"], c["train_l4"], c["val_emb"], c["val_bin"]
        print(f"训练嵌入(缓存): {tr_emb.shape}")
        enc = tok = None
    else:
        enc, tok = v8train.load_v7_encoder(dev)
        def rd(p):
            T, B, L = [], [], []
            for line in open(p):
                it = json.loads(line); T.append(it["text"]); lab = it["label"]
                B.append(0 if lab in [0, 2] else 1); L.append(lab)
            return T, np.array(B), np.array(L)
        trT, tr_bin, tr_l4 = rd(Path(args.data) / "train.jsonl")
        vaT, va_bin, _ = rd(Path(args.data) / "val.jsonl")
        tr_emb = v8train.extract_embeddings(enc, tok, trT, dev)
        va_emb = v8train.extract_embeddings(enc, tok, vaT, dev)

    # 评测集嵌入(每集一次, 三行复用)
    if enc is None:
        enc, tok = v8train.load_v7_encoder(dev)
    ds_emb = {}
    for nm, cfg in EV.DATASETS.items():
        txt = EV.load_texts(cfg)
        if not txt:
            print(f"  ✗ {nm} 跳过"); continue
        ds_emb[nm] = v8train.extract_embeddings(enc, tok, txt, dev).astype(np.float32)
    del enc; gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    rows = []

    # ── 行1: learned (现有 v9, 参考) ──
    ld = Path(args.learned_dir)
    proj = M.LearnedCSProjection(768, 32)
    proj.load_state_dict(torch.load(str(ld / "cs_projection_32d.pt"), map_location="cpu")); proj.eval()
    det = M.DualMultiCentroidDetector.load(str(ld / "detector_32d_1c.npz"))
    tr_json = json.load(open(ld / "training_results.json")) if (ld / "training_results.json").exists() else {}
    thr = tr_json.get("32d_1c", {}).get("threshold", 0.0)
    projL = lambda e: proj(torch.tensor(e, dtype=torch.float32)).detach().numpy()
    rows.append(summarize("learned-32D (v9 现状)", eval_detector(det, thr, ds_emb, projL)))
    print("done: learned-32D")

    # ── 行2/3: 每个随机维度 → kmeans + proto ──
    for d in dims:
        R = random_proj(768, d, seed=42)
        projR = lambda e, R=R: l2(e @ R)
        zc_tr, zc_va = projR(tr_emb), projR(va_emb)

        # random + kmeans (无学习)
        detk = M.DualMultiCentroidDetector(args.nproto, args.nproto)
        detk.fit(zc_tr[tr_bin == 1], zc_tr[tr_bin == 0])
        thr_k, _ = v8train.optimize_threshold(detk, zc_va, va_bin)
        rows.append(summarize(f"random+kmeans-{d}D", eval_detector(detk, thr_k, ds_emb, projR)))
        print(f"done: random+kmeans-{d}D (thr={thr_k:.3f})")

        # random + 可学习原型 + 边界 margin (重设计)
        ph, pb = train_prototypes(zc_tr, tr_bin, tr_l4, d, args.nproto, args.lam, args.margin)
        detp = ProtoDetector(ph, pb)
        thr_p, _ = v8train.optimize_threshold(detp, zc_va, va_bin)
        rows.append(summarize(f"random+proto-{d}D (重设计)", eval_detector(detp, thr_p, ds_emb, projR)))
        print(f"done: random+proto-{d}D (thr={thr_p:.3f})")

    # ── 输出表 ──
    cols = ["config", "AdvBench_DR", "HarmBench_DR", "mean_attack_DR", "JBB_Benign_FPR", "Alpaca_FPR", "mean_benign_FPR", "Youden_J"]
    print("\n" + " | ".join(f"{c}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]}" for c in cols))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nWROTE {args.out}")
    print("RP_DONE")


if __name__ == "__main__":
    main()
