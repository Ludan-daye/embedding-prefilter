#!/usr/bin/env python3
"""
V9 边界感知学习型 CS 训练。
复用 v8 的编码器/SupCon/阈值优化；改动：argparse、读 v9_training(带4类标签)、可选边界 margin 损失。

用法:
  python scripts/v9_boundary/train.py --loss supcon  --dims 32 --out models/v9_boundary
  python scripts/v9_boundary/train.py --loss boundary --lam 0.5 --margin 0.2 --dims 32 --out models/v9_a2_l0.5_m0.2
"""
import os, sys, json, time, argparse, gc
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

BASE = Path(__file__).parent.parent.parent
V9_DIR = Path(__file__).parent
sys.path.insert(0, str(V9_DIR))
sys.path.insert(0, str(BASE / "scripts" / "v8_cs_supcon"))
# 复用 v8 组件
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa: E402
from train import (supervised_contrastive_loss, load_v7_encoder,  # noqa: E402
                   extract_embeddings, optimize_threshold)
from loss import boundary_margin_loss  # noqa: E402


def load_training_data_v9(data_dir):
    """读 v9_training，返回 (texts, label_binary, label4) for train/val。"""
    def read(path):
        texts, b, l4 = [], [], []
        for line in open(path):
            it = json.loads(line)
            texts.append(it["text"])
            lab = it["label"]
            b.append(0 if lab in [0, 2] else 1)
            l4.append(lab)
        return texts, np.array(b), np.array(l4)
    tr = read(Path(data_dir) / "train.jsonl")
    va = read(Path(data_dir) / "val.jsonl")
    print(f"训练集 {len(tr[0])} (bin benign={int((tr[1]==0).sum())}, harmful={int((tr[1]==1).sum())}, "
          f"gray_benign={int((tr[2]==2).sum())})")
    print(f"验证集 {len(va[0])}")
    return tr, va


def train_cs_projection_v9(train_emb, train_bin, train_l4, val_emb, val_bin,
                           target_dim, loss_mode, lam, margin,
                           epochs=500, lr=0.001, batch_size=1024, temperature=0.07):
    device = torch.device('cpu')  # 投影 ~25K 参数, CPU 足够
    emb_t = torch.tensor(train_emb, dtype=torch.float32)
    bin_t = torch.tensor(train_bin, dtype=torch.long)
    l4_t = torch.tensor(train_l4, dtype=torch.long)
    val_emb_t = torch.tensor(val_emb, dtype=torch.float32)
    val_bin_t = torch.tensor(val_bin, dtype=torch.long)

    model = LearnedCSProjection(train_emb.shape[1], target_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = float('inf'), None
    n = len(emb_t)
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)[:batch_size]
        z = model(emb_t[idx])
        loss = supervised_contrastive_loss(z, bin_t[idx], temperature)
        if loss_mode == "boundary":
            loss = loss + lam * boundary_margin_loss(z, l4_t[idx], margin=margin)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vloss = supervised_contrastive_loss(model(val_emb_t), val_bin_t, temperature)
            if vloss.item() < best_val:
                best_val = vloss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep{epoch+1:4d} loss={loss.item():.4f} val={vloss.item():.4f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["supcon", "boundary"], default="supcon")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--dims", default="32")  # 逗号分隔
    ap.add_argument("--data", default=str(BASE / "datasets/v9_training"))
    ap.add_argument("--out", default=str(BASE / "models/v9_boundary"))
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    # 嵌入缓存按数据共享（A1 与 A2 网格共用同一份 v9_training 的嵌入，避免重复编码）
    cache_file = Path(args.data) / "_v9_emb_cache.npz"

    print("=" * 60)
    print(f"V9 训练 | loss={args.loss} lam={args.lam} margin={args.margin} dims={dims}")
    print("=" * 60)

    if cache_file.exists():
        c = np.load(str(cache_file))
        train_emb, train_bin, train_l4 = c["train_emb"], c["train_bin"], c["train_l4"]
        val_emb, val_bin = c["val_emb"], c["val_bin"]
        print(f"加载缓存 embeddings: train {train_emb.shape}")
    else:
        device = get_device()
        print(f"加载 V7 编码器 (device={device})...")
        encoder, tok = load_v7_encoder(device)
        (tr_txt, train_bin, train_l4), (va_txt, val_bin, _val_l4) = load_training_data_v9(args.data)
        print("提取 768D embeddings...")
        train_emb = extract_embeddings(encoder, tok, tr_txt, device)
        val_emb = extract_embeddings(encoder, tok, va_txt, device)
        np.savez(str(cache_file), train_emb=train_emb, train_bin=train_bin,
                 train_l4=train_l4, val_emb=val_emb, val_bin=val_bin)
        del encoder; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {}
    for dim in dims:
        print(f"\n=== 训练投影 768D -> {dim}D ===")
        t0 = time.time()
        cs = train_cs_projection_v9(train_emb, train_bin, train_l4, val_emb, val_bin,
                                    target_dim=dim, loss_mode=args.loss, lam=args.lam,
                                    margin=args.margin, epochs=args.epochs)
        print(f"  训练耗时 {time.time()-t0:.1f}s")
        torch.save(cs.state_dict(), str(outdir / f"cs_projection_{dim}d.pt"))
        with torch.no_grad():
            train_comp = cs(torch.tensor(train_emb, dtype=torch.float32)).numpy()
            val_comp = cs(torch.tensor(val_emb, dtype=torch.float32)).numpy()
        for nc in [1, 3]:
            key = f"{dim}d_{nc}c"
            det = DualMultiCentroidDetector(n_harmful_centroids=nc, n_benign_centroids=nc)
            hmask, bmask = train_bin == 1, train_bin == 0
            det.fit(train_comp[hmask], train_comp[bmask])
            thr, vm = optimize_threshold(det, val_comp, val_bin)
            det.save(str(outdir / f"detector_{key}.npz"))
            tr_pred, _ = det.predict(train_comp, threshold=thr)
            results[key] = {"target_dim": dim, "n_centroids": nc, "threshold": round(thr, 3),
                            "val_metrics": vm, "train_f1": round(f1_score(train_bin, tr_pred, zero_division=0), 4)}
            print(f"  {key}: thr={thr:.3f} ValF1={vm['f1']:.4f} ValFPR={vm['fpr']:.4f}")

    json.dump(results, open(outdir / "training_results.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n完成. 结果 -> {outdir/'training_results.json'}")


if __name__ == "__main__":
    main()
