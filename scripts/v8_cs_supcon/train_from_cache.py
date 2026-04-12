#!/usr/bin/env python3
"""
V8 CS 训练（从缓存 embedding 开始，不需要加载 V7 encoder）
用 Supervised Contrastive Loss 训练 CS 投影。
先运行 extract_embeddings.py 生成缓存，再运行本脚本。
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device


def supervised_contrastive_loss(z, labels, temperature=0.07):
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    sim = z @ z.T / temperature
    sim_max = sim.max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()

    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & self_mask

    exp_sim = torch.exp(sim) * self_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.float().sum(dim=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1)

    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return loss[valid].mean()


def train_cs_projection(train_emb, train_labels, val_emb, val_labels,
                        target_dim, epochs=500, lr=0.001, batch_size=1024,
                        temperature=0.07):
    device = torch.device('cpu')
    input_dim = train_emb.shape[1]

    emb_t = torch.tensor(train_emb, dtype=torch.float32).to(device)
    lab_t = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_emb_t = torch.tensor(val_emb, dtype=torch.float32).to(device)
    val_lab_t = torch.tensor(val_labels, dtype=torch.long).to(device)

    model = LearnedCSProjection(input_dim, target_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    n_samples = len(emb_t)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n_samples)[:batch_size]
        z = model(emb_t[idx])
        loss = supervised_contrastive_loss(z, lab_t[idx], temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_z = model(val_emb_t)
                val_loss = supervised_contrastive_loss(val_z, val_lab_t, temperature)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:4d} | loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def optimize_threshold(detector, val_compressed, val_labels, fpr_constraint=0.05):
    best_threshold = 0.0
    best_f1 = 0.0
    best_metrics = None

    for t in np.arange(-0.3, 0.31, 0.01):
        preds, _ = detector.predict(val_compressed, threshold=t)
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()
        tn = ((preds == 0) & (val_labels == 0)).sum()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if fpr <= fpr_constraint and f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {'threshold': round(t, 3), 'f1': round(f1, 4),
                            'precision': round(prec, 4), 'recall': round(rec, 4),
                            'fpr': round(float(fpr), 4)}

    if best_metrics is None:
        for t in np.arange(-0.3, 0.31, 0.01):
            preds, _ = detector.predict(val_compressed, threshold=t)
            f1 = f1_score(val_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        preds, _ = detector.predict(val_compressed, threshold=best_threshold)
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()
        tn = ((preds == 0) & (val_labels == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        best_metrics = {'threshold': round(best_threshold, 3), 'f1': round(best_f1, 4),
                        'precision': round(prec, 4), 'recall': round(rec, 4),
                        'fpr': round(float(fpr), 4), 'note': 'relaxed_fpr'}

    return best_threshold, best_metrics


def main():
    output_dir = BASE_DIR / "models" / "v8_cs_supcon"
    cache_file = output_dir / "cache" / "v7_embeddings.npz"

    if not cache_file.exists():
        print(f"错误: 缓存文件不存在: {cache_file}")
        print("请先运行 embedding 提取脚本生成缓存。")
        sys.exit(1)

    print("=" * 70)
    print("V8 CS 压缩感知分类器 — Supervised Contrastive Training")
    print("=" * 70)

    # 加载缓存
    print(f"\n加载缓存 embeddings: {cache_file}")
    cached = np.load(str(cache_file))
    train_emb = cached['train_emb']
    train_labels = cached['train_labels']
    val_emb = cached['val_emb']
    val_labels = cached['val_labels']
    print(f"  训练集: {train_emb.shape} (benign={sum(train_labels==0)}, harmful={sum(train_labels==1)})")
    print(f"  验证集: {val_emb.shape} (benign={sum(val_labels==0)}, harmful={sum(val_labels==1)})")

    # 维度 × 质心扫描
    target_dims = [8, 16, 32, 64, 128]
    centroid_counts = [1, 3, 5]
    all_results = {}

    for dim in target_dims:
        print(f"\n{'=' * 50}")
        print(f"训练 CS 投影: 768D → {dim}D")
        print(f"{'=' * 50}")

        t0 = time.time()
        cs_model = train_cs_projection(
            train_emb, train_labels, val_emb, val_labels,
            target_dim=dim, epochs=500, lr=0.001, batch_size=1024
        )
        print(f"  训练耗时: {time.time() - t0:.1f}s")

        # 保存投影权重
        proj_path = output_dir / f"cs_projection_{dim}d.pt"
        torch.save(cs_model.state_dict(), str(proj_path))

        # 投影所有 embeddings
        with torch.no_grad():
            train_comp = cs_model(torch.tensor(train_emb, dtype=torch.float32)).numpy()
            val_comp = cs_model(torch.tensor(val_emb, dtype=torch.float32)).numpy()

        for nc in centroid_counts:
            key = f"{dim}d_{nc}c"
            print(f"\n  拟合质心: {dim}D, harmful={nc}, benign={nc}")

            harmful_mask = train_labels == 1
            benign_mask = train_labels == 0
            actual_nc_h = min(nc, harmful_mask.sum())
            actual_nc_b = min(nc, benign_mask.sum())

            detector = DualMultiCentroidDetector(actual_nc_h, actual_nc_b)
            detector.fit(train_comp[harmful_mask], train_comp[benign_mask])

            # 阈值优化
            threshold, val_metrics = optimize_threshold(detector, val_comp, val_labels)
            print(f"    最优阈值: {threshold:.3f} | Val F1={val_metrics['f1']:.4f} | Val FPR={val_metrics['fpr']:.4f}")

            # 训练集评估
            train_preds, _ = detector.predict(train_comp, threshold=threshold)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)
            train_acc = accuracy_score(train_labels, train_preds)

            # 保存 detector
            det_path = output_dir / f"detector_{key}.npz"
            detector.save(str(det_path))

            all_results[key] = {
                'target_dim': dim,
                'n_centroids': nc,
                'threshold': round(threshold, 3),
                'val_metrics': val_metrics,
                'train_f1': round(train_f1, 4),
                'train_acc': round(train_acc, 4),
            }

    # 保存汇总
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("训练完成! 结果汇总:")
    print(f"{'=' * 70}")
    print(f"\n{'配置':<12} {'Train F1':>10} {'Val F1':>10} {'Val FPR':>10} {'Threshold':>10}")
    print("-" * 55)
    for key, r in sorted(all_results.items()):
        print(f"{key:<12} {r['train_f1']:>10.4f} {r['val_metrics']['f1']:>10.4f} "
              f"{r['val_metrics']['fpr']:>10.4f} {r['threshold']:>10.3f}")

    print(f"\n结果已保存至: {results_path}")
    print(f"模型已保存至: {output_dir}")


if __name__ == '__main__':
    main()
