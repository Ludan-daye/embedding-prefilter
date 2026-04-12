#!/usr/bin/env python3
"""
V8 CS 压缩感知分类器训练
用 Supervised Contrastive Loss 训练 CS 投影，替代 V6 的 MSE 保距损失。
用 V7 encoder + V7 训练数据（含真实对话）。
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# 路径设置
BASE_DIR = Path(__file__).parent.parent.parent  # CSonEmbedding/
V8_DIR = Path(__file__).parent

# V8 组件
sys.path.insert(0, str(V8_DIR))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device


def _load_v7_module():
    """延迟加载 V7 model 模块（避免 import 时 segfault）"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v7_model", str(BASE_DIR / "scripts" / "v7_classifier" / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════
# Supervised Contrastive Loss
# ═══════════════════════════════════════════════

def supervised_contrastive_loss(z, labels, temperature=0.07):
    """
    Supervised InfoNCE loss on L2-normalized embeddings.

    z: [B, D] L2-normalized
    labels: [B] binary (0=benign, 1=harmful)
    temperature: scalar
    """
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    sim = z @ z.T / temperature  # [B, B]

    # 数值稳定
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


# ═══════════════════════════════════════════════
# Embedding 提取
# ═══════════════════════════════════════════════

def load_v7_encoder(device):
    """加载 V7 encoder（冻结）"""
    v7_mod = _load_v7_module()
    model_path = BASE_DIR / "models" / "v7_classifier"

    with open(model_path / "config.json") as f:
        config = json.load(f)

    model = v7_mod.V6HarmfulDetector(
        model_name=config['model_name'],
        projection_dim=config['projection_dim']
    )

    checkpoint = torch.load(str(model_path / "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = v7_mod.load_tokenizer(config['model_name'])
    return model, tokenizer


def load_training_data():
    """加载 V7 训练数据，返回 (texts, binary_labels) for train/val"""
    data_path = BASE_DIR / "datasets" / "v7_training"

    def read_jsonl(path):
        texts, labels = [], []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item['text'])
                # 4类 → 二分类: {0,2}→0(benign), {1,3}→1(harmful)
                labels.append(0 if item['label'] in [0, 2] else 1)
        return texts, np.array(labels)

    train_texts, train_labels = read_jsonl(data_path / "train.jsonl")
    val_texts, val_labels = read_jsonl(data_path / "val.jsonl")

    print(f"训练集: {len(train_texts)} 条 (benign={sum(train_labels==0)}, harmful={sum(train_labels==1)})")
    print(f"验证集: {len(val_texts)} 条 (benign={sum(val_labels==0)}, harmful={sum(val_labels==1)})")
    return train_texts, train_labels, val_texts, val_labels


def extract_embeddings(model, tokenizer, texts, device, batch_size=32):
    """用 V7 encoder 提取 768D embeddings"""
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="提取 embeddings"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.encode(inputs['input_ids'], inputs['attention_mask'])
        all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


# ═══════════════════════════════════════════════
# CS 投影训练
# ═══════════════════════════════════════════════

def train_cs_projection(train_emb, train_labels, val_emb, val_labels,
                        target_dim, epochs=500, lr=0.001, batch_size=1024,
                        temperature=0.07):
    """训练 LearnedCSProjection，用 Supervised Contrastive Loss"""
    # CS 投影只有 ~25K 参数，用 CPU 避免 MPS segfault
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
        # 随机采样一个 batch
        idx = torch.randperm(n_samples, device=device)[:batch_size]
        z = model(emb_t[idx])
        loss = supervised_contrastive_loss(z, lab_t[idx], temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 验证 (每 50 epoch)
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


# ═══════════════════════════════════════════════
# 阈值优化
# ═══════════════════════════════════════════════

def optimize_threshold(detector, val_compressed, val_labels, fpr_constraint=0.05):
    """在验证集上扫 threshold，找 F1@FPR<=constraint 的最优点"""
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
                            'fpr': round(fpr, 4)}

    # 如果 fpr_constraint 太严找不到，放宽到最佳 f1
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


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def main():
    device = get_device()
    output_dir = BASE_DIR / "models" / "v8_cs_supcon"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "v7_embeddings.npz"

    print("=" * 70)
    print("V8 CS 压缩感知分类器 — Supervised Contrastive Training")
    print("=" * 70)

    # ── Step 1-3: 提取 embeddings（有缓存则跳过）──
    if cache_file.exists():
        print(f"\n[1-3/7] 加载缓存 embeddings: {cache_file}")
        cached = np.load(str(cache_file))
        train_emb = cached['train_emb']
        train_labels = cached['train_labels']
        val_emb = cached['val_emb']
        val_labels = cached['val_labels']
        print(f"  训练集: {train_emb.shape}, 验证集: {val_emb.shape}")
    else:
        print("\n[1/7] 加载 V7 encoder...")
        emb_device = torch.device('cpu')
        encoder, tokenizer = load_v7_encoder(emb_device)

        print("\n[2/7] 加载 V7 训练数据...")
        train_texts, train_labels, val_texts, val_labels = load_training_data()

        print("\n[3/7] 提取 768D embeddings...")
        t0 = time.time()
        train_emb = extract_embeddings(encoder, tokenizer, train_texts, emb_device)
        val_emb = extract_embeddings(encoder, tokenizer, val_texts, emb_device)
        print(f"  训练集: {train_emb.shape}, 验证集: {val_emb.shape}")
        print(f"  耗时: {time.time() - t0:.1f}s")

        # 保存缓存
        np.savez(str(cache_file),
                 train_emb=train_emb, train_labels=train_labels,
                 val_emb=val_emb, val_labels=val_labels)
        print(f"  缓存已保存: {cache_file}")

        # 释放 encoder
        del encoder
        import gc; gc.collect()

    # ── Step 4-6: 维度 × 质心扫描 ──
    target_dims = [8, 16, 32, 64, 128]
    centroid_counts = [1, 3, 5]
    all_results = {}

    for dim in target_dims:
        print(f"\n{'=' * 50}")
        print(f"[4/7] 训练 CS 投影: 768D → {dim}D")
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

        # 投影所有 embeddings (CPU)
        with torch.no_grad():
            cs_model = cs_model.cpu()
            train_comp = cs_model(torch.tensor(train_emb, dtype=torch.float32)).numpy()
            val_comp = cs_model(torch.tensor(val_emb, dtype=torch.float32)).numpy()

        for nc in centroid_counts:
            key = f"{dim}d_{nc}c"
            print(f"\n  [5/7] 拟合质心: {dim}D, harmful={nc}, benign={nc}")

            # 拟合检测器
            detector = DualMultiCentroidDetector(n_harmful_centroids=nc, n_benign_centroids=nc)
            harmful_mask = train_labels == 1
            benign_mask = train_labels == 0

            # 质心数不能超过样本数
            actual_nc_h = min(nc, harmful_mask.sum())
            actual_nc_b = min(nc, benign_mask.sum())
            if actual_nc_h != nc or actual_nc_b != nc:
                detector = DualMultiCentroidDetector(actual_nc_h, actual_nc_b)

            detector.fit(train_comp[harmful_mask], train_comp[benign_mask])

            # 阈值优化
            print(f"  [6/7] 优化阈值...")
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

    # ── Step 7: 保存汇总 ──
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
