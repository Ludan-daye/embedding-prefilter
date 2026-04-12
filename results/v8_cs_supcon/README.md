# V8 CS 压缩感知分类器 — 实验报告

## 概述

V8 是基于 **Supervised Contrastive Loss** 训练的真正嵌入压缩分类器，将 768 维 BGE 嵌入压缩至 32 维（24× 压缩），通过余弦相似度质心匹配进行恶意输入检测。

### 与 V6 CS / V7 分类器的区别

| 特性 | V6 CS (旧) | V7 分类器 | **V8 CS (新)** |
|------|-----------|----------|---------------|
| 投影训练损失 | MSE 保距（无监督） | InfoNCE（仅训练时） | **Supervised InfoNCE** |
| 推理是否压缩 | ✅ 32-128D | ❌ 用 768D | **✅ 32D** |
| 推理方式 | 多质心余弦匹配 | 线性分类头 | **双侧质心余弦匹配** |
| 训练数据 | V6（无真实对话） | V7（含真实对话） | **V7（含真实对话）** |
| Encoder | V6 微调 BGE | V7 微调 BGE | **V7 微调 BGE** |
| ToxicChat FPR | **78.1%** | 5.0% | **9.3%** |

### 核心改进

1. **损失函数**：MSE → Supervised Contrastive Loss（显式拉近同类、推远异类）
2. **双侧质心**：harmful 和 benign 各用 KMeans 聚类（V6 只有 harmful 多质心）
3. **训练数据**：使用 V7 的 5643 条数据（含 DailyDialog/WildChat 真实对话）
4. **Encoder**：使用 V7 微调后的 BGE（已适配真实对话分布）

---

## 模型架构

```
用户输入
  ↓
V7 微调 BGE-base-en-v1.5 encoder → 768维 embedding (L2归一化)
  ↓
LearnedCSProjection: Linear(768, 32, bias=False) + L2归一化 → 32维
  ↓
DualMultiCentroidDetector:
  score = max(cos_sim(z, harmful_centroids)) - max(cos_sim(z, benign_centroids))
  pred = "harmful" if score > threshold else "benign"
```

- **投影层参数**：24,576 个（~100KB），占 BGE encoder 的 0.02%
- **推理表征**：32 维（768 维的 4.2%，24× 压缩）
- **存储**：128 字节/样本（vs NeMo Guard 3072 字节/样本）

---

## 训练配置

| 参数 | 值 |
|------|-----|
| Encoder | BAAI/bge-base-en-v1.5（V7 微调权重，冻结） |
| 投影层 | Linear(768, target_dim, bias=False) |
| 损失函数 | Supervised InfoNCE（temperature=0.07） |
| 优化器 | AdamW（lr=0.001, weight_decay=1e-4） |
| 学习率调度 | CosineAnnealingLR（T_max=500） |
| Batch size | 1024 |
| Epochs | 500 |
| 训练数据 | 5079 条（V7 train split，含真实对话） |
| 验证数据 | 564 条（V7 val split） |
| 维度扫描 | {8, 16, 32, 64, 128} |
| 质心数扫描 | {1, 3, 5}（双侧对称） |

### 训练结果（验证集）

| 配置 | Val F1 | Val FPR | Train F1 | Threshold |
|------|--------|---------|----------|-----------|
| 8d_1c | 0.9367 | 4.00% | 0.9426 | -0.030 |
| 16d_1c | 0.9407 | 3.38% | 0.9547 | 0.010 |
| **32d_1c** | **0.9496** | **3.38%** | **0.9615** | **0.000** |
| 64d_1c | 0.9478 | 4.00% | 0.9652 | -0.030 |
| 128d_1c | 0.9518 | 3.38% | 0.9692 | -0.020 |
| 32d_3c | 0.9172 | 0.92% | 0.9328 | 0.010 |

**关键发现**：
- 32D 已达到性能饱和（val F1=0.9496 vs 128D 的 0.9518，差距 0.2%）
- 单质心 > 多质心（SupCon 已整理好空间，KMeans 多质心反而过拟合）
- 8D（96× 压缩）仍可用（val F1=0.9367）

---

## 评测结果

### 推荐配置：32d_1c（24× 压缩）

#### 攻击检测率（Detection Rate ↑）

| 数据集 | 样本数 | V8 (32D) | V7 (768D) | 差异 | 说明 |
|--------|--------|----------|-----------|------|------|
| **AdvBench** | 200 | **97.5%** | 85.0% | +12.5% | 直接有害指令 |
| **HarmBench** | 200 | **96.5%** | 82.0% | +14.5% | 有害行为 |
| **GCG** | 100 | **96.0%** | 84.0% | +12.0% | 梯度优化对抗后缀 |
| **PAIR** | 86 | **90.7%** | 47.7% | +43.0% | 语义越狱攻击 |
| **JailbreakHub** | 79 | **81.0%** | 64.6% | +16.4% | 手动越狱模板 |
| **BeaverTails** | 300 | **99.0%** | 85.3% | +13.7% | 有害对话 |
| ToxicChat* | 300 | 20.0% | 13.3% | +6.7% | *标注错配 |

#### 误报率（FPR ↓）

| 数据集 | 样本数 | V8 (32D) | V7 (768D) | 差异 | 说明 |
|--------|--------|----------|-----------|------|------|
| **Alpaca** | 200 | **2.0%** | 1.5% | +0.5% | 常规指令 |
| **ToxicChat benign** | 300 | 9.3% | 5.0% | +4.3% | 真实无害对话 |
| JBB-Benign | 100 | 67.0% | 34.0% | +33.0% | 边界良性样本 |
| BeaverTails benign* | 300 | 87.3% | — | — | *标注错配 |

*ToxicChat_harmful 和 BeaverTails_benign 存在标注错配问题（标注的是模型回复安全性，非 prompt 攻击意图），不宜直接对比。

### 与现有方法对比

#### 攻击检测率

| 方法 | 参数量 | 推理维度 | GCG | PAIR | AdvBench | HarmBench | 独立部署 |
|------|--------|---------|-----|------|---------|-----------|---------|
| **V8 CS (32D)** | **25K** | **32D** | **96.0%** | **90.7%** | **97.5%** | **96.5%** | ✅ |
| NeMo+RF | 109M+RF | 768D | — | — | — | — | ✅ |
| PromptGuard | 86M | 全模型 | — | — | — | — | ✅ |
| Gradient Cuff | 需目标LLM | — | 98.8% | 77.0% | — | — | ❌ |
| Perplexity | ~0 | — | 96.2%* | 0.0% | — | — | ✅ |

*Perplexity 仅对机器生成的 GCG 攻击有效，人工编写越狱检测率 0%。

#### 效率对比

| 方法 | 推理表征维度 | 投影参数量 | 推理延迟 |
|------|------------|-----------|---------|
| **V8 CS** | **32D** | **25K (100KB)** | **<10ms** |
| NeMo Guard | 768D | 109M+RF | ~10ms |
| PromptGuard | — (全模型) | 86M | ~20ms |
| PromptGuard 2 | — (全模型) | 86M | ~92ms |
| InjecGuard | — (全模型) | 184M | ~15ms |

### 维度扫描（所有 1-centroid 配置）

| 维度 | 压缩比 | GCG | PAIR | AdvBench | HarmBench | BeaverTails | Alpaca FPR |
|------|--------|-----|------|---------|-----------|------------|-----------|
| 8D | 96× | 97.0% | 83.7% | 98.0% | 96.0% | 98.7% | 3.0% |
| 16D | 48× | 96.0% | 83.7% | 95.5% | 94.0% | 98.7% | 1.5% |
| **32D** | **24×** | **96.0%** | **90.7%** | **97.5%** | **96.5%** | **99.0%** | **2.0%** |
| 64D | 12× | 97.0% | 91.9% | 98.0% | 98.0% | 99.3% | 3.5% |
| 128D | 6× | 97.0% | 91.9% | 97.5% | 98.0% | 99.7% | 3.5% |

---

## 文件结构

```
models/v8_cs_supcon/
├── cs_projection_32d.pt        # 推荐配置：32D 投影权重 (~100KB)
├── cs_projection_{8,16,64,128}d.pt  # 其他维度投影权重
├── detector_32d_1c.npz         # 推荐配置：32D 单质心检测器
├── detector_{dim}d_{n}c.npz    # 其他配置检测器
├── training_results.json       # 15 种配置的训练/验证指标
├── eval_results.json           # 11 个基准数据集的评测结果
└── cache/                      # 缓存（不提交到 git）
    ├── v7_embeddings.npz       # 训练数据 embedding 缓存
    └── eval_*.npy              # 评测数据 embedding 缓存

scripts/v8_cs_supcon/
├── model.py                    # LearnedCSProjection + DualMultiCentroidDetector
├── train.py                    # 完整训练脚本（含 embedding 提取）
├── train_from_cache.py         # 从缓存训练（跳过 embedding 提取）
└── run_training.sh             # 运行脚本
```

---

## 已知局限

1. **ToxicChat harmful DR=20%**：ToxicChat 标注的是 LLM 回复毒性，非 prompt 攻击意图，属于任务错配
2. **JBB-Benign FPR=67%**：边界样本（看起来恶意但合法），嵌入方法的固有弱点
3. **BeaverTails benign FPR=87%**：`is_safe` 标注模型回复安全性，非 prompt 无害性，应排除
4. **ToxicChat benign FPR=9.3%**：高于 V7 的 5.0%，质心方法对真实对话分布的边界判定不如分类头精细

---

## 复现

```bash
# 1. 提取 V7 encoder embeddings（需要 V7 权重）
# 生成 models/v8_cs_supcon/cache/v7_embeddings.npz
python3 -c "..." # 见 train.py 中的 embedding 提取部分

# 2. 训练 CS 投影 + 质心检测器
OMP_NUM_THREADS=1 python3 scripts/v8_cs_supcon/train_from_cache.py

# 3. 评测
OMP_NUM_THREADS=1 python3 scripts/v8_cs_supcon/evaluate.py
```

注意：M1 Mac 上需要 `OMP_NUM_THREADS=1` 避免 sklearn/torch 线程冲突导致的 segfault。
