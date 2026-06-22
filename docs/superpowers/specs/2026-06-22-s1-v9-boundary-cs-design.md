# S1 设计文档：v9 — 边界感知的学习型压缩感知（Boundary-aware Learned CS）

- 日期：2026-06-22
- 项目：embedding-prefilter（CSonEmbedding）
- 子项目：**S1**（路线 S0 → **S1** → S2 → S3 的第二步）
- 状态：待用户评审
- 依赖：S0（已完成，v8 在 A100 上可复现，弱点已量化，见 `results/REPRODUCTION_S0.md`）

---

## 1. 背景与根因

S0 证实 v8 的核心弱点：**边界正常样本假阳率灾难**——`32d_1c` 实测 JBB-Benign FPR=0.75、BeaverTails_benign=0.89，而普通正常 Alpaca 仅 0.04。

根因诊断（强证据）：v7 训练集 5643 条里**边界正常（gray-benign）只有 400 条（7%）**。模型能分开"普通正常 vs 有害"，但几乎没见过"看着像有害、其实正常"的边界样本，于是在**学习型压缩感知投影**把 768→32 维时，把这些边界正常样本压到了 harmful 质心一侧。

本子项目把这当作**学习型压缩感知（learned CS）的几何问题**来解决：改进感知矩阵 `LearnedCSProjection` 的训练，使其在低维测量空间里把边界正常样本正确地落到 benign 一侧。

## 2. 目标

在 **32d 同压缩率（24×）**下，把 v8 的 **DR–FPR 帕累托曲线整体往外推**：核心是大幅降低 JBB-Benign FPR（当前 0.75），同时尽量不掉攻击检测率（AdvBench/HarmBench/PAIR）。产出新变体 **v9**，与 v8 同协议对比。

成功判据：v9 的 (平均攻击 DR vs JBB-Benign FPR) 帕累托曲线在同一压缩率下**整体压在 v8 曲线外侧**（即任一 FPR 水平上 v9 的 DR ≥ v8，或任一 DR 水平上 v9 的 FPR ≤ v8）。不预设单点硬指标。

## 3. 方法：边界感知 SupCon 训练学习型 CS 投影

核心对象：`LearnedCSProjection(768 → 32)`（无偏置线性 + L2 归一化，即学习型压缩感知的"测量矩阵"）。分两步，天然构成"数据 vs 目标"消融：

### A1 — 边界正常数据扩充（隔离"数据效应"）
- **数据源**：`数据集/or_bench.parquet`（OR-Bench 过度拒绝基准，几千条"看着危险其实安全"的请求）+ XSTest-safe（~250）+ 现有 400 gray-benign。
- **去污染**：OR-Bench/XSTest 训练样本与 **JBB-Benign 测试集做精确 + 近重复去重**（归一化文本精确匹配 + 高余弦近邻阈值），保证零泄漏。JBB-Benign 全程**只测不训**。
- **重平衡**：gray-benign 从 400 扩到 ~2000–4000，连同原 v7 数据写入**新训练集 `datasets/v9_training/{train,val}.jsonl`**（保留 4 类标签，不动 v7_training）。
- **A1 跑**：用**未改动的 v8 SupCon 目标**、只换这份新数据，重训 32d 学习型 CS 投影 → 画帕累托曲线。隔离"数据效应"。

### A2 — 边界感知 SupCon 目标（隔离"目标效应"）
- 在现有 supervised contrastive（InfoNCE on L2-normalized z，二分类标签）基础上，**加一个困难负样本 margin 项**，作用于 4 类标签里的 gray-benign：
  - 对每个 gray-benign 锚点 g，取批内**最相似的 harmful 样本** h\*（hardest negative）与 benign 类中心 c_b；
  - margin 损失 `L_b = relu(margin + sim(g, h*) − sim(g, c_b))`，总损失 `L = L_supcon + λ · mean(L_b)`。
  - 直觉：显式把边界正常从有害一侧推回正常一侧，正是检测器 `max_harmful_sim − max_benign_sim` 依赖的几何。
- 对称地可选加 gray-harmful 项（次要）。`λ`、`margin` 做小网格扫描（用帕累托曲线选）。精确公式与实现细节在 plan 里敲定。

## 4. 评测（帕累托式）

- 对 **{v8 基线, v9-A1, v9-A2}** 各自**扫检测阈值**（如 `np.arange(-0.4, 0.41, 0.02)`），计算 (平均攻击 DR vs JBB-Benign FPR) 点列，画曲线，证明 v9 曲线压在 v8 外侧。
- 同时复用 S0 的 11 数据集表（`evaluate.py`）；新增 **OR-Bench 留出片 + XSTest** 作为额外 benign 指标。
- 固定 **32d** 公平对比；可选扫 8/16/64/128d。
- 新增 `pareto_eval.py` 产出曲线数据（JSON）+（可选）matplotlib 图。

## 5. 组件 / 产物（均放 v9 命名，不覆盖 v8）

- `scripts/v9_boundary/prepare_data.py` — 挖 OR-Bench + XSTest、去污染（vs JBB-Benign）、写 `datasets/v9_training/`。
- `scripts/v9_boundary/train.py` — 复用 v8 训练流程；`--loss {supcon, boundary}` 切换 A1/A2；输出 `models/v9_boundary/cs_projection_{dim}d.pt` + `detector_*.npz` + `training_results.json`。
- `scripts/v9_boundary/model.py` — 复用/继承 v8 的 `LearnedCSProjection` + `DualMultiCentroidDetector`；新增边界感知损失函数。
- `scripts/v9_boundary/pareto_eval.py` — 扫阈值出 DR–FPR 曲线，对比 v8/A1/A2。
- `models/v9_boundary/`、`datasets/v9_training/`、`results/v9_boundary/`（含 `S1_REPORT.md`）。

## 6. 范围

**In**：A1 数据管线（含去污染）、A1/A2 训练、帕累托评测、与 v8 同协议对比报告（含"数据 vs 目标"消融）。
**Out（留后续）**：B 决策层校准头、C 多质心几何、多语言、对抗鲁棒性、改论文正文（S3）、外部基线复现（S2）。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| margin/λ 太强 → 掉攻击 DR | 帕累托曲线监控；λ、margin 小网格扫描选点 |
| OR-Bench 与 JBB-Benign 重叠/标签噪声 | 精确+近重复去重；人工抽检若干样本 |
| A1 数据扩充导致普通 benign(Alpaca) 退化 | 评测保留 Alpaca FPR 作为护栏指标 |
| 重训方差 | 固定 seed；与 v8 同协议；报告多次或多配置 |
| 算力/网络 | A100 + 国内镜像（沿用 S0 经验：sparse 克隆、HF_ENDPOINT=hf-mirror、HF_HUB_OFFLINE、setsid nohup） |

## 8. 成功标准（验收）

- [ ] 产出去污染后的 `datasets/v9_training/`（gray-benign ≥ 2000，且与 JBB-Benign 零重叠，有去重报告）。
- [ ] v9-A1、v9-A2 在 32d 训练完成，产物落 `models/v9_boundary/`。
- [ ] 帕累托曲线显示 **v9（A1 和/或 A2）整体优于 v8**；JBB-Benign FPR 在可比攻击 DR 下显著下降。
- [ ] 护栏：Alpaca FPR 不显著恶化（仍 ≤ ~0.05）。
- [ ] `results/v9_boundary/S1_REPORT.md`：v8 vs A1 vs A2 帕累托 + 11 数据集表 + 数据/目标消融结论。

## 9. 路线上下文

| 子项目 | 内容 | 状态 |
|---|---|---|
| S0 落地复现 | 搬上 A100、复现、量化弱点 | ✅ 完成 |
| **S1 v9 边界感知 CS**（本文档） | 用 learned CS + 边界感知 SupCon 修边界假阳率 | 设计中 |
| S2 基线+消融 | P0/P1 基线 + BGE+SVM 消融 | 待 |
| S3 校正+成稿 | 维度错标、标签不匹配、补写论文 | 待 |
