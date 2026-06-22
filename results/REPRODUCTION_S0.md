# S0 复现报告：v8 CS-SupCon（A100 重训复现）

- 日期：2026-06-22
- 执行环境：远程 A100 80GB 服务器，conda `torch` 环境（torch 2.8.0+cu128, transformers 4.46.3, scikit-learn 1.5.1）
- 对应 spec / plan：`docs/superpowers/specs/2026-06-22-s0-server-reproduction-design.md`、`docs/superpowers/plans/2026-06-22-s0-server-reproduction.md`

## 0. 结论速览

- **复现成功。** 在 A100 上**从零重训** v8，并用重建的 `evaluate.py` 在 11 个数据集上评测，头条检测率与原 `eval_results.json` **逐项吻合（多数 ±2pp 内，且普遍略高）**。
- **V8 的核心弱点被独立证实，且更严重**：边界正常样本假阳率 **JBB_Benign FPR=0.75（原声称 0.67）**、**BeaverTails_benign FPR=0.89（原 0.873）**。一个会误杀 3/4 边界正常请求的 prefilter **不可直接部署**——这正是 S1 要解决的问题。
- 验证集指标复现度高：重训 **32d_1c Val F1=0.9480**（原 committed ≈0.9496）。

## 1. 权重来源（重要）

- HuggingFace `ludandaye/embedding-prefilter` 上**只有 v6/v7/v7.1 的 best_model.pt，没有任何 v8 权重** → v8 **必须重训**。
- v8 训练与评估都依赖 **v7 微调编码器**（`BAAI/bge-base-en-v1.5`, 768D）。`models/v7_classifier/best_model.pt`（751M）从 **hf-mirror.com** 下载（服务器直连 huggingface.co 不通）。
- 因此本报告所有 v8 数字来自 **A100 重训权重**（非下载），这是最忠实的"端到端可复现"验证。

## 2. 方法学

- 仓库**缺失 v8 评估脚本**（README 引用的 `scripts/v8_cs_supcon/evaluate.py` 从未提交）→ 按 `scripts/test_comprehensive_v7.py` 的方法学**重建**：相同的 11 个数据集、相同的 `random.seed(42)` 子采样、相同的指标公式，模型换成 v8 的 `LearnedCSProjection(768→dim)` + `DualMultiCentroidDetector`，阈值取自重训的 `training_results.json`。
- 重训：`scripts/v8_cs_supcon/train.py`（嵌入提取改用 A100 cuda，投影训练仍 CPU；嵌入 device 无关）。产出 5 个投影 `.pt` + 15 个检测器 `.npz` + `training_results.json`。
- 指标定义：有害集 `DR = 1 - 漏检率`；正常集 `FPR = 误判为有害的比例`。

## 3. 复现核对（配置 32d_1c，阈值=-0.05）

| 数据集 | 类型 | 声称(committed) | 实测(重训) | 差异 |
|---|---|---:|---:|---:|
| GCG | DR | 0.9600 | 0.9700 | +0.0100 |
| PAIR | DR | 0.9070 | 0.9186 | +0.0116 |
| JailbreakHub | DR | 0.8101 | 0.8608 | +0.0507 |
| AdvBench | DR | 0.9750 | 0.9900 | +0.0150 |
| HarmBench | DR | 0.9650 | 0.9850 | +0.0200 |
| ToxicChat_harmful | DR | 0.2000 | 0.3000 | +0.1000 |
| BeaverTails_harmful | DR | 0.9900 | 0.9967 | +0.0067 |
| **JBB_Benign** | **FPR** | **0.6700** | **0.7500** | **+0.0800** |
| Alpaca | FPR | 0.0200 | 0.0400 | +0.0200 |
| ToxicChat_benign | FPR | 0.0933 | 0.1333 | +0.0400 |
| **BeaverTails_benign** | **FPR** | **0.8733** | **0.8900** | **+0.0167** |

- 平均有害 DR：声称 0.8296 → 实测 **0.8602**；平均正常 FPR：声称 0.4142 → 实测 **0.4533**。
- 最大单项差异 0.10 出现在 **ToxicChat_harmful**（已知**标注语义不匹配**数据集：标的是 model 回复的安全性，不是 prompt 意图，不应计入头条）。

### 差异解释（为何系统性偏高）

实测**有害 DR 与正常 FPR 同步上移**，根因是重训得到的 **32d_1c 阈值=-0.05（原 0.0）更激进**：阈值更低 → 更多样本判为有害 → 有害集 DR↑、正常集 FPR↑。叠加子样本随机性，属正常重训方差。**方向与量级一致 = 复现成立**。

## 4. 跨配置概览（实测）

| 配置 | AdvBench DR | PAIR DR | JBB_Benign FPR | BeaverTails_benign FPR |
|---|---:|---:|---:|---:|
| 8d_1c | 0.975 | 0.872 | 0.650 | 0.843 |
| 16d_1c | 0.960 | 0.884 | 0.650 | 0.857 |
| **32d_1c** | **0.990** | **0.919** | **0.750** | **0.890** |
| 32d_3c | 0.975 | 0.849 | 0.710 | 0.850 |
| 64d_1c | 0.970 | 0.907 | 0.720 | 0.877 |
| 128d_1c | 0.985 | 0.919 | 0.740 | 0.897 |

→ **边界正常假阳率在所有配置都高（JBB 0.65–0.75，BeaverTails_benign 0.84–0.90）**，不是单一配置的偶然。检测率更高的配置（32d_1c/128d_1c）假阳率也更高；唯一假阳率略低的 32d_3c 以 PAIR 掉到 0.849 为代价——典型的检测/假阳率 tradeoff。

## 5. 仅记录、不修复的问题（留给后续子项目）

- **维度/版本错标**：README/REPORT/comparison_table 仍写"384→19, 5% 压缩"，而 v8 实为 **768→32**；`detect.py` 原本加载的是 bge-small 19 维老原型。本 S0 已把 `detect.py` 改指向真实 v8（见 §6），但文档层面的错标留给 **S3** 统一校正。
- **ToxicChat / BeaverTails 标注语义不匹配**：标的是"回复安全性"而非"prompt 意图"，导致 ToxicChat_harmful DR 偏低、BeaverTails_benign FPR 虚高。头条对比应剔除或单列，留给 **S3**。

## 6. 工程可运行性（已修）

- `detect.py` 重写为加载**真实 v8（32d_1c）**：v7 微调编码器(768D) → `LearnedCSProjection(768→32)` → `DualMultiCentroidDetector`，单条交互检测。
- 延迟基准 `scripts/bench_latency.py`：见 §7（Task 7 完成后填）。

## 7. 延迟实测（A100, N=200, 32d_1c）

| 测法 | mean | p50 | p95 |
|---|---:|---:|---:|
| 端到端单条（编码器+投影+检测） | 13.57 ms | 13.53 | 13.87 |
| 仅投影+检测（嵌入已算，= v8 压缩开销） | 0.279 ms | 0.274 | 0.31 |
| 批量 200 端到端摊销 | — | — | **0.646 ms/条** |

- **detect.py 黄金路径验证通过**："How to make a bomb at home" → score 0.2395 → 有害；"What is the capital of France" → score -0.2782 → 正常。
- **"<10ms" 声称核验**：对**批量摊销（0.65ms/条）**和 **v8 自身的压缩+检测开销（0.28ms）成立**；但**单条端到端在 A100 上约 13.6ms，编码器前向占绝对主导**。即"v8 压缩极轻量"为真，整体延迟取决于编码器与是否批量化——这是诚实的 nuance，文档不应笼统宣称单条 <10ms。

## 8. S0 → S1 衔接

**S1 要解决的精确问题（基于本次实测）**：v8 `32d_1c` 在"明确正常"样本上假阳率可接受（**Alpaca FPR=0.04**），但在**边界/对抗式正常**样本上假阳率灾难性：

- **JBB_Benign FPR = 0.75**（故意写得像有害的正常请求）
- **BeaverTails_benign FPR = 0.89**（含标注语义问题，可能虚高）
- ToxicChat_benign FPR = 0.13（中等）

**S1 目标**：在**不显著降低**攻击检测率（当前 AdvBench 0.99 / HarmBench 0.985 / PAIR 0.919）的前提下，把 **JBB_Benign FPR 从 0.75 降到可部署区间（目标 ≤0.10–0.15）**。

**候选方向（留待 S1 brainstorm，不在 S0 内实现）**：
1. 把 V6 已有的 **gray-benign 模板**（教育/安全/创作类边界正常）作为困难负样本加入训练；
2. 质心打分与**校准线性头**融合，替代纯余弦阈值；
3. 在 JBB-Benign 上**重新校准阈值/温度**（注意检测率 tradeoff）；
4. 先**厘清 BeaverTails_benign / ToxicChat 的标注语义**（可能让 FPR 虚高），把头条评测口径修正（与 S3 协同）。

---

> **复现结论一句话**：v8 的"高检测率"在 A100 重训下**忠实复现**，但其"边界正常假阳率过高、不可部署"的硬伤也被**独立证实且更严重**——这就是 S1 的起点。
