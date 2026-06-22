# S2 设计文档：v9 统一协议基线对比 + BGE+SVM 消融

- 日期：2026-06-22
- 项目：embedding-prefilter（CSonEmbedding）
- 子项目：**S2**（路线 S0 → S1 → **S2** → S3 的第三步）
- 状态：待用户评审
- 依赖：S0（v8 复现）、S1（v9 = a2_l1m5，已 Pareto 改进 v8）

---

## 1. 背景

S0 在 A100 上复现了 v8；S1 产出 **v9（a2_l1m5）**，在 32d 同压缩率下帕累托改进 v8。S2 的任务是**把 v9 放到统一协议下与一组基线横向对比**，并完成**最关键的 BGE+SVM 消融**——证明提升来自 CS 压缩本身，而非编码器或数据。

仓库已有 `results/v8_cs_supcon/BASELINES_TO_REPRODUCE.md`（P0/P1/P2 清单 + 统一协议），本 spec 在其基础上**适配现状**：① 对比对象用 v9 而非 v8；② 我们有 A100 + vllm，梯度方法/LLM 护栏可跑；③ **补 LlamaGuard（清单遗漏但 related_work 反复引用）**；④ **复用已实现的 floor 基线**（`scripts/baseline_comparison.py` 已含 Keyword/TF-IDF+LR/BGE+Cosine/BGE+SVM）。

## 2. 目标

产出**投稿级横向对比表** + **BGE+SVM 消融结论**：在统一 11 数据集协议（seed 42）下，v9 与下列基线在攻击检测率（DR）/良性误报率（FPR）/参数量/维度/延迟上的逐项对比。

成功 = ① 消融明确回答"CS 压缩相对同编码器同数据的 SVM 是否有增益"；② 一张可直接进论文 §4.3 的对比表覆盖所有"必做"基线；③ 每个基线都是**自测**（统一协议），不混用原论文数字。

## 3. 基线全集

| 类别 | 方法 | 参数/依赖 | 来源 | 阶段 |
|---|---|---|---|---|
| 本方法 | **v9 (a2_l1m5)** | 25K 投影 + 共享 BGE | 复用 S1 | A |
| 消融（灵魂） | **BGE+SVM** | BGE-base 768D + SVM | 复用 `baseline_comparison.py`，改用 v9_training 数据 | A |
| floor | BGE+Cosine、TF-IDF+LR、Keyword | 轻 | 复用 `baseline_comparison.py` | A |
| 嵌入+轻分类器 | NeMo Guard | Arctic-Embed-M-Long 109M + RF | HF 下载 | B |
| 微调分类器 | InjecGuard (184M)、ProtectAI-DeBERTa (184M) | DeBERTa | HF 下载 | B |
| 微调分类器 | PromptGuard v1 (86M)、PG2 (86M/22M) | mDeBERTa | HF 下载（**门控**） | B |
| 统计 | Perplexity + LightGBM | GPT-2 + LGBM | 自实现（训练 AdvBench-GCG + Alpaca） | B |
| LLM 护栏 | **LlamaGuard-2 或 3（8B）** | vllm 推理 | HF 下载 | C |
| 梯度 | Gradient Cuff、GradSafe | 挂 **Vicuna-7B-v1.5**（免门控） | clone 官方 repo | C |

**仅引用（不自测）**：WildGuard、GuardReasoner（P2 重型 LLM）、ShieldGemma / Granite Guardian / Aegis / Lakera（更新/闭源）。在 related_work 引官方数字。

## 4. 架构（统一基线评测框架）

- **`scripts/baselines/harness.py`**：核心。给定 `predict_batch(texts)->list[int]`（1=有害），在 11 数据集上跑（复用 `scripts/v8_cs_supcon/evaluate.py` 的 DATASETS/load_texts/seed=42），算 DR/FPR/F1 + 延迟，写 `results/baselines/<method>/metrics.json`（含每数据集 + 每条 pred/score）。
- **每个基线一个小适配器**（`scripts/baselines/<method>.py`），只暴露 `predict_batch`：
  - `ours_v9.py`、`bge_svm.py`、`bge_cosine.py`、`tfidf_lr.py`、`keyword.py`（A，多数从 `baseline_comparison.py` 抽取）
  - `nemo_guard.py`、`injecguard.py`、`protectai.py`、`promptguard.py`、`perplexity_lgb.py`（B）
  - `llamaguard.py`（vllm）、`gradient_cuff.py`、`gradsafe.py`（C）
- **`scripts/baselines/make_table.py`**：汇总所有 `metrics.json` → `results/baselines/comparison_table.md`（方法/参数量/维度/各攻击 DR/各 benign FPR/延迟）。

文件边界清晰：harness 不知道任何基线细节；适配器只实现 `predict_batch`；make_table 只读 metrics.json。新增基线 = 加一个适配器，不动 harness。

## 5. 统一协议与口径

- 11 数据集、`random.seed(42)`，与 `evaluate.py` 完全一致；攻击集报 DR、benign 集报 FPR。
- 每个基线用**它自己的原生二分类判定**（在 metrics 里注明各自判定规则/阈值，如 PromptGuard 用 P(INJECTION)+P(JAILBREAK)>0.5）。
- 标注语义不匹配的 ToxicChat_benign / BeaverTails_benign **单列标注**，不进头条平均。
- 公平性声明：v9 复用 S1 权重；BGE+SVM 用**与 v9 相同的 v9_training 数据 + 相同 BGE-base 编码器**，仅换分类器——这是隔离 CS 压缩贡献的关键。

## 6. 分阶段执行（每阶段独立可交付，对比表增量增长）

- **阶段 A — 消融 + floor（最先，最便宜）**：v9 + BGE+SVM + BGE+Cosine + TF-IDF+LR + Keyword。主要复用现成代码，接到 harness。产出第一版对比表 + **消融结论**。
- **阶段 B — 轻量分类器**：NeMo Guard、InjecGuard、ProtectAI、Perplexity+LGBM（+ PromptGuard v1/v2 若能下）。HF 经 hf-mirror 下载，A100 推理。
- **阶段 C — 梯度 + LLM 护栏（最后，最重）**：LlamaGuard（vllm）；Gradient Cuff + GradSafe（挂 Vicuna-7B，clone 官方 repo），先 10 条冒烟再全量。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| PromptGuard 门控（Meta 协议）hf-mirror 下不到 | 能下就跑；下不到如实标注"gated"，引官方数字 |
| 梯度方法 repo 复现坑多 + 慢（每条 prompt 要梯度） | 放最后；Vicuna-7B 统一目标；先冒烟；若超预算降级为子集并诚实标注 |
| 7B/8B 模型经 mirror 下载慢/大 | 用 hf-mirror；后台 setsid 下载；vllm 加载一次复用 |
| 各基线判定口径不同 | 统一为原生二分类；metrics 注明各自规则 |
| 服务器连接频繁掉 | 沿用 S0/S1：setsid nohup + 轮询标记 + 启动后验证日志 |
| 各基线依赖冲突 | 适配器独立；必要时各自 pip 安装到 torch env |

## 8. 成功标准（验收）

- [ ] `scripts/baselines/harness.py` + make_table 可用；新增基线只需加适配器。
- [ ] 阶段 A 完成：v9 vs BGE+SVM 消融结论明确（同数据同编码器下 CS 压缩的增益量化）。
- [ ] 阶段 B 完成：NeMo/InjecGuard/ProtectAI/Perplexity（+PromptGuard 若可）自测数字入表。
- [ ] 阶段 C 完成：LlamaGuard 自测入表；Gradient Cuff/GradSafe 至少冒烟通过并尽量全量（超预算则子集+标注）。
- [ ] `results/baselines/comparison_table.md`（投稿级）+ `results/baselines/S2_REPORT.md`（消融 + 横向结论 + 诚实标注未跑/降级项）。

## 9. 范围

**In**：阶段 A/B/C 全部基线 + v9，统一协议，对比表 + 消融报告。
**Out**：WildGuard/GuardReasoner/ShieldGemma 等仅引用；改论文正文（S3）；继续改 v9 模型（已判定固有上限）。

## 10. 路线上下文

| 子项目 | 状态 |
|---|---|
| S0 落地复现 | ✅ |
| S1 v9 边界感知 CS | ✅ |
| **S2 基线对比 + 消融**（本文档） | 设计中 |
| S3 校正 + 成稿 | 待 |
