# V8 论文实验 — 需要复现的基线方法清单

> **目的**：V8 论文实验节需要在**同一批数据集**（AdvBench / HarmBench / GCG / PAIR / JailbreakHub / BeaverTails / Alpaca / ToxicChat / JBB-Benign）上跑完以下所有基线，与 V8 做公平横向对比。**不接受**仅引用原论文数字——因为各论文使用的测试集、样本数、评测协议都不同，结果不可比。
>
> 每条基线包含：① 论文出处 ② 官方模型/代码 ③ 训练数据集 ④ 复现难度 ⑤ 复现步骤。

---

## 优先级说明

| 优先级 | 含义 | 方法数 |
|---|---|---|
| 🔴 P0（必做） | 与 V8 同类型，论文故事依赖的直接对手 | 4 |
| 🟡 P1（强烈建议） | SoK 排名靠前/工业标杆 | 3 |
| 🟢 P2（可选） | 完整性/敏感度分析 | 4 |

---

## 🔴 P0：必做基线（4 项）

### 1. NeMo Guard JailbreakDetect

**核心地位**：V8 的最直接对标——同样是"BGE 类嵌入 + 轻量分类器"路线。V8 声称"维度压缩 24×"的核心对比就是它。

| 项目 | 信息 |
|---|---|
| **论文** | Galinkin & Sablotny, "Improved Large Language Model Jailbreak Detection via Pretrained Embeddings", AICS 2025. arXiv:2412.01547 |
| **arXiv** | https://arxiv.org/abs/2412.01547 |
| **官方模型** | `nvidia/nemoguard-jailbreak-detect`（HF Hub）或 NVIDIA NeMo-Guardrails 仓库 |
| **架构** | Snowflake Arctic Embed M Long（109M，768 维）→ 随机森林分类器 |
| **训练数据** | JailbreakHub + 自构造样本（具体见论文 §4） |
| **官方仓库** | https://github.com/NVIDIA/NeMo-Guardrails |
| **复现难度** | 🟢 低（官方已开源权重和 RF pickle） |
| **预估时间** | 1-2 小时（下载 + 在 11 个数据集上跑完） |

**复现步骤**：
```bash
# 1. 下载模型
huggingface-cli download Snowflake/snowflake-arctic-embed-m-long
# 2. 下载 NeMo Guard 的 RF 分类器 pickle
wget <nemo guard rf model>
# 3. 对每个测试集：
#    text → Snowflake encoder (768D) → RF.predict_proba() → threshold
# 4. 记录 DR / FPR
```

**V8 论文需要的对比数字**：GCG, PAIR, JailbreakHub, AdvBench, HarmBench, BeaverTails 的 DR + Alpaca/ToxicChat/JBB-Benign 的 FPR。

---

### 2. PromptGuard (Meta, 86M)

**核心地位**：工业界最知名的轻量前置分类器，Meta 官方方案。论文必须有它的对比。

| 项目 | 信息 |
|---|---|
| **论文** | Meta PurpleLlama Team, "PromptGuard Model Card", 2024. （非正式论文；随 Llama 系列发布） |
| **官方模型** | `meta-llama/Prompt-Guard-86M`（HF Hub，需申请 Llama 协议） |
| **架构** | mDeBERTa-v3-base 微调（86M 参数），全模型推理 |
| **训练数据** | Meta 内部 jailbreak + injection 数据集（未完全公开） |
| **官方仓库** | https://github.com/meta-llama/PurpleLlama |
| **复现难度** | 🟢 低（直接加载 HF 模型，推理即可） |
| **预估时间** | 2 小时（含模型下载 ~350MB） |

**复现步骤**：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("meta-llama/Prompt-Guard-86M")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Prompt-Guard-86M")
# 输出 3 类概率: [BENIGN, INJECTION, JAILBREAK]
# 判定: P(INJECTION) + P(JAILBREAK) > threshold → harmful
```

**注意**：PromptGuard 的 OOD 退化是论文的核心论据之一——用 V8 的 11 个测试集跑它，预期会看到明显的性能下降（已知 JailbreakHub F1=0.303）。这个对比对 V8 极其有利。

---

### 3. PromptGuard 2 (Meta, 86M / 22M)

**核心地位**：PromptGuard 的升级版，修复了 OOD 退化问题。论文里既要和 V1 比（证明 V8 参数少），也要和 V2 比（证明 V8 不只是欺负旧版本）。

| 项目 | 信息 |
|---|---|
| **论文** | Meta, "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents", arXiv:2505.03574, 2025 |
| **arXiv** | https://arxiv.org/abs/2505.03574 |
| **官方模型** | `meta-llama/Llama-Prompt-Guard-2-86M` 和 `meta-llama/Llama-Prompt-Guard-2-22M`（HF Hub） |
| **架构** | 86M: mDeBERTa-base + 能量损失函数；22M: DeBERTa-xsmall（仅英文） |
| **训练数据** | Meta 扩展后的 jailbreak + injection 集（比 V1 大） |
| **复现难度** | 🟢 低 |
| **预估时间** | 2 小时 × 2 个模型 = 4 小时 |

**复现步骤**：同 PromptGuard V1，但需要分别跑两个规模。

**为什么两个规模都要跑**：
- 86M 版本：精度天花板对比（PG2 的 Recall@1%FPR=97.5%）
- 22M 版本：与 V8 参数量最接近的同类（22M vs V8 的 25K，V8 仍小 880×）

---

### 4. InjecGuard

**核心地位**：DeBERTa 184M 参数路线的代表，"Over-defense mitigation" 论文，在相关工作必然出现。

| 项目 | 信息 |
|---|---|
| **论文** | Zhao et al., "InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models", arXiv:2410.22770, 2024 |
| **arXiv** | https://arxiv.org/abs/2410.22770 |
| **官方模型** | `leolee99/InjecGuard`（HF Hub） |
| **架构** | DeBERTa-v3-base + NotInject 训练数据（184M 参数） |
| **训练数据** | PINT + 作者构造的 NotInject 边界集 |
| **官方仓库** | https://github.com/leolee99/InjecGuard |
| **复现难度** | 🟢 低 |
| **预估时间** | 2 小时 |

**复现步骤**：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("leolee99/InjecGuard")
model = AutoModelForSequenceClassification.from_pretrained("leolee99/InjecGuard")
# 二分类: [SAFE, INJECTION]
```

**V8 的对比亮点**：InjecGuard 184M vs V8 的 25K 投影层 + 共享 BGE encoder = 参数量差 **7300×**。InjecGuard 声称解决了 over-defense，要在 XSTest / OR-Bench / JBB-Benign 上和它直接比 FPR。

---

## 🟡 P1：强烈建议（3 项）

### 5. Gradient Cuff

**核心地位**：SoK 统一评测中**轻量方法排名第一**（平均 ASR=0.148），但需要目标 LLM 梯度——这正是 V8 强调"无需目标 LLM"的对照组。

| 项目 | 信息 |
|---|---|
| **论文** | Hu et al., "Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes", NeurIPS 2024. arXiv:2403.00867 |
| **arXiv** | https://arxiv.org/abs/2403.00867 |
| **官方仓库** | https://github.com/TrustAIRLab/gradient-cuff |
| **模型** | 无独立权重——**需要**一个目标 LLM（论文用 LLaMA-2-7B-Chat 和 Vicuna-7B） |
| **训练数据** | 无训练（零样本/梯度推断） |
| **复现难度** | 🔴 高（需要 LLM 推理资源 + 反向传播） |
| **预估时间** | 需要 GPU；在 V8 11 个测试集上跑约 8-16 小时（每条 prompt 都要 LLM 梯度计算） |

**复现步骤**：
```bash
git clone https://github.com/TrustAIRLab/gradient-cuff
cd gradient-cuff
# 需要 LLaMA-2-7B-Chat 权重（申请 Meta 许可）
python gradient_cuff_detect.py \
  --target_model meta-llama/Llama-2-7b-chat-hf \
  --test_data <your_test_sets>
```

**M1 Mac 能跑吗？** 勉强（LLaMA-2-7B int4 量化约 4GB）。建议找 A100 机器。

**论文价值**：报告 Gradient Cuff 在同样 11 个数据集上的数字，**直接说明 V8 不需要目标 LLM 即可达到可比 DR**（尤其在 PAIR 上 V8=90.7% vs GCuff 引用论文的 77.0%）。

---

### 6. GradSafe

**核心地位**：另一个基于目标 LLM 梯度的方法，ACL 2024，经常和 Gradient Cuff 一起引用。

| 项目 | 信息 |
|---|---|
| **论文** | Xie et al., "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis", ACL 2024. arXiv:2402.13494 |
| **arXiv** | https://arxiv.org/abs/2402.13494 |
| **官方仓库** | https://github.com/xyq7/GradSafe |
| **模型** | 同 Gradient Cuff，需目标 LLM |
| **训练数据** | 零样本 |
| **复现难度** | 🔴 高 |
| **预估时间** | 类似 Gradient Cuff |

**复现步骤**：
```bash
git clone https://github.com/xyq7/GradSafe
# 配置目标 LLM，在测试集上运行
```

**论文价值**：SoK 评测中 GradSafe 平均 ASR=0.224（比 Gradient Cuff 差但仍是常见 baseline）。如果资源有限只能选一个梯度方法，选 Gradient Cuff（排名更高）。

---

### 7. Perplexity Filter + LightGBM

**核心地位**：最简单的"无参数"基线，0 参数也能做安全检测。是 V8 维度压缩故事的另一个极端对照——V8 压到 32 维，Perplexity 连嵌入都不需要。

| 项目 | 信息 |
|---|---|
| **论文** | Alon & Kamfonas, "Detecting Language Model Attacks with Perplexity", arXiv:2308.14132, 2023 |
| **arXiv** | https://arxiv.org/abs/2308.14132 |
| **官方模型** | 无（论文里没发布权重，但方法简单可自实现） |
| **架构** | GPT-2 计算困惑度 → LightGBM 分类 |
| **训练数据** | AdvBench GCG 攻击（2000 条） + Alpaca（2000 条） |
| **复现难度** | 🟢 低 |
| **预估时间** | 2-3 小时（训练 + 评测） |

**复现步骤**：
```python
# 1. GPT-2 small 算困惑度
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
model = GPT2LMHeadModel.from_pretrained("gpt2")
tok = GPT2TokenizerFast.from_pretrained("gpt2")

def perplexity(text):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return torch.exp(loss).item()

# 2. 训练 LightGBM
import lightgbm as lgb
# 特征: [perplexity, log(length), char_diversity, ...]
# 标签: 0/1
clf = lgb.LGBMClassifier().fit(X_train, y_train)
```

**已知结论**：Perplexity 对 GCG 等机器生成攻击检测率 96.2%，对人工越狱 0%。论文要显示**V8 在两类攻击上都稳定**的互补优势。

---

## 🟢 P2：可选（4 项）

### 8. WildGuard

| 项目 | 信息 |
|---|---|
| **论文** | Han et al., "WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs", NeurIPS 2024 |
| **官方模型** | `allenai/wildguard`（HF Hub，7B 参数） |
| **类型** | LLM-based guardrail（SoK 排名第 2） |
| **难度** | 🟡 中（7B LLM，需要 GPU） |

**价值**：作为"重型 LLM guardrail"的代表，证明 V8（0.025M）能以千分之一的参数量达到可比精度。

---

### 9. GuardReasoner

| 项目 | 信息 |
|---|---|
| **论文** | Liu et al., "GuardReasoner: Towards Reasoning-based LLM Safeguards", arXiv:2501.18492, 2025 |
| **官方模型** | `yueliu1999/GuardReasoner-8B`（HF Hub） |
| **类型** | LLM + CoT 推理（SoK 排名第 1） |
| **难度** | 🟡 中（8B 模型） |

**价值**：当前 SoK 第一名，对比"V8 的 25K 参数 vs 当前 SoTA 的 8B" = 32 万× 参数差距。

---

### 10. ProtectAI DeBERTa v2

| 项目 | 信息 |
|---|---|
| **模型** | `protectai/deberta-v3-base-prompt-injection-v2`（HF Hub） |
| **训练数据** | 未完全公开（ProtectAI 内部数据集） |
| **架构** | DeBERTa-v3-base 微调（184M） |
| **难度** | 🟢 低 |

**价值**：HuggingFace 上最多下载量的 prompt injection 分类器（社区标杆）。

---

### 11. BGE + SVM（自实现基线）

| 项目 | 信息 |
|---|---|
| **架构** | BGE-base-en-v1.5（768D）→ Linear SVM（sklearn） |
| **训练数据** | V7 训练集（5643 条，与 V8 完全一致） |
| **难度** | 🟢 极低 |

**价值**：**最关键的消融对照**——同 encoder、同训练数据，只换分类器。V8（CS 投影 + 双侧质心）vs SVM，证明 CS 压缩本身带来的提升（而不是 encoder 或数据的功劳）。建议**必做**。

---

## 评测数据集统一协议

**所有基线必须在完全相同的 11 个测试集上跑**，样本数和采样种子固定：

| 数据集 | 样本数 | 类型 | 来源 |
|---|---|---|---|
| GCG | 100 | harmful | `datasets/jailbreakbench/jbb_gcg_all.csv` |
| PAIR | 86 | harmful | `datasets/jailbreakbench/jbb_pair_all.csv` |
| JailbreakHub | 79 | harmful | `datasets/jailbreakhub/jailbreakhub.csv` |
| AdvBench | 200 | harmful | `datasets/advbench/advbench_harmful_behaviors.csv` |
| HarmBench | 200 | harmful | `datasets/harmbench/harmbench_behaviors.csv` |
| ToxicChat_harmful | 300 | harmful | `datasets/gcg_attacks/toxic_chat_full.csv` (human_annotation=True) |
| BeaverTails_harmful | 300 | harmful | `datasets/beavertails/beavertails_test.csv` (is_safe=False) |
| JBB_Benign | 100 | benign | `datasets/gcg_attacks/jbb_benign_behaviors.csv` |
| Alpaca | 200 | benign | `datasets/normal/alpaca.jsonl` |
| ToxicChat_benign | 300 | benign | `datasets/gcg_attacks/toxic_chat_full.csv` (human_annotation=False) |
| BeaverTails_benign | 300 | benign | `datasets/beavertails/beavertails_test.csv` (is_safe=True) ⚠️标注错配，可选 |

`random.seed(42)` 固定采样。

---

## 建议的复现工作流

### 阶段 1（1-2 天）：P0 闭环
跑完 NeMo Guard、PromptGuard V1/V2、InjecGuard 四个。**这是论文投稿前最底线的 baseline**，缺一个都会被审稿人质疑。全部是 HF Hub 可下载模型，M1 Mac 可跑。

### 阶段 2（2-3 天）：P1 补全
跑 Gradient Cuff + Perplexity Filter。Gradient Cuff 需要 LLaMA-2-7B + GPU，建议借 A100。

### 阶段 3（1 天）：消融基线
实现 **BGE + SVM**（同 encoder/数据/测试集，只换分类器），作为 V8 核心论点的消融证据。

### 阶段 4（可选）：SoK 高排名方法
跑 WildGuard / GuardReasoner 补充"重量级 LLM guardrail"对比，非必须。

---

## 输出格式

每个基线跑完后，产出：

```
results/baselines/<method_name>/
├── predictions_<dataset>.json   # 每条样本的 pred + score
├── metrics.json                 # DR / FPR / F1 / latency
└── README.md                    # 模型版本 + 配置 + 评测日期
```

最终汇总到 `results/baselines/comparison_table.md`，格式：

| 方法 | 参数量 | 维度 | AdvBench DR | HarmBench DR | GCG DR | PAIR DR | JailbreakHub DR | Alpaca FPR | ToxicChat_benign FPR |
|------|--------|------|-------------|--------------|--------|---------|-----------------|------------|----------------------|
| V8 (Ours) | 25K | 32D | 97.5% | 96.5% | 96.0% | 90.7% | 81.0% | 2.0% | 9.3% |
| NeMo Guard | 109M | 768D | ? | ? | ? | ? | ? | ? | ? |
| PromptGuard | 86M | - | ? | ? | ? | ? | ? | ? | ? |
| PromptGuard 2 (86M) | 86M | - | ? | ? | ? | ? | ? | ? | ? |
| PromptGuard 2 (22M) | 22M | - | ? | ? | ? | ? | ? | ? | ? |
| InjecGuard | 184M | - | ? | ? | ? | ? | ? | ? | ? |
| Gradient Cuff | 0 (LLM) | - | ? | ? | ? | ? | ? | ? | ? |
| GradSafe | 0 (LLM) | - | ? | ? | ? | ? | ? | ? | ? |
| Perplexity + LGB | ~0 | - | ? | ? | ? | ? | ? | ? | ? |
| BGE + SVM (消融) | 109M | 768D | ? | ? | ? | ? | ? | ? | ? |
| **V8 (Ours, 32D)** | **25K proj** | **32D** | **97.5%** | **96.5%** | **96.0%** | **90.7%** | **81.0%** | **2.0%** | **9.3%** |

---

## 硬件要求总结

| 方法 | 最小资源 | 推荐资源 |
|---|---|---|
| NeMo Guard | M1 Mac 8GB | — |
| PromptGuard V1/V2 (86M/22M) | M1 Mac 8GB | — |
| InjecGuard (184M) | M1 Mac 16GB | — |
| ProtectAI DeBERTa (184M) | M1 Mac 16GB | — |
| Perplexity + LGB | M1 Mac 8GB | — |
| BGE + SVM | M1 Mac 8GB | — |
| Gradient Cuff (LLaMA-2-7B) | A100 40GB | A100 80GB |
| GradSafe (LLaMA-2-7B) | A100 40GB | A100 80GB |
| WildGuard (7B) | A100 40GB | A100 40GB |
| GuardReasoner (8B) | A100 40GB | A100 80GB |

**结论**：P0 + P1 中除了梯度方法外全部能在 M1 Mac 16GB 上跑完。**梯度方法（Gradient Cuff / GradSafe / 重型 LLM guardrail）需要一次性借 A100**。

---

## 论文章节映射

| V8 论文章节 | 需要的基线 |
|---|---|
| §2.1 相关工作（前置分类器） | PromptGuard, PromptGuard 2, InjecGuard, ProtectAI |
| §2.1 相关工作（LLM 内部信号） | Gradient Cuff, GradSafe |
| §2.1 相关工作（统计特征） | Perplexity Filter |
| §2.2 嵌入安全检测 | NeMo Guard（必须！同路线对手） |
| §4.3 主实验对比表 | **全部 P0 + P1**（11 方法横向） |
| §4.4 消融实验 | BGE + SVM（同 encoder 不同分类器） |
| §4.5 效率对比 | 所有方法的参数量/延迟/维度 |
| §4.6 SoK 评测位置 | WildGuard, GuardReasoner（引论文数字即可，不必自测） |

---

## 最后提醒

1. **"引论文数字" ≠ "自测数字"**：审稿人会问"为什么你的 NeMo 数字和原论文不一样？"——答案必须是"我们用了统一的评测协议"。不统一的数字不可比。

2. **BGE + SVM 的消融是 V8 论文的灵魂**：如果同 encoder + 同数据下 SVM 的 AdvBench DR 也是 97%+，那 V8 的 CS 压缩就只是"不劣化"而不是"带来提升"——这会直接决定论文的核心卖点。**必做且必须尽早做**。

3. **Gradient Cuff 虽然难跑但不能省**：它是 SoK 中轻量路线第一名。论文如果回避它会被直接拒稿（"作者故意不和 SoTA 比"）。找台 A100 跑 8-16 小时是必要投资。
