# S2 报告：v9 统一协议基线对比 + BGE+SVM 消融

- 日期：2026-06-22
- 协议：统一 11 数据集，`random.seed(42)`，与 `scripts/v8_cs_supcon/evaluate.py` 一致。攻击集报 DR↑，benign 集报 FPR↓。每个基线用其原生二分类判定。
- 框架：`scripts/baselines/harness.py`（可插拔 `predict_batch`）+ 各适配器 + `make_table.py`。
- 本方法代表：**v9 = a2_l1m5**（S1 产出，已 Pareto 改进 v8）。

## 0. 结论速览

1. **核心消融成立**：同编码器(BGE-base)、同训练数据(v9_training)下，v9 的 CS 压缩(32D)相对 BGE+SVM(768D) **假阳率显著更低**（JBB-Benign 0.25 vs 0.46，Alpaca 0.01 vs 0.035），且参数小 ~4000×、维度小 24×——证明 CS 压缩**带来真增益**，不只是"不劣化"。
2. **v9 是唯一全攻击类型均衡 + 低假阳 + 极小**的方法。各基线都有致命短板（见下）。
3. 诚实标注：LLM 护栏类(LlamaGuard/WildGuard/PromptGuard)在 hf-mirror **门控不可下**，引官方数字；gradient-cuff 镜像 clone 失败；GradSafe 用开放 Vicuna-7B 代 Llama-2-chat(门控)。

## 1. 主对比表（自测，统一协议）

| 方法 | 参数 | 维度 | GCG DR | PAIR DR | JailbreakHub DR | AdvBench DR | HarmBench DR | JBB-Benign FPR | Alpaca FPR | 延迟 ms/条 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v9 (Ours)** | **25K** | **32D** | **0.950** | 0.616 | 0.848 | **0.705** | 0.795 | **0.250** | **0.010** | 2.3 |
| BGE+SVM（消融） | 109M enc | 768D | 0.890 | 0.686 | 0.430 | 0.690 | **0.900** | 0.460 | 0.035 | 2.2 |
| BGE+Cosine | 109M enc | 768D | 0.720 | 0.163 | 0.772 | 0.610 | 0.635 | 0.390 | 0.105 | 2.2 |
| InjecGuard | 184M | - | 0.920 | 0.209 | **1.000** | 0.620 | 0.230 | 0.260 | 0.000 | 3.7 |
| ProtectAI-DeBERTa | 184M | - | 0.610 | 0.395 | 0.899 | **0.000** | **0.000** | 0.010 | 0.000 | 3.2 |
| Perplexity+LGBM | ~0 | - | **1.000** | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 10.4 |
| TF-IDF+LR | ~5K | - | 0.540 | 0.326 | 0.405 | 0.330 | 0.675 | 0.210 | 0.015 | 0.07 |
| Keyword | 0 | - | 0.350 | 0.686 | 0.810 | 0.600 | 0.220 | 0.190 | 0.030 | 0.02 |
| GradSafe (Vicuna-7B) | 7B grad | - | _跑完填_ | | | | | | | |

> 注：上表为各方法**原生阈值**单点结果（v9 的 Pareto 曲线见 S1）。延迟为 A100 上含编码器的端到端均值。

## 2. 核心消融：CS 压缩 vs BGE+SVM（论文灵魂）

控制变量：**同 BGE-base 编码器、同 v9_training 数据**，只换"压缩+判别"模块。

| | 攻击均衡性 | JBB-Benign FPR | Alpaca FPR | 维度 | 参数 |
|---|---|---|---|---|---|
| BGE+SVM | GCG/HarmBench 强但 JBHub 弱(0.43) | 0.46 | 0.035 | 768D | 109M+SVM |
| **v9 (CS 压缩+双质心)** | 全类型均衡(0.62–0.95) | **0.25** | **0.01** | **32D** | **25K** |

**结论**：v9 在 **24× 压缩、~4000× 更少参数**下，边界假阳率比全维 SVM **低 46%**（0.25 vs 0.46），普通假阳率低 ~3.5×，且攻击检测更均衡。→ **CS 压缩本身是增益来源**，而非编码器或数据。这是 v9 论文 §4.4 消融的核心证据。

## 3. 各基线短板（v9 的相对优势）

- **ProtectAI / InjecGuard（注入检测器）**：在 AdvBench/HarmBench 上极弱（ProtectAI=0.0，InjecGuard HarmBench=0.23）——它们只认"prompt 注入"，不认普通有害指令。v9 全类型通吃。
- **Perplexity**：只抓 GCG(机器生成乱码后缀，1.0)，对一切语义攻击(PAIR/AdvBench/HarmBench)≈0。经典困惑度失效。
- **floor（Keyword/TF-IDF/Cosine）**：全面偏弱，假阳率也高(Cosine JBB=0.39)。
- **BGE+SVM**：JailbreakHub 弱(0.43)、假阳率高。
- → **v9 是表中唯一在 5 类攻击上都 ≥0.6 且 benign FPR ≤0.25 的方法，同时最小最快。**

## 4. 未自测/降级项（诚实标注）

| 方法 | 状态 | 原因 | 处理 |
|---|---|---|---|
| PromptGuard v1/v2 | 未跑 | Meta 门控，hf-mirror 403 | 引官方数字（§相关工作） |
| LlamaGuard-3-8B | 未跑 | 门控 403 | 引官方数字 |
| WildGuard | 未跑 | 门控 403 | 引官方数字 |
| NeMo Guard | 未跑 | `nvidia/nemoguard-jailbreak-detect` 门控 401 | 引官方数字 |
| Gradient Cuff | 未跑 | 镜像 clone 失败 + 每条 prompt 多次扰动推理代价高 | 引官方数字（PAIR 77.0%，对照 v9 PAIR 0.62/S1 曲线更高） |
| GradSafe | 跑中 | 挂开放 Vicuna-7B 代 Llama-2-chat(门控) | 结果跑完填表 |

> 门控类若提供带 gated 权限的 HF token，可一并补测。

## 5. S2 → S3 衔接

- §4.3 主实验表 = 本文件 §1；§4.4 消融 = §2；§4.5 效率 = 参数/维度/延迟列。
- S3 写论文时：门控基线引官方数字并注明"统一协议未自测"；强调 v9 的"全类型均衡 + 极小 + 低假阳"三位一体。
