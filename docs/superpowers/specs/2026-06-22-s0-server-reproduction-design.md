# S0 设计文档：服务器落地与复现（Server Onboarding & Reproduction）

- 日期：2026-06-22
- 项目：embedding-prefilter（仓库内部代号 CSonEmbedding）
- 子项目：**S0**（改进路线 S0 → S1 → S2 → S3 的第一步）
- 状态：待用户评审

---

## 1. 背景

这是用户本人的科研项目：一个放在 LLM 推理**之前**的轻量级恶意/越狱输入检测器，核心是用「学习到的投影矩阵」把句向量压到极低维度，同时靠质心 + 余弦相似度保持/提升检测能力。项目迭代到 v8。

经过对 GitHub 仓库（7-agent 并行精读 + 完整性审查）和 GPU 服务器的探测，确认了几个**对本子项目至关重要**的事实：

- **GitHub 仓库是唯一真源**，含 v7/v8；服务器上的 `~/ludan/CSonEmbedding` 是 v6 之前的旧拷贝（无 git、无 v7/v8、无 `models/`、无 `paper/`），只能当参考。
- **A100 80GB 环境基本就绪**：conda `torch` 环境有 `torch 2.5.1+cu121`、`scikit-learn 1.5.1`、`transformers 4.46`、`vllm 0.11`；缺 `sentence-transformers`（pip 即可）。
- **权重哪里都没有现成完整版**：v8 检测器质心/阈值 `.npz` 在仓库里，但**投影矩阵 `.pt`、v7 分类头权重不在仓库**（在 HF `ludandaye/embedding-prefilter` 或需重训）。训练脚本与数据齐全，A100 上重训可行。
- **已知问题（S0 只记录、不修）**：
  - 维度/版本错标——README/REPORT/对比表称「384→19, 5% 压缩」，但 v7 实为 768→128、v8 实为 768→32；`detect.py` 仍加载老的 bge-small 19 维原型（用 `fastembed`），与 v7/v8 脱节。
  - V8 边界正常样本假阳率疑似很高（JBB-Benign FPR≈67%、BeaverTails-benign≈89%，据仓库自带 `eval_results.json`）。
  - 「<10ms」延迟无任何实测脚本；无训练/测试集去重审计。

## 2. 目标（Goal）

把当前 GitHub 版本（v8）在 A100 服务器上**跑通并忠实复现** README/`results/` 中的头条数字，产出一份「声称 vs 实测」的复现报告；同时把仓库变成端到端可运行（修 `detect.py` 入口 + 延迟实测）。为 S1（修假阳率）提供**可信基线和真实数字**。

S0 不追求改进任何指标，只追求「跑通 + 复现 + 如实记录」。

## 3. 范围

**In scope**
- 服务器工作区 + conda 环境搭建（独立环境，不污染现有 env）。
- 拿到/重训 **v8** 权重（本轮**只做 v8**，不碰 v7——用户 2026-06-22 确认聚焦 v8）。
- 复现 **v8** 在 11 个数据集上的检测率 / FPR / AUC。
- 复现报告：claimed vs measured，逐项标注差异，尤其**核验** JBB-Benign / BeaverTails-benign 的真实 FPR、确认维度错标。
- 轻量工程修复：`detect.py`（或新入口）指向真实 v8 模型；延迟基准脚本。

**Out of scope（留给后续子项目）**
- 实际修复 FPR → S1。
- 复现外部基线（PromptGuard / NeMo / Gradient Cuff）+ BGE+SVM 消融 → S2。
- 修正维度/版本错标、清理标签不匹配数据集、补写论文 → S3。（S0 只**记录**这些问题。）

## 4. 工作区与环境

- 工作目录：`~/ludan/embeddingprofilter`（`git clone` 当前 GitHub 仓库到此）。
- conda 环境：新建 `embprefilter`（基于 `torch` env 克隆或新建），补装：`sentence-transformers`、`fastembed`、`lightgbm`、`faiss-cpu`（可选）、`datasets`、`huggingface_hub`。版本固定写入 `requirements-lock.txt`。
- 编码器：BGE 系列首次使用会从 HF 联网下载（A100 可联网）。

## 5. 权重策略

1. 先用 `huggingface_hub` 从 `ludandaye/embedding-prefilter` 尝试下载 **v8** 投影矩阵 `.pt`（检测器质心/阈值 `.npz` 已在仓库）。
2. 若 HF 上缺失或不全 → 在 A100 上用 `scripts/v8_cs_supcon/train.py` **重训** v8；有训练脚本 + `datasets/` + `embedding_db/` 缓存，重训也更符合「可复现」目标。
3. 报告中明确记录：用的是 **HF 下载权重** 还是 **重训权重**（直接影响复现结果的解读）。

## 6. 执行阶段

- **Phase A — 落地**：clone 仓库到服务器 → 建环境 → 装依赖 → 验证 `import torch, sentence_transformers` 通过、GPU 可见。
- **Phase B — 权重**：按第 5 节拿到 v8 权重，记录来源。
- **Phase C — 复现**：跑 **v8** 评估覆盖 11 数据集（AdvBench/HarmBench/GCG/PAIR/BeaverTails/JailbreakBench/JailbreakHub/Alpaca/ToxicChat/MaliciousInstruct/JBB-Benign），收集 DR/FPR/AUC。
- **Phase D — 报告**：写 `results/REPRODUCTION_S0.md`——每数据集 claimed vs measured，标注差异；明确量化 FPR 弱点与维度错标。
- **Phase E — 工程修复（轻量）**：`detect.py`（或新增 `run_detect.py`）指向真实 v8 模型；写 `scripts/bench_latency.py` 实测单条/批量延迟，核验「<10ms」。

## 7. 验收标准（Success Criteria）

- [ ] A100 上 `import torch, sentence_transformers` 通过，`torch.cuda.is_available()==True`。
- [ ] v8（至少 `32d_1c` 及关键配置）能加载并对样本输出预测。
- [ ] 复现 v8 在 AdvBench/HarmBench/GCG/PAIR 等的检测率，与 `models/v8_cs_supcon/eval_results.json` 差距在合理容差内（约 ±2–3 个百分点）；**若差距大，报告如实记录并分析原因**（这本身是有价值的发现）。
- [ ] 给出 JBB-Benign / BeaverTails-benign 的实测 FPR（验证 67% / 89% 是否属实）。
- [ ] 产出 `results/REPRODUCTION_S0.md` 与延迟实测数字。
- [ ] `detect.py` 能用真实 v8 模型跑通一条真实检测。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| HF 权重缺失/不全 | 在 A100 重训（预计分钟级） |
| 复现数字对不上 | 如实记录——说明 README 数字不可复现，成为 S3 校正的输入 |
| 编码器差异（fastembed vs sentence-transformers、bge-small vs base）导致数值/维度不一致 | 报告中固定并记录编码器与维度 |
| 服务器环境冲突 | 独立 conda env，不动现有环境 |

## 9. 交付物

- A100 上可端到端运行的项目（`~/ludan/embeddingprofilter`）。
- `results/REPRODUCTION_S0.md` 复现报告（claimed vs measured）。
- `scripts/bench_latency.py` 延迟基准 + 实测结果。
- 修复后的 `detect.py` 入口（指向真实 v8）。
- **S0 → S1 衔接说明**：用实测数字给出 S1（修假阳率）的精确问题定义。

## 10. 完整路线（上下文）

| 子项目 | 内容 | 依赖 |
|---|---|---|
| **S0 落地复现**（本文档） | 搬上 A100、复现、跑通、记录问题 | 无 |
| S1 修假阳率 | 解决 V8 边界 FPR 67% 硬伤，检测率/假阳率双达标 | S0 |
| S2 基线+消融 | 统一协议复现 P0/P1 基线 + BGE+SVM 消融 | S0 |
| S3 校正+成稿 | 修错标、清理标签不匹配数据集、补写论文章节 | S1, S2 |
