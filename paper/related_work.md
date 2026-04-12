# 相关工作

## 2.1 LLM 安全前置分类器

随着大语言模型在生产环境中的广泛部署，恶意输入检测已成为保障系统安全的关键环节。前置分类器作为部署在 LLM 推理之前的轻量安全模块，旨在以极低的计算开销拦截越狱提示、有害指令和提示注入等攻击，从而避免对每条输入都消耗高昂的大模型推理资源。

**基于微调 Transformer 的方法**是当前前置分类器的主流路线。Meta 提出的 PromptGuard [1] 在 mDeBERTa-v3-base（86M 参数）上进行微调，在内分布测试中取得了 AUC = 0.997 的优异表现；然而，独立评测表明其在分布外数据上性能急剧退化——在 JailbreakHub 数据集上 F1 仅为 0.303，误报率高达 50.7% [2]，且存在 99.8% 的白字符间距绕过漏洞。其后续版本 PromptGuard 2 [3] 引入能量损失函数和对抗分词修复，将 Recall@1%FPR 提升至 97.5%，但推理延迟增至 92.4ms。InjecGuard [4] 在 DeBERTa-v3-base（184M 参数）的基础上，通过构建 NotInject 训练数据缓解了过度防御问题，平均准确率达 83.5%，但其 184M 的参数规模仍然带来较大的部署成本。ProtectAI DeBERTa-v2 在内分布评测中 F1 达 0.955，但在对抗扰动下绕过率高达 67.9% [4]。总体而言，微调 Transformer 方法依赖完整模型推理，参数量通常在 22M–184M 之间，且对分布外数据和对抗攻击的鲁棒性不足。

**基于目标 LLM 内部信号的方法**利用大模型自身的梯度或损失信息进行检测。GradSafe [5]（ACL 2024）计算目标 LLM 对合规回复"Sure."的梯度，通过与安全样本梯度的余弦相似度实现零样本越狱检测，在 XSTest 上取得 F1 = 0.900；但该方法对 SAA、DrAttack 等新型攻击完全失效 [6]。Gradient Cuff [7]（NeurIPS 2024）通过分析拒绝损失景观和梯度范数实现两步检测，在 SoK 统一评测中取得了轻量方法中最低的平均 ASR = 0.148 [8]，其中 GCG 攻击的 ASR 低至 1.2%。然而，此类方法的根本局限在于必须访问目标 LLM 的内部状态（梯度或损失），无法作为独立模块部署在任意 LLM 之前。

**基于统计特征的方法**试图利用文本表面特征进行快速检测。Alon 和 Kamfonas [9] 提出基于 GPT-2 困惑度的检测方法，利用 GCG 等梯度优化攻击产生的异常高困惑度（>1000）进行识别，对机器生成攻击的检测率达 96.2%。然而，该方法对人工编写的语义越狱攻击检测率为 0%，在 SoK 统一评测中平均 ASR 为 0.239，几乎等同于无防御 [8]。这表明基于表面统计特征的方法无法应对语义层面的对抗攻击。

上述方法或依赖大规模 Transformer 推理、或依赖目标 LLM 梯度、或仅对特定攻击类型有效，且均未在 AdvBench、HarmBench 等标准有害指令基准上进行系统评测。本文旨在探索一种基于嵌入压缩的轻量路径，以更小的表征维度和更低的推理延迟实现对多类型攻击的有效检测。

## 2.2 文本嵌入在安全检测中的应用

预训练文本嵌入模型（如 BGE [10]、Snowflake Arctic Embed [11]、Sentence-BERT [12]）将文本映射到连续向量空间，使语义相关的文本具有较高的余弦相似度。近年来，研究者开始探索利用嵌入空间的几何性质进行安全检测。

Galinkin 和 Sablotny [2] 提出 NeMo Guard JailbreakDetect，使用 Snowflake Arctic Embed M Long 模型提取 768 维嵌入，然后以随机森林分类器进行越狱检测，在 JailbreakHub 上取得 F1 = 0.960 的优异表现。该工作证明了预训练嵌入本身包含丰富的安全判别信息，即使不微调嵌入模型，仅在嵌入之上训练传统机器学习分类器也能取得高性能。Ayub 和 Majumdar [13] 进一步验证了这一发现，在 OpenAI、GTE、MiniLM 等多种嵌入模型上训练逻辑回归、XGBoost 和随机森林分类器用于提示注入检测，最优 F1 达 0.867。

然而，现有基于嵌入的方法直接使用原始高维嵌入（768 维或更高）作为分类特征，带来两方面问题：一是存储开销大，每个样本需要存储 768 × 4 = 3KB 的浮点向量，规模化部署时质心库和缓存的内存需求显著；二是高维空间中存在大量与安全判别无关的冗余维度。我们的实验观察表明，恶意文本在嵌入空间中呈现显著的低维聚集性——其 PCA 有效维度仅约 15 维，内部平均余弦相似度达 0.62，显著高于正常文本的 0.55（有效维度约 45 维）。这意味着安全判别信息集中在一个远低于原始维度的子空间中，为有监督的嵌入压缩提供了理论依据。

## 2.3 余弦相似度与嵌入空间降维

余弦相似度（CS）是文本嵌入模型的核心度量方式。主流嵌入模型（BGE、GTE、E5 等）在训练阶段即以余弦相似度作为对比学习的优化目标，使得模型输出空间中的方向信息承载了主要的语义判别能力，而向量幅值的影响被 L2 归一化消除 [10]。这一特性使得余弦相似度成为嵌入空间中最自然的距离度量。

在嵌入降维领域，经典的 Johnson-Lindenstrauss 引理 [14] 证明了高维空间中的点集可以通过随机投影映射到 O(log n / ε²) 维空间，同时保持任意两点间的欧氏距离在 (1±ε) 倍范围内。然而，随机投影是无监督的，不考虑下游任务的判别需求，且保持的是欧氏距离而非嵌入模型原生的余弦相似度结构。PCA 等线性降维方法虽然可以捕获主要方差方向，但其优化目标是最大化重建方差，而非保持类间的余弦相似度差异。

本文选择余弦相似度作为压缩的优化目标，有三方面考量。第一，余弦相似度与嵌入模型的训练目标一致：嵌入模型以 CS 度量语义关联，因此在压缩过程中保持 CS 结构等价于保持语义判别能力。第二，CS 仅依赖向量方向而非幅值，这意味着压缩后的低维空间只需要编码方向信息，所需的维度远少于需同时编码方向和幅值的欧氏空间。第三，恶意文本的低维聚集性（CS = 0.62）与正常文本的分散分布（CS = 0.55）之间的 CS 差异，为有监督的 CS 保持投影提供了明确的优化信号——通过 InfoNCE 对比损失在投影空间中拉近同类 CS、推远异类 CS，可以在极低维度（19 维，仅为原始 384 维的 5%）中保留甚至增强安全相关的方向判别能力。

---

## 参考文献

[1] Meta. Prompt-Guard-86M Model Card, PurpleLlama, 2024.

[2] Galinkin, E. & Sablotny, M. Improved Large Language Model Jailbreak Detection via Pretrained Embeddings. AICS 2025, arXiv:2412.01547.

[3] Meta. LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents. arXiv:2505.03574, 2025.

[4] InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models. arXiv:2410.22770, 2024.

[5] Xie, Z. et al. GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis. ACL 2024, arXiv:2402.13494.

[6] JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation. USENIX Security 2025.

[7] Hu, X., Chen, H. & Ho, T.-Y. Gradient Cuff: Detecting Jailbreak Attacks on LLMs by Exploring Refusal Loss Landscapes. NeurIPS 2024, arXiv:2403.00867.

[8] SoK: Evaluating Jailbreak Guardrails in Large Language Models. arXiv:2506.10597, 2025.

[9] Alon, G. & Kamfonas, M. Detecting Language Model Attacks with Perplexity. arXiv:2308.14132, 2023.

[10] Xiao, S. et al. C-Pack: Packaged Resources to Advance General Chinese Embedding. arXiv:2309.07597, 2023. (BGE)

[11] Merrick, L. et al. Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models. arXiv:2405.05374, 2024.

[12] Reimers, N. & Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

[13] Ayub, R. & Majumdar, S. Embedding-based Classifiers Can Detect Prompt Injection Attacks. arXiv:2410.22284, 2024.

[14] Johnson, W. & Lindenstrauss, J. Extensions of Lipschitz Mappings into a Hilbert Space. Contemporary Mathematics, 1984.
