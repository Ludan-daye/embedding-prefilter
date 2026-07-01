# CHANGELOG

## 2026-07-01 — 表1 + 表4 5 种子确认(主结果表全部钉死)

- 新增 [`scripts/v9_boundary/v16_decision_seeds.py`](scripts/v9_boundary/v16_decision_seeds.py) + [`results/v9_boundary/v16_decision_seeds_results.json`](results/v9_boundary/v16_decision_seeds_results.json):表1 判别头 2×2,5 种子。BGE 双质心→判别头 XSTest 0.656±.010→**0.013±.012**、攻击 DR 0.865→0.913;std 远小于差距 → 决策规则结论铁证。
- 新增 [`scripts/baselines/pareto_seeds.py`](scripts/baselines/pareto_seeds.py) + [`results/baselines/pareto_seeds_results.json`](results/baselines/pareto_seeds_results.json):表4 统一主表,5 种子。提速=基线每集全量推理一次、每种子取子集;最终变体每种子重训。最终 3 变体霸占 J 前三(0.839±.006 / 0.802±.024 / 0.801±.028),到第 4 名(0.515/0.508)差 ~10 std。
- **至此表1/2/4 全部 5 种子 mean±std**,`paper/FINAL_SYSTEM.md` §1/§4/§6/§7/§9 全部更新,§9 主实验缺口标记闭合。轻量变体统一为 PCA-128(6×)。
- 修了 pareto_seeds 的 EMB_KEY 映射 bug(最终变体用 jbb/xs/alp 键,基线用 JBB_Benign/XSTest/Alpaca 键)。

## 2026-07-01 — v15 压缩 sweep 5 种子定论(压缩 vs 不压缩)

- 新增 [`scripts/v9_boundary/v15_compress_seeds.py`](scripts/v9_boundary/v15_compress_seeds.py) + [`results/v9_boundary/v15_compress_seeds_results.json`](results/v9_boundary/v15_compress_seeds_results.json)。BGE 编码一次,5 种子(42–46)× {1×,3×,5×,6×,12×,24×} × {随机,PCA},报 mean±std + 配对 ΔJ vs 不压缩。
- **定论**:①无任何压缩变体在综合 J 上严格优于不压缩(此前单种子"PCA-32 更好"是噪声);②PCA ~6×(128D)统计免费(J 0.753±.03 vs 不压缩 0.741±.04),头小 6×;③PCA 在每个比例都优于随机,压得越狠差距越大。
- 据此修正 `paper/FINAL_SYSTEM.md`:推荐轻量变体改为 **PCA-128(6×)**;§4 表2 换成种子表;§1/§5/§6/§7/§9 全部同步"压缩不严格更优,卖点=小6×无损"。
- 修了脚本 bug:val 打分处 `C()` 双重压缩(768→256 再 @256×768)。

## 2026-07-01 — baseline 并轨:统一留出协议主表

- 新增 [`scripts/baselines/pareto_final.py`](scripts/baselines/pareto_final.py) + 结果 [`results/baselines/pareto_final_results.json`](results/baselines/pareto_final_results.json)。
- **干净留出协议**(重叠数据集 70/30 seed42,所有方法同批 30% 留出评测;ToxicChat/BeaverTails 全集),消除训练/测试泄漏。
- 一张主表并轨:最终系统 3 变体(E5×判别头 / BGE×判别头 / BGE×PCA-32)+ 8 基线(ours_v9 / bge_svm / bge_cosine / tfidf / keyword / perplexity / injecguard / protectai),含 XSTest 过度拒绝列。
- **结果**:三个最终变体霸占 Youden's J 前三(0.849/0.846/0.755),远超 InjecGuard(0.533)、旧 v9(0.510);同 BGE 编码器下判别头把 XSTest 过度拒绝从 0.587–0.773 塌到 **0.027**,攻击 DR 反升。旧 v9 在统一留出集上被严格支配。
- 已写入 `paper/FINAL_SYSTEM.md` §4 表4(核心对比)、§7 可追溯、§9(并轨缺口已闭合,剩种子确认)。
- 过程修了两个脚本 bug:`del _bge/_e5` UnboundLocalError;floor 基线返回字符串标签的 `int('benign')` 类型错误(改用 `_to01`)。

## 2026-07-01 — 论文写作:最终版系统整合

- 新增 [`paper/FINAL_SYSTEM.md`](paper/FINAL_SYSTEM.md):当前**唯一权威**的系统与结果基准。含最终系统定义、论文叙事、三个部署变体(Pareto)、精确训练/评测协议、4 张主结果表(数字逐一核对 v13/v14 JSON)、4 条贡献、诚实局限、每个数字→支撑脚本的可追溯映射、待种子确认缺口。
- 新增 [`TODO.md`](TODO.md):论文写作阶段任务表(P00 已完成;P01 补种子为最高优硬缺口;P02–P07 各章节写作)。
- 明确**作废** `paper/main.tex` / `methodology.tex` / `abstract.md`(2026-06-22,基于已被推翻的 CS 框架),重写以 FINAL_SYSTEM 为准。
- 核心结论沉淀(来自 v13/v14):嵌入式前置过滤器的过度拒绝**主要是决策规则伪影**;判别头 + 多样攻击 + 边界训练是零代价严格改进;E5-large 把过度拒绝压到近零;PCA-32(24×)保留修复。
- 未跑任何新实验;纯文档整合。种子确认(P01)待用户同意后在 A100 执行。

## 2026-06-23 — README 诚实重写 + v13/v14 突破(见 memory: project-state-and-roadmap)

- commit dd0766c:README 重写为 v13/v14 决策规则主线并推送 GitHub。
- v13(`v13_eval.py`)、v14(`v14_compress.py`)、v12(`v12_eval.py`)、encoder_sweep、margin_datatest、v11 微调(负结果)脚本与结果 JSON 全部提交。
