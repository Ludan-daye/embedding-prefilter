# TODO — 论文写作阶段(最终版)

> 系统与结果的**唯一权威基准**:[`paper/FINAL_SYSTEM.md`](paper/FINAL_SYSTEM.md)。
> 旧的 `paper/main.tex` / `methodology.tex` / `abstract.md` 基于已推翻的 CS 框架,**作废**,勿照抄。
> 当前投稿稿件:英文 [`paper/aaai2026/main_aaai.tex`](paper/aaai2026/main_aaai.tex)(AAAI 2026 匿名版)+ 中文工作稿 [`paper/main_cn.tex`](paper/main_cn.tex)(内容迭代用,与英文稿逐段对齐)。
> 规则:标 `[x]` 前必须有可验证证据;远程 GPU 实验须用户明确同意才跑。

## 主任务表

| ID | 状态 | 任务 | 优先级 | 相关文件 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|---|---|
| P00 | `[x]` | 整合最终系统基准 | 高 | [FINAL_SYSTEM.md](paper/FINAL_SYSTEM.md) | 最终系统/叙事/全部真实结果表/贡献/局限/可追溯来源落成一份权威文档 | 已完成,数字逐一核对 v13/v14 JSON | 2026-07-01 |
| P01 | `[x]` | 补随机种子确认 | 高 | [v16_decision_seeds.py](scripts/v9_boundary/v16_decision_seeds.py)、[v15_compress_seeds.py](scripts/v9_boundary/v15_compress_seeds.py)、[v17_e5_compress_seeds.py](scripts/v9_boundary/v17_e5_compress_seeds.py)、[pareto_seeds.py](scripts/baselines/pareto_seeds.py) | 表1/2/2b/4 全部 ≥5 种子 mean±std,结论方向不因种子翻转 | 已完成:4 份种子结果 JSON 落盘(`results/v9_boundary/v1{5,6,7}_*_seeds_results.json`、`results/baselines/pareto_seeds_results.json`),std 远小于关键差距;唯一修正:单种子"PCA-32 更好"是噪声,改为"PCA-128 6× 统计免费" | 2026-07-02 |
| P02 | `[x]` | 定稿 Abstract + Intro | 高 | [main_aaai.tex](paper/aaai2026/main_aaai.tex)、[main_cn.tex](paper/main_cn.tex) | 以"过度拒绝=决策规则伪影"为主线,4 条贡献清晰 | 已完成:英文摘要+引言(RQ1/RQ2 框架、4 条完整句贡献)与中文版逐段对齐,双双编译通过(main_aaai.pdf / main_cn.pdf, 07-02);遗留 Fig.1 见 P08 | 2026-07-02 |
| P03 | `[ ]` | 写 Method 章节 | 高 | [FINAL_SYSTEM.md §2/§3](paper/FINAL_SYSTEM.md)、`paper/aaai2026/main_aaai.tex` §Method | 威胁模型 + 决策规则诊断 + 判别头 recipe + PCA-128 压缩 + 训练/评测协议,无 CS 叙述 | 待验证 | 2026-07-02 |
| P04 | `[ ]` | 写 Experiments 章节 | 高 | [FINAL_SYSTEM.md §4](paper/FINAL_SYSTEM.md)、`paper/aaai2026/main_aaai.tex` §Experiments | 表1(2×2 消融)+表2(BGE 压缩)+表2b(E5 压不动)+表4(统一留出主表)入稿,均为 5 种子 mean±std | 待验证 | 2026-07-02 |
| P05 | `[ ]` | 写 Limitations + 负结果 | 中 | [FINAL_SYSTEM.md §6](paper/FINAL_SYSTEM.md) | CS 推翻/margin 无效/微调失败/评测集小/PAIR 弱/压缩不严格更优 如实入稿 | 待验证 | 2026-07-02 |
| P06 | `[x]` | Related Work + 定位 | 中 | [main_aaai.tex](paper/aaai2026/main_aaai.tex)、[verified_citations.md](related_work/verified_citations.md)、[refs.bib](paper/aaai2026/refs.bib) | 五段结构(LLM护栏/白盒/轻量同族/过度拒绝/几何+压缩);RCS 白盒对比、refusal-direction 写"利用"非"发现",无 overclaim | 已完成:94-agent 工作流检索+逐篇对抗验证 82 条引用(0 造假),refs.bib 全部为验证版;中英文稿双双编译通过,bibtex 0 警告 0 未定义;引言"双质心"误归 galinkin 已修正 | 2026-07-02 |
| P07 | `[x]` | 选定投稿目标与篇幅 | 中 | [paper/aaai2026/](paper/aaai2026/) | 确定会议/期刊,据此裁剪 | 已完成:AAAI 2026(匿名投稿模板已配置并编译通过);风格基准 JBShield (USENIX Sec'25) | 2026-07-02 |
| P08 | `[ ]` | 画 Fig.1(pipeline + 消融柱状图) | 中 | `paper/aaai2026/main_aaai.tex`(Intro 内 TODO 标记) | 左:text→冻结 BGE→PCA-128→判别头 pipeline;右:XSTest FPR 几何 vs 判别头 2×2 柱状图 | 待验证 | 2026-07-02 |
| P09 | `[ ]` | 写 Conclusion + 全文自查 | 中 | `paper/aaai2026/main_aaai.tex` | 结论章成稿;全文数字逐一对照 FINAL_SYSTEM §7 可追溯表;refs.bib 无缺失引用;编译零警告 | 待验证 | 2026-07-02 |

## 高优任务详情

### P03: 写 Method 章节

| 项目 | 内容 |
|---|---|
| 状态 | `[ ]` 未开始 |
| 目标 | 把 FINAL_SYSTEM §2(系统定义)+ §3(训练/评测协议)写成 AAAI 稿 Method 章 |
| 相关文件 | [FINAL_SYSTEM.md §2/§3](paper/FINAL_SYSTEM.md);写入 `paper/aaai2026/main_aaai.tex`,同步 `paper/main_cn.tex` |
| 完成标准 | ①威胁模型(可参考旧稿);②决策规则诊断(encoder sweep AUC≈0.999 + 表征侧三路失败);③判别头 recipe(公式+训练数据配比);④PCA-128 压缩;⑤留出协议与阈值选择(DR≥0.90 最低 FPR)。全程无 CS 叙述 |
| 当前进展 | 素材齐备,章节占位已在稿内 |
| 验证方式 | 通读能否独立复现;协议数字与 FINAL_SYSTEM §3 逐一一致;编译通过 |
| 验证结果 | 待验证 |

### P04: 写 Experiments 章节

| 项目 | 内容 |
|---|---|
| 状态 | `[ ]` 未开始 |
| 目标 | 四张主表入稿并配读法分析,全部 5 种子 mean±std |
| 相关文件 | [FINAL_SYSTEM.md §4](paper/FINAL_SYSTEM.md);数据源 `results/v9_boundary/v1{5,6,7}_*_seeds_results.json`、`results/baselines/pareto_seeds_results.json` |
| 完成标准 | 表1 决策消融、表2 BGE 压缩、表2b E5 压不动、表4 统一主表(11 方法,J 前三);每表配"读法"段;诚实 caveat 入脚注(训练型基线未重训、ToxicChat/BeaverTails 语义不匹配、门控护栏引官方数字) |
| 验证方式 | 每个数字对照 FINAL_SYSTEM §7 可追溯表核对一遍 |
| 验证结果 | 待验证 |
