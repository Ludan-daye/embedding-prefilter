# TODO — 论文写作阶段(最终版)

> 系统与结果的**唯一权威基准**:[`paper/FINAL_SYSTEM.md`](paper/FINAL_SYSTEM.md)。
> 旧的 `paper/main.tex` / `methodology.tex` / `abstract.md` 基于已推翻的 CS 框架,**作废**,勿照抄。
> 规则:标 `[x]` 前必须有可验证证据;远程 GPU 实验须用户明确同意才跑。

## 主任务表

| ID | 状态 | 任务 | 优先级 | 相关文件 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|---|---|
| P00 | `[x]` | 整合最终系统基准 | 高 | [FINAL_SYSTEM.md](paper/FINAL_SYSTEM.md) | 最终系统/叙事/全部真实结果表/贡献/局限/可追溯来源落成一份权威文档 | 已完成,数字逐一核对 v13/v14 JSON | 2026-07-01 |
| P01 | `[ ]` | 补随机种子确认 v13/v14 | 高 | [v13_eval.py](scripts/v9_boundary/v13_eval.py)、[v14_compress.py](scripts/v9_boundary/v14_compress.py) | 3–5 种子重跑,报均值±std,判别头三项全胜/PCA-32 三赢站稳 | 待验证(**需用户同意在 A100 上跑**) | 2026-07-01 |
| P02 | `[ ]` | 定稿 Abstract + Intro | 高 | [FINAL_SYSTEM.md §1/§5](paper/FINAL_SYSTEM.md)、`paper/abstract.md`(重写) | 以"过度拒绝=决策规则伪影"为主线,4 条贡献清晰 | 待验证 | 2026-07-01 |
| P03 | `[ ]` | 重写 Method 章节 | 高 | [FINAL_SYSTEM.md §2/§3](paper/FINAL_SYSTEM.md)、`paper/methodology.tex`(重写) | 判别头 recipe + 训练/评测协议 + pipeline 图,去掉 CS 叙述 | 待验证 | 2026-07-01 |
| P04 | `[ ]` | 写 Experiments 章节 | 高 | [FINAL_SYSTEM.md §4](paper/FINAL_SYSTEM.md) | 表1(2×2)+表2(压缩)+表3(每类)+表4(基线/Youden's J)入稿 | 待验证 | 2026-07-01 |
| P05 | `[ ]` | 写 Limitations + 负结果 | 中 | [FINAL_SYSTEM.md §6](paper/FINAL_SYSTEM.md) | CS 推翻/margin 无效/微调失败/评测集小/PAIR 弱 如实入稿 | 待验证 | 2026-07-01 |
| P06 | `[ ]` | Related Work + 定位 | 中 | [FINAL_SYSTEM.md §5 定位](paper/FINAL_SYSTEM.md)、`paper/related_work.md` | RCS(激活压缩)/ NemoGuard / 轻量BERT / refusal-direction 区分清楚,不 overclaim | 待验证 | 2026-07-01 |
| P07 | `[ ]` | 选定投稿目标与篇幅 | 中 | — | 确定会议/期刊(应用向 or 测量+负结果向),据此裁剪 | 待验证 | 2026-07-01 |

## 高优任务详情

### P01: 补随机种子确认 v13/v14

| 项目 | 内容 |
|---|---|
| 状态 | `[ ]` 未开始 |
| 目标 | 把 v13(判别头三项全胜)、v14(PCA-32 三赢)从单种子小样本升级为多种子均值±std 铁证 |
| 相关文件 | `scripts/v9_boundary/v13_eval.py`、`v14_compress.py`;结果入 `results/v9_boundary/` |
| 完成标准 | ≥3 种子;报每格均值±std;结论方向不因种子翻转 |
| 当前进展 | 脚本就绪,单种子结果已出;评测集小(JBB30/XSTest75/各攻击24–60)是唯一硬缺口 |
| 验证方式 | 跑完看 std 是否小于关键差距(如 XSTest 0.773→0.133 的差) |
| 验证结果 | 待验证 —— **须用户明确同意才在 A100 上跑**(memory: ask-before-running-experiments) |

### P02: 定稿 Abstract + Intro

| 项目 | 内容 |
|---|---|
| 状态 | `[ ]` 未开始 |
| 目标 | 用"没有造更强表示,而是发现过度拒绝是次优几何决策人为制造的,并给出简单严格更优且可极限压缩的修复"为叙事 |
| 相关文件 | `paper/FINAL_SYSTEM.md` §1/§5;重写 `paper/abstract.md` |
| 完成标准 | 一段话讲清问题→发现→修复→部署;4 条贡献列清;无 CS overclaim |
| 当前进展 | 素材已在 FINAL_SYSTEM 与 README 就绪 |
| 验证方式 | 通读能否独立讲清故事;贡献与结果表一一对应 |
| 验证结果 | 待验证 |
