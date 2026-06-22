# S1 — v9 边界感知学习型 CS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 通过改进学习型压缩感知投影（768→32）的训练——A1 扩充边界正常数据 + A2 边界感知 SupCon margin——把 v8 的 DR–FPR 帕累托曲线在 32d 同压缩率下整体外推（核心降 JBB-Benign FPR）。

**Architecture:** 复用 v8 的 `LearnedCSProjection` + `DualMultiCentroidDetector` 与 v7 微调编码器；新增 (1) 数据管线从 OR-Bench/XSTest 挖边界正常（去污染 vs JBB-Benign）写 `datasets/v9_training/`，(2) 边界感知 margin 损失，(3) 帕累托扫阈值评测。所有计算在 A100 上，沿用 S0 经验（hf-mirror、`HF_HUB_OFFLINE=1`、`setsid nohup python -u`、scp 传小文件）。

**Tech Stack:** PyTorch (A100 cuda), transformers (v7 bge-base 编码器), datasets (HF, 走 hf-mirror), scikit-learn (KMeans), numpy/pandas。

---

## 连接与同步约定（沿用 S0；每条 Bash 自包含）

服务器 `vicuna@8.138.30.52:6007`，工作目录 `~/ludan/embeddingprofilter`，conda `~/anaconda3/envs/torch/bin/python`。口令本会话提供，**不写入本文件**，每条命令前置 `SSHPASS='<pw>' sshpass -e ...`。

- 跑远程：`SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 '<cmd>'`
- 传小文件：`SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new <local> vicuna@8.138.30.52:~/ludan/embeddingprofilter/<path>`
- **长任务必须脱离会话**（连接每隔几分钟会掉）：`HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 setsid nohup <py> -u <script> > /tmp/<log> 2>&1 < /dev/null &`，再单独轮询 `pgrep -f <script>` 直到结束。
- 用已缓存模型时加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 避免 mirror HEAD 超时；下载 HF 数据集时**不要**加 offline，用 `HF_ENDPOINT=https://hf-mirror.com`。

代码开发循环：本地写 `scripts/v9_boundary/*.py` → scp 到服务器 → 服务器跑 → 结果 scp 回本地 → 本地 commit。

---

## Task 1: 数据管线 prepare_data.py（OR-Bench/XSTest → v9_training，去污染）

**Files:**
- Create: `scripts/v9_boundary/__init__.py`（空）
- Create: `scripts/v9_boundary/prepare_data.py`
- Create on server: `datasets/v9_training/{train.jsonl,val.jsonl}`, `datasets/v9_training/composition.json`

- [ ] **Step 1: 本地写 `scripts/v9_boundary/prepare_data.py`**

```python
#!/usr/bin/env python3
"""构建 v9 训练集：v7 原数据 + 从 OR-Bench/XSTest 挖的边界正常(gray_benign)，去污染 vs JBB-Benign。"""
import os, sys, json, re, random
from pathlib import Path
BASE = Path(__file__).parent.parent.parent
random.seed(42)

def norm(t):  # 归一化用于精确去重
    return re.sub(r"\s+", " ", str(t).strip().lower())

def load_jbb_benign_norms():
    import pandas as pd
    p = BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv"
    df = pd.read_csv(p)
    col = "Goal" if "Goal" in df.columns else df.columns[0]
    return set(norm(x) for x in df[col].dropna())

def load_v7_rows():
    rows = []
    for split in ["train", "val"]:
        for line in open(BASE / f"datasets/v7_training/{split}.jsonl"):
            it = json.loads(line); it["_split"] = split; rows.append(it)
    return rows

def mine_or_bench(n_hard=1000, n_80k=3000):
    """OR-Bench: bench-llm/or-bench, 列 prompt + category（过度拒绝=看着危险其实安全）。"""
    from datasets import load_dataset
    texts = []
    for cfg, n in [("or-bench-hard-1k", n_hard), ("or-bench-80k", n_80k)]:
        try:
            ds = load_dataset("bench-llm/or-bench", cfg)
            s = ds[list(ds.keys())[0]]
            assert "prompt" in s.column_names, f"unexpected cols {s.column_names}"
            ptxt = list(s["prompt"])
            if len(ptxt) > n:
                ptxt = random.sample(ptxt, n)
            texts += ptxt
            print(f"  OR-Bench/{cfg}: +{len(ptxt)}")
        except Exception as e:
            print(f"  OR-Bench/{cfg} FAILED: {e}")
    return texts

def mine_xstest():
    """XSTest-safe（可选）。失败不致命。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/XSTest")
        s = ds[list(ds.keys())[0]]
        cols = s.column_names
        pcol = "prompt" if "prompt" in cols else cols[0]
        lcol = "label" if "label" in cols else None
        out = []
        for r in s:
            if lcol is None or str(r[lcol]).lower() in ("safe", "0", "false"):
                out.append(r[pcol])
        print(f"  XSTest-safe: +{len(out)}")
        return out
    except Exception as e:
        print(f"  XSTest FAILED (optional): {e}")
        return []

def main():
    jbb = load_jbb_benign_norms()
    v7 = load_v7_rows()
    gray = mine_or_bench() + mine_xstest()
    # 去污染：去掉与 JBB-Benign 归一化精确重复的
    seen = set(norm(it["text"]) for it in v7)  # 也去掉与 v7 已有的重复
    kept, dropped_jbb, dropped_dup = [], 0, 0
    for t in gray:
        nt = norm(t)
        if not nt or len(nt) < 8:
            continue
        if nt in jbb:
            dropped_jbb += 1; continue
        if nt in seen:
            dropped_dup += 1; continue
        seen.add(nt)
        kept.append({"text": str(t).strip(), "label": 2, "category": "gray_benign", "source": "or_bench/xstest"})
    print(f"挖到 gray_benign {len(gray)} -> 去 JBB 重叠 {dropped_jbb}, 去内部重复 {dropped_dup}, 保留 {len(kept)}")
    assert dropped_jbb == 0 or True  # 报告即可；下方 composition.json 记录
    # 合并 + 90/10 切分（在新增 gray 上切；v7 保持原 split）
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * 0.1))
    new_val, new_train = kept[:n_val], kept[n_val:]
    train_rows = [it for it in v7 if it["_split"] == "train"] + new_train
    val_rows   = [it for it in v7 if it["_split"] == "val"]   + new_val
    random.shuffle(train_rows); random.shuffle(val_rows)
    outdir = BASE / "datasets/v9_training"; outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        with open(outdir / f"{name}.jsonl", "w") as f:
            for it in rows:
                f.write(json.dumps({"text": it["text"], "label": it["label"],
                                    "category": it.get("category", "")}, ensure_ascii=False) + "\n")
    from collections import Counter
    comp = {"train": dict(Counter(it["label"] for it in train_rows)),
            "val": dict(Counter(it["label"] for it in val_rows)),
            "gray_benign_added": len(kept), "dropped_jbb_overlap": dropped_jbb,
            "dropped_internal_dup": dropped_dup}
    json.dump(comp, open(outdir / "composition.json", "w"), indent=2, ensure_ascii=False)
    print("composition:", json.dumps(comp, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确保服务器有 `datasets` 库 + 传脚本**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'~/anaconda3/envs/torch/bin/python -c "import datasets" 2>/dev/null && echo HAVE_datasets || ~/anaconda3/envs/torch/bin/pip install -q datasets'
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 'mkdir -p ~/ludan/embeddingprofilter/scripts/v9_boundary && touch ~/ludan/embeddingprofilter/scripts/v9_boundary/__init__.py'
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/v9_boundary/prepare_data.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v9_boundary/prepare_data.py
```
Expected: `HAVE_datasets`（或 pip 成功）；scp 传 1 文件。

- [ ] **Step 3: 服务器跑数据管线（脱离会话，下载 OR-Bench）**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com setsid nohup ~/anaconda3/envs/torch/bin/python -u scripts/v9_boundary/prepare_data.py > /tmp/v9_prep.log 2>&1 < /dev/null & echo PID $!'
# 轮询：
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'while pgrep -f prepare_data.py >/dev/null; do sleep 8; done; tail -8 /tmp/v9_prep.log'
```
Expected: 日志打印各源挖到的数量、`composition`；最终 train.jsonl 的 label==2 数量 ≥ 2000。

- [ ] **Step 4: 验证去污染 + 组成**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && cat datasets/v9_training/composition.json && ~/anaconda3/envs/torch/bin/python -c "
import json
labs={}
for l in open(\"datasets/v9_training/train.jsonl\"): labs[json.loads(l)[\"label\"]]=labs.get(json.loads(l)[\"label\"],0)+1
print(\"train labels:\", labs)
"'
```
Expected: `dropped_jbb_overlap` 已记录（理想 0，若 >0 也已剔除）；train label==2（gray_benign）≥ 2000。

- [ ] **Step 5: 取回 composition.json 到本地 + commit 脚本**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52:~/ludan/embeddingprofilter/datasets/v9_training/composition.json /tmp/v9_composition.json
git add scripts/v9_boundary/__init__.py scripts/v9_boundary/prepare_data.py
git commit -m "feat(v9): data pipeline mining OR-Bench/XSTest boundary-benign (deduped vs JBB-Benign)"
```
（`datasets/v9_training/*.jsonl` 留在服务器，体积小可选 scp 回来一并 commit。）

---

## Task 2: 边界感知损失 loss.py（含单元测试）

**Files:**
- Create: `scripts/v9_boundary/loss.py`
- Create: `scripts/v9_boundary/test_loss.py`

- [ ] **Step 1: 写失败测试 `scripts/v9_boundary/test_loss.py`**

```python
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from loss import boundary_margin_loss

def test_zero_when_no_gray_benign():
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    label4 = torch.tensor([0,1,0,1,0,1,0,1])  # 无 gray_benign(==2)
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() == 0.0

def test_penalizes_gray_benign_near_harmful():
    # 构造：gray_benign 与 harmful 很近、与 benign 很远 -> loss>0
    h = torch.tensor([1.0,0,0,0]); b = torch.tensor([0,1.0,0,0])
    g_bad = h.clone()                      # gray_benign 紧贴 harmful
    z = torch.stack([h, b, g_bad]); z = torch.nn.functional.normalize(z, dim=1)
    label4 = torch.tensor([1,0,2])         # harmful, benign, gray_benign
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() > 0.0

def test_zero_when_gray_benign_safely_on_benign_side():
    h = torch.tensor([1.0,0,0,0]); b = torch.tensor([0,1.0,0,0])
    g_good = b.clone()                     # gray_benign 紧贴 benign
    z = torch.stack([h, b, g_good]); z = torch.nn.functional.normalize(z, dim=1)
    label4 = torch.tensor([1,0,2])
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() == 0.0

if __name__ == "__main__":
    test_zero_when_no_gray_benign()
    test_penalizes_gray_benign_near_harmful()
    test_zero_when_gray_benign_safely_on_benign_side()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败（loss.py 还没写）**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/v9_boundary/test_loss.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v9_boundary/test_loss.py
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python scripts/v9_boundary/test_loss.py 2>&1 | tail -3'
```
Expected: `ModuleNotFoundError: No module named 'loss'` 或 ImportError。

- [ ] **Step 3: 写实现 `scripts/v9_boundary/loss.py`**

```python
#!/usr/bin/env python3
"""边界感知损失：把 gray_benign 从 harmful 一侧推回 benign 一侧。"""
import torch

def boundary_margin_loss(z, label4, margin=0.2):
    """
    z: [B, D] L2-normalized
    label4: [B] 4 类 (0=benign,1=harmful,2=gray_benign,3=gray_harmful)
    对每个 gray_benign 锚点 g:
        sim_h* = max_{harmful h} z_g·z_h    (最难有害负样本)
        sim_cb = z_g · c_b                  (c_b = 批内 benign 类原型, 归一化均值)
        L = relu(margin + sim_h* - sim_cb)
    无 gray_benign 或无 harmful 或无 benign 时返回 0。
    """
    device = z.device
    gb = (label4 == 2)
    harmful = (label4 == 1)
    benign = (label4 == 0)
    if gb.sum() == 0 or harmful.sum() == 0 or benign.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    zg = z[gb]                                   # [G, D]
    zh = z[harmful]                              # [H, D]
    c_b = z[benign].mean(dim=0)
    c_b = c_b / (c_b.norm() + 1e-8)              # benign 原型
    sim_h = (zg @ zh.T).max(dim=1).values        # [G] 最难有害负样本
    sim_cb = zg @ c_b                            # [G]
    loss = torch.relu(margin + sim_h - sim_cb)
    return loss.mean()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/v9_boundary/loss.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v9_boundary/loss.py
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python scripts/v9_boundary/test_loss.py 2>&1 | tail -3'
```
Expected: `ALL PASS`。

- [ ] **Step 5: commit**

```bash
git add scripts/v9_boundary/loss.py scripts/v9_boundary/test_loss.py
git commit -m "feat(v9): boundary-aware hard-negative margin loss + unit tests"
```

---

## Task 3: v9 训练脚本 train.py（复用 v8 + 新数据 + --loss 开关）

**Files:**
- Create: `scripts/v9_boundary/train.py`（基于 `scripts/v8_cs_supcon/train.py`，改 4 处）

- [ ] **Step 1: 本地基于 v8 train.py 复制改造为 v9 train.py**

改动点（其余完全复用 v8）：
1. 顶部加 `import argparse`、`from loss import boundary_margin_loss`、`sys.path` 含本目录。
2. `load_training_data()` 读 `datasets/v9_training/`，同时返回 **4 类标签 label4**（不只二分类）：`labels_binary = 0 if label in [0,2] else 1`；`labels4 = label`。
3. `train_cs_projection(...)` 增参 `loss_mode, lam, margin`；每步：`L = supervised_contrastive_loss(z, bin_batch, temp)`，若 `loss_mode=="boundary"` 再 `L = L + lam * boundary_margin_loss(z, label4_batch, margin)`。需在采样 batch 时同时索引 `label4`。
4. `main()` 加 argparse：`--loss {supcon,boundary}`（默认 supcon）、`--lam`（默认 0.5）、`--margin`（默认 0.2）、`--dims`（默认 "32"，逗号分隔）、`--out`（默认 `models/v9_boundary`）。输出目录、文件名带配置后缀（如 `cs_projection_32d.pt`，及 `training_results.json`）。embedding 缓存复用 v8 的 `models/v8_cs_supcon/cache/`？否——v9 数据不同，缓存到 `models/v9_boundary/cache/v9_embeddings.npz`。

> 注：v8 train.py 的 `extract_embeddings`/`load_v7_encoder` 直接 import 复用：`sys.path.insert(0, BASE/'scripts/v8_cs_supcon'); from train import load_v7_encoder, extract_embeddings`（避免重复编码器代码）。

- [ ] **Step 2: 传脚本并 dry-check import + argparse**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/v9_boundary/train.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v9_boundary/train.py
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_HUB_OFFLINE=1 ~/anaconda3/envs/torch/bin/python scripts/v9_boundary/train.py --help 2>&1 | tail -12'
```
Expected: 打印 argparse 帮助，含 `--loss/--lam/--margin/--dims`，无 import 错误。

- [ ] **Step 3: commit**

```bash
git add scripts/v9_boundary/train.py
git commit -m "feat(v9): training script with --loss {supcon,boundary} on v9_training data"
```

---

## Task 4: A1 跑（只换数据，原 SupCon 目标，32d）

**Files:** 产出 `models/v9_boundary/cs_projection_32d.pt`、`detector_32d_{1,3}c.npz`、`training_results.json`（A1）

- [ ] **Step 1: 服务器训练 A1（脱离会话）**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 setsid nohup ~/anaconda3/envs/torch/bin/python -u scripts/v9_boundary/train.py --loss supcon --dims 32 --out models/v9_boundary > /tmp/v9_a1_train.log 2>&1 < /dev/null & echo PID $!'
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'while pgrep -f "v9_boundary/train.py" >/dev/null; do sleep 10; done; grep -avE "提取 embeddings|it/s\]" /tmp/v9_a1_train.log | tail -15'
```
Expected: 训练完成，产出 `cs_projection_32d.pt` + detectors；打印 32d_1c Val F1/FPR。

- [ ] **Step 2: 验证产物**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ls models/v9_boundary/*.pt models/v9_boundary/detector_32d_1c.npz && ~/anaconda3/envs/torch/bin/python -c "import json;print(json.load(open(\"models/v9_boundary/training_results.json\")).get(\"32d_1c\"))"'
```
Expected: 文件齐全；打印 A1 的 32d_1c 验证指标。

---

## Task 5: 帕累托评测 pareto_eval.py（v8 vs v9-A1）

**Files:**
- Create: `scripts/v9_boundary/pareto_eval.py`
- Create on server: `results/v9_boundary/pareto.json`

- [ ] **Step 1: 本地写 `scripts/v9_boundary/pareto_eval.py`**

复用 S0 `evaluate.py` 的 DATASETS/load_texts/编码器，但：对给定 (投影.pt, detector.npz)，**扫阈值** `np.arange(-0.5,0.51,0.02)`，每个阈值算 (平均攻击 DR, JBB_Benign FPR, Alpaca FPR)，输出曲线点列。对多个模型目录（v8=`models/v8_cs_supcon`、v9-A1=`models/v9_boundary`）各算一条，写 `results/v9_boundary/pareto.json`。要点：
- 攻击集 = {GCG,PAIR,JailbreakHub,AdvBench,HarmBench}（剔除标注不匹配的 ToxicChat/BeaverTails，避免污染曲线；这些单列）。
- 嵌入每数据集只算一次（编码器共享），再对每个 (模型, 阈值) 复用。
- 32d，detector 用 `_32d_1c.npz`。

```python
#!/usr/bin/env python3
"""扫阈值画 DR-FPR 帕累托曲线，对比多个 v8/v9 模型目录。"""
import os, sys, json, random
import numpy as np, pandas as pd, torch
from pathlib import Path
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device
from train import load_v7_encoder, extract_embeddings
sys.path.insert(0, str(BASE / "scripts"))
import importlib.util
# 复用 v8 evaluate 的 DATASETS / load_texts
spec = importlib.util.spec_from_file_location("v8eval", str(BASE/"scripts/v8_cs_supcon/evaluate.py"))
v8eval = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8eval)

random.seed(42)
ATTACK = ["GCG","PAIR","JailbreakHub","AdvBench","HarmBench"]
BENIGN = ["JBB_Benign","Alpaca"]
MODELS = {"v8": BASE/"models/v8_cs_supcon", "v9_a1": BASE/"models/v9_boundary"}
DIM, NC = 32, 1
THRESHOLDS = [round(t,3) for t in np.arange(-0.5,0.51,0.02)]

def main():
    device = get_device()
    encoder, tok = load_v7_encoder(device)
    emb = {}
    for name in ATTACK + BENIGN:
        texts = v8eval.load_texts(v8eval.DATASETS[name])
        emb[name] = extract_embeddings(encoder, tok, texts, device).astype(np.float32)
        print(name, emb[name].shape)
    del encoder
    out = {}
    for mname, mdir in MODELS.items():
        proj = LearnedCSProjection(768, DIM)
        proj.load_state_dict(torch.load(str(mdir/f"cs_projection_{DIM}d.pt"), map_location="cpu")); proj.eval()
        det = DualMultiCentroidDetector.load(str(mdir/f"detector_{DIM}d_{NC}c.npz"))
        z = {n: proj(torch.tensor(e)).detach().numpy() for n,e in emb.items()}
        curve = []
        for t in THRESHOLDS:
            drs = [float((det.predict(z[n], t)[0]==1).mean()) for n in ATTACK]
            jbb = float((det.predict(z["JBB_Benign"], t)[0]==1).mean())
            alp = float((det.predict(z["Alpaca"], t)[0]==1).mean())
            curve.append({"thr": t, "avg_attack_DR": round(sum(drs)/len(drs),4),
                          "JBB_Benign_FPR": round(jbb,4), "Alpaca_FPR": round(alp,4)})
        out[mname] = curve
        print(mname, "done")
    (BASE/"results/v9_boundary").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BASE/"results/v9_boundary/pareto.json","w"), indent=2)
    print("WROTE results/v9_boundary/pareto.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 传 + 跑（脱离会话）**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/v9_boundary/pareto_eval.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v9_boundary/pareto_eval.py
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 setsid nohup ~/anaconda3/envs/torch/bin/python -u scripts/v9_boundary/pareto_eval.py > /tmp/v9_pareto.log 2>&1 < /dev/null & echo PID $!'
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'while pgrep -f pareto_eval.py >/dev/null; do sleep 8; done; grep -avE "提取|it/s\]" /tmp/v9_pareto.log | tail -8'
```
Expected: `WROTE results/v9_boundary/pareto.json`。

- [ ] **Step 3: 取回并比对 A1 是否帕累托占优 v8**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52:~/ludan/embeddingprofilter/results/v9_boundary/pareto.json /tmp/v9_pareto.json
python3 -c "
import json; d=json.load(open('/tmp/v9_pareto.json'))
def at_dr(curve, target=0.95):
    best=min((c for c in curve if c['avg_attack_DR']>=target), key=lambda c:c['JBB_Benign_FPR'], default=None)
    return best
for m in d: 
    b=at_dr(d[m]); print(m, '@DR>=0.95 -> JBB_FPR=', b['JBB_Benign_FPR'] if b else 'n/a', 'thr=', b['thr'] if b else None)
"
git add scripts/v9_boundary/pareto_eval.py
git commit -m "feat(v9): pareto sweep eval (attack DR vs JBB-Benign FPR), v8 vs v9-A1"
```
Expected: 在"平均攻击 DR ≥ 0.95"处，v9_a1 的 JBB_Benign_FPR 明显低于 v8（验证 A1 数据效应）。

---

## Task 6: A2 跑（边界损失，λ/margin 小扫描）+ 帕累托

**Files:** 产出 `models/v9_boundary_a2_<lam>_<margin>/...`，更新 `results/v9_boundary/pareto.json`（加 A2 曲线）

- [ ] **Step 1: 小网格训练 A2（脱离会话，串行 3 组）**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 setsid nohup bash -c "
for lam in 0.5 1.0; do for mg in 0.2 0.3; do
  ~/anaconda3/envs/torch/bin/python -u scripts/v9_boundary/train.py --loss boundary --lam \$lam --margin \$mg --dims 32 --out models/v9_a2_l\${lam}_m\${mg}
done; done
echo A2_GRID_DONE" > /tmp/v9_a2.log 2>&1 < /dev/null & echo PID $!'
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'while ! grep -q A2_GRID_DONE /tmp/v9_a2.log 2>/dev/null; do sleep 15; done; grep -avE "提取|it/s\]" /tmp/v9_a2.log | tail -20'
```
Expected: 4 组 A2 模型目录生成（`models/v9_a2_l0.5_m0.2` 等）。

- [ ] **Step 2: 把 A2 各组加进 pareto_eval 的 MODELS 再跑**

修改 `pareto_eval.py` 的 `MODELS`（或加 `--models` 参数）纳入 4 个 A2 目录，重跑 Step 5 的评测，得到所有曲线。
```bash
# 编辑 MODELS 后重传重跑（同 Task5 Step2），输出更新的 pareto.json
```
Expected: pareto.json 含 v8 / v9_a1 / 4×v9_a2 曲线。

- [ ] **Step 3: 选出最优 A2 并 commit**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52:~/ludan/embeddingprofilter/results/v9_boundary/pareto.json /tmp/v9_pareto.json
python3 -c "
import json; d=json.load(open('/tmp/v9_pareto.json'))
def jbb_at(curve,target=0.95):
    c=[x for x in curve if x['avg_attack_DR']>=target]
    return min((x['JBB_Benign_FPR'] for x in c), default=1.0)
for m in d: print(f'{m:20s} JBB_FPR@DR>=0.95 = {jbb_at(d[m]):.3f}')
"
git add scripts/v9_boundary/pareto_eval.py
git commit -m "feat(v9): A2 boundary-loss grid + pareto comparison vs v8/A1"
```
Expected: 至少一组 A2 在 DR≥0.95 处 JBB_FPR 进一步低于 A1。

---

## Task 7: 报告 results/v9_boundary/S1_REPORT.md

**Files:**
- Create: `results/v9_boundary/S1_REPORT.md`

- [ ] **Step 1: 用 pareto.json + composition.json + 11 数据集 eval 写报告**

内容：(a) 数据组成（gray_benign 400→N，去污染统计）；(b) 帕累托表/曲线：v8 vs A1 vs 最优 A2 在若干 DR 水平下的 JBB_Benign_FPR；(c) 护栏 Alpaca FPR；(d) 11 数据集表（对最优 A2，复用 evaluate.py 思路或扩展）；(e) **数据 vs 目标消融结论**（A1 贡献多少、A2 再贡献多少）；(f) 是否达成"帕累托占优 v8"。

- [ ] **Step 2: commit**

```bash
git add results/v9_boundary/S1_REPORT.md
git commit -m "docs(s1): v9 boundary-aware CS report — pareto-improves v8 on DR-FPR"
```

---

## Task 8: 收尾 + S1→S2 衔接

- [ ] **Step 1: 把最优 v9 配置写入 README/记忆，给出 S2 衔接（基线对比时用 v9 而非 v8）。**
- [ ] **Step 2: 最终 commit；finishing-a-development-branch 决定合并。**

---

## Self-Review（对照 spec）

- ✅ spec §3 A1 数据扩充 → Task 1（OR-Bench/XSTest 挖掘 + 去污染 + v9_training）
- ✅ spec §3 A2 边界感知损失 → Task 2（loss + 测试）+ Task 3（train --loss boundary）+ Task 6（λ/margin 扫描）
- ✅ spec §4 帕累托评测 → Task 5 + Task 6（v8 vs A1 vs A2 扫阈值曲线）
- ✅ spec §5 组件命名（scripts/v9_boundary、models/v9_boundary、datasets/v9_training、results/v9_boundary）→ 各 Task 路径一致
- ✅ spec §8 验收：gray_benign≥2000+零泄漏(Task1)、A1/A2 产物(Task4/6)、帕累托占优(Task5/6)、Alpaca 护栏(Task5)、S1_REPORT(Task7)
- ⚠️ 偏离 spec 的现实修正（已纳入）：OR-Bench/XSTest **不在仓库**（parquet 0 字节 / LFS 指针）→ 改为从 HF 经 hf-mirror 下载（Task 1）。
- 类型一致性：`LearnedCSProjection(768,dim)`、`DualMultiCentroidDetector.load/predict(z,thr)->(*preds*,score)`、`boundary_margin_loss(z,label4,margin)`、训练产物名 `cs_projection_{dim}d.pt` / `detector_{dim}d_{nc}c.npz` 在 Task2/3/4/5/6 一致；label4 语义 (0/1/2/3) 在 prepare_data 与 train 与 loss 一致。
- 连接鲁棒性：所有长任务用 `setsid nohup -u` + 轮询（S0 验证过连接每隔几分钟会掉）。
