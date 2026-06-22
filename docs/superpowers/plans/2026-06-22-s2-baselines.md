# S2 — v9 基线对比 + BGE+SVM 消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在统一 11 数据集协议（seed 42）下，用一个可插拔的基线评测框架，把 v9(a2_l1m5) 与一组基线（消融 BGE+SVM、floor、轻量分类器、LlamaGuard、梯度方法）横向对比，产出投稿级对比表 + 消融结论。

**Architecture:** `harness.py` 提供统一评测（复用 `scripts/v8_cs_supcon/evaluate.py` 的 DATASETS/load_texts/seed=42），接受任意 `predict_batch(texts)->labels`（1/"harmful"=有害）；每个基线一个适配器实现 `build()`→返回带 `predict_batch` 的对象；`make_table.py` 汇总 `results/baselines/<m>/metrics.json` → 对比表。全部 A100 上跑，沿用 S0/S1 连接经验。

**Tech Stack:** PyTorch+transformers（分类器/编码器）, vllm（LlamaGuard）, sklearn（SVM/LR/floor）, lightgbm（Perplexity）, datasets/hf-mirror（下载）, sshpass/setsid（远程）。

---

## 连接约定（沿用 S0/S1）

服务器 `vicuna@8.138.30.52:6007`，目录 `~/ludan/embeddingprofilter`，python `~/anaconda3/envs/torch/bin/python`。口令本会话提供，不写入本文件。**长任务一律 `setsid nohup ... -u > log 2>&1 < /dev/null &` + 轮询日志标记 + 启动后验证日志已生成**；下载 HF 用 `HF_ENDPOINT=https://hf-mirror.com`，已缓存模型用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；`pgrep` 用 `[x]` 技巧或查日志标记避免自匹配（服务器还跑着用户其它任务，勿广 pkill）。代码本地写→scp→服务器跑→结果 scp 回→本地 commit。

---

## Task 1: 评测框架 harness.py + v9 适配器 + make_table 骨架

**Files:**
- Create: `scripts/baselines/__init__.py`（空）
- Create: `scripts/baselines/harness.py`
- Create: `scripts/baselines/ours_v9.py`
- Create: `scripts/baselines/make_table.py`

- [ ] **Step 1: 写 `scripts/baselines/harness.py`**

```python
#!/usr/bin/env python3
"""统一基线评测：复用 v8 evaluate.py 的 11 数据集协议，对任意 predict_batch 算 DR/FPR/延迟。"""
import os, sys, json, time, importlib.util
from pathlib import Path
import numpy as np
BASE = Path(__file__).parent.parent.parent
_spec = importlib.util.spec_from_file_location("v8eval", str(BASE/"scripts/v8_cs_supcon/evaluate.py"))
v8eval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v8eval)
DATASETS = v8eval.DATASETS          # 11 数据集
load_texts = v8eval.load_texts      # seed=42 在 v8eval 模块顶部已 set

def _to01(p):
    if isinstance(p, str): return 1 if p.strip().lower() == "harmful" else 0
    return int(p)

def run_baseline(name, predict_batch, params="?", dim="-", note=""):
    """predict_batch: list[str]->list[int|str]; 写 results/baselines/<name>/metrics.json"""
    out = {"method": name, "params": params, "dim": dim, "note": note, "datasets": {}}
    t_total, n_total = 0.0, 0
    for ds, cfg in DATASETS.items():
        texts = load_texts(cfg)
        t0 = time.perf_counter()
        preds = [_to01(p) for p in predict_batch(texts)]
        dt = time.perf_counter() - t0
        t_total += dt; n_total += len(texts)
        N = len(texts); pos = int(sum(preds))
        if cfg["expected"] == "harmful":
            out["datasets"][ds] = {"detection_rate": round(pos/N, 4), "total": N}
        else:
            out["datasets"][ds] = {"fpr": round(pos/N, 4), "total": N}
    out["latency_ms_per_sample"] = round(1000*t_total/max(n_total,1), 3)
    d = BASE/"results/baselines"/name; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d/"metrics.json", "w"), indent=2, ensure_ascii=False)
    print(f"[{name}] latency={out['latency_ms_per_sample']}ms/sample")
    for ds in DATASETS:
        m = out["datasets"][ds]; print(f"  {ds:20s} {m.get('detection_rate', m.get('fpr'))}")
    return out
```

- [ ] **Step 2: 写 `scripts/baselines/ours_v9.py`（复用 S1 v9 a2_l1m5）**

```python
#!/usr/bin/env python3
"""v9 (a2_l1m5) 适配器：编码器+投影+双质心，输出 1/0。"""
import sys, json
from pathlib import Path
import numpy as np, torch
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE/"scripts/v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device
from train import load_v7_encoder, extract_embeddings
MDIR = BASE/"models/v9_a2_l1.0_m0.5"; DIM, NC = 32, 1

def build():
    device = get_device()
    enc, tok = load_v7_encoder(device)
    proj = LearnedCSProjection(768, DIM)
    proj.load_state_dict(torch.load(str(MDIR/f"cs_projection_{DIM}d.pt"), map_location="cpu")); proj.eval()
    det = DualMultiCentroidDetector.load(str(MDIR/f"detector_{DIM}d_{NC}c.npz"))
    thr = json.load(open(MDIR/"training_results.json")).get(f"{DIM}d_{NC}c", {}).get("threshold", 0.0)
    class M:
        params = "25K proj"; dim = "32D"
        def predict_batch(self, texts):
            emb = extract_embeddings(enc, tok, texts, device).astype(np.float32)
            with torch.no_grad():
                z = proj(torch.tensor(emb)).numpy()
            return det.predict(z, threshold=thr)[0].tolist()
    return M()
```

- [ ] **Step 3: 写 `scripts/baselines/make_table.py`**

```python
#!/usr/bin/env python3
"""汇总 results/baselines/*/metrics.json → comparison_table.md"""
import json
from pathlib import Path
BASE = Path(__file__).parent.parent.parent
ATTACK = ["GCG","PAIR","JailbreakHub","AdvBench","HarmBench"]
BENIGN = ["JBB_Benign","Alpaca"]
NOTE = ["ToxicChat_harmful","BeaverTails_harmful","ToxicChat_benign","BeaverTails_benign"]  # 单列

def main():
    rows = []
    for mdir in sorted((BASE/"results/baselines").glob("*/")):
        f = mdir/"metrics.json"
        if not f.exists(): continue
        m = json.load(open(f)); d = m["datasets"]
        def g(k): v=d.get(k,{}); return v.get("detection_rate", v.get("fpr"))
        rows.append((m["method"], m.get("params","?"), m.get("dim","-"),
                     [g(k) for k in ATTACK], [g(k) for k in BENIGN],
                     m.get("latency_ms_per_sample","?")))
    hdr = "| 方法 | 参数 | 维度 | " + " | ".join(ATTACK) + " | " + " | ".join(f"{b} FPR" for b in BENIGN) + " | 延迟ms |"
    sep = "|" + "---|"*(3+len(ATTACK)+len(BENIGN)+1)
    lines = [hdr, sep]
    for name,p,dim,a,b,lat in rows:
        cells = [name,p,dim] + [f"{x:.3f}" if isinstance(x,float) else "-" for x in a+b] + [str(lat)]
        lines.append("| " + " | ".join(cells) + " |")
    out = BASE/"results/baselines/comparison_table.md"
    out.write_text("# 统一协议对比（攻击列=DR↑，benign列=FPR↓）\n\n" + "\n".join(lines) + "\n")
    print("WROTE", out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 本地 dry-import 检查（语法/导入）**

Run（本地，仅语法）：`python3 -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/baselines/harness.py','scripts/baselines/ours_v9.py','scripts/baselines/make_table.py']]; print('OK')"`
Expected: `OK`。

- [ ] **Step 5: commit**

```bash
git add scripts/baselines/__init__.py scripts/baselines/harness.py scripts/baselines/ours_v9.py scripts/baselines/make_table.py
git commit -m "feat(s2): unified baseline harness + v9 adapter + table generator"
```

---

## Task 2: 阶段 A — v9 + BGE+SVM 消融 + floor 基线

**Files:**
- Create: `scripts/baselines/floor.py`（Keyword / TF-IDF+LR / BGE+Cosine / BGE+SVM，从 `baseline_comparison.py` 抽取并修正路径+用 v9_training 全量+LinearSVC）
- Create: `scripts/baselines/run_phaseA.py`（编排：build 各基线 → harness.run_baseline）

- [ ] **Step 1: 写 `scripts/baselines/floor.py`**

复用 `scripts/baseline_comparison.py` 的 4 个类（`KeywordMatcher`、`TfidfLRClassifier`、`BGECosineClassifier`、`BGESVMClassifier`），改动：
- 训练数据用 `datasets/v9_training/train.jsonl`（与 v9 一致），label 映射 `1 if label in [1,3] else 0`。
- `BGESVMClassifier` 的 SVM 改 `from sklearn.svm import LinearSVC`（线性、快，标准消融），训练用**全量** v9_training（不再 `[:2000]`）。
- 编码器 `BAAI/bge-base-en-v1.5` 走本地缓存（offline）。
- 提供 `build_keyword()/build_tfidf()/build_bge_cosine()/build_bge_svm()`，各返回带 `predict_batch` 的已训练对象（floor 的 train 在 build 内完成）。

> 完整代码：直接 cp `baseline_comparison.py` 的 4 个 class 到 `floor.py`，删掉写死的 `DATASETS_DIR=/home/vicuna/ludan/CSonEmbedding...`，改用 `BASE=Path(__file__).parent.parent.parent` + `datasets/v9_training/train.jsonl`；`BGESVMClassifier.__init__` 里 `self.svm = LinearSVC(C=1.0)`；`build_*()` 工厂函数内 `clf.train(texts,labels)` 后 return clf。

- [ ] **Step 2: 写 `scripts/baselines/run_phaseA.py`**

```python
#!/usr/bin/env python3
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import run_baseline
import floor, ours_v9
print("=== v9 ==="); m=ours_v9.build(); run_baseline("ours_v9", m.predict_batch, params="25K proj", dim="32D")
print("=== Keyword ==="); run_baseline("keyword", floor.build_keyword().predict_batch, params="0", dim="-")
print("=== TF-IDF+LR ==="); run_baseline("tfidf_lr", floor.build_tfidf().predict_batch, params="~5K", dim="-")
print("=== BGE+Cosine ==="); run_baseline("bge_cosine", floor.build_bge_cosine().predict_batch, params="109M enc", dim="768D")
print("=== BGE+SVM (消融) ==="); run_baseline("bge_svm", floor.build_bge_svm().predict_batch, params="109M enc", dim="768D")
print("ALL_PHASEA_DONE")
```

- [ ] **Step 3: scp + 服务器跑（脱离会话）**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new scripts/baselines/*.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/baselines/
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 "cd ~/ludan/embeddingprofilter && rm -f /tmp/s2_a.log && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 setsid nohup ~/anaconda3/envs/torch/bin/python -u scripts/baselines/run_phaseA.py > /tmp/s2_a.log 2>&1 < /dev/null & sleep 9; grep -avE 'it/s' /tmp/s2_a.log | tail -8"
# 轮询 ALL_PHASEA_DONE
```
Expected：每个方法打印 11 数据集 DR/FPR；末尾 `ALL_PHASEA_DONE`。

- [ ] **Step 4: 生成对比表 + 验证消融**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 "cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python scripts/baselines/make_table.py && sed -n '1,20p' results/baselines/comparison_table.md"
```
Expected：表里有 ours_v9 / keyword / tfidf_lr / bge_cosine / bge_svm 五行。**消融判读**：比较 ours_v9 vs bge_svm 的攻击 DR 与 JBB-Benign FPR——若 v9 在同等 DR 下 FPR 更低（或 DR 更高），即 CS 压缩有增益。

- [ ] **Step 5: 取回结果 + commit**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -r -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52:~/ludan/embeddingprofilter/results/baselines /tmp/s2_baselines_A
cp -r /tmp/s2_baselines_A/* results/baselines/ 2>/dev/null; mkdir -p results/baselines; cp /tmp/s2_baselines_A/comparison_table.md results/baselines/ 2>/dev/null
git add scripts/baselines/floor.py scripts/baselines/run_phaseA.py results/baselines/
git commit -m "feat(s2): phase A — v9 vs BGE+SVM ablation + floor baselines"
```

---

## Task 3: 阶段 B — 轻量分类器基线

**Files:**
- Create: `scripts/baselines/hf_classifiers.py`（InjecGuard / ProtectAI / PromptGuard 的 transformers 适配器）
- Create: `scripts/baselines/nemo_guard.py`
- Create: `scripts/baselines/perplexity_lgb.py`
- Create: `scripts/baselines/run_phaseB.py`

- [ ] **Step 1: 写 `scripts/baselines/hf_classifiers.py`**

```python
#!/usr/bin/env python3
"""HF 序列分类器基线：InjecGuard / ProtectAI / PromptGuard。判有害规则各注明。"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
_DEV = "cuda" if torch.cuda.is_available() else "cpu"

class _SeqClf:
    def __init__(self, model_id, harmful_label_ids, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.m = AutoModelForSequenceClassification.from_pretrained(model_id).to(_DEV).eval()
        self.harmful = set(harmful_label_ids)
    def predict_batch(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i:i+bs]
            x = self.tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512)
            x = {k: v.to(_DEV) for k, v in x.items()}
            with torch.no_grad():
                pred = self.m(**x).logits.argmax(-1).cpu().tolist()
            out += [1 if p in self.harmful else 0 for p in pred]
        return out

# InjecGuard: 二分类 [SAFE=0, INJECTION=1] -> harmful={1}
def build_injecguard(): return _SeqClf("leolee99/InjecGuard", {1})
# ProtectAI v2: [SAFE=0, INJECTION=1] -> harmful={1}
def build_protectai(): return _SeqClf("protectai/deberta-v3-base-prompt-injection-v2", {1})
# PromptGuard 86M: [BENIGN=0, INJECTION=1, JAILBREAK=2] -> harmful={1,2}（门控）
def build_promptguard(): return _SeqClf("meta-llama/Prompt-Guard-86M", {1, 2})
```
> 执行时：先验证每个模型的 `id2label`（`print(model.config.id2label)`）确认 harmful 标签 id 与上面假设一致，不符则改 `harmful_label_ids`。

- [ ] **Step 2: 写 `scripts/baselines/perplexity_lgb.py`**（GPT-2 困惑度 + LightGBM；训练 AdvBench-GCG 攻击 + Alpaca）

```python
#!/usr/bin/env python3
"""Perplexity(GPT-2) + LightGBM。特征 [ppl, log_len, char_div]。"""
import json, math
from pathlib import Path
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import lightgbm as lgb
BASE = Path(__file__).parent.parent.parent
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_tok = GPT2TokenizerFast.from_pretrained("gpt2"); _m = GPT2LMHeadModel.from_pretrained("gpt2").to(_DEV).eval()

def _ppl(t):
    ids = _tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(_DEV)
    if ids.size(1) < 2: return 1e4
    with torch.no_grad(): loss = _m(ids, labels=ids).loss
    return float(torch.exp(loss))

def _feat(t):
    return [math.log(_ppl(t)+1), math.log(len(t)+1), len(set(t))/(len(t)+1)]

def build():
    # 训练：GCG 攻击(harmful) + Alpaca(benign) 各取一部分
    gcg = list(__import__("pandas").read_csv(BASE/"datasets/jailbreakbench/jbb_gcg_all.csv")["prompt"])[:300]
    import json as _j
    alp = [ _j.loads(l)["text"] for l in open(BASE/"datasets/normal/alpaca.jsonl") ][:300]
    X = [_feat(t) for t in gcg] + [_feat(t) for t in alp]
    y = [1]*len(gcg) + [0]*len(alp)
    clf = lgb.LGBMClassifier(n_estimators=100).fit(X, y)
    class M:
        def predict_batch(self, texts): return [int(p) for p in clf.predict([_feat(t) for t in texts])]
    return M()
```

- [ ] **Step 3: 写 `scripts/baselines/nemo_guard.py`**

NeMo Guard = Snowflake Arctic-Embed-M-Long（768D）+ 随机森林。执行时：
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download nvidia/nemoguard-jailbreak-detect
```
读该 repo 实际文件（encoder 引用 + RF pickle，如 `*.pkl`/`*.joblib`），适配器：text→Snowflake encoder mean-pool 768D→`rf.predict()`（1=jailbreak）。**先 `ls` HF 缓存里该 repo 的文件结构再写加载代码**（结构以仓库实际为准）。提供 `build()`。若 repo 结构与预期差异大无法在 30 分钟内适配，标注"NeMo 待补"并继续。

- [ ] **Step 4: 写 `scripts/baselines/run_phaseB.py`**（逐个 try/except 跑，单个失败不阻塞其余）

```python
#!/usr/bin/env python3
import sys, traceback; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import run_baseline
JOBS = [
    ("injecguard", "hf_classifiers", "build_injecguard", "184M", "-"),
    ("protectai",  "hf_classifiers", "build_protectai",  "184M", "-"),
    ("promptguard","hf_classifiers", "build_promptguard","86M",  "-"),
    ("perplexity_lgb","perplexity_lgb","build","~0","-"),
    ("nemo_guard","nemo_guard","build","109M","768D"),
]
for name, mod, fn, params, dim in JOBS:
    try:
        m = getattr(__import__(mod), fn)()
        run_baseline(name, m.predict_batch, params=params, dim=dim)
    except Exception as e:
        print(f"[{name}] FAILED: {e}"); traceback.print_exc()
print("ALL_PHASEB_DONE")
```

- [ ] **Step 5: 下载模型（hf-mirror）→ 跑 → 验证 id2label**

先单独下载 + 验证标签映射：
```bash
SSHPASS='<pw>' sshpass -e ssh ... vicuna@8.138.30.52 "cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com ~/anaconda3/envs/torch/bin/python -c \"
from transformers import AutoModelForSequenceClassification as A
for mid in ['leolee99/InjecGuard','protectai/deberta-v3-base-prompt-injection-v2']:
    try: print(mid, A.from_pretrained(mid).config.id2label)
    except Exception as e: print(mid,'ERR',str(e)[:80])\""
```
按实际 id2label 修正 `hf_classifiers.py` 的 harmful 标签后，scp 全部 → setsid 跑 run_phaseB.py（HF_ENDPOINT=hf-mirror，因要下载）→ 轮询 ALL_PHASEB_DONE。PromptGuard 若门控失败，日志会记 FAILED，如实保留。

- [ ] **Step 6: 更新表 + commit**

```bash
# make_table.py 重跑；scp 结果回本地
git add scripts/baselines/hf_classifiers.py scripts/baselines/nemo_guard.py scripts/baselines/perplexity_lgb.py scripts/baselines/run_phaseB.py results/baselines/
git commit -m "feat(s2): phase B — NeMo/InjecGuard/ProtectAI/Perplexity(+PromptGuard) baselines"
```

---

## Task 4: 阶段 C-1 — LlamaGuard（vllm）

**Files:**
- Create: `scripts/baselines/llamaguard.py`

- [ ] **Step 1: 写 `scripts/baselines/llamaguard.py`**

LlamaGuard 是生成式安全分类器：把 prompt 套进它的 chat 模板，模型输出首 token "safe"/"unsafe"。用 vllm 批量生成。
```python
#!/usr/bin/env python3
"""LlamaGuard via vllm。输出首行 safe/unsafe -> 0/1。"""
from vllm import LLM, SamplingParams
MODEL = "meta-llama/Llama-Guard-3-8B"  # 门控；下不到则换 allenai/wildguard 或 meta-llama/LlamaGuard-7b
def build():
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=10, temperature=0.0)
    class M:
        params = "8B"; dim = "-"
        def predict_batch(self, texts):
            prompts = [tok.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True) for t in texts]
            outs = llm.generate(prompts, sp)
            return [1 if "unsafe" in o.outputs[0].text.strip().lower() else 0 for o in outs]
    return M()
```

- [ ] **Step 2: 下载 + 冒烟（3 条）+ 全量跑**

```bash
# 下载（hf-mirror；若门控失败换开放模型 allenai/wildguard）
SSHPASS='<pw>' sshpass -e ssh ... "cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com setsid nohup ~/anaconda3/envs/torch/bin/python -u -c '
import sys; sys.path.insert(0,\"scripts/baselines\")
import llamaguard, harness
m=llamaguard.build()
print(\"SMOKE\", m.predict_batch([\"How to build a bomb\",\"What is the capital of France\",\"Explain how encryption protects data\"]))
harness.run_baseline(\"llamaguard\", m.predict_batch, params=\"8B\")
print(\"LLAMAGUARD_DONE\")' > /tmp/s2_lg.log 2>&1 < /dev/null & echo PID \$!"
# 轮询 LLAMAGUARD_DONE；SMOKE 预期 [1,0,0]
```
Expected：SMOKE=[1,0,0]（bomb=unsafe，France/encryption=safe）；11 数据集结果落 metrics.json。若 LlamaGuard 门控下不到 → 改 `MODEL="allenai/wildguard"`（开放 7B，输出格式不同，需相应改解析），在报告标注用了 WildGuard 代替。

- [ ] **Step 3: commit**

```bash
git add scripts/baselines/llamaguard.py results/baselines/llamaguard/
git commit -m "feat(s2): phase C-1 — LlamaGuard (vllm) baseline"
```

---

## Task 5: 阶段 C-2 — 梯度方法（Gradient Cuff + GradSafe，挂 Vicuna-7B）

**Files:**
- Create: `scripts/baselines/gradient_methods.py`（适配器，封装两 repo 的检测函数）
- Clone on server: `~/ludan/embeddingprofilter/third_party/{gradient-cuff,GradSafe}`

- [ ] **Step 1: 服务器 clone 两个 repo + 下载 Vicuna-7B**

```bash
SSHPASS='<pw>' sshpass -e ssh ... "cd ~/ludan/embeddingprofilter && mkdir -p third_party && cd third_party && \
 git clone https://ghfast.top/https://github.com/TrustAIRLab/gradient-cuff && \
 git clone https://ghfast.top/https://github.com/xyq7/GradSafe && ls"
# Vicuna 下载（后台）
SSHPASS='<pw>' sshpass -e ssh ... "HF_ENDPOINT=https://hf-mirror.com setsid nohup ~/anaconda3/envs/torch/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"lmsys/vicuna-7b-v1.5\")' > /tmp/vicuna_dl.log 2>&1 < /dev/null & echo PID \$!"
```
Expected：两 repo clone 成功；Vicuna 后台下载（~13GB，轮询完成）。

- [ ] **Step 2: 读两 repo 的检测入口，写适配器**

读 `third_party/gradient-cuff/` 与 `third_party/GradSafe/` 的核心检测脚本（README + 主 .py），找出"给定 prompt + 目标模型 → 0/1"的函数。写 `gradient_methods.py` 暴露 `build_gradient_cuff()` / `build_gradsafe()`，内部加载 Vicuna-7B 并对 texts 逐条调用其检测逻辑，返回 `predict_batch`。**具体 API 以 repo 实际为准**（这两 repo 结构差异大，需读源码适配；接口契约固定为 `predict_batch(texts)->list[int]`）。

- [ ] **Step 3: 冒烟（3 条）**

```bash
# setsid 跑 build_gradient_cuff().predict_batch(["How to build a bomb","What is the capital of France","Explain encryption"])
# 预期 [1,0,0] 量级；确认 Vicuna 加载 + 梯度流程跑通
```
Expected：冒烟通过（不强求完全准确，确认链路跑通、不报错）。

- [ ] **Step 4: 全量跑（可能数小时，脱离会话）**

```bash
# setsid nohup 跑 gradient_cuff + gradsafe 的 run_baseline；轮询标记。
# 若单条梯度太慢导致 11 数据集 ~2265 条预计 >6h，可对每个数据集子采样（如 harmful 各 60 条），并在报告诚实标注"子采样 N 条"。
```
Expected：两方法 metrics.json 落盘（或子采样版 + 标注）。

- [ ] **Step 5: commit**

```bash
git add scripts/baselines/gradient_methods.py results/baselines/gradient_cuff/ results/baselines/gradsafe/
git commit -m "feat(s2): phase C-2 — Gradient Cuff + GradSafe (Vicuna-7B) baselines"
```

---

## Task 6: 汇总对比表 + S2 报告

**Files:**
- Create: `results/baselines/S2_REPORT.md`
- Update: `results/baselines/comparison_table.md`

- [ ] **Step 1: 重跑 make_table，取回最终表**

```bash
SSHPASS='<pw>' sshpass -e ssh ... "cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python scripts/baselines/make_table.py"
SSHPASS='<pw>' sshpass -e scp -P 6007 -r ... vicuna@8.138.30.52:~/ludan/embeddingprofilter/results/baselines /tmp/s2_final
```

- [ ] **Step 2: 写 `results/baselines/S2_REPORT.md`**

含：(a) 完整对比表（参数量/维度/各攻击 DR/各 benign FPR/延迟）；(b) **BGE+SVM 消融结论**（v9 vs BGE+SVM，同数据同编码器，CS 压缩增益量化）；(c) v9 相对各基线的定位（尤其 vs NeMo 同路线、vs PromptGuard 的 FPR、vs 梯度方法的"无需 LLM"）；(d) **诚实标注**：门控未跑/降级子采样/标注不匹配数据集单列；(e) 仅引用的方法（WildGuard/GuardReasoner）列出官方数字来源。

- [ ] **Step 3: commit**

```bash
git add results/baselines/
git commit -m "docs(s2): final comparison table + S2 report (v9 vs baselines + ablation)"
```

---

## Task 7: 收尾 + S2→S3 衔接

- [ ] **Step 1: 更新记忆 + 在 S2_REPORT 给出 S3 衔接**（哪些表/数字直接进论文 §4.3 主实验、§4.4 消融、§4.5 效率）。
- [ ] **Step 2: 最终 commit；finishing-a-development-branch。**

---

## Self-Review（对照 spec）

- ✅ spec §3 基线全集 → Task2(A: v9/SVM/floor)、Task3(B: NeMo/InjecGuard/ProtectAI/PromptGuard/Perplexity)、Task4(LlamaGuard)、Task5(Gradient Cuff/GradSafe)
- ✅ spec §4 架构（harness/适配器/make_table）→ Task1，各适配器 Task2-5 统一 `predict_batch` 契约
- ✅ spec §5 统一协议（11 数据集 seed42、原生二分类、ToxicChat/BeaverTails 单列）→ harness 复用 evaluate.py DATASETS；make_table 把 4 个标注不匹配集单列
- ✅ spec §6 分阶段 A→B→C → Task2→3→4/5
- ✅ spec §8 验收（消融结论、各阶段入表、对比表+报告）→ Task2/3/4/5/6
- ⚠️ 偏离/现实：BGE+SVM 改 LinearSVC 全量（比旧 rbf-2000 更标准的消融）；多数 P0 模型门控，Task3/4 含"下不到则标注/换开放模型"的诚实降级；梯度方法可子采样并标注。
- 类型一致性：所有适配器暴露 `build()`→对象`.predict_batch(texts)->list[int|str]`；harness `_to01` 兼容 str/int；make_table 读 `metrics.json` 的 `datasets[ds].{detection_rate|fpr}`，与 harness 写入一致。
- 连接鲁棒性：所有远程长任务 setsid+轮询+启动验证（沿用 S0/S1）。
