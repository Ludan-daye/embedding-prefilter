# S0 — v8 服务器落地与复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 A100 服务器上重训并端到端复现 v8 CS-SupCon（`32d_1c`）在 11 个数据集上的检测率/FPR，产出「声称 vs 实测」复现报告，并把 `detect.py` 修成可用真实 v8 模型 + 实测延迟。

**Architecture:** 本地仓库（`/root/ludandaye/reaserch/embedding-prefiter`，分支 `improve/s0-server-reproduction`）是开发与真源；所有计算在远程 A100（`~/ludan/embeddingprofilter`）上跑。开发循环：本地用编辑器写代码 → `rsync` 同步到服务器 → SSH 跑 → 结果 `scp` 回本地 → 本地提交。v8 权重 HF 上没有，必须重训；v7 微调编码器 `best_model.pt` 从 hf-mirror 下载（v8 的 embedding 提取与评估都依赖它）。eval 脚本仓库里缺失，需按 `test_comprehensive_v7.py` 方法学重建。

**Tech Stack:** Python3, PyTorch 2.5.1+cu121 (A100), transformers 4.46 (AutoModel, 用于 v7 微调 bge-base 编码器), scikit-learn (KMeans/f1), numpy/pandas, huggingface_hub (走 hf-mirror.com), sshpass/rsync。

---

## 连接与同步约定（每个 Bash 调用自包含）

服务器：`vicuna@8.138.30.52`，端口 `6007`，工作目录 `~/ludan/embeddingprofilter`。
口令在本会话中由用户提供（**不写入本文件**）。每条命令前置 `SSHPASS='<会话口令>'` 并用 `sshpass -e`。

- 跑远程命令：
  `SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 '<cmd>'`
- 本地→服务器同步仓库：
  `SSHPASS='<pw>' sshpass -e rsync -az --delete --exclude '.git' -e "ssh -p 6007 -o StrictHostKeyChecking=accept-new" /root/ludandaye/reaserch/embedding-prefiter/ vicuna@8.138.30.52:~/ludan/embeddingprofilter/`
- 服务器→本地取结果（单文件）：
  `SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52:~/ludan/embeddingprofilter/<path> <local-path>`

环境约定（每条 python 命令前置）：`HF_ENDPOINT=https://hf-mirror.com`（HF 直连不通），`OMP_NUM_THREADS=1`，conda 用 `~/anaconda3/envs/torch/bin/python`。

---

## Task 1: 同步仓库到服务器 + 准备依赖

**Files:**
- 无新增；在服务器 `~/ludan/embeddingprofilter/` 落地本地仓库内容。

- [ ] **Step 1: 同步本地仓库到服务器**

```bash
SSHPASS='<pw>' sshpass -e rsync -az --delete --exclude '.git' --exclude '__pycache__' \
  -e "ssh -p 6007 -o StrictHostKeyChecking=accept-new" \
  /root/ludandaye/reaserch/embedding-prefiter/ \
  vicuna@8.138.30.52:~/ludan/embeddingprofilter/
```
Expected: rsync 传输若干文件，无 error；结束码 0。

- [ ] **Step 2: 验证关键依赖在 `torch` env 可用**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'~/anaconda3/envs/torch/bin/python -c "import torch,transformers,sklearn,numpy,pandas; print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"tf\",transformers.__version__,\"sk\",sklearn.__version__)"'
```
Expected: `torch 2.5.1+cu121 cuda True tf 4.46.x sk 1.5.1`（pandas/numpy 不报错）。若 pandas/tqdm 缺失 → `~/anaconda3/envs/torch/bin/pip install pandas tqdm`。

- [ ] **Step 3: 安装 huggingface_hub（拉权重用）**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'~/anaconda3/envs/torch/bin/pip install -q "huggingface_hub>=0.23" && ~/anaconda3/envs/torch/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"'
```
Expected: 打印版本号（如 `0.2x.x`），无 error。

（本任务无代码改动，不 commit。）

---

## Task 2: 下载 v7 微调编码器 best_model.pt（hf-mirror）

v8 的 embedding 提取与评估都依赖 `models/v7_classifier/best_model.pt`（786MB，仓库 gitignore 了，但 HF 上有）。

**Files:**
- Create on server: `~/ludan/embeddingprofilter/models/v7_classifier/best_model.pt`

- [ ] **Step 1: 从 hf-mirror 下载 best_model.pt 到指定路径**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com ~/anaconda3/envs/torch/bin/python -c "
from huggingface_hub import hf_hub_download
p=hf_hub_download(repo_id=\"ludandaye/embedding-prefilter\", filename=\"models/v7_classifier/best_model.pt\", local_dir=\".\")
print(\"downloaded:\", p)
"'
```
Expected: 打印 `downloaded: .../models/v7_classifier/best_model.pt`；下载约 786MB（几十秒~分钟级）。

- [ ] **Step 2: 验证文件大小与可加载性**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ls -lh models/v7_classifier/best_model.pt && ~/anaconda3/envs/torch/bin/python -c "
import torch; sd=torch.load(\"models/v7_classifier/best_model.pt\", map_location=\"cpu\"); 
print(\"type:\", type(sd)); 
ks=list(sd.keys()) if hasattr(sd,\"keys\") else []
print(\"num keys:\", len(ks)); print(\"sample keys:\", ks[:5])
"'
```
Expected: 文件 ~786M；打印 state_dict 的 key 数量与若干 key 名（含 encoder 与 projection 层）。

（无本地代码改动，不 commit。）

---

## Task 3: A100 上重训 v8（产出 projection + detectors）

**Files:**
- Create on server: `models/v8_cs_supcon/cs_projection_{8,16,32,64,128}d.pt`, `detector_{dim}d_{nc}c.npz`（15 个）, 覆盖写 `training_results.json`。

- [ ] **Step 1: 先 dry-check —— train.py 能 import 且找到 v7 编码器**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python -c "
import json; c=json.load(open(\"models/v7_classifier/config.json\")); print(\"encoder:\",c.get(\"model_name\"),\"proj:\",c.get(\"projection_dim\"))
import os; print(\"best_model.pt exists:\", os.path.exists(\"models/v7_classifier/best_model.pt\"))
print(\"train data:\", os.path.exists(\"datasets/v7_training/train.jsonl\"), os.path.exists(\"datasets/v7_training/val.jsonl\"))
"'
```
Expected: `encoder: BAAI/bge-base-en-v1.5 proj: 128`，`best_model.pt exists: True`，两个训练文件 True。

- [ ] **Step 2: 跑训练（后台，存日志）**

> 注：`train.py` 会先用 v7 编码器（A100 加速）对 5643 条样本抽 768d embedding，再在 CPU 上训 5 个维度的投影（~25K 参数）+ KMeans 检测器 + 阈值扫描。整体分钟级。首次会从 hf-mirror 拉 `BAAI/bge-base-en-v1.5` 的 tokenizer/config。

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
 nohup ~/anaconda3/envs/torch/bin/python scripts/v8_cs_supcon/train.py > /tmp/v8_train.log 2>&1 &
 echo "PID $!"'
```
Expected: 打印 `PID <n>`。

- [ ] **Step 3: 轮询直到训练结束**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'tail -n 15 /tmp/v8_train.log; echo "---"; ls -1 ~/ludan/embeddingprofilter/models/v8_cs_supcon/*.pt 2>/dev/null | wc -l'
```
Expected：日志显示训练完成；`*.pt` 计数变为 `5`（8/16/32/64/128d）。若未完成则隔一会儿重跑此 Step。

- [ ] **Step 4: 验证产物齐全 + 与已提交的 training_results 粗对比**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && echo "pt:" && ls models/v8_cs_supcon/*.pt && echo "npz count:" && ls models/v8_cs_supcon/detector_*.npz | wc -l && \
~/anaconda3/envs/torch/bin/python -c "
import json; r=json.load(open(\"models/v8_cs_supcon/training_results.json\"))
k=\"32d_1c\" if \"32d_1c\" in r else list(r)[0]
print(\"retrained 32d_1c:\", r.get(\"32d_1c\", r.get(k)))
"'
```
Expected：5 个 `.pt`、15 个 `.npz`；打印 32d_1c 的 `{threshold, val_f1, val_fpr}`，`val_f1` 应在 ~0.95 附近（与原 `training_results.json` 同量级；KMeans/训练随机性可有小差异）。

（产物是服务器端二进制，不进 git；不 commit。）

---

## Task 4: 重建 v8 评估脚本 `evaluate.py`

仓库无 v8 eval。基于 `scripts/test_comprehensive_v7.py` 的方法学（同款 DATASETS / 抽样 / 指标公式）重建，把分类器换成 v8 投影 + 双质心检测器。

**Files:**
- Read: `scripts/v8_cs_supcon/train.py`（复用其「加载 v7 编码器 + `.encode()` 抽 768d」代码块）
- Read: `scripts/test_comprehensive_v7.py`（复用 DATASETS 字典、`random.seed(42)`+`random.sample`、指标公式）
- Read: `scripts/v8_cs_supcon/model.py`（`LearnedCSProjection`, `DualMultiCentroidDetector`）
- Create: `scripts/v8_cs_supcon/evaluate.py`

- [ ] **Step 1: 本地读取三处源码，确认编码器加载与 DATASETS 配置的确切写法**

Run（本地）：
```bash
sed -n '60,140p' scripts/v8_cs_supcon/train.py        # 编码器加载 + encode
grep -n "DATASETS\|random.sample\|detection_rate\|fpr\|label_col\|text_col" scripts/test_comprehensive_v7.py | head -40
```
Expected：拿到 `V6HarmfulDetector` 构造与 `encode()` 调用方式、`DATASETS` 各项 (path,text_col,filter,n)。

- [ ] **Step 2: 本地写 `scripts/v8_cs_supcon/evaluate.py`**

要点（按下列结构组装；编码器加载与 DATASETS 直接复用 Step 1 读到的真实写法）：

```python
#!/usr/bin/env python3
"""复现 v8 CS-SupCon 在 11 个数据集上的 DR/FPR。重建自 test_comprehensive_v7.py 方法学。"""
import os, sys, json, random
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'v7_classifier'))
from model import LearnedCSProjection, DualMultiCentroidDetector   # scripts/v8_cs_supcon/model.py
# 复用 train.py 里加载 v7 微调编码器的同款代码 → 得到对象 encoder，提供 encoder.encode(list[str])->np.ndarray[N,768]

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
random.seed(42)

# 直接照搬 test_comprehensive_v7.py 的 DATASETS（path/text_col/label_col/label_val/n/kind∈{harmful,benign}）
DATASETS = {
  'GCG':                {'path':'datasets/jailbreakbench/jbb_gcg_all.csv','text':'prompt','n':100,'kind':'harmful'},
  'PAIR':               {'path':'datasets/jailbreakbench/jbb_pair_all.csv','text':'prompt','n':86,'kind':'harmful'},
  'JailbreakHub':       {'path':'datasets/jailbreakhub/jailbreakhub.csv','text':'prompt','n':79,'kind':'harmful'},
  'AdvBench':           {'path':'datasets/advbench/advbench_harmful_behaviors.csv','text':'goal','n':200,'kind':'harmful'},
  'HarmBench':          {'path':'datasets/harmbench/harmbench_behaviors.csv','text':'Behavior','n':200,'kind':'harmful'},
  'ToxicChat_harmful':  {'path':'datasets/gcg_attacks/toxic_chat_full.csv','text':'user_input','filter':('human_annotation','true'),'n':300,'kind':'harmful'},
  'BeaverTails_harmful':{'path':'datasets/beavertails/beavertails_test.csv','text':'prompt','filter':('is_safe','false'),'n':300,'kind':'harmful'},
  'JBB_Benign':         {'path':'datasets/gcg_attacks/jbb_benign_behaviors.csv','text':'Goal','n':100,'kind':'benign'},
  'Alpaca':             {'path':'datasets/normal/alpaca.jsonl','text':'text','n':200,'kind':'benign'},
  'ToxicChat_benign':   {'path':'datasets/gcg_attacks/toxic_chat_full.csv','text':'user_input','filter':('human_annotation','false'),'n':300,'kind':'benign'},
  'BeaverTails_benign': {'path':'datasets/beavertails/beavertails_test.csv','text':'prompt','filter':('is_safe','true'),'n':300,'kind':'benign'},
}

def load_texts(cfg):
    p = os.path.join(ROOT, cfg['path'])
    if p.endswith('.jsonl'):
        rows=[json.loads(l) for l in open(p) if l.strip()]; df=pd.DataFrame(rows)
    else:
        df=pd.read_csv(p)
    if 'filter' in cfg:
        col,val=cfg['filter']; df=df[df[col].astype(str).str.strip().str.lower()==str(val).strip().lower()]
    texts=df[cfg['text']].astype(str).tolist()
    if len(texts)>cfg['n']: texts=random.sample(texts, cfg['n'])
    return texts

def main():
    CONFIGS=[('8d_1c',8,1),('16d_1c',16,1),('32d_1c',32,1),('32d_3c',32,3),('64d_1c',64,1),('128d_1c',128,1)]
    tr=json.load(open(os.path.join(ROOT,'models/v8_cs_supcon/training_results.json')))
    # encoder = <复用 train.py 加载 v7 编码器的代码>
    results={}
    for name,dim,nc in CONFIGS:
        proj=LearnedCSProjection(768,dim); proj.load_state_dict(torch.load(os.path.join(ROOT,f'models/v8_cs_supcon/cs_projection_{dim}d.pt'),map_location='cpu')); proj.eval()
        det=DualMultiCentroidDetector.load(os.path.join(ROOT,f'models/v8_cs_supcon/detector_{dim}d_{nc}c.npz'))
        thr=tr[name]['threshold']
        res={}
        for ds,cfg in DATASETS.items():
            texts=load_texts(cfg)
            emb=encoder.encode(texts)                                  # [N,768] np
            with torch.no_grad():
                z=proj(torch.tensor(np.asarray(emb),dtype=torch.float32)).numpy()  # [N,dim] L2-normed
            pred=det.predict(z, threshold=thr)                          # True=harmful
            N=len(texts)
            if cfg['kind']=='harmful':
                dr=float((pred==True).sum())/N
                res[ds]={'detection_rate':round(dr,4),'asr':round(1-dr,4),'total':N}
            else:
                fpr=float((pred==True).sum())/N
                res[ds]={'fpr':round(fpr,4),'total':N}
        results[name]=res
        print(name,'done')
    json.dump(results, open(os.path.join(ROOT,'models/v8_cs_supcon/eval_results.json'),'w'), indent=2)
    print('WROTE eval_results.json')

if __name__=='__main__':
    main()
```

> 用 Step 1 读到的**真实**编码器加载代码替换 `# encoder = ...` 行；若 `det.predict` 返回值类型/维度与上面假设不符，按 `model.py` 的真实签名调整（report 已确认 `predict(embeddings, threshold)` 返回布尔数组，`score=max_harmful_sim-max_benign_sim`）。

- [ ] **Step 3: 同步到服务器**

```bash
SSHPASS='<pw>' sshpass -e rsync -az -e "ssh -p 6007 -o StrictHostKeyChecking=accept-new" \
  scripts/v8_cs_supcon/evaluate.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/scripts/v8_cs_supcon/evaluate.py
```
Expected: 传输 1 个文件。

- [ ] **Step 4: 烟雾测试——只跑 2 个数据集确认能加载/打分**

在服务器临时把 DATASETS 缩到 `AdvBench`+`JBB_Benign` 跑一遍（或加 `--smoke` 支持），验证编码器加载、投影、打分、指标计算全链路无异常，并打印这两个数的 DR/FPR。
```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 \
 ~/anaconda3/envs/torch/bin/python - <<PY
import scripts.v8_cs_supcon.evaluate as E
E.DATASETS={k:E.DATASETS[k] for k in ["AdvBench","JBB_Benign"]}
E.main()
PY'
```
Expected: 打印 `32d_1c done` 等；AdvBench DR 接近 ~0.97，JBB_Benign FPR 接近 ~0.67（粗略即可，证明链路正确）。

- [ ] **Step 5: 本地 commit 脚本**

```bash
git add scripts/v8_cs_supcon/evaluate.py
git commit -m "feat(v8): reconstruct evaluate.py for 11-dataset reproduction"
```

---

## Task 5: 运行完整 v8 评估并做复现核对

**Files:**
- Create on server: `models/v8_cs_supcon/eval_results.json`（覆盖）
- Reference (committed target): 仓库内原 `models/v8_cs_supcon/eval_results.json`（已先备份）

- [ ] **Step 1: 备份仓库自带的 eval_results.json 作为复现目标**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && cp models/v8_cs_supcon/eval_results.json /tmp/eval_results_committed.json && echo backed-up'
```
Expected: `backed-up`。

- [ ] **Step 2: 跑完整 11 数据集 × 6 配置评估**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 \
 ~/anaconda3/envs/torch/bin/python scripts/v8_cs_supcon/evaluate.py 2>&1 | tail -20'
```
Expected: 6 个配置各 `done`，末尾 `WROTE eval_results.json`。

- [ ] **Step 3: 自动比对 实测 vs 声称（32d_1c 关键项）**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && ~/anaconda3/envs/torch/bin/python -c "
import json
new=json.load(open(\"models/v8_cs_supcon/eval_results.json\"))[\"32d_1c\"]
old=json.load(open(\"/tmp/eval_results_committed.json\"))[\"32d_1c\"]
def g(d,k): v=d[k]; return v.get(\"detection_rate\",v.get(\"fpr\"))
for k in old:
    o,n=g(old,k),g(new,k); print(f\"{k:22s} claimed={o:.4f} measured={n:.4f} diff={n-o:+.4f}\")
"'
```
Expected: 各项 `diff` 在 ±0.03 内算复现成功；**JBB_Benign 与 BeaverTails_benign 的 measured FPR 应分别 ≈0.67、≈0.87**（核验弱点属实）。差异大者记录到报告。

---

## Task 6: 写复现报告 REPRODUCTION_S0.md

**Files:**
- Create: `results/REPRODUCTION_S0.md`

- [ ] **Step 1: 取回两份 eval json 到本地**

```bash
SSHPASS='<pw>' sshpass -e scp -P 6007 -o StrictHostKeyChecking=accept-new \
  vicuna@8.138.30.52:~/ludan/embeddingprofilter/models/v8_cs_supcon/eval_results.json /tmp/eval_new.json
cp models/v8_cs_supcon/eval_results.json /tmp/eval_old.json
```
Expected: 两文件就位。

- [ ] **Step 2: 本地写报告**（含：环境与权重来源「重训」声明；32d_1c 全 11 项 claimed vs measured 表；6 配置概览；明确结论：JBB-Benign/BeaverTails-benign 高 FPR 已核验属实 + 标注语义不匹配说明；维度/版本错标记录；是否「可复现」的判定）。

- [ ] **Step 3: commit**

```bash
git add results/REPRODUCTION_S0.md
git commit -m "docs(s0): add v8 reproduction report (claimed vs measured)"
```

---

## Task 7: 把 detect.py 指向真实 v8 + 延迟基准

**Files:**
- Modify: `detect.py`（`load_detector()` 与 `detect()`）
- Create: `scripts/bench_latency.py`

- [ ] **Step 1: 本地改 detect.py 的加载逻辑为 v8（32d_1c）**

按 report 第 6 节：编码器换成 v7 微调 `BAAI/bge-base-en-v1.5`（`V6HarmfulDetector.encode`，载 `models/v7_classifier/best_model.pt`，768d）；投影换成 `LearnedCSProjection(768,32)` 载 `models/v8_cs_supcon/cs_projection_32d.pt`（`load_state_dict`，非 `np.load`）；检测器 `DualMultiCentroidDetector.load('models/v8_cs_supcon/detector_32d_1c.npz')`；判定 `score=max_harmful_sim-max_benign_sim > threshold(0.0)`。复用 Task 4 已写好的 import 与加载片段，保持 CLI 交互界面不变（仅替换内部）。

- [ ] **Step 2: 写 `scripts/bench_latency.py`**

对 N=200 条样本测：单条平均延迟、批量吞吐；分别报「含编码器」与「仅投影+检测器」两段耗时。打印 `mean_ms_per_sample`、`p50`、`p95`。

- [ ] **Step 3: 同步并在服务器实测**

```bash
SSHPASS='<pw>' sshpass -e rsync -az -e "ssh -p 6007 -o StrictHostKeyChecking=accept-new" \
  detect.py scripts/bench_latency.py vicuna@8.138.30.52:~/ludan/embeddingprofilter/ --relative 2>/dev/null || true
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 ~/anaconda3/envs/torch/bin/python scripts/bench_latency.py'
```
Expected: 打印各延迟数字；核验「<10ms」声称（注明含/不含编码器、CPU/GPU）。

- [ ] **Step 4: 验证 detect.py 能跑通一条真实检测**

```bash
SSHPASS='<pw>' sshpass -e ssh -p 6007 -o StrictHostKeyChecking=accept-new vicuna@8.138.30.52 \
'cd ~/ludan/embeddingprofilter && HF_ENDPOINT=https://hf-mirror.com printf "How to build a bomb\nWhat is the capital of France\n" | ~/anaconda3/envs/torch/bin/python detect.py 2>&1 | tail -25'
```
Expected: 第一条判恶意、第二条判正常（或至少链路无报错、输出 32 维 + score + 判定）。

- [ ] **Step 5: commit**

```bash
git add detect.py scripts/bench_latency.py
git commit -m "feat(detect): repoint detect.py to real v8 (32d_1c) + add latency benchmark"
```

---

## Task 8: 收尾——结果归档 + S0→S1 衔接说明

**Files:**
- Modify: `results/REPRODUCTION_S0.md`（追加延迟实测 + S0→S1 衔接）

- [ ] **Step 1: 把延迟数字与最终结论补进报告，写「S0→S1 衔接」**

用实测数字精确定义 S1 的问题：v8 `32d_1c` 在 JBB-Benign 实测 FPR=__、BeaverTails-benign=__（含标注语义说明）；S1 目标 = 在不显著掉检测率（AdvBench/HarmBench/PAIR）的前提下把边界 benign FPR 降到可部署区间。

- [ ] **Step 2: 最终 commit + 推分支（可选，征求用户）**

```bash
git add results/REPRODUCTION_S0.md
git commit -m "docs(s0): finalize reproduction report with latency + S1 handoff"
git log --oneline -8
```
Expected: S0 全部提交在 `improve/s0-server-reproduction` 分支；是否 push 到 GitHub 由用户决定。

---

## Self-Review（对照 spec）

- ✅ spec §4 工作区/环境 → Task 1
- ✅ spec §5 权重策略（HF 无 v8 → 重训；需先下 v7 编码器）→ Task 2+3
- ✅ spec §6 Phase C 复现 11 数据集 → Task 4+5（含重建缺失的 evaluate.py）
- ✅ spec §6 Phase D 复现报告 → Task 6+8
- ✅ spec §6 Phase E detect.py + 延迟 → Task 7
- ✅ spec §7 验收：import/GPU(Task1)、v8 加载预测(Task4-5)、容差核对(Task5)、JBB/BeaverTails FPR 实测(Task5)、报告(Task6)、detect 跑通(Task7)
- ⚠️ 偏离 spec 的新发现（已纳入计划，非占位符）：(a) v8 eval 脚本仓库缺失 → 需重建（Task4）；(b) HF 无 v8 权重但有 v7 编码器 → 必须重训且先下 v7 编码器（Task2-3）；(c) HF 直连不通 → 全程 hf-mirror。
- 类型一致性：`LearnedCSProjection(768,dim)` / `DualMultiCentroidDetector.load(npz)` / `det.predict(z, threshold)` 在 Task4/7 用法一致；threshold 一律取自 `training_results.json`（npz 不含阈值）。
