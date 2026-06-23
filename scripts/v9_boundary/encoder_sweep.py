#!/usr/bin/env python3
"""0.86 边界上限是任务内禀,还是 v7-finetuned BGE-base 这一个 encoder 的局限?
换若干 off-the-shelf encoder,只测边界判别(有害 vs JBB 边界正常)。
若某个 encoder 显著 > 0.86 -> 上限非内禀,'冲新意'有戏;若都 ~0.86 -> 内禀重叠铁证。
每个 encoder: boundary-AUC(logistic 线性 + rbf-SVM 非线性) + easy-AUC(参考)。逐个加载,用完即释放。"""
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from train import load_v7_encoder, extract_embeddings  # noqa
from model import get_device  # noqa
np.random.seed(42)

# (名字, hf_id, 输入前缀)  —— None 表示用 v7-finetuned 参考
ENCODERS = [
    ("v7-finetuned-bge-base", None, ""),
    ("bge-base (un-tuned)", "BAAI/bge-base-en-v1.5", ""),
    ("bge-large", "BAAI/bge-large-en-v1.5", ""),
    ("e5-large-v2", "intfloat/e5-large-v2", "query: "),
    ("gte-large", "thenlper/gte-large", ""),
]


def load():
    adv = pd.read_csv(BASE / "datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()[:200]
    harm = pd.read_csv(BASE / "datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()[:200]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:400]
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    return adv + harm, alp, jbb, xs


def embed_hf(hf_id, prefix, texts, dev, bs=32):
    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModel.from_pretrained(hf_id).to(dev).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = [prefix + t for t in texts[i:i + bs]]
        x = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.no_grad():
            H = mdl(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        emb = (H * m).sum(1) / m.sum(1).clamp(min=1e-9)   # 统一 mean-pool
        out.append(emb.cpu().numpy())
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


def auc_lin(X, y):
    return cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                           StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean()


def auc_svm(X, y):
    return cross_val_score(SVC(kernel="rbf", class_weight="balanced"),
                           StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean()


def main():
    dev = get_device()
    harm_t, easy_t, jbb_t, xs_t = load()
    y_easy = np.array([1] * len(harm_t) + [0] * len(easy_t))
    y_jbb = np.array([1] * len(harm_t) + [0] * len(jbb_t))
    y_xs = np.array([1] * len(harm_t) + [0] * len(xs_t))
    print(f"harmful={len(harm_t)} easy={len(easy_t)} JBB={len(jbb_t)} XSTest={len(xs_t)}", flush=True)
    print(f"\n{'encoder':<24}{'easy':>8}{'JBB(lin)':>9}{'JBB(rbf)':>9}{'XS(lin)':>9}{'XS(rbf)':>9}", flush=True)
    v7enc = None
    for name, hf_id, pref in ENCODERS:
        try:
            if hf_id is None:
                if v7enc is None:
                    v7enc = load_v7_encoder(dev)
                m, t = v7enc
                emb = lambda txt: extract_embeddings(m, t, txt, dev).astype(np.float64)
            else:
                emb = lambda txt, _id=hf_id, _p=pref: embed_hf(_id, _p, txt, dev)
            H, Eb, Jb, Xb = emb(harm_t), emb(easy_t), emb(jbb_t), emb(xs_t)
            print(f"{name:<24}{auc_lin(np.vstack([H,Eb]),y_easy):>8.3f}"
                  f"{auc_lin(np.vstack([H,Jb]),y_jbb):>9.3f}{auc_svm(np.vstack([H,Jb]),y_jbb):>9.3f}"
                  f"{auc_lin(np.vstack([H,Xb]),y_xs):>9.3f}{auc_svm(np.vstack([H,Xb]),y_xs):>9.3f}", flush=True)
        except Exception as e:
            print(f"{name:<24}  SKIP: {type(e).__name__}: {str(e)[:70]}", flush=True)
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
