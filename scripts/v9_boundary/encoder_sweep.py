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
    return adv + harm, alp, jbb


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
    harm_t, easy_t, bound_t = load()
    yb = np.array([1] * len(harm_t) + [0] * len(bound_t))
    ye = np.array([1] * len(harm_t) + [0] * len(easy_t))
    print(f"harmful={len(harm_t)} easy={len(easy_t)} boundary={len(bound_t)}")
    print(f"\n{'encoder':<26}{'easy(lin)':>10}{'bound(lin)':>11}{'bound(rbf)':>11}")
    v7enc = None
    for name, hf_id, pref in ENCODERS:
        try:
            if hf_id is None:
                if v7enc is None:
                    v7enc = load_v7_encoder(dev)
                m, t = v7enc
                H = extract_embeddings(m, t, harm_t, dev).astype(np.float64)
                Eb = extract_embeddings(m, t, easy_t, dev).astype(np.float64)
                Bb = extract_embeddings(m, t, bound_t, dev).astype(np.float64)
            else:
                H = embed_hf(hf_id, pref, harm_t, dev)
                Eb = embed_hf(hf_id, pref, easy_t, dev)
                Bb = embed_hf(hf_id, pref, bound_t, dev)
            Xe, Xb = np.vstack([H, Eb]), np.vstack([H, Bb])
            print(f"{name:<26}{auc_lin(Xe, ye):>10.4f}{auc_lin(Xb, yb):>11.4f}{auc_svm(Xb, yb):>11.4f}", flush=True)
        except Exception as e:
            print(f"{name:<26}  SKIP: {type(e).__name__}: {str(e)[:60]}", flush=True)
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
