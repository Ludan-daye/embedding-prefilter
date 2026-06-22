#!/usr/bin/env python3
"""Perplexity(GPT-2) + LightGBM。特征 [log_ppl, log_len, char_div]。训练 GCG 攻击 + Alpaca。"""
import json, math
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import lightgbm as lgb
import pandas as pd

BASE = Path(__file__).parent.parent.parent
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_tok = GPT2TokenizerFast.from_pretrained("gpt2")
_m = GPT2LMHeadModel.from_pretrained("gpt2").to(_DEV).eval()


def _ppl(t):
    ids = _tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(_DEV)
    if ids.size(1) < 2:
        return 1e4
    with torch.no_grad():
        loss = _m(ids, labels=ids).loss
    return float(torch.exp(loss))


def _feat(t):
    return [math.log(_ppl(t) + 1), math.log(len(t) + 1), len(set(t)) / (len(t) + 1)]


def build():
    gcg = list(pd.read_csv(BASE / "datasets/jailbreakbench/jbb_gcg_all.csv")["prompt"])[:300]
    alp = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:300]
    X = [_feat(t) for t in gcg] + [_feat(t) for t in alp]
    y = [1] * len(gcg) + [0] * len(alp)
    clf = lgb.LGBMClassifier(n_estimators=100).fit(X, y)

    class M:
        params = "~0"; dim = "-"

        def predict_batch(self, texts):
            return [int(p) for p in clf.predict([_feat(t) for t in texts])]
    return M()
