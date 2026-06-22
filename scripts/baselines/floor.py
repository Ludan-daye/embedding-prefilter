#!/usr/bin/env python3
"""Floor / 消融基线：Keyword / TF-IDF+LR / BGE+Cosine / BGE+SVM。
抽取自 scripts/baseline_comparison.py，改：相对路径、用 v9_training 全量、SVM 改 LinearSVC。
全部用 BAAI/bge-base-en-v1.5（与 v9 同编码器），输出 'harmful'/'benign'。"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel

BASE = Path(__file__).parent.parent.parent
_DEV = "cuda" if torch.cuda.is_available() else "cpu"

HARMFUL_KEYWORDS = [
    "bomb", "explosive", "weapon", "kill", "murder", "hack", "steal",
    "drug", "cocaine", "heroin", "meth", "poison", "attack", "terrorist",
    "malware", "virus", "ransomware", "phishing", "exploit", "vulnerability",
    "jailbreak", "bypass", "ignore previous", "dan", "do anything now",
    "pretend", "roleplay", "hypothetical", "illegal", "harmful",
    "suicide", "self-harm", "violence", "abuse", "torture",
]


def load_v9_training():
    texts, labels = [], []
    with open(BASE / "datasets/v9_training/train.jsonl") as f:
        for line in f:
            it = json.loads(line)
            texts.append(it["text"])
            labels.append(1 if it["label"] in [1, 3] else 0)
    return texts, labels


def _bge():
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    m = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(_DEV).eval()
    return tok, m


def _encode(tok, m, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]
        x = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512)
        x = {k: v.to(_DEV) for k, v in x.items()}
        with torch.no_grad():
            o = m(**x).last_hidden_state
            mask = x["attention_mask"].unsqueeze(-1).expand(o.size()).float()
            emb = (o * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(emb.cpu().numpy())
    return np.vstack(out)


class KeywordMatcher:
    def __init__(self): self.kw = [k.lower() for k in HARMFUL_KEYWORDS]
    def predict_batch(self, texts):
        return ["harmful" if any(k in t.lower() for k in self.kw) else "benign" for t in texts]


class TfidfLRClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000)
    def train(self, texts, labels):
        self.clf.fit(self.vec.fit_transform(texts), labels)
    def predict_batch(self, texts):
        return ["harmful" if p == 1 else "benign" for p in self.clf.predict(self.vec.transform(texts))]


class BGECosineClassifier:
    def __init__(self): self.tok, self.m = _bge(); self.hc = None; self.bc = None
    def train(self, texts, labels):
        e = _encode(self.tok, self.m, texts); y = np.array(labels)
        self.hc = e[y == 1].mean(0); self.bc = e[y == 0].mean(0)
    def predict_batch(self, texts):
        e = _encode(self.tok, self.m, texts); out = []
        for v in e:
            hs = v @ self.hc / (np.linalg.norm(v) * np.linalg.norm(self.hc) + 1e-9)
            bs = v @ self.bc / (np.linalg.norm(v) * np.linalg.norm(self.bc) + 1e-9)
            out.append("harmful" if hs > bs else "benign")
        return out


class BGESVMClassifier:
    def __init__(self): self.tok, self.m = _bge(); self.svm = LinearSVC(C=1.0)
    def train(self, texts, labels):
        self.svm.fit(_encode(self.tok, self.m, texts), labels)
    def predict_batch(self, texts):
        return ["harmful" if p == 1 else "benign" for p in self.svm.predict(_encode(self.tok, self.m, texts))]


def build_keyword():
    return KeywordMatcher()

def build_tfidf():
    t, y = load_v9_training(); c = TfidfLRClassifier(); c.train(t, y); return c

def build_bge_cosine():
    t, y = load_v9_training(); c = BGECosineClassifier(); c.train(t, y); return c

def build_bge_svm():
    t, y = load_v9_training(); c = BGESVMClassifier(); print(f"  BGE+SVM 训练 {len(t)} 条..."); c.train(t, y); return c
