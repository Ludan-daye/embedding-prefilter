#!/usr/bin/env python3
"""HF 序列分类器基线：InjecGuard / ProtectAI / PromptGuard。判有害规则各注明。
执行前请验证各模型 config.id2label，确认 harmful 标签 id。"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _SeqClf:
    def __init__(self, model_id, harmful_label_ids, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True).to(_DEV).eval()
        self.harmful = set(harmful_label_ids)
        self.max_len = max_len
        print(f"  {model_id} id2label={self.m.config.id2label}")

    def predict_batch(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i:i + bs]
            x = self.tok(b, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
            x = {k: v.to(_DEV) for k, v in x.items()}
            with torch.no_grad():
                pred = self.m(**x).logits.argmax(-1).cpu().tolist()
            out += [1 if p in self.harmful else 0 for p in pred]
        return out


# InjecGuard: 二分类 [SAFE=0, INJECTION=1] -> harmful={1}
def build_injecguard():
    return _SeqClf("leolee99/InjecGuard", {1})

# ProtectAI v2: [SAFE=0, INJECTION=1] -> harmful={1}
def build_protectai():
    return _SeqClf("protectai/deberta-v3-base-prompt-injection-v2", {1})

# PromptGuard 86M: [BENIGN=0, INJECTION=1, JAILBREAK=2] -> harmful={1,2}（门控）
def build_promptguard():
    return _SeqClf("meta-llama/Prompt-Guard-86M", {1, 2})
