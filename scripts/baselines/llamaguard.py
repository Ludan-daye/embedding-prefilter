#!/usr/bin/env python3
"""LlamaGuard via transformers（避开 vllc/transformers 版本冲突）。
apply_chat_template 生成 LlamaGuard 审核提示，模型输出 safe / unsafe\\nSx -> 0/1。
门控下不到则设 LG_MODEL=allenai/wildguard 等开放模型（其模板不同，需另解析）。"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("LG_MODEL", "meta-llama/Llama-Guard-3-8B")
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(_DEV).eval()

    class M:
        params = "8B"; dim = "-"

        def predict_batch(self, texts, bs=8):
            res = []
            for i in range(0, len(texts), bs):
                b = texts[i:i + bs]
                prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False)
                           for t in b]
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=4096).to(_DEV)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                gen = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                res += [1 if "unsafe" in g.lower() else 0 for g in gen]
            return res
    return M()
