#!/usr/bin/env python3
"""阶段 B：轻量分类器基线。逐个 try/except，单个失败不阻塞其余。"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import run_baseline

JOBS = [
    ("injecguard", "hf_classifiers", "build_injecguard", "184M", "-"),
    ("protectai", "hf_classifiers", "build_protectai", "184M", "-"),
    ("promptguard", "hf_classifiers", "build_promptguard", "86M", "-"),
    ("perplexity_lgb", "perplexity_lgb", "build", "~0", "-"),
]

for name, mod, fn, params, dim in JOBS:
    try:
        print(f"=== {name} ===")
        m = getattr(__import__(mod), fn)()
        run_baseline(name, m.predict_batch, params=params, dim=dim)
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        traceback.print_exc()
print("ALL_PHASEB_DONE")
