#!/usr/bin/env python3
"""阶段 A：v9 + BGE+SVM 消融 + floor 基线，统一协议评测。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import run_baseline
import floor, ours_v9

print("=== v9 (a2_l1m5) ===")
m = ours_v9.build(); run_baseline("ours_v9", m.predict_batch, params="25K proj", dim="32D")
print("=== Keyword ===")
run_baseline("keyword", floor.build_keyword().predict_batch, params="0", dim="-")
print("=== TF-IDF+LR ===")
run_baseline("tfidf_lr", floor.build_tfidf().predict_batch, params="~5K", dim="-")
print("=== BGE+Cosine ===")
run_baseline("bge_cosine", floor.build_bge_cosine().predict_batch, params="109M enc", dim="768D")
print("=== BGE+SVM (消融) ===")
run_baseline("bge_svm", floor.build_bge_svm().predict_batch, params="109M enc", dim="768D")
print("ALL_PHASEA_DONE")
