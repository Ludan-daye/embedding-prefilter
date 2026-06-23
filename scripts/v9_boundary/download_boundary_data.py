#!/usr/bin/env python3
"""下载过度拒绝/边界(像攻击的正常)数据集到 datasets/overrefusal/,从 hf-mirror。
多候选源逐个 try,统一存成 jsonl: {text, label, source}。用于贡献② 的跨基准 margin 实验。"""
import os, json, sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset, get_dataset_config_names  # noqa

BASE = Path(__file__).parent.parent.parent
OUT = BASE / "datasets" / "overrefusal"
OUT.mkdir(parents=True, exist_ok=True)

# (保存名, hf_id, config, 候选文本列, 候选标签列)
SOURCES = [
    ("xstest", "walledai/XSTest", None, ["prompt", "text"], ["label", "type"]),
    ("xstest_v2", "natolambert/xstest-v2-copy", None, ["prompt", "text"], ["label", "type"]),
    ("orbench_hard", "bench-llm/or-bench", "or-bench-hard-1k", ["prompt", "text"], ["category"]),
    ("orbench_toxic", "bench-llm/or-bench", "or-bench-toxic", ["prompt", "text"], ["category"]),
    ("phtest", "furonghuang-lab/PHTest", None, ["prompt", "text", "instruction"], ["label", "type"]),
    ("phtest_alt", "JailbreakBench/PHTest", None, ["prompt", "text"], ["label"]),
]


def pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None


def grab(name, hid, cfg, txtc, labc):
    try:
        ds = load_dataset(hid, cfg) if cfg else load_dataset(hid)
    except Exception as e:
        print(f"  [{name}] load 失败: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return 0
    rows = []
    for split in ds:
        d = ds[split]
        tc = pick(d.column_names, txtc)
        lc = pick(d.column_names, labc)
        if tc is None:
            print(f"  [{name}/{split}] 找不到文本列, 有: {d.column_names}", flush=True); continue
        for r in d:
            t = str(r.get(tc, "")).strip()
            if t:
                rows.append({"text": t, "label": (str(r.get(lc, "")) if lc else ""), "source": name, "split": split})
    if rows:
        p = OUT / f"{name}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 标签分布
        from collections import Counter
        labs = Counter(r["label"] for r in rows)
        print(f"  [{name}] OK {len(rows)} 条 -> {p.name}  cols={d.column_names}  labels(top)={dict(list(labs.items())[:6])}", flush=True)
    return len(rows)


def main():
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}  OUT={OUT}", flush=True)
    total = {}
    for name, hid, cfg, txtc, labc in SOURCES:
        print(f"== {name} ({hid}{'/'+cfg if cfg else ''}) ==", flush=True)
        total[name] = grab(name, hid, cfg, txtc, labc)
    print("\n=== 汇总 ===", flush=True)
    for k, v in total.items():
        print(f"  {k}: {v}", flush=True)
    print("DL_DONE", flush=True)


if __name__ == "__main__":
    main()
