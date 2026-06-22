#!/usr/bin/env python3
"""构建 v9 训练集：v7 原数据 + 从 OR-Bench/XSTest 挖的边界正常(gray_benign)，去污染 vs JBB-Benign。"""
import os, sys, json, re, random
from pathlib import Path
BASE = Path(__file__).parent.parent.parent
random.seed(42)


def norm(t):  # 归一化用于精确去重
    return re.sub(r"\s+", " ", str(t).strip().lower())


def load_jbb_benign_norms():
    import pandas as pd
    p = BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv"
    df = pd.read_csv(p)
    col = "Goal" if "Goal" in df.columns else df.columns[0]
    return set(norm(x) for x in df[col].dropna())


def load_v7_rows():
    rows = []
    for split in ["train", "val"]:
        for line in open(BASE / f"datasets/v7_training/{split}.jsonl"):
            it = json.loads(line); it["_split"] = split; rows.append(it)
    return rows


def mine_or_bench(n_hard=1000, n_80k=3000):
    """OR-Bench: bench-llm/or-bench, 列 prompt + category（过度拒绝=看着危险其实安全）。"""
    from datasets import load_dataset
    texts = []
    for cfg, n in [("or-bench-hard-1k", n_hard), ("or-bench-80k", n_80k)]:
        try:
            ds = load_dataset("bench-llm/or-bench", cfg)
            s = ds[list(ds.keys())[0]]
            assert "prompt" in s.column_names, f"unexpected cols {s.column_names}"
            ptxt = list(s["prompt"])
            if len(ptxt) > n:
                ptxt = random.sample(ptxt, n)
            texts += ptxt
            print(f"  OR-Bench/{cfg}: +{len(ptxt)}")
        except Exception as e:
            print(f"  OR-Bench/{cfg} FAILED: {e}")
    return texts


def mine_xstest():
    """XSTest-safe（可选）。失败不致命。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/XSTest")
        s = ds[list(ds.keys())[0]]
        cols = s.column_names
        pcol = "prompt" if "prompt" in cols else cols[0]
        lcol = "label" if "label" in cols else None
        out = []
        for r in s:
            if lcol is None or str(r[lcol]).lower() in ("safe", "0", "false"):
                out.append(r[pcol])
        print(f"  XSTest-safe: +{len(out)}")
        return out
    except Exception as e:
        print(f"  XSTest FAILED (optional): {e}")
        return []


def main():
    jbb = load_jbb_benign_norms()
    print(f"JBB-Benign 去污染参照: {len(jbb)} 条")
    v7 = load_v7_rows()
    print(f"v7 原数据: {len(v7)} 条")
    gray = mine_or_bench() + mine_xstest()
    # 去污染：去掉与 JBB-Benign 归一化精确重复的；也去掉与 v7 已有的重复
    seen = set(norm(it["text"]) for it in v7)
    kept, dropped_jbb, dropped_dup = [], 0, 0
    for t in gray:
        nt = norm(t)
        if not nt or len(nt) < 8:
            continue
        if nt in jbb:
            dropped_jbb += 1; continue
        if nt in seen:
            dropped_dup += 1; continue
        seen.add(nt)
        kept.append({"text": str(t).strip(), "label": 2, "category": "gray_benign",
                     "source": "or_bench/xstest"})
    print(f"挖到 gray_benign {len(gray)} -> 去 JBB 重叠 {dropped_jbb}, 去内部重复 {dropped_dup}, 保留 {len(kept)}")
    # 合并 + 90/10 切分（在新增 gray 上切；v7 保持原 split）
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * 0.1))
    new_val, new_train = kept[:n_val], kept[n_val:]
    train_rows = [it for it in v7 if it["_split"] == "train"] + new_train
    val_rows = [it for it in v7 if it["_split"] == "val"] + new_val
    random.shuffle(train_rows); random.shuffle(val_rows)
    outdir = BASE / "datasets/v9_training"; outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        with open(outdir / f"{name}.jsonl", "w") as f:
            for it in rows:
                f.write(json.dumps({"text": it["text"], "label": it["label"],
                                    "category": it.get("category", "")}, ensure_ascii=False) + "\n")
    from collections import Counter
    comp = {"train": dict(sorted(Counter(it["label"] for it in train_rows).items())),
            "val": dict(sorted(Counter(it["label"] for it in val_rows).items())),
            "gray_benign_added": len(kept), "dropped_jbb_overlap": dropped_jbb,
            "dropped_internal_dup": dropped_dup}
    json.dump(comp, open(outdir / "composition.json", "w"), indent=2, ensure_ascii=False)
    print("composition:", json.dumps(comp, ensure_ascii=False))


if __name__ == "__main__":
    main()
