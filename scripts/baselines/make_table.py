#!/usr/bin/env python3
"""汇总 results/baselines/*/metrics.json → comparison_table.md"""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
ATTACK = ["GCG", "PAIR", "JailbreakHub", "AdvBench", "HarmBench"]
BENIGN = ["JBB_Benign", "Alpaca"]
NOTE = ["ToxicChat_harmful", "BeaverTails_harmful", "ToxicChat_benign", "BeaverTails_benign"]  # 标注不匹配，单列


def main():
    rows = []
    for mdir in sorted((BASE / "results/baselines").glob("*/")):
        f = mdir / "metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f)); d = m["datasets"]

        def g(k):
            v = d.get(k, {}); return v.get("detection_rate", v.get("fpr"))
        rows.append((m["method"], m.get("params", "?"), m.get("dim", "-"),
                     [g(k) for k in ATTACK], [g(k) for k in BENIGN],
                     [g(k) for k in NOTE], m.get("latency_ms_per_sample", "?")))

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else "-"

    # 主表
    hdr = "| 方法 | 参数 | 维度 | " + " | ".join(f"{a} DR" for a in ATTACK) + " | " + \
          " | ".join(f"{b} FPR" for b in BENIGN) + " | 延迟ms |"
    sep = "|" + "---|" * (3 + len(ATTACK) + len(BENIGN) + 1)
    lines = ["# 统一协议对比（攻击列=DR↑，benign列=FPR↓；seed=42）", "",
             "## 主表（攻击检测 + 边界/普通 benign）", "", hdr, sep]
    for name, p, dim, a, b, note, lat in rows:
        cells = [name, p, dim] + [fmt(x) for x in a + b] + [str(lat)]
        lines.append("| " + " | ".join(cells) + " |")
    # 单列标注不匹配数据集
    lines += ["", "## 附表（标注语义不匹配，单列不进头条）", "",
              "| 方法 | " + " | ".join(NOTE) + " |", "|" + "---|" * (1 + len(NOTE))]
    for name, p, dim, a, b, note, lat in rows:
        lines.append("| " + name + " | " + " | ".join(fmt(x) for x in note) + " |")

    out = BASE / "results/baselines/comparison_table.md"
    out.write_text("\n".join(lines) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
