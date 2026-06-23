#!/usr/bin/env python3
"""v11 = 边界感知编码器微调。数据证明边界过度拒绝是 encoder 层问题(投影/margin救不动,0.86天花板),
所以解冻 BGE 顶层、用 SupCon+边界margin 直接重塑 768D 表示,让"像攻击的正常"在编码器里就离有害远。

对比微调前(=v9冻结)vs 微调后(v11),同下游(768D双质心+DR≥0.90阈值),报留出边界FPR(JBB/XSTest)。
留出: XSTest-safe 30% + JBB 30% 只评测,不进训练;攻击DR用GCG/PAIR/JailbreakHub(未训练的攻击类型)。"""
import os, sys, json, gc, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))   # 只挂 v8, 避免 model.py 重名
import importlib.util
import model as M                                   # v8: DualMultiCentroidDetector, get_device
import train as v8train                             # supervised_contrastive_loss
import evaluate as EV                               # DATASETS, load_texts
# v7 编码器(可微调)单独按文件路径加载, 避免与 v8 的 model 模块冲突
_spec = importlib.util.spec_from_file_location("v7model", str(BASE / "scripts/v7_classifier/model.py"))
_v7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v7)
V6HarmfulDetector, load_tokenizer = _v7.V6HarmfulDetector, _v7.load_tokenizer

ATTACK_EVAL = ["GCG", "PAIR", "JailbreakHub"]       # 未训练的攻击类型, 用于锁工作点+报DR


def boundary_margin_loss(z, l4, margin):
    gb, h, b = (l4 == 2), (l4 == 1), (l4 == 0)
    if int(gb.sum()) == 0 or int(h.sum()) == 0 or int(b.sum()) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    zg, zh = z[gb], z[h]
    cb = z[b].mean(0); cb = cb / (cb.norm() + 1e-8)
    return torch.relu(margin + (zg @ zh.T).max(1).values - zg @ cb).mean()


def encode(model, tok, texts, dev, bs=128, grad=False, max_len=128):
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            x = {k: v.to(dev) for k, v in x.items()}
            out.append(model.encode(x["input_ids"], x["attention_mask"]))
    return torch.cat(out) if grad else torch.cat(out).cpu().numpy()


def thr_target_dr(scores, ybin, target=0.90):
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bf, ok = 0.0, 2.0, False
    for t in np.unique(scores):
        p = scores > t
        if (p & (ybin == 1)).sum() / P >= target:
            f = (p & (ybin == 0)).sum() / N
            if f < bf: bf, bt, ok = f, float(t), True
    return (bt, True) if ok else (float(np.min(scores)) - 1e-6, False)


def evaluate_model(model, tok, dev, data, tag):
    """768D 双质心 + DR≥0.90 阈值, 报留出边界 FPR + 攻击 DR。"""
    he = encode(model, tok, data["harm_tr"], dev); be = encode(model, tok, data["ben_tr"], dev)
    det = M.DualMultiCentroidDetector(1, 1); det.fit(he, be)
    # val(从训练有害/正常各留一截)选阈值
    ve = np.vstack([encode(model, tok, data["harm_val"], dev), encode(model, tok, data["ben_val"], dev)])
    vy = np.array([1]*len(data["harm_val"]) + [0]*len(data["ben_val"]))
    _, vs = det.predict(ve, 0.0); thr, ok = thr_target_dr(vs, vy)
    def fpr(txt):
        e = encode(model, tok, txt, dev); return round(float((det.predict(e, thr)[0] == 1).sum())/len(txt), 4)
    def dr(txt):
        e = encode(model, tok, txt, dev); return round(1 - float((det.predict(e, thr)[0] == 0).sum())/len(txt), 4)
    adr = round(float(np.mean([dr(data["atk"][k]) for k in data["atk"]])), 4)
    r = {"tag": tag, "JBB_FPR": fpr(data["jbb_eval"]), "XSTest_FPR": fpr(data["xs_eval"]), "attack_DR": adr, "DR_met": ok}
    print(f"[{tag}] JBB={r['JBB_FPR']} XSTest={r['XSTest_FPR']} 攻击DR={r['attack_DR']}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.5); ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    dev = M.get_device(); rng = np.random.RandomState(42)

    # ---- 数据 ----
    adv = pd.read_csv(BASE/"datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()
    harm = pd.read_csv(BASE/"datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()
    harmful = adv + harm
    benign = [json.loads(l)["text"] for l in open(BASE/"datasets/normal/alpaca.jsonl")][:1500]
    orb = [json.loads(l)["text"] for l in open(BASE/"datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE/"datasets/overrefusal/xstest.csv"); xs = xs[xs["label"]=="safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE/"datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()

    def split(lst, frac=0.7):
        idx = rng.permutation(len(lst)); k = int(len(lst)*frac)
        return [lst[i] for i in idx[:k]], [lst[i] for i in idx[k:]]
    xs_tr, xs_ev = split(xs); jbb_tr, jbb_ev = split(jbb)
    gray_tr = orb + xs_tr + jbb_tr                    # 训练边界灰
    h_tr, h_val = split(harmful, 0.85); b_tr, b_val = split(benign, 0.85)
    atk = {k: EV.load_texts(EV.DATASETS[k]) for k in ATTACK_EVAL if EV.load_texts(EV.DATASETS[k])}
    data = {"harm_tr": h_tr, "ben_tr": b_tr, "harm_val": h_val, "ben_val": b_val,
            "jbb_eval": jbb_ev, "xs_eval": xs_ev, "atk": atk}
    print(f"有害{len(harmful)} 正常{len(benign)} 灰(训){len(gray_tr)} | 留出 JBB{len(jbb_ev)} XSTest{len(xs_ev)} | 攻击评测{list(atk)}", flush=True)

    # ---- 模型 ----
    mp = BASE/"models/v7_classifier"; cfg = json.load(open(mp/"config.json"))
    model = V6HarmfulDetector(cfg["model_name"], cfg["projection_dim"])
    ck = torch.load(str(mp/"best_model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.to(dev)
    tok = load_tokenizer(cfg["model_name"])

    results = {}
    model.eval(); results["v9_frozen(before)"] = evaluate_model(model, tok, dev, data, "v9冻结(微调前)")

    # ---- 微调顶层 ----
    model.unfreeze_encoder_layers(args.layers)
    train_txt = h_tr + b_tr + gray_tr
    train_l4 = np.array([1]*len(h_tr) + [0]*len(b_tr) + [2]*len(gray_tr))
    train_bin = np.where(train_l4 == 1, 1, 0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    n = len(train_txt); steps = (n + args.bs - 1)//args.bs
    print(f"微调: 顶{args.layers}层 可训练参数={model.get_trainable_params():,} epochs={args.epochs} steps/ep={steps}", flush=True)
    for ep in range(args.epochs):
        model.train(); order = rng.permutation(n); tot = 0.0
        for s in range(steps):
            bidx = order[s*args.bs:(s+1)*args.bs]
            bt = [train_txt[i] for i in bidx]
            z = encode(model, tok, bt, dev, bs=args.bs, grad=True)
            yb = torch.tensor(train_bin[bidx], dtype=torch.long, device=dev)
            l4 = torch.tensor(train_l4[bidx], dtype=torch.long, device=dev)
            loss = v8train.supervised_contrastive_loss(z, yb, 0.07) + args.lam * boundary_margin_loss(z, l4, args.margin)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print(f"  ep{ep+1}/{args.epochs} loss={tot/steps:.4f}", flush=True)

    model.eval(); results["v11_finetuned(after)"] = evaluate_model(model, tok, dev, data, "v11微调后")

    # ---- 判定 ----
    a, c = results["v9_frozen(before)"], results["v11_finetuned(after)"]
    print("\n=== v11 编码器微调 判定(留出边界 FPR,越低越好) ===", flush=True)
    print(f"{'':22s}{'JBB_FPR':>9}{'XSTest_FPR':>11}{'攻击DR':>8}", flush=True)
    for k in ["v9_frozen(before)", "v11_finetuned(after)"]:
        r = results[k]; print(f"{k:22s}{r['JBB_FPR']:>9}{r['XSTest_FPR']:>11}{r['attack_DR']:>8}", flush=True)
    dj, dx = c["JBB_FPR"]-a["JBB_FPR"], c["XSTest_FPR"]-a["XSTest_FPR"]
    print(f"\nΔ(微调-冻结): JBB {dj:+.3f}  XSTest {dx:+.3f}  (负=改善)", flush=True)
    print(f"初判: {'编码器微调有效(边界FPR下降)' if (dj+dx) < -0.08 else '编码器微调收效有限'}", flush=True)
    json.dump(results, open(BASE/"results/v9_boundary/v11_finetune_results.json", "w"), indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), str(BASE/"models/v11_finetune_top%d.pt" % args.layers))
    print("V11_DONE", flush=True)


if __name__ == "__main__":
    main()
