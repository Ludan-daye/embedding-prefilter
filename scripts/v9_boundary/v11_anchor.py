#!/usr/bin/env python3
"""v11-careful = 锚点防遗忘的边界感知编码器微调。

v11 第一版失败: 顶层微调灾难性遗忘, 边界FPR降了但攻击DR从0.79崩到0.475(欠检测换的)。
本版修复: 锚点损失把【有害+正常】embedding 钉在冻结编码器位置(保住攻击检测), 只让【边界margin】
推"像攻击的正常"远离有害。最小扰动: 顶1层, lr 5e-6, 少步, 攻击DR守门(< 冻结-0.05 视为退化)。

L = λa·anchor(‖z_ft - z_frozen‖² on 有害+正常) + λm·boundary_margin(gray远离harmful)
留出 JBB30% + XSTest30% 只评测; 攻击DR用 GCG/PAIR/JailbreakHub(未训练攻击类型)。"""
import os, sys, json, gc, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
import importlib.util
import model as M
import evaluate as EV
_spec = importlib.util.spec_from_file_location("v7model", str(BASE / "scripts/v7_classifier/model.py"))
_v7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v7)
V6HarmfulDetector, load_tokenizer = _v7.V6HarmfulDetector, _v7.load_tokenizer
ATTACK_EVAL = ["GCG", "PAIR", "JailbreakHub"]


def enc_batch(model, tok, texts, dev, grad, max_len=128):
    x = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    x = {k: v.to(dev) for k, v in x.items()}
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return model.encode(x["input_ids"], x["attention_mask"])


def encode_all(model, tok, texts, dev, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        out.append(enc_batch(model, tok, texts[i:i+bs], dev, grad=False).cpu().numpy())
    return np.vstack(out)


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
    he, be = encode_all(model, tok, data["harm_tr"], dev), encode_all(model, tok, data["ben_tr"], dev)
    det = M.DualMultiCentroidDetector(1, 1); det.fit(he, be)
    ve = np.vstack([encode_all(model, tok, data["harm_val"], dev), encode_all(model, tok, data["ben_val"], dev)])
    vy = np.array([1]*len(data["harm_val"]) + [0]*len(data["ben_val"]))
    thr, ok = thr_target_dr(det.predict(ve, 0.0)[1], vy)
    fpr = lambda txt: round(float((det.predict(encode_all(model, tok, txt, dev), thr)[0] == 1).sum())/len(txt), 4)
    dr = lambda txt: round(1 - float((det.predict(encode_all(model, tok, txt, dev), thr)[0] == 0).sum())/len(txt), 4)
    adr = round(float(np.mean([dr(data["atk"][k]) for k in data["atk"]])), 4)
    r = {"tag": tag, "JBB_FPR": fpr(data["jbb_eval"]), "XSTest_FPR": fpr(data["xs_eval"]), "attack_DR": adr, "DR_met": ok}
    print(f"[{tag}] JBB={r['JBB_FPR']} XSTest={r['XSTest_FPR']} 攻击DR={r['attack_DR']}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1); ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--lam_a", type=float, default=1.0)
    ap.add_argument("--lam_m", type=float, default=1.0); ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    dev = M.get_device(); rng = np.random.RandomState(42)

    adv = pd.read_csv(BASE/"datasets/advbench/advbench_harmful_behaviors.csv")["goal"].astype(str).tolist()
    harm = pd.read_csv(BASE/"datasets/harmbench/harmbench_behaviors.csv")["Behavior"].astype(str).tolist()
    harmful = adv + harm
    benign = [json.loads(l)["text"] for l in open(BASE/"datasets/normal/alpaca.jsonl")][:1500]
    orb = [json.loads(l)["text"] for l in open(BASE/"datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE/"datasets/overrefusal/xstest.csv"); xs = xs[xs["label"]=="safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE/"datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()

    def split(lst, frac=0.7):
        idx = rng.permutation(len(lst)); k = int(len(lst)*frac); return [lst[i] for i in idx[:k]], [lst[i] for i in idx[k:]]
    xs_tr, xs_ev = split(xs); jbb_tr, jbb_ev = split(jbb)
    gray_tr = orb + xs_tr + jbb_tr
    h_tr, h_val = split(harmful, 0.85); b_tr, b_val = split(benign, 0.85)
    atk = {k: EV.load_texts(EV.DATASETS[k]) for k in ATTACK_EVAL if EV.load_texts(EV.DATASETS[k])}
    data = {"harm_tr": h_tr, "ben_tr": b_tr, "harm_val": h_val, "ben_val": b_val,
            "jbb_eval": jbb_ev, "xs_eval": xs_ev, "atk": atk}
    print(f"有害{len(harmful)} 正常{len(benign)} 灰(训){len(gray_tr)} | 留出JBB{len(jbb_ev)} XSTest{len(xs_ev)}", flush=True)

    mp = BASE/"models/v7_classifier"; cfg = json.load(open(mp/"config.json"))
    model = V6HarmfulDetector(cfg["model_name"], cfg["projection_dim"])
    model.load_state_dict(torch.load(str(mp/"best_model.pt"), map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(dev); tok = load_tokenizer(cfg["model_name"])

    results = {}
    model.eval(); results["v9_frozen"] = evaluate_model(model, tok, dev, data, "v9冻结")

    z0_h = torch.tensor(encode_all(model, tok, h_tr, dev), device=dev)
    z0_b = torch.tensor(encode_all(model, tok, b_tr, dev), device=dev)

    model.unfreeze_encoder_layers(args.layers)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    print(f"微调: 顶{args.layers}层 可训练={model.get_trainable_params():,} steps={args.steps} lr={args.lr} λa={args.lam_a} λm={args.lam_m}", flush=True)
    nh, nb, ng = len(h_tr), len(b_tr), len(gray_tr)
    for st in range(args.steps):
        model.train()
        ih = rng.randint(0, nh, args.bs); ib = rng.randint(0, nb, args.bs); ig = rng.randint(0, ng, args.bs)
        zh = enc_batch(model, tok, [h_tr[i] for i in ih], dev, grad=True)
        zb = enc_batch(model, tok, [b_tr[i] for i in ib], dev, grad=True)
        zg = enc_batch(model, tok, [gray_tr[i] for i in ig], dev, grad=True)
        L_anchor = F.mse_loss(zh, z0_h[ih]) + F.mse_loss(zb, z0_b[ib])
        cb = zb.mean(0); cb = cb / (cb.norm() + 1e-8)
        L_margin = torch.relu(args.margin + (zg @ zh.T).max(1).values - zg @ cb).mean()
        loss = args.lam_a * L_anchor + args.lam_m * L_margin
        opt.zero_grad(); loss.backward(); opt.step()
        if (st+1) % 30 == 0:
            print(f"  step{st+1}/{args.steps} L={float(loss):.4f} (anchor={float(L_anchor):.4f} margin={float(L_margin):.4f})", flush=True)

    model.eval(); results["v11_anchor"] = evaluate_model(model, tok, dev, data, "v11锚点微调后")

    a, c = results["v9_frozen"], results["v11_anchor"]
    dj, dx, dd = c["JBB_FPR"]-a["JBB_FPR"], c["XSTest_FPR"]-a["XSTest_FPR"], c["attack_DR"]-a["attack_DR"]
    print("\n=== v11 锚点微调 判定 ===", flush=True)
    print(f"{'':16s}{'JBB_FPR':>9}{'XSTest_FPR':>11}{'攻击DR':>8}", flush=True)
    for k in ["v9_frozen", "v11_anchor"]:
        r = results[k]; print(f"{k:16s}{r['JBB_FPR']:>9}{r['XSTest_FPR']:>11}{r['attack_DR']:>8}", flush=True)
    print(f"\nΔ: JBB {dj:+.3f}  XSTest {dx:+.3f}  攻击DR {dd:+.3f}", flush=True)
    if dd < -0.05:
        verdict = "退化: 攻击DR掉太多(遗忘未防住)"
    elif (dj + dx) < -0.06:
        verdict = "成功: 守住攻击DR的同时降了边界FPR ✅"
    else:
        verdict = "无效: 攻击DR守住了但边界FPR没降(锚点太紧/margin不迁移)"
    print(f"初判: {verdict}", flush=True)
    json.dump(results, open(BASE/"results/v9_boundary/v11_anchor_results.json", "w"), indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), str(BASE / ("models/v11_anchor_top%d.pt" % args.layers)))
    print("V11A_DONE", flush=True)


if __name__ == "__main__":
    main()
