#!/usr/bin/env python3
"""统一【留出协议】主表:最终系统3变体 vs 全部基线,一次跑完,含 XSTest 过度拒绝。

为什么要留出协议:判别头/ v9/ bge_svm/ perplexity 等都在我们的数据上训练过。统一 harness 恰在这些
全集上评测 → 训练测试泄漏,数字虚高。本脚本对【任何训练型方法会用到的数据集】(5攻击 + JBB + Alpaca
+ XSTest)按 70/30(每集独立 seed=42)划分,所有方法只在 30% 留出上评测;判别头只在 70% 上训。
ToxicChat / BeaverTails 无人训练 → 用全集。→ 公平、无泄漏、可发表。

评测方法:
  最终系统 A: E5-large × 判别头(logistic)
  最终系统 B: BGE-base(v7ft) × 判别头
  最终系统 C: BGE-base(v7ft) × PCA-32 × 判别头
  基线: ours_v9(双质心) / bge_svm / bge_cosine / tfidf_lr / keyword / perplexity_lgb / injecguard / protectai
每个基线在 try/except 内构建,单个失败不影响整表。"""
import os, sys, json, gc, time, random
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
sys.path.insert(0, str(BASE / "scripts/baselines"))
import model as M          # noqa: E402
import train as v8train    # noqa: E402
import evaluate as EV      # noqa: E402

ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]  # 重叠有害 → 划分
NONOVL_HARM = ["ToxicChat_harmful", "BeaverTails_harmful"]          # 无人训练 → 全集
NONOVL_BENIGN = ["ToxicChat_benign", "BeaverTails_benign"]          # 无人训练 → 全集
dev = M.get_device()


def split(lst, f=0.7, seed=42):
    r = np.random.RandomState(seed); idx = r.permutation(len(lst)); k = int(len(lst) * f)
    return [lst[i] for i in idx[:k]], [lst[i] for i in idx[k:]]


def load_ds(name):
    random.seed(42)
    return EV.load_texts(EV.DATASETS[name])


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def thr_target_dr(scores, ybin, target=0.90):
    P, N = max(int((ybin == 1).sum()), 1), max(int((ybin == 0).sum()), 1)
    bt, bf, ok = 0.0, 2.0, False
    for t in np.unique(scores):
        p = scores > t
        if (p & (ybin == 1)).sum() / P >= target:
            f = (p & (ybin == 0)).sum() / N
            if f < bf:
                bf, bt, ok = f, float(t), True
    return (bt, True) if ok else (float(np.min(scores)) - 1e-6, False)


# ---------- 编码器(加载一次,缓存) ----------
_bge = None
def enc_bge(texts):
    global _bge
    if _bge is None:
        _bge = v8train.load_v7_encoder(dev)
    m, t = _bge
    return l2(v8train.extract_embeddings(m, t, texts, dev).astype(np.float64))


_e5 = None
def enc_e5(texts, bs=32):
    global _e5
    if _e5 is None:
        tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        mdl = AutoModel.from_pretrained("intfloat/e5-large-v2").to(dev).eval()
        _e5 = (tok, mdl)
    tok, mdl = _e5
    out = []
    for i in range(0, len(texts), bs):
        x = tok(["query: " + t for t in texts[i:i + bs]], padding=True, truncation=True,
                max_length=256, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.no_grad():
            H = mdl(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        out.append(((H * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return l2(np.vstack(out).astype(np.float64))


def main():
    random.seed(42); np.random.seed(42)

    # ============ 1. 构建训练 / 留出评测集 ============
    atk_tr, atk_ev = {}, {}
    for k in ATTACKS:
        tr, ev = split(load_ds(k))
        atk_tr[k], atk_ev[k] = tr, ev
    harm_tr_all = sum(atk_tr.values(), [])

    alp_all = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")]
    alp_pool = alp_all[:200]                      # 评测池(与 harness 同量级)
    alp_tr_pool, alp_ev = split(alp_pool)         # 留出 60 评测
    benign_train = alp_tr_pool + alp_all[200:1700]  # 训练用大 benign,与评测 60 不相交

    orb = [json.loads(l)["text"] for l in open(BASE / "datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    xs_tr, xs_ev = split(xs)
    jbb_tr, jbb_ev = split(jbb)
    gray_tr = orb + xs_tr + jbb_tr

    # 判别头训练集(=v13 口径)
    h_tr, h_val = split(harm_tr_all, 0.85)
    b_tr, b_val = split(benign_train, 0.85)

    # 留出评测字典: name -> (texts, kind)  kind in {"harmful","benign"}
    EVAL = {}
    for k in ATTACKS:
        EVAL[k] = (atk_ev[k], "harmful")
    for k in NONOVL_HARM:
        EVAL[k] = (load_ds(k), "harmful")
    EVAL["JBB_Benign"] = (jbb_ev, "benign")
    EVAL["XSTest"] = (xs_ev, "benign")
    EVAL["Alpaca"] = (alp_ev, "benign")
    for k in NONOVL_BENIGN:
        EVAL[k] = (load_ds(k), "benign")

    print(f"训练: 有害{len(h_tr)}(5类) benign{len(b_tr)} 灰{len(gray_tr)} | 阈值val h{len(h_val)} b{len(b_val)}", flush=True)
    print("留出评测集大小: " + ", ".join(f"{k}={len(v[0])}" for k, v in EVAL.items()), flush=True)

    results = {}   # name -> {"params","dim","per":{ds:val}}

    # ============ 2. 最终系统 3 变体 ============
    # 预编码(每编码器一次): 训练/val/所有留出
    def encode_all(enc_fn):
        E = {"h_tr": enc_fn(h_tr), "b_tr": enc_fn(b_tr), "gray": enc_fn(gray_tr),
             "h_val": enc_fn(h_val), "b_val": enc_fn(b_val)}
        E_ev = {k: enc_fn(v[0]) for k, v in EVAL.items()}
        return E, E_ev

    def fit_head_and_eval(E, E_ev, name, params, dim, compressor=None):
        def C(x):
            return x if compressor is None else compressor(x)
        Xtr = np.vstack([C(E["h_tr"]), C(E["b_tr"]), C(E["gray"])])
        ytr = np.array([1] * len(E["h_tr"]) + [0] * (len(E["b_tr"]) + len(E["gray"])))
        Xtr = l2(Xtr)
        sca = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
        sc = lambda X: clf.decision_function(sca.transform(l2(C(X))))
        Xval = np.vstack([E["h_val"], E["b_val"]]); yval = np.array([1] * len(E["h_val"]) + [0] * len(E["b_val"]))
        thr, ok = thr_target_dr(sc(Xval), yval)
        per = {}
        for ds, emb in E_ev.items():
            pred = (sc(emb) > thr).astype(int)
            kind = EVAL[ds][1]
            per[ds] = round(float(pred.mean()), 4)   # harmful→DR, benign→FPR
        results[name] = {"params": params, "dim": dim, "per": per}
        print(f"  [{name}] thr_ok={ok} " + " ".join(f"{k}={v}" for k, v in per.items()), flush=True)

    try:
        Eb, Eb_ev = encode_all(enc_bge)
        fit_head_and_eval(Eb, Eb_ev, "final_BGE_logistic", "~800", "768D")
        pca = PCA(n_components=32, random_state=42).fit(l2(np.vstack([Eb["h_tr"], Eb["b_tr"], Eb["gray"]])))
        fit_head_and_eval(Eb, Eb_ev, "final_BGE_PCA32", "~32", "32D", compressor=lambda x: pca.transform(l2(x)))
    except Exception as e:
        print(f"BGE 变体失败: {type(e).__name__}: {str(e)[:120]}", flush=True)

    try:
        Ee, Ee_ev = encode_all(enc_e5)
        fit_head_and_eval(Ee, Ee_ev, "final_E5_logistic", "~1024", "1024D")
    except Exception as e:
        print(f"E5 变体失败: {type(e).__name__}: {str(e)[:120]}", flush=True)

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ============ 3. 基线(复用适配器,留出评测) ============
    def _to01(p):
        if isinstance(p, str):
            return 1 if p.strip().lower() == "harmful" else 0
        return int(p)

    def eval_predictor(name, predict_batch, params, dim):
        per = {}
        for ds, (texts, kind) in EVAL.items():
            preds = [_to01(p) for p in predict_batch(texts)]
            per[ds] = round(float(np.mean(preds)), 4)
        results[name] = {"params": params, "dim": dim, "per": per}
        print(f"  [{name}] " + " ".join(f"{k}={v}" for k, v in per.items()), flush=True)

    def try_baseline(name, builder, params, dim):
        try:
            t0 = time.time()
            m = builder()
            eval_predictor(name, m.predict_batch, params, dim)
            print(f"    {name} 完成 {time.time()-t0:.0f}s", flush=True)
            del m; gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{name}] SKIP: {type(e).__name__}: {str(e)[:120]}", flush=True)

    import floor, ours_v9
    try_baseline("ours_v9", ours_v9.build, "25K proj", "32D")
    try_baseline("bge_svm", floor.build_bge_svm, "109M enc", "768D")
    try_baseline("bge_cosine", floor.build_bge_cosine, "109M enc", "768D")
    try_baseline("tfidf_lr", floor.build_tfidf, "~5K", "-")
    try_baseline("keyword", floor.build_keyword, "0", "-")
    try:
        import perplexity_lgb
        try_baseline("perplexity_lgb", perplexity_lgb.build, "~0", "-")
    except Exception as e:
        print(f"  [perplexity_lgb] IMPORT SKIP: {str(e)[:80]}", flush=True)
    try:
        import hf_classifiers
        try_baseline("injecguard", hf_classifiers.build_injecguard, "184M", "-")
        try_baseline("protectai", hf_classifiers.build_protectai, "184M", "-")
    except Exception as e:
        print(f"  [hf_classifiers] IMPORT SKIP: {str(e)[:80]}", flush=True)

    # ============ 4. 汇总表 ============
    cols_atk = ATTACKS + NONOVL_HARM
    cols_ben = ["JBB_Benign", "XSTest", "Alpaca"] + NONOVL_BENIGN
    for name, r in results.items():
        per = r["per"]
        atk_main = [per[k] for k in ATTACKS if k in per]
        r["attack_DR_5"] = round(float(np.mean(atk_main)), 4) if atk_main else None
        r["youden_J"] = round((r["attack_DR_5"] or 0) - float(np.mean([per.get(k, 0) for k in ["JBB_Benign", "XSTest", "Alpaca"]])), 4)

    print("\n" + "=" * 120, flush=True)
    print("=== 统一留出协议主表(攻击列=DR↑, benign列=FPR↓; XSTest/JBB/Alpaca 为过度拒绝) ===", flush=True)
    hdr = f"{'方法':22s}{'参数':>8}{'维度':>7}" + "".join(f"{c[:9]:>10}" for c in cols_atk) + "".join(f"{c[:11]:>12}" for c in cols_ben) + f"{'攻击DR5':>9}{'J':>7}"
    print(hdr, flush=True)
    for name, r in results.items():
        per = r["per"]
        line = f"{name:22s}{r['params']:>8}{r['dim']:>7}"
        line += "".join(f"{per.get(c, float('nan')):>10.3f}" for c in cols_atk)
        line += "".join(f"{per.get(c, float('nan')):>12.3f}" for c in cols_ben)
        line += f"{r['attack_DR_5']:>9.3f}{r['youden_J']:>7.3f}"
        print(line, flush=True)

    out = {"protocol": "heldout-30pct (overlap sets split 70/30 seed42; ToxicChat/BeaverTails full)",
           "eval_sizes": {k: len(v[0]) for k, v in EVAL.items()}, "results": results}
    json.dump(out, open(BASE / "results/baselines/pareto_final_results.json", "w"), indent=2, ensure_ascii=False)
    print("\nPARETO_FINAL_DONE", flush=True)


if __name__ == "__main__":
    main()
