#!/usr/bin/env python3
"""表4(统一留出协议主表)的种子确认。5 种子,报 mean±std + 按 J 排名。

提速关键:基线模型对每个数据集【全集只推理一次】,缓存 per-text 0/1 预测;每种子只做"取 30%
留出子集 + 求均值"。最终 3 变体(BGE×判别头 / BGE×PCA128 / E5×判别头)每种子重训判别头(embedding
已缓存,很便宜)。协议同 pareto_final:重叠集 70/30(每集 seed)划分,ToxicChat/BeaverTails 全集。
J = attack_DR5 − mean(JBB, XSTest, Alpaca)。"""
import os, sys, json, gc, random
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

ATTACKS = ["AdvBench", "HarmBench", "GCG", "PAIR", "JailbreakHub"]
NONOVL = ["ToxicChat_harmful", "BeaverTails_harmful", "ToxicChat_benign", "BeaverTails_benign"]
SEEDS = [42, 43, 44, 45, 46]
dev = M.get_device()


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _to01(p):
    if isinstance(p, str):
        return 1 if p.strip().lower() == "harmful" else 0
    return int(p)


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


def split_idx(n, f, seed):
    r = np.random.RandomState(seed)
    idx = r.permutation(n)
    k = int(n * f)
    return idx[:k], idx[k:]


def load_ds(name):
    random.seed(42)
    return EV.load_texts(EV.DATASETS[name])


def main():
    atk_txt = {k: load_ds(k) for k in ATTACKS}
    alp_all = [json.loads(l)["text"] for l in open(BASE / "datasets/normal/alpaca.jsonl")][:1700]
    orb = [json.loads(l)["text"] for l in open(BASE / "datasets/overrefusal/orbench_hard.jsonl")]
    xs = pd.read_csv(BASE / "datasets/overrefusal/xstest.csv")
    xs = xs[xs["label"] == "safe"]["prompt"].astype(str).tolist()
    jbb = pd.read_csv(BASE / "datasets/gcg_attacks/jbb_benign_behaviors.csv")["Goal"].astype(str).tolist()
    nonovl_txt = {k: load_ds(k) for k in NONOVL}

    KIND = {**{k: "harmful" for k in ATTACKS}, "JBB_Benign": "benign", "XSTest": "benign", "Alpaca": "benign",
            "ToxicChat_harmful": "harmful", "BeaverTails_harmful": "harmful",
            "ToxicChat_benign": "benign", "BeaverTails_benign": "benign"}
    # 重叠集全量 texts(每种子对其取 30% 留出)
    SPLIT_POOL = {**{k: atk_txt[k] for k in ATTACKS}, "JBB_Benign": jbb, "XSTest": xs, "Alpaca": alp_all[:200]}

    # ---------- 1. 基线: 每集全量推理一次, 缓存 per-text 0/1 ----------
    def cache_baseline(name, builder):
        try:
            m = builder()
            pf = {}
            for ds, txt in {**SPLIT_POOL, **nonovl_txt}.items():
                pf[ds] = np.array([_to01(p) for p in m.predict_batch(txt)])
            del m; gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  缓存 {name}", flush=True)
            return pf
        except Exception as e:
            print(f"  [{name}] SKIP: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return None

    import floor, ours_v9
    baseline_preds = {}
    baseline_meta = {"ours_v9": "25K/32D", "bge_svm": "109M/768D", "bge_cosine": "109M/768D",
                     "tfidf_lr": "~5K", "keyword": "0", "perplexity_lgb": "~0", "injecguard": "184M", "protectai": "184M"}
    baseline_preds["ours_v9"] = cache_baseline("ours_v9", ours_v9.build)
    baseline_preds["bge_svm"] = cache_baseline("bge_svm", floor.build_bge_svm)
    baseline_preds["bge_cosine"] = cache_baseline("bge_cosine", floor.build_bge_cosine)
    baseline_preds["tfidf_lr"] = cache_baseline("tfidf_lr", floor.build_tfidf)
    baseline_preds["keyword"] = cache_baseline("keyword", floor.build_keyword)
    try:
        import perplexity_lgb
        baseline_preds["perplexity_lgb"] = cache_baseline("perplexity_lgb", perplexity_lgb.build)
    except Exception as e:
        print(f"  perplexity import SKIP {str(e)[:60]}", flush=True)
    try:
        import hf_classifiers
        baseline_preds["injecguard"] = cache_baseline("injecguard", hf_classifiers.build_injecguard)
        baseline_preds["protectai"] = cache_baseline("protectai", hf_classifiers.build_protectai)
    except Exception as e:
        print(f"  hf_classifiers import SKIP {str(e)[:60]}", flush=True)

    # ---------- 2. 最终变体: 编码一次(BGE, E5) ----------
    def encode_all(enc_fn):
        d = {k: enc_fn(atk_txt[k]) for k in ATTACKS}
        d["alp"] = enc_fn(alp_all); d["orb"] = enc_fn(orb); d["xs"] = enc_fn(xs); d["jbb"] = enc_fn(jbb)
        for k in NONOVL:
            d[k] = enc_fn(nonovl_txt[k])
        return d

    def enc_bge_fn():
        enc, tok = v8train.load_v7_encoder(dev)
        return lambda t: l2(v8train.extract_embeddings(enc, tok, t, dev).astype(np.float64)), (enc, tok)

    def enc_e5_fn():
        tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        mdl = AutoModel.from_pretrained("intfloat/e5-large-v2").to(dev).eval()

        def f(texts, bs=32):
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
        return f, (mdl, tok)

    fb, hb = enc_bge_fn(); EB = encode_all(fb); del hb; gc.collect(); torch.cuda.empty_cache()
    fe, he = enc_e5_fn(); EE = encode_all(fe); del he; gc.collect(); torch.cuda.empty_cache()
    print("最终变体编码完成(BGE + E5)", flush=True)

    # ---------- 3. 逐种子评测 ----------
    all_methods = ["final_BGE_logistic", "final_BGE_PCA128", "final_E5_logistic"] + \
                  [m for m in baseline_preds if baseline_preds[m] is not None]
    per_seed = {m: [] for m in all_methods}
    EVAL_DS = ATTACKS + ["ToxicChat_harmful", "BeaverTails_harmful", "JBB_Benign", "XSTest", "Alpaca",
                         "ToxicChat_benign", "BeaverTails_benign"]
    EMB_KEY = {**{k: k for k in ATTACKS}, "JBB_Benign": "jbb", "XSTest": "xs", "Alpaca": "alp",
               **{k: k for k in NONOVL}}   # EVAL_DS 名 -> 编码字典键

    def metrics_from_preds(preds_by_ds):
        adr = float(np.mean([preds_by_ds[k].mean() for k in ATTACKS]))
        jbb = float(preds_by_ds["JBB_Benign"].mean()); xsf = float(preds_by_ds["XSTest"].mean())
        alp = float(preds_by_ds["Alpaca"].mean())
        J = adr - (jbb + xsf + alp) / 3
        return {"adr": adr, "jbb": jbb, "xs": xsf, "alp": alp, "J": J}

    for seed in SEEDS:
        atk_ev_i = {k: split_idx(len(atk_txt[k]), 0.7, seed)[1] for k in ATTACKS}
        atk_tr_i = {k: split_idx(len(atk_txt[k]), 0.7, seed)[0] for k in ATTACKS}
        alp_tr_i, alp_ev_i = split_idx(200, 0.7, seed)
        xs_tr_i, xs_ev_i = split_idx(len(xs), 0.7, seed)
        jbb_tr_i, jbb_ev_i = split_idx(len(jbb), 0.7, seed)
        EVI = {**{k: atk_ev_i[k] for k in ATTACKS}, "JBB_Benign": jbb_ev_i, "XSTest": xs_ev_i, "Alpaca": alp_ev_i}

        # --- 基线: 取子集 ---
        for mname, pf in baseline_preds.items():
            if pf is None:
                continue
            pbd = {}
            for ds in EVAL_DS:
                pbd[ds] = pf[ds][EVI[ds]] if ds in EVI else pf[ds]
            per_seed[mname].append(metrics_from_preds(pbd))

        # --- 最终变体: 重训 ---
        def run_final(E, name, compress=None):
            Htr = np.vstack([E[k][atk_tr_i[k]] for k in ATTACKS])
            benign_train = np.vstack([E["alp"][alp_tr_i], E["alp"][200:1700]])
            gray = np.vstack([E["orb"], E["xs"][xs_tr_i], E["jbb"][jbb_tr_i]])
            hti, hvi = split_idx(len(Htr), 0.85, seed); bti, bvi = split_idx(len(benign_train), 0.85, seed)
            Htr2, Hval = Htr[hti], Htr[hvi]; Btr, Bval = benign_train[bti], benign_train[bvi]
            proj = (lambda X: X) if compress is None else compress(np.vstack([Htr2, Btr, gray]))
            C = lambda X: l2(proj(X))
            Xtr = np.vstack([C(Htr2), C(Btr), C(gray)]); ytr = np.array([1] * len(Htr2) + [0] * (len(Btr) + len(gray)))
            sca = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sca.transform(Xtr), ytr)
            sc = lambda X: clf.decision_function(sca.transform(C(X)))
            yval = np.array([1] * len(Hval) + [0] * len(Bval))
            thr, _ = thr_target_dr(np.concatenate([sc(Hval), sc(Bval)]), yval)
            pbd = {}
            for ds in EVAL_DS:
                ek = EMB_KEY[ds]
                emb = E[ek][EVI[ds]] if ds in EVI else E[ek]
                pbd[ds] = (sc(emb) > thr).astype(int)
            per_seed[name].append(metrics_from_preds(pbd))

        def pca_fit(Xtr_raw):
            p = PCA(n_components=128, random_state=seed).fit(Xtr_raw)
            return lambda X, p=p: p.transform(X)

        run_final(EB, "final_BGE_logistic")
        run_final(EB, "final_BGE_PCA128", compress=pca_fit)
        run_final(EE, "final_E5_logistic")
        print(f"seed {seed} 完成", flush=True)

    # ---------- 4. 汇总 ----------
    def ms(vals):
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    summary = {}
    for m, rows in per_seed.items():
        if not rows:
            continue
        summary[m] = {k: ms([r[k] for r in rows]) for k in ["adr", "jbb", "xs", "alp", "J"]}
    ranked = sorted(summary, key=lambda m: -summary[m]["J"][0])

    print("\n" + "=" * 100, flush=True)
    print(f"=== 表4 统一留出协议主表({len(SEEDS)} 种子 mean±std, 按 J 降序). J=攻击DR5−mean(JBB,XSTest,Alpaca) ===", flush=True)
    print(f"{'方法':22s}{'攻击DR5':>15}{'JBB↓':>13}{'XSTest↓':>13}{'Alpaca↓':>13}{'J↑':>14}", flush=True)
    for m in ranked:
        s = summary[m]
        print(f"{m:22s}{s['adr'][0]:>8.3f}±{s['adr'][1]:<5.3f}{s['jbb'][0]:>6.3f}±{s['jbb'][1]:<4.3f}"
              f"{s['xs'][0]:>6.3f}±{s['xs'][1]:<4.3f}{s['alp'][0]:>6.3f}±{s['alp'][1]:<4.3f}"
              f"{s['J'][0]:>7.3f}±{s['J'][1]:<5.3f}", flush=True)

    json.dump({"seeds": SEEDS, "meta": baseline_meta, "summary": summary, "per_seed": per_seed},
              open(BASE / "results/baselines/pareto_seeds_results.json", "w"), indent=2, ensure_ascii=False)
    print("\nPARETO_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()
