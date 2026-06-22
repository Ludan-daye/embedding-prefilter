#!/usr/bin/env python3
"""v9 (a2_l1m5) 适配器：编码器+投影+双质心，输出 1/0。"""
import sys, json
from pathlib import Path
import numpy as np, torch

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "scripts/v8_cs_supcon"))
from model import LearnedCSProjection, DualMultiCentroidDetector, get_device  # noqa: E402
from train import load_v7_encoder, extract_embeddings  # noqa: E402

MDIR = BASE / "models/v9_a2_l1.0_m0.5"
DIM, NC = 32, 1


def build():
    device = get_device()
    enc, tok = load_v7_encoder(device)
    proj = LearnedCSProjection(768, DIM)
    proj.load_state_dict(torch.load(str(MDIR / f"cs_projection_{DIM}d.pt"), map_location="cpu")); proj.eval()
    det = DualMultiCentroidDetector.load(str(MDIR / f"detector_{DIM}d_{NC}c.npz"))
    thr = json.load(open(MDIR / "training_results.json")).get(f"{DIM}d_{NC}c", {}).get("threshold", 0.0)

    class M:
        params = "25K proj"; dim = "32D"

        def predict_batch(self, texts):
            emb = extract_embeddings(enc, tok, texts, device).astype(np.float32)
            with torch.no_grad():
                z = proj(torch.tensor(emb)).numpy()
            return det.predict(z, threshold=thr)[0].tolist()
    return M()
