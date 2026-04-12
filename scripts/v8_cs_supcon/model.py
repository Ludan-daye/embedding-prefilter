#!/usr/bin/env python3
"""
V8 CS 压缩感知模型组件
- LearnedCSProjection: 有监督线性投影 (768D -> low-D)
- DualMultiCentroidDetector: 双侧多质心检测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def get_device():
    """MPS / CUDA / CPU 三路自动检测"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class LearnedCSProjection(nn.Module):
    """学习型 CS 压缩投影: Linear(input_dim, output_dim, bias=False) + L2 归一化"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0, std=1 / np.sqrt(output_dim))

    def forward(self, x):
        z = self.proj(x)
        return F.normalize(z, p=2, dim=-1)


class DualMultiCentroidDetector:
    """双侧多质心检测器: harmful 和 benign 各用 KMeans 聚类"""

    def __init__(self, n_harmful_centroids=3, n_benign_centroids=3):
        self.n_harmful = n_harmful_centroids
        self.n_benign = n_benign_centroids
        self.harmful_centroids = None  # [n_harmful, dim]
        self.benign_centroids = None   # [n_benign, dim]

    def fit(self, harmful_embeddings, benign_embeddings):
        # Harmful 侧
        km_h = KMeans(n_clusters=self.n_harmful, random_state=42, n_init=10)
        km_h.fit(harmful_embeddings)
        self.harmful_centroids = km_h.cluster_centers_
        self.harmful_centroids /= np.linalg.norm(self.harmful_centroids, axis=1, keepdims=True)

        # Benign 侧
        km_b = KMeans(n_clusters=self.n_benign, random_state=42, n_init=10)
        km_b.fit(benign_embeddings)
        self.benign_centroids = km_b.cluster_centers_
        self.benign_centroids /= np.linalg.norm(self.benign_centroids, axis=1, keepdims=True)

    def predict(self, embeddings, threshold=0.0):
        max_harmful_sim = np.max(embeddings @ self.harmful_centroids.T, axis=1)
        max_benign_sim = np.max(embeddings @ self.benign_centroids.T, axis=1)
        score = max_harmful_sim - max_benign_sim
        preds = (score > threshold).astype(int)
        return preds, score

    def save(self, path):
        np.savez(path,
                 harmful_centroids=self.harmful_centroids,
                 benign_centroids=self.benign_centroids)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        det = cls(
            n_harmful_centroids=data['harmful_centroids'].shape[0],
            n_benign_centroids=data['benign_centroids'].shape[0],
        )
        det.harmful_centroids = data['harmful_centroids']
        det.benign_centroids = data['benign_centroids']
        return det
