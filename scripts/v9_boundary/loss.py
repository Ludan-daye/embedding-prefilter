#!/usr/bin/env python3
"""边界感知损失：把 gray_benign 从 harmful 一侧推回 benign 一侧。"""
import torch


def boundary_margin_loss(z, label4, margin=0.2):
    """
    z: [B, D] L2-normalized
    label4: [B] 4 类 (0=benign,1=harmful,2=gray_benign,3=gray_harmful)
    对每个 gray_benign 锚点 g:
        sim_h* = max_{harmful h} z_g·z_h    (最难有害负样本)
        sim_cb = z_g · c_b                  (c_b = 批内 benign 类原型, 归一化均值)
        L = relu(margin + sim_h* - sim_cb)
    无 gray_benign 或无 harmful 或无 benign 时返回 0。
    """
    device = z.device
    if not torch.is_tensor(label4):
        label4 = torch.as_tensor(label4, device=device)
    gb = (label4 == 2)
    harmful = (label4 == 1)
    benign = (label4 == 0)
    if int(gb.sum()) == 0 or int(harmful.sum()) == 0 or int(benign.sum()) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    zg = z[gb]                                   # [G, D]
    zh = z[harmful]                              # [H, D]
    c_b = z[benign].mean(dim=0)
    c_b = c_b / (c_b.norm() + 1e-8)              # benign 原型
    sim_h = (zg @ zh.T).max(dim=1).values        # [G] 最难有害负样本
    sim_cb = zg @ c_b                            # [G]
    loss = torch.relu(margin + sim_h - sim_cb)
    return loss.mean()
