import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from loss import boundary_margin_loss


def test_zero_when_no_gray_benign():
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    label4 = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # 无 gray_benign(==2)
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() == 0.0


def test_penalizes_gray_benign_near_harmful():
    # gray_benign 紧贴 harmful、远离 benign -> loss>0
    h = torch.tensor([1.0, 0, 0, 0]); b = torch.tensor([0, 1.0, 0, 0])
    g_bad = h.clone()
    z = torch.stack([h, b, g_bad]); z = torch.nn.functional.normalize(z, dim=1)
    label4 = torch.tensor([1, 0, 2])  # harmful, benign, gray_benign
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() > 0.0


def test_zero_when_gray_benign_safely_on_benign_side():
    h = torch.tensor([1.0, 0, 0, 0]); b = torch.tensor([0, 1.0, 0, 0])
    g_good = b.clone()  # gray_benign 紧贴 benign
    z = torch.stack([h, b, g_good]); z = torch.nn.functional.normalize(z, dim=1)
    label4 = torch.tensor([1, 0, 2])
    out = boundary_margin_loss(z, label4, margin=0.2)
    assert out.item() == 0.0


if __name__ == "__main__":
    test_zero_when_no_gray_benign()
    test_penalizes_gray_benign_near_harmful()
    test_zero_when_gray_benign_safely_on_benign_side()
    print("ALL PASS")
