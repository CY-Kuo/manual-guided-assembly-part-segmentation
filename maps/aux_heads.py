import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxMaskHead(nn.Module):
    """
    A light decoder placed on a tapped feature map (student side).
    Predicts per-pixel mask logits for num_classes (1 = binary).
    """
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        mid = max(32, min(256, in_channels))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1)
        )
    def forward(self, x, out_hw):
        y = self.block(x)
        if y.shape[-2:] != out_hw:
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y  # (B,C,H,W)

def dice_loss(logits, target, eps=1e-6):
    # target: (B,1,H,W) in {0,1}
    probs = torch.sigmoid(logits)
    num = 2*(probs*target).sum(dim=(2,3)) + eps
    den = (probs.pow(2)+target.pow(2)).sum(dim=(2,3)) + eps
    loss = 1 - (num/den).mean()
    return loss

def bce_logits(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target)

