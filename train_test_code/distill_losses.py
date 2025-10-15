# distill_losses.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

def _safe_name(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_]+', '_', s)

def _resize_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

def _inst_norm_like(x):
    return F.instance_norm(x)

# ---------------------------
# NEW: stats helpers
# ---------------------------
def _per_channel_stats(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (B,C,H,W) -> per-channel mean/std over H,W
    B, C, H, W = x.shape
    mu  = x.mean(dim=(2,3))                                # (B,C)
    var = x.var (dim=(2,3), unbiased=False)                # (B,C)
    std = torch.sqrt(var + eps)
    return mu, std

def _per_channel_stats_masked(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (B,C,H,W), m: (B,1 or C,H,W) in {0,1} weights
    if m.shape[1] == 1:
        m = m.expand(-1, x.shape[1], -1, -1)               # (B,C,H,W)
    wsum = m.sum(dim=(2,3)).clamp_min(eps)                 # (B,C)
    mu = (x * m).sum(dim=(2,3)) / wsum                     # (B,C)
    var = ((x - mu[...,None,None])**2 * m).sum(dim=(2,3)) / wsum
    std = torch.sqrt(var + eps)
    return mu, std

def channel_stat_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # detach teacher, instance-norm for scale stability
    t = t.detach()
    if s.shape[-2:] != t.shape[-2:]:
        t = _resize_to(t, s)
    sm, ss = _per_channel_stats(_inst_norm_like(s))
    tm, ts = _per_channel_stats(_inst_norm_like(t))
    return F.mse_loss(sm, tm) + F.mse_loss(ss, ts)

def region_stat_loss(s: torch.Tensor, t: torch.Tensor,
                     cam_mask: Optional[torch.Tensor], man_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # Region‑aware stats (foreground). If teacher manual mask is missing, use global stats on teacher.
    # cam_mask/man_mask expected shape (B,1,H,W) in {0,1} (can be float). We upsample to match s/t.
    t = t.detach()
    # upsample masks
    cm = None
    if cam_mask is not None:
        cm = (F.interpolate(cam_mask.float(), size=s.shape[-2:], mode="nearest") > 0.5).float()  # (B,1,h,w)
    mm = None
    if man_mask is not None:
        mm = (F.interpolate(man_mask.float(), size=t.shape[-2:], mode="nearest") > 0.5).float()

    s_n = _inst_norm_like(s)
    t_n = _inst_norm_like(t)

    if cm is not None:
        sm, ss = _per_channel_stats_masked(s_n, cm)
    else:
        sm, ss = _per_channel_stats(s_n)

    if mm is not None:
        tm, ts = _per_channel_stats_masked(t_n, mm)
    else:
        tm, ts = _per_channel_stats(t_n)

    return F.mse_loss(sm, tm) + F.mse_loss(ss, ts)

# ---------------------------
# Your existing modules
# ---------------------------
class FeatureAlign(nn.Module):
    def __init__(self, c_s, c_t):
        super().__init__()
        c = min(c_s, c_t)
        self.ps = nn.Conv2d(c_s, c, 1, bias=False)
        self.pt = nn.Conv2d(c_t, c, 1, bias=False)
    def forward(self, s, t):
        t = t.detach()
        if s.shape[-2:] != t.shape[-2:]:
            t = _resize_to(t, s)
        s_p = self.ps(s)
        t_p = self.pt(t)
        return F.mse_loss(_inst_norm_like(s_p), _inst_norm_like(t_p))

def spatial_attention(x, p=2):
    a = x.abs().pow(p).sum(1, keepdim=True)
    a = F.normalize(a.flatten(1), dim=1).view_as(a)
    return a

def attention_transfer_loss(s, t, p=2):
    t = t.detach()
    if s.shape[-2:] != t.shape[-2:]:
        t = _resize_to(t, s)
    return F.mse_loss(spatial_attention(s,p), spatial_attention(t,p))

class CrossAttentionAlign(nn.Module):
    def __init__(self, c_s, c_t, heads=4, dim_head=32, pool=1):
        super().__init__()
        self.h = heads
        self.d = dim_head
        self.pool = int(pool)
        hd = heads * dim_head
        self.q = nn.Conv2d(c_s, hd, 1, bias=False)
        self.k = nn.Conv2d(c_t, hd, 1, bias=False)
        self.v = nn.Conv2d(c_t, hd, 1, bias=False)
        self.proj = nn.Conv2d(hd, c_s, 1, bias=False)
        self.tts = nn.Conv2d(c_t, c_s, 1, bias=False)

    def forward(self, s, t):
        if s.shape[-2:] != t.shape[-2:]:
            t = _resize_to(t, s)
        if self.pool > 1:
            s_d = F.avg_pool2d(s, self.pool)
            t_d = F.avg_pool2d(t, self.pool).detach()
        else:
            s_d = s
            t_d = t.detach()

        B, Cs, H, W = s_d.shape
        N = H * W
        h, d = self.h, self.d

        q = self.q(s_d).view(B, h, d, N).transpose(2, 3)   # (B,h,N,d)
        k = self.k(t_d).view(B, h, d, N)                    # (B,h,d,N)
        v = self.v(t_d).view(B, h, d, N).transpose(2, 3)    # (B,h,N,d)

        attn = (q @ k) / math.sqrt(d)
        attn = attn.softmax(dim=-1)
        out  = attn @ v
        out  = out.transpose(2, 3).contiguous().view(B, h * d, H, W)
        out  = self.proj(out)
        tproj = self.tts(t_d)
        pred  = s_d + out

        pred  = _inst_norm_like(pred)
        tproj = _inst_norm_like(tproj)
        return pred, tproj

def cross_attention_align_loss(s, t, blk: CrossAttentionAlign):
    pred, target = blk(s, t)
    norm = pred.numel() / pred.shape[0]
    return F.mse_loss(pred, target, reduction="sum") / norm

# ---------------------------
# CFG + Distiller
# ---------------------------
@dataclass
class DistillCfg:
    lambda_feat: float
    lambda_at: float
    lambda_xattn: float
    xattn_heads: int
    xattn_dim_head: int
    xattn_pool: int = 1
    # NEW:
    lambda_stat: float = 0.0
    lambda_roi:  float = 0.0

class Distiller(nn.Module):
    def __init__(self, cfg: DistillCfg):
        super().__init__()
        self.cfg = cfg
        self.aligners = nn.ModuleDict()
        self.xblocks  = nn.ModuleDict()

    def _al(self, key, cs, ct):
        sk = _safe_name(key)
        if sk not in self.aligners:
            self.aligners[sk] = FeatureAlign(cs, ct)
        return self.aligners[sk]

    def _xb(self, key, cs, ct):
        sk = _safe_name(key)
        if sk not in self.xblocks:
            self.xblocks[sk] = CrossAttentionAlign(
                cs, ct,
                heads=self.cfg.xattn_heads,
                dim_head=self.cfg.xattn_dim_head,
                pool=getattr(self.cfg, "xattn_pool", 1),
            )
        return self.xblocks[sk]

    def compute(self,
                s_feats: Dict[str,torch.Tensor],
                t_feats: Dict[str,torch.Tensor],
                s_heads: Dict[str,torch.Tensor],
                t_heads: Dict[str,torch.Tensor],
                cam_mask: Optional[torch.Tensor] = None,
                man_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        L = {}

        # (1) feature alignment
        if self.cfg.lambda_feat > 0:
            acc = 0; n = 0
            for k in s_feats:
                if k in t_feats:
                    al = self._al("F:"+k, s_feats[k].shape[1], t_feats[k].shape[1]).to(s_feats[k].device)
                    acc = acc + al(s_feats[k], t_feats[k]); n += 1
            if n: L["feat"] = (acc / n) * self.cfg.lambda_feat

        # (2) attention transfer
        if self.cfg.lambda_at > 0:
            acc = 0; n = 0
            for k in s_feats:
                if k in t_feats:
                    acc = acc + attention_transfer_loss(s_feats[k], t_feats[k]); n += 1
            if n: L["attn"] = (acc / n) * self.cfg.lambda_at

        # (3) cross-attention on heads
        if self.cfg.lambda_xattn > 0:
            acc = 0; n = 0
            for k in s_heads:
                if k in t_heads:
                    xb = self._xb("X:"+k, s_heads[k].shape[1], t_heads[k].shape[1]).to(s_heads[k].device)
                    acc = acc + cross_attention_align_loss(s_heads[k], t_heads[k], xb); n += 1
            if n: L["xattn"] = (acc / n) * self.cfg.lambda_xattn

        # (4) NEW: channel-stat (position-invariant)
        if self.cfg.lambda_stat > 0:
            acc = 0; n = 0
            for k in s_feats:
                if k in t_feats:
                    acc = acc + channel_stat_loss(s_feats[k], t_feats[k]); n += 1
            if n: L["stat"] = (acc / n) * self.cfg.lambda_stat

        # (5) NEW: region-stat (mask-aware, object-centric)
        if self.cfg.lambda_roi > 0:
            acc = 0; n = 0
            for k in s_feats:
                if k in t_feats:
                    acc = acc + region_stat_loss(s_feats[k], t_feats[k], cam_mask, man_mask); n += 1
            if n: L["roi"] = (acc / n) * self.cfg.lambda_roi

        return L
