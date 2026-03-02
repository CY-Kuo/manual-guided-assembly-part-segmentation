# baselines/direct_fusion_baseline.py
# Baseline 2: Direct Fusion (MAPS-matched taps + StructureFusion, end-to-end, no teacher/distillation).
# Manual encoder branch mirrors student backbone; same fusion locations (model.6, 10, 23) and module (StructureFusion).

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from train_ablation import StructureFusion, resolve_canonical_tap
    from feature_hooks import FeatureHooker
    from aux_heads import AuxMaskHead
except ImportError:
    from ..train_ablation import StructureFusion, resolve_canonical_tap
    from ..feature_hooks import FeatureHooker
    from ..aux_heads import AuxMaskHead


def _tap_safe_key(tap: str) -> str:
    """ModuleDict keys cannot contain '.'; use underscore."""
    return tap.replace(".", "_")


class DirectFusionBaseline(nn.Module):
    """
    Manual-aware baseline: same fusion taps and StructureFusion as MAPS, but manual
    features come from a learnable manual encoder (no frozen teacher, no distillation).
    """
    def __init__(
        self,
        student_model_path: str,
        img_size: int,
        aux_head_tap: str = "model.23",
        fusion_taps: Optional[List[str]] = None,
        fusion_dims: Optional[Dict[str, int]] = None,
        fusion_tokens: Optional[Dict[str, int]] = None,
        num_classes: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.aux_head_tap = aux_head_tap
        self.fusion_taps = fusion_taps or ["model.6", "model.10", "model.23"]
        self.fusion_dims = fusion_dims or {}
        self.fusion_tokens = fusion_tokens or {}
        self.num_classes = num_classes
        self.img_size = img_size
        dev = device or torch.device("cpu")

        self.camera_branch = YOLO(student_model_path).model.to(dev)
        self.manual_branch = copy.deepcopy(self.camera_branch)
        for p in self.manual_branch.parameters():
            p.requires_grad_(True)

        _tmp = YOLO(student_model_path).model.to(dev)
        resolved_cam: Dict[str, str] = {}
        for tap in self.fusion_taps:
            try:
                resolved_cam[tap] = resolve_canonical_tap(_tmp, tap, img_size, prefer=("proto",), device=dev)
            except Exception:
                pass
        del _tmp

        self.resolved = resolved_cam
        self.s_hook = FeatureHooker(self.camera_branch, list(resolved_cam.values()), use_regex=False)
        self.m_hook = FeatureHooker(self.manual_branch, list(resolved_cam.values()), use_regex=False)

        with torch.no_grad():
            _ = self.camera_branch(torch.zeros(1, 3, img_size, img_size, device=dev))
            _ = self.manual_branch(torch.zeros(1, 3, img_size, img_size, device=dev))
        s_channels = {tap: self.s_hook.buffers[resolved_cam[tap]].shape[1] for tap in self.fusion_taps if tap in resolved_cam}
        self.fusions = nn.ModuleDict()
        self.tap_to_safe_name = {}  # Map "model.6" -> "model_6" for ModuleDict keys
        for tap in self.fusion_taps:
            if tap not in resolved_cam:
                continue
            safe_key = _tap_safe_key(tap)
            self.tap_to_safe_name[tap] = safe_key
            c_s = s_channels[tap]
            d = self.fusion_dims.get(tap, 128)
            nt = self.fusion_tokens.get(tap, 4)
            self.fusions[safe_key] = StructureFusion(c_s, c_s, d=d, num_tokens=nt, use_film=True)
        s_aux_name = resolved_cam.get(aux_head_tap)
        if s_aux_name is None:
            raise RuntimeError(f"aux_head_tap {aux_head_tap} not in resolved taps")
        Cs = self.s_hook.buffers[s_aux_name].shape[1]
        self.seg_adapter = AuxMaskHead(in_channels=Cs, num_classes=num_classes)
        self.s_hook.clear()
        self.m_hook.clear()

    def forward(
        self,
        cam: torch.Tensor,
        man: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        cam: (B, 3, H, W), man: (B, 3, H, W).
        Returns seg logits (B, num_classes, out_H, out_W).
        """
        self.s_hook.clear()
        self.m_hook.clear()
        _ = self.camera_branch(cam)
        _ = self.manual_branch(man)
        fused_aux = None
        for tap in self.fusion_taps:
            if tap not in self.resolved or tap not in self.tap_to_safe_name:
                continue
            safe_key = self.tap_to_safe_name[tap]
            if safe_key not in self.fusions:
                continue
            s_name = self.resolved[tap]
            s_feat = self.s_hook.buffers.get(s_name)
            m_feat = self.m_hook.buffers.get(s_name)
            if s_feat is None or m_feat is None:
                continue
            fused = self.fusions[safe_key](s_feat, m_feat)
            if tap == self.aux_head_tap:
                fused_aux = fused
        if fused_aux is None:
            s_name = self.resolved.get(self.aux_head_tap)
            fused_aux = self.s_hook.buffers.get(s_name)
        if fused_aux is None:
            raise RuntimeError("No fused feature for aux head")
        if out_hw is None:
            out_hw = (cam.shape[2], cam.shape[3])
        return self.seg_adapter(fused_aux, out_hw=out_hw)
