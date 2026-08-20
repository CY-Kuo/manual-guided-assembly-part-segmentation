"""Direct camera/manual fusion baseline without a frozen teacher or distillation."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from maps.aux_heads import AuxMaskHead
from maps.feature_hooks import FeatureHooker
from maps.fusion import StructureFusion


def _resolve_canonical_tap(model, anchor: str, img_size: int, device: torch.device) -> str:
    features: Dict[str, tuple] = {}

    def hook_factory(name):
        def hook(_, __, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                features[name] = tuple(output.shape)
        return hook

    handles = []
    for name, module in model.named_modules():
        if name == anchor or name.startswith(anchor + "."):
            handles.append(module.register_forward_hook(hook_factory(name)))
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        for handle in handles:
            handle.remove()

    if anchor in features:
        return anchor
    direct_children = {
        name: shape for name, shape in features.items()
        if name.startswith(anchor + ".") and name.count(".") == anchor.count(".") + 1
    }
    preferred = f"{anchor}.proto"
    if preferred in direct_children:
        return preferred
    if not direct_children:
        raise RuntimeError(f"Anchor '{anchor}' produced no 4D feature map")
    return sorted(
        direct_children.items(), key=lambda item: (-(item[1][2] * item[1][3]), item[0])
    )[0][0]


class DirectFusionBaseline(nn.Module):
    """Learnable camera/manual branches fused at the MAPS comparison taps."""

    def __init__(
        self,
        student_model_path: str,
        img_size: int = 640,
        aux_head_tap: str = "model.23",
        fusion_taps: Optional[List[str]] = None,
        fusion_dims: Optional[Dict[str, int]] = None,
        fusion_tokens: Optional[Dict[str, int]] = None,
        num_classes: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from ultralytics import YOLO

        dev = device or torch.device("cpu")
        self.aux_head_tap = aux_head_tap
        self.fusion_taps = fusion_taps or ["model.6", "model.10", "model.23"]
        self.fusion_dims = fusion_dims or {}
        self.fusion_tokens = fusion_tokens or {}

        self.camera_branch = YOLO(student_model_path).model.to(dev)
        self.manual_branch = copy.deepcopy(self.camera_branch)
        for parameter in self.manual_branch.parameters():
            parameter.requires_grad_(True)

        self.resolved = {
            tap: _resolve_canonical_tap(self.camera_branch, tap, img_size, dev)
            for tap in self.fusion_taps
        }
        resolved_names = list(self.resolved.values())
        self.camera_hook = FeatureHooker(self.camera_branch, resolved_names)
        self.manual_hook = FeatureHooker(self.manual_branch, resolved_names)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=dev)
            self.camera_branch(dummy)
            self.manual_branch(dummy)

        self.fusions = nn.ModuleDict()
        for tap, name in self.resolved.items():
            camera_feature = self.camera_hook.buffers[name]
            manual_feature = self.manual_hook.buffers[name]
            self.fusions[tap.replace(".", "_")] = StructureFusion(
                int(camera_feature.shape[1]),
                int(manual_feature.shape[1]),
                d=int(self.fusion_dims.get(tap, 128)),
                num_tokens=int(self.fusion_tokens.get(tap, 4)),
                use_film=True,
            )

        aux_name = self.resolved[self.aux_head_tap]
        aux_channels = int(self.camera_hook.buffers[aux_name].shape[1])
        self.seg_adapter = AuxMaskHead(aux_channels, num_classes=num_classes)
        self.camera_hook.clear()
        self.manual_hook.clear()

    def forward(
        self,
        camera: torch.Tensor,
        manual: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        self.camera_hook.clear()
        self.manual_hook.clear()
        self.camera_branch(camera)
        self.manual_branch(manual)

        fused_aux = None
        for tap, name in self.resolved.items():
            fused = self.fusions[tap.replace(".", "_")](
                self.camera_hook.buffers[name], self.manual_hook.buffers[name]
            )
            if tap == self.aux_head_tap:
                fused_aux = fused

        if fused_aux is None:
            raise RuntimeError("Auxiliary fusion tap was not produced")
        target_hw = out_hw or (camera.shape[2], camera.shape[3])
        return self.seg_adapter(fused_aux, target_hw)
