# baselines/dataset_early_fusion.py
# Dataset that returns 6-channel fused input (Ic || Im) for early-fusion baseline.
# Reuses PairedSegDataset; same augmentations apply to both camera and manual.

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    from dataset_pairs import PairedSegDataset
except ImportError:
    from ..dataset_pairs import PairedSegDataset


class EarlyFusionDataset(Dataset):
    """
    Wraps PairedSegDataset and returns fused 6-channel input: concat(cam, man) => (6, H, W).
    Keys: "fused" (6,H,W), "mask", "cam_path", "man_path", and optionally "cam", "man" for debugging.
    """
    def __init__(
        self,
        camera_root: str,
        images_camera_dirname: str = "camera",
        images_manual_dirname: str = "manual",
        mask_root_camera: Optional[str] = None,
        mask_root_manual: Optional[str] = None,
        img_exts: List[str] = [".jpg", ".jpeg", ".png"],
        img_size: int = 640,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mask_suffix: str = "_rle.json",
        return_cam_man: bool = False,
    ):
        self._base = PairedSegDataset(
            camera_root=camera_root,
            images_camera_dirname=images_camera_dirname,
            images_manual_dirname=images_manual_dirname,
            mask_root_camera=mask_root_camera,
            mask_root_manual=mask_root_manual,
            img_exts=img_exts,
            img_size=img_size,
            mean=mean,
            std=std,
            mask_suffix=mask_suffix,
        )
        self.return_cam_man = return_cam_man

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        out = self._base[idx]
        cam = out["cam"]   # (3, H, W)
        man = out["man"]   # (3, H, W)
        fused = torch.cat([cam, man], dim=0)  # (6, H, W)
        result = {
            "fused": fused,
            "mask": out["mask"],
            "cam_path": out["cam_path"],
            "man_path": out["man_path"],
        }
        if self.return_cam_man:
            result["cam"] = cam
            result["man"] = man
        return result
