import os
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from rle_utils import load_rle_json

def _is_img(p: Path, exts: List[str]) -> bool:
    return p.suffix.lower() in {e.lower() for e in exts}

def _swap_last_dir(p: Path, src_name: str, dst_name: str) -> Path:
    parts = list(p.parts)
    idxs = [i for i, s in enumerate(parts) if s == src_name]
    if idxs:
        parts[idxs[-1]] = dst_name
    else:
        # If 'camera' not found in parts, replace parent name as a fallback
        parts[-2] = dst_name
    return Path(*parts)

class PairedSegDataset(Dataset):
    """
    Walks camera_root recursively; for each camera image:
      - manual image path = replace last folder 'camera' -> 'manual'
      - camera mask path  = mask_root_camera / <rel_to_camera_root> with name '<stem>_rle.json'
    """
    def __init__(
        self,
        camera_root: str,
        images_camera_dirname: str = "camera",
        images_manual_dirname: str = "manual",
        mask_root_camera: Optional[str] = None,
        mask_root_manual: Optional[str] = None,  # optional, not used by default
        img_exts: List[str] = [".jpg", ".jpeg", ".png"],
        img_size: int = 640,
        mean=(0,0,0),
        std=(1,1,1),
        mask_suffix: str = "_rle.json",
    ):
        self.camera_root = Path(camera_root).resolve()
        self.images_camera_dirname = images_camera_dirname
        self.images_manual_dirname = images_manual_dirname
        self.mask_root_camera = Path(mask_root_camera).resolve() if mask_root_camera else None
        self.mask_root_manual = Path(mask_root_manual).resolve() if mask_root_manual else None
        self.mask_suffix = mask_suffix

        self.img_paths: List[Tuple[Path, Path]] = []  # (cam_img, man_img)
        for cam in self.camera_root.rglob("*"):
            if cam.is_file() and _is_img(cam, img_exts):
                man = _swap_last_dir(cam, images_camera_dirname, images_manual_dirname)
                self.img_paths.append((cam, man))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found under {self.camera_root}")

        self.H = self.W = img_size
        self.tf_cam = T.Compose([T.Resize((self.H, self.W)), T.ToTensor(), T.Normalize(mean, std)])
        self.tf_man = T.Compose([T.Resize((self.H, self.W)), T.ToTensor(), T.Normalize(mean, std)])
        self.tf_mask = T.Compose([T.Resize((self.H, self.W), interpolation=T.InterpolationMode.NEAREST)])

        print(f"[Dataset] Found {len(self.img_paths)} camera images under {self.camera_root}")
        if self.mask_root_camera:
            print(f"[Dataset] Using camera masks from {self.mask_root_camera}")

    def _mask_cam_path(self, cam_path: Path) -> Optional[Path]:
        if self.mask_root_camera is None:
            return None
        rel = cam_path.relative_to(self.camera_root)            # e.g., '0000.jpg' or 'sub/0000.jpg'
        mask_name = f"{rel.stem}{self.mask_suffix}"             # '0000_rle.json'
        return (self.mask_root_camera / rel.parent / mask_name)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        cam_path, man_path = self.img_paths[idx]

        # images
        cam_img = Image.open(cam_path).convert("RGB")
        try:
            man_img = Image.open(man_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Dataset] WARN manual missing for {cam_path} -> expected {man_path}")
            man_img = cam_img

        cam_t = self.tf_cam(cam_img)
        man_t = self.tf_man(man_img)

        # camera mask
        mask = torch.zeros(1, self.H, self.W, dtype=torch.float32)
        mpath = self._mask_cam_path(cam_path)
        if mpath and mpath.exists():
            m_np = load_rle_json(str(mpath))  # (H0,W0) in {0,1} merged foreground
            if m_np is not None:
                m_pil = Image.fromarray((m_np*255).astype(np.uint8))
                m_rs = self.tf_mask(m_pil)  # PIL->Tensor NEAREST
                m_arr = np.array(m_rs)
                if m_arr.ndim == 3:
                    m_arr = m_arr[0]
                mask = torch.from_numpy((m_arr > 0).astype(np.float32)).unsqueeze(0)

        return {
            "cam": cam_t,
            "man": man_t,
            "cam_path": str(cam_path),
            "man_path": str(man_path),
            "mask": mask
        }
