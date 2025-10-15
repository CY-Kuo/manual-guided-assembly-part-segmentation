# train_dualinput_ablate.py
# Dual-input manual-guided mask generator with ablation runner over fusion combos.
# - Per-combo folders under cfg.out_dir/{ablation_name}/  (metrics.csv + ckpts)
# - Visualizations disabled (no training/fixed viz)
# - Eval mimics training pipeline (same fusion switches)
# - Checkpoints include exact resolved layers used for fusion

import os, yaml, csv, json, re, dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

# project-local modules
from dataset_pairs import PairedSegDataset
from feature_hooks import FeatureHooker
from distill_losses import DistillCfg, Distiller
from aux_heads import AuxMaskHead, dice_loss  # BCE handled below

try:
    from pycocotools import mask as mask_utils  # for instance GT decode
except Exception:
    mask_utils = None


# -------------------------
# Config loading & helpers
# -------------------------
def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _safe_name(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]+', '_', name)

def get_ultralytics_version():
    try:
        import ultralytics as _u
        return getattr(_u, "__version__", "unknown")
    except Exception:
        return "unknown"

def resolve_canonical_tap(model, anchor: str, img_size: int, prefer: Tuple[str,...]=("proto",), device=None) -> str:
    """
    Resolve a logical anchor (e.g., 'model.23') to a concrete 4D tensor-producing leaf on THIS model.
    """
    device = device or next(model.parameters()).device

    feats: Dict[str, Tuple[int,int,int,int]] = {}
    def hook_factory(n):
        def fn(_, __, out):
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                feats[n] = tuple(out.shape)  # (B,C,H,W)
        return fn

    handles = []
    for n, m in model.named_modules():
        if n == anchor or n.startswith(anchor + "."):
            handles.append(m.register_forward_hook(hook_factory(n)))

    try:
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        for h in handles:
            h.remove()

    if anchor in feats:
        return anchor

    base_dots = anchor.count(".")
    children = {n: sh for n, sh in feats.items()
                if n.startswith(anchor + ".") and n.count(".") == base_dots + 1}
    if not children:
        raise RuntimeError(
            f"[resolve_canonical_tap] Anchor '{anchor}' produced no 4D outputs on this model."
        )

    for p in prefer:
        cand = f"{anchor}.{p}"
        if cand in children:
            return cand

    def area(sh): return sh[2] * sh[3]
    best = sorted(children.items(), key=lambda kv: (-area(kv[1]), kv[0]))[0][0]
    return best


# -------------------------
# Experiment config
# -------------------------
@dataclass
class Cfg:
    # models
    student_model: str
    teacher_model: str

    # images + masks (train)
    camera_root: str
    images_camera_dirname: str
    images_manual_dirname: str
    mask_root_camera: Optional[str]
    mask_root_manual: Optional[str]
    img_exts: List[str]
    mask_suffix: str

    # preprocessing
    img_size: int
    num_classes: int
    mean: List[float]
    std: List[float]

    # taps (kept for YAML-compat; we resolve at runtime)
    distill_student_layers: List[str]
    distill_teacher_layers: List[str]
    head_student_layers: List[str]
    head_teacher_layers: List[str]
    aux_head_tap: str  # e.g., "model.23"

    # loss weights
    lambda_aux: float
    lambda_feat: float
    lambda_at: float
    lambda_xattn: float
    lambda_stat: float
    lambda_roi:  float

    # xattn hypers
    xattn: Dict[str, int]  # heads, dim_head, pool

    # optim
    epochs: int
    batch_size: int
    workers: int
    lr: float
    weight_decay: float
    amp: bool
    device: int

    # io
    out_dir: str
    save_every: int

    # ----- EVAL -----
    val_camera_root: Optional[str] = None
    val_mask_root_camera: Optional[str] = None
    eval_every: int = 0
    eval_max_images: Optional[int] = None
    eval_conf_thres: float = 0.25
    eval_iou_nms: float = 0.7

    # per-tap fusion hyperparams
    fusion_dims: Optional[Dict[str, int]] = None
    fusion_tokens: Optional[Dict[str, int]] = None
    lambda_fuse_reg: Optional[Dict[str, float]] = None

    # ablations to run (None -> run all 7)
    ablations: Optional[List[str]] = None

    # optional: restrict distillation taps to exactly the fused mids
    align_distill_to_fusion: bool = True


# -------------------------
# Ablation utilities
# -------------------------
@dataclass
class AblationFlags:
    name: str
    use_fuse_aux: bool
    use_fuse_10: bool
    use_fuse_6: bool

ABLATION_GRID: List[AblationFlags] = [
    #AblationFlags(name="fuse_6_only",    use_fuse_aux=False, use_fuse_10=False, use_fuse_6=True),
    #AblationFlags(name="fuse_10_only",   use_fuse_aux=False, use_fuse_10=True,  use_fuse_6=False),
    #AblationFlags(name="fuse_6_10",      use_fuse_aux=False, use_fuse_10=True,  use_fuse_6=True),
    AblationFlags(name="fuse_aux_only",  use_fuse_aux=True,  use_fuse_10=False, use_fuse_6=False),
    AblationFlags(name="fuse_6_aux",     use_fuse_aux=True,  use_fuse_10=False, use_fuse_6=True),
    AblationFlags(name="fuse_10_aux",    use_fuse_aux=True,  use_fuse_10=True,  use_fuse_6=False),
    AblationFlags(name="fuse_6_10_aux",  use_fuse_aux=True,  use_fuse_10=True,  use_fuse_6=True),
    AblationFlags(name="base",           use_fuse_aux=False,  use_fuse_10=False,  use_fuse_6=False),
]


# -------------------------
# Small helpers
# -------------------------
def to_device(x, device):
    return x.to(device, non_blocking=True)

def bce_logits_weighted(logits, target, w_bg=1.2, w_fg=1.0):
    w = torch.where(target > 0.5,
                    torch.as_tensor(w_fg, device=logits.device),
                    torch.as_tensor(w_bg, device=logits.device))
    return F.binary_cross_entropy_with_logits(logits, target, weight=w)

def _ensure_one_hot(target, num_classes):
    if target.ndim == 3:
        B,H,W = target.shape
        oh = torch.zeros(B, num_classes, H, W, device=target.device, dtype=torch.float32)
        oh.scatter_(1, target.unsqueeze(1).long(), 1.0)
        return oh
    elif target.ndim == 4 and target.shape[1] == num_classes:
        return target.float()
    else:
        raise ValueError(f"mask_multi must be (B,H,W) or (B,{num_classes},H,W), got {tuple(target.shape)}")

def soft_dice_loss_multiclass(logits, one_hot, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    probs_fg = probs[:, 1:, :, :]
    onehot_fg = one_hot[:, 1:, :, :]
    num = 2 * (probs_fg * onehot_fg).sum(dim=(2,3)) + eps
    den = (probs_fg.pow(2) + onehot_fg.pow(2)).sum(dim=(2,3)) + eps
    return (1.0 - (num / den)).mean()


# -------------------------
# StructureFusion
# -------------------------
class StructureFusion(nn.Module):
    def __init__(self, s_channels: int, t_channels: int, d: int = 128, num_tokens: int = 4, use_film: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.d = d
        self.use_film = use_film

        self.t_gap = nn.AdaptiveAvgPool2d(1)
        self.t_token = nn.Sequential(
            nn.Conv2d(t_channels, d * num_tokens, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.q_proj = nn.Conv2d(s_channels, d, 1, bias=True)
        self.out_proj = nn.Conv2d(d, s_channels, 1, bias=True)

        if use_film:
            self.film = nn.Sequential(
                nn.Conv2d(t_channels, s_channels * 2, 1, bias=True)
            )
        self.temperature = d ** 0.5

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        B, Cs, H, W = s_feat.shape
        t_g = self.t_gap(t_feat)
        tokens = self.t_token(t_g).view(B, self.num_tokens, self.d)

        q = self.q_proj(s_feat)
        q_flat = q.view(B, self.d, H * W).transpose(1, 2)

        k = tokens
        v = tokens

        attn = torch.softmax(torch.matmul(q_flat, k.transpose(1, 2)) / self.temperature, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).view(B, self.d, H, W)
        out = self.out_proj(out)

        fused = s_feat + out
        if self.use_film:
            gamma_beta = self.film(t_g)
            gamma, beta = gamma_beta.split(Cs, dim=1)
            fused = fused * torch.sigmoid(gamma) + beta
        return fused


# -------------------------
# EVAL UTILITIES
# -------------------------
def _prep_image(path: str, size=640):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return x

def _decode_instances_from_json(json_path: str) -> List[np.ndarray]:
    if mask_utils is None:
        return []
    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            data = yaml.safe_load(f)
    items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
    masks = []
    for it in items:
        rle_like = it.get("segmentation", it) if isinstance(it, dict) else None
        if not isinstance(rle_like, dict):
            continue
        if "counts" in rle_like and "size" in rle_like:
            counts = rle_like["counts"]; size = rle_like["size"]
            rle = {"counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
                   "size": [int(size[0]), int(size[1])]}
            m = mask_utils.decode(rle)
            if m.ndim == 3: m = m[...,0]
            masks.append((m>0).astype(np.uint8))
    return masks

def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(inter) / float(union)

def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    r = np.linspace(0, 1, 101)
    p_env = np.maximum.accumulate(precisions[::-1])[::-1]
    out = []
    for rv in r:
        inds = np.where(recalls >= rv)[0]
        out.append(p_env[inds[0]] if inds.size > 0 else 0.0)
    return float(np.mean(out))

def _eval_map(preds: List[Dict], gts: Dict[int, List[np.ndarray]], iou_thresholds=np.arange(0.5, 0.96, 0.05)) -> Tuple[float,float]:
    preds = sorted(preds, key=lambda x: -x["score"])
    total_gts = sum(len(v) for v in gts.values())
    aps = {}
    for thr in iou_thresholds:
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        matched = {img_id: np.zeros(len(gts[img_id]), dtype=bool) for img_id in gts.keys()}
        for i, p in enumerate(preds):
            img_id = p["img_id"]
            gt_masks = gts.get(img_id, [])
            if len(gt_masks) == 0:
                fp[i] = 1.0
                continue
            ious = np.array([_mask_iou(p["mask"], g) if not matched[img_id][j] else -1.0
                             for j, g in enumerate(gt_masks)], dtype=np.float32)
            j_best = int(ious.argmax()) if ious.size > 0 else -1
            best = float(ious[j_best]) if j_best >= 0 else 0.0
            if j_best >= 0 and best >= thr and not matched[img_id][j_best]:
                tp[i] = 1.0
                matched[img_id][j_best] = True
            else:
                fp[i] = 1.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(1, total_gts)
        precisions = tp_cum / np.maximum(1, (tp_cum + fp_cum))
        aps[thr] = _compute_ap(recalls, precisions)
    map_50_95 = float(np.mean(list(aps.values()))) if aps else 0.0
    map_50 = float(aps.get(0.5, 0.0))
    return map_50, map_50_95

def _dice_iou(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps=1e-6) -> Tuple[float,float]:
    inter = (pred_bin * gt_bin).sum().item()
    p_sum = pred_bin.sum().item()
    t_sum = gt_bin.sum().item()
    union = p_sum + t_sum - inter
    dice = (2*inter + eps) / (p_sum + t_sum + eps)
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)

def _resize_bool_mask(mask_np: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    H, W = out_hw
    if mask_np.shape == (H, W):
        return mask_np.astype(bool)
    m = (mask_np > 0.5).astype(np.uint8) * 255
    pil = Image.fromarray(m)
    pil = pil.resize((W, H), Image.NEAREST)
    return (np.array(pil) > 127)

def _paired_manual_path(cam_path: Path, camera_root: Path, manual_dirname: str) -> Optional[Path]:
    try:
        rel = cam_path.resolve().relative_to(camera_root.resolve())
    except Exception:
        rel = Path(os.path.basename(cam_path))
    manual_root = camera_root.resolve().parent / manual_dirname
    man_path = manual_root / rel
    return man_path if man_path.exists() else None


# -------------------------
# Eval (mirrors ablation switches)
# -------------------------
def run_eval(cfg: Cfg, student_state: Dict[str, torch.Tensor], device: torch.device, out_dir: str, epoch: int,
             seg_adapter_state: Optional[Dict[str, torch.Tensor]] = None,
             fusions_state: Optional[Dict[str, torch.Tensor]] = None,
             distiller_state: Optional[Dict[str, torch.Tensor]] = None,
             fusion_meta: Optional[Dict] = None,
             ablate: Optional[AblationFlags] = None) -> Dict[str, float]:

    name_map = (fusion_meta or {}).get("name_map", {}) or {}
    if not name_map:
        raise RuntimeError("[Eval] Missing fusion_meta.name_map in checkpoint.")

    aux_anchor = cfg.aux_head_tap
    aux_key = _safe_name(aux_anchor)
    if aux_key not in name_map:
        found = [k for k,v in name_map.items() if isinstance(v, dict) and v.get("anchor") == aux_anchor]
        if not found:
            raise RuntimeError(f"[Eval] fusion_meta has no entry for aux anchor '{aux_anchor}'.")
        aux_key = found[0]
    aux_entry = name_map[aux_key]
    s_aux = aux_entry["student"]
    t_aux = aux_entry["teacher"]

    # Which mids to distill during eval?
    if getattr(cfg, "align_distill_to_fusion", False) and (ablate is not None):
        mids_for_distill = []
        if ablate.use_fuse_10: mids_for_distill.append("model.10")
        if ablate.use_fuse_6:  mids_for_distill.append("model.6")
    else:
        mids_for_distill = ["model.10", "model.6"]

    # Filter name_map mids_present to only those mids
    mids_present = []
    for a in mids_for_distill:
        k = _safe_name(a)
        if k in name_map:
            mids_present.append((a, k, name_map[k]["student"], name_map[k]["teacher"]))

    if not cfg.val_camera_root or not cfg.val_mask_root_camera:
        print("[Eval] val_camera_root / val_mask_root_camera not set; skipping.")
        return {}

    camera_root = Path(cfg.val_camera_root)
    mask_root = Path(cfg.val_mask_root_camera)
    if not camera_root.exists():
        print(f"[Eval] val_camera_root not found: {camera_root} ; skipping.")
        return {}

    img_paths = [p for p in camera_root.rglob("*") if p.suffix.lower() in {e.lower() for e in cfg.img_exts}]
    img_paths = sorted(img_paths)
    if cfg.eval_max_images is not None:
        img_paths = img_paths[:cfg.eval_max_images]
    if not img_paths:
        print(f"[Eval] no images under {camera_root}")
        return {}

    # student
    yolo_eval = YOLO(cfg.student_model)
    yolo_eval.model.load_state_dict(student_state, strict=False)
    yolo_eval.model.to(device).eval()

    # teacher (frozen)
    teacher_eval = YOLO(cfg.teacher_model).model.to(device).eval()
    for p in teacher_eval.parameters():
        p.requires_grad_(False)

    s_feat_layers = [s for (_, _, s, _) in mids_present]
    t_feat_layers = [t for (_, _, _, t) in mids_present]
    s_feat_hook = FeatureHooker(yolo_eval.model, s_feat_layers, use_regex=False)
    s_head_hook = FeatureHooker(yolo_eval.model, [s_aux],    use_regex=False)
    t_feat_hook = FeatureHooker(teacher_eval,   t_feat_layers, use_regex=False)
    t_head_hook = FeatureHooker(teacher_eval,   [t_aux],     use_regex=False)

    distiller_eval = Distiller(DistillCfg(
        lambda_feat=cfg.lambda_feat,
        lambda_at=cfg.lambda_at,
        lambda_xattn=cfg.lambda_xattn,
        xattn_heads=cfg.xattn.get("heads", 2),
        xattn_dim_head=cfg.xattn.get("dim_head", 16),
        xattn_pool=cfg.xattn.get("pool", 1),
        lambda_stat=getattr(cfg, "lambda_stat", 0.0),
        lambda_roi =getattr(cfg, "lambda_roi",  0.0),
    )).to(device).eval()
    if distiller_state is not None:
        try:
            distiller_eval.load_state_dict(distiller_state, strict=False)
        except Exception as e:
            print(f"[Eval] distiller load warning: {e}")

    with torch.no_grad():
        _ = yolo_eval.model(torch.zeros(1,3,cfg.img_size,cfg.img_size, device=device))
        _ = teacher_eval(torch.zeros(1,3,cfg.img_size,cfg.img_size, device=device))

    seg_adapter_eval = None
    seg_sample_s = s_head_hook.buffers.get(s_aux, None)
    seg_sample_t = t_head_hook.buffers.get(t_aux, None)
    if isinstance(seg_sample_s, torch.Tensor) and isinstance(seg_sample_t, torch.Tensor):
        Cs = int(seg_sample_s.shape[1])
        seg_adapter_eval = AuxMaskHead(in_channels=Cs, num_classes=cfg.num_classes).to(device).eval()
        if seg_adapter_state is not None:
            try:
                seg_adapter_eval.load_state_dict(seg_adapter_state, strict=False)
            except Exception as e:
                print(f"[Eval] aux head load warning: {e}")
    else:
        raise RuntimeError("[Eval] Resolved aux taps did not produce 4D tensors — cannot eval.")

    fusions_eval = nn.ModuleDict()
    def _get_fusion_for(key: str, s_tensor: torch.Tensor, t_tensor: torch.Tensor):
        if key not in fusions_eval:
            anchor = name_map[key]["anchor"]
            d = (fusion_meta.get("dims", {}) or {}).get(anchor, 128)
            k_tok = (fusion_meta.get("tokens", {}) or {}).get(anchor, 4)
            fusions_eval[key] = StructureFusion(
                s_channels=int(s_tensor.shape[1]),
                t_channels=int(t_tensor.shape[1]),
                d=int(d), num_tokens=int(k_tok), use_film=True
            ).to(device).eval()
        return fusions_eval[key]

    if fusions_state:
        try:
            fusions_eval.load_state_dict(fusions_state, strict=False)
        except Exception as e:
            print(f"[Eval] WARN: could not pre-load fusion weights: {e}")

    dice_list, iou_list = [], []
    gt_instances_by_img: Dict[int, List[np.ndarray]] = {}
    predictions: List[Dict] = []
    d_totals = {"feat":0.0, "attn":0.0, "xattn":0.0, "stat":0.0, "roi":0.0}
    n_d = 0

    with torch.no_grad():
        for img_id, cam_path in enumerate(img_paths):
            mjson = mask_root / cam_path.relative_to(camera_root).parent / (cam_path.stem + cfg.mask_suffix)
            if not mjson.exists():
                continue
            if mask_utils is None:
                continue

            try:
                with open(mjson, "r") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
                merged = None
                for it in items:
                    r = it.get("segmentation", it)
                    if isinstance(r, dict) and "counts" in r and "size" in r:
                        rle = {"counts": r["counts"].encode("utf-8") if isinstance(r["counts"], str) else r["counts"],
                               "size": [int(r["size"][0]), int(r["size"][1])]}
                        m = mask_utils.decode(rle)
                        if m.ndim == 3: m = m[...,0]
                        m = (m>0).astype(np.uint8)
                        merged = m if merged is None else np.maximum(merged, m)
                if merged is None: continue
                gt_merged = merged
            except Exception:
                continue

            H0, W0 = gt_merged.shape
            gt_bin_t = torch.from_numpy(gt_merged.astype(np.float32)).view(1,1,H0,W0).to(device)

            gt_instances = _decode_instances_from_json(str(mjson))
            gt_instances_by_img[img_id] = [(g>0).astype(bool) for g in gt_instances]

            man_path = _paired_manual_path(cam_path, camera_root, cfg.images_manual_dirname)
            if man_path is None:
                continue

            x_cam = _prep_image(str(cam_path), cfg.img_size).to(device)
            x_man = _prep_image(str(man_path), cfg.img_size).to(device)

            s_feat_hook.clear(); s_head_hook.clear()
            t_feat_hook.clear(); t_head_hook.clear()
            _ = yolo_eval.model(x_cam)
            _ = teacher_eval(x_man)

            # --- Aux path: honor ablation flag ---
            s23 = s_head_hook.buffers[s_aux]
            t23 = t_head_hook.buffers[t_aux]
            key23 = _safe_name(cfg.aux_head_tap)

            if (ablate is None) or ablate.use_fuse_aux:
                f23 = _get_fusion_for(key23, s23, t23)
                s23_for_head = f23(s23, t23)
            else:
                s23_for_head = s23

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg.amp)):
                pred_logits = seg_adapter_eval(s23_for_head, out_hw=(H0, W0))

            if cfg.num_classes == 1 or pred_logits.shape[1] == 1:
                pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
                gate_logits = pred_logits
            else:
                fg_logits = pred_logits[:, 1:, :, :].amax(dim=1, keepdim=True)
                pred_bin = (torch.sigmoid(fg_logits) > 0.5).float()
                gate_logits = fg_logits

            d, j = _dice_iou(pred_bin, gt_bin_t)
            dice_list.append(d); iou_list.append(j)

            # --- Distill stats: fuse mids only if enabled ---
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda" and cfg.amp)):
                p = torch.sigmoid(gate_logits)
                hard_cover = (p > 0.30).float().mean().item()
                if hard_cover < 0.02: gate = p
                elif hard_cover <= 0.40: gate = (p > 0.10).float()
                else: gate = (p > 0.30).float()

                def _apply_gate(feat_dict):
                    out = {}
                    for k, v in feat_dict.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 4:
                            g = F.interpolate(gate, size=v.shape[-2:], mode="nearest")
                            out[k] = v * g
                        else:
                            out[k] = v
                    return out

                s_feats_mod = dict(s_feat_hook.buffers)
                s_heads_mod = dict(s_head_hook.buffers)
                s_heads_mod[s_aux] = s23_for_head

                for a, k, s_name, t_name in mids_present:
                    if a == "model.10" and ablate and not ablate.use_fuse_10:
                        continue
                    if a == "model.6"  and ablate and not ablate.use_fuse_6:
                        continue
                    if (s_name in s_feats_mod) and (t_name in t_feat_hook.buffers):
                        s_mid = s_feats_mod[s_name]
                        t_mid = t_feat_hook.buffers[t_name]
                        f_mid = _get_fusion_for(k, s_mid, t_mid)
                        s_mid_fused = f_mid(s_mid, t_mid)
                        s_feats_mod[s_name] = s_mid_fused

                s_feats_g = _apply_gate(s_feats_mod)
                s_heads_g = _apply_gate(s_heads_mod)

                extra = distiller_eval.compute(
                    s_feats=s_feats_g,
                    t_feats=dict(t_feat_hook.buffers),
                    s_heads=s_heads_g,
                    t_heads=dict(t_head_hook.buffers),
                    cam_mask=gt_bin_t,
                    man_mask=None
                )

            for k in ["feat","attn","xattn","stat","roi"]:
                if k in extra and isinstance(extra[k], torch.Tensor):
                    d_totals[k] += float(extra[k].detach().cpu())
            n_d += 1

            # YOLO instance AP (student-only)
            res = yolo_eval.predict(
                source=str(cam_path), imgsz=cfg.img_size, conf=cfg.eval_conf_thres,
                iou=cfg.eval_iou_nms, device=device.index if device.type=="cuda" else None, verbose=False
            )[0]
            if getattr(res, "masks", None) is not None and res.masks is not None:
                pmasks = res.masks.data.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy() if getattr(res, "boxes", None) is not None else np.ones(len(pmasks))
                if pmasks.ndim == 3:
                    for k in range(pmasks.shape[0]):
                        mk = (pmasks[k] > 0.5)
                        mk_bool = _resize_bool_mask(mk, (H0, W0))
                        predictions.append({"img_id": img_id, "score": float(scores[k]), "mask": mk_bool})

    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    mean_iou  = float(np.mean(iou_list))  if iou_list else 0.0
    ap50, map_5095 = _eval_map(predictions, gt_instances_by_img)
    distill_means = {k: (d_totals[k]/max(1,n_d)) for k in d_totals}

    print("\n====== VAL (Dual-input: ablation-matched pipeline) ======")
    print(f"Epoch {epoch} — Images: {len(gt_instances_by_img)}  (paired samples: {n_d})")
    print(f"Aux (image-wise):  Dice={mean_dice:.4f}  IoU={mean_iou:.4f}")
    print(f"Seg head (instance):  AP50={ap50:.4f}  mAP50-95={map_5095:.4f}")
    print(f"Distill@eval (means):  feat={distill_means['feat']:.4f}  attn={distill_means['attn']:.4f}  "
          f"xattn={distill_means['xattn']:.4f}  stat={distill_means['stat']:.4f}  roi={distill_means['roi']:.4f}")
    print("==============================================================\n")

    s_feat_hook.remove(); s_head_hook.remove()
    t_feat_hook.remove(); t_head_hook.remove()

    return {"dice":mean_dice, "iou":mean_iou, "ap50":ap50, "map5095":map_5095,
            "d_feat":distill_means["feat"], "d_attn":distill_means["attn"],
            "d_xattn":distill_means["xattn"], "d_stat":distill_means["stat"], "d_roi":distill_means["roi"]}


# -------------------------
# Single ablation training
# -------------------------
def run_one_training(cfg: Cfg, ablate: AblationFlags, out_dir: str):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    # ----- DATA -----
    ds = PairedSegDataset(
        camera_root=cfg.camera_root,
        images_camera_dirname=cfg.images_camera_dirname,
        images_manual_dirname=cfg.images_manual_dirname,
        mask_root_camera=cfg.mask_root_camera,
        mask_root_manual=cfg.mask_root_manual,
        img_exts=cfg.img_exts,
        img_size=cfg.img_size,
        mean=tuple(cfg.mean),
        std=tuple(cfg.std),
        mask_suffix=cfg.mask_suffix
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)

    # ----- MODELS -----
    student = YOLO(cfg.student_model).model.to(device).train()
    teacher = YOLO(cfg.teacher_model).model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ----- TAP RESOLUTION (resolve concrete 4D nodes) -----
    aux_anchor = cfg.aux_head_tap
    resolved = {"aux": {}, "mid": {"student": {}, "teacher": {}}}

    # Resolve aux tap on student & teacher (fail fast if missing)
    resolved["aux"]["student"] = resolve_canonical_tap(
        student, aux_anchor, cfg.img_size, prefer=("proto",), device=device
    )
    resolved["aux"]["teacher"] = resolve_canonical_tap(
        teacher, aux_anchor, cfg.img_size, prefer=("proto",), device=device
    )

    # Decide which mids to use for distillation (and thus which hooks to attach)
    if getattr(cfg, "align_distill_to_fusion", False):
        mids_for_distill = []
        if ablate.use_fuse_10: mids_for_distill.append("model.10")
        if ablate.use_fuse_6:  mids_for_distill.append("model.6")
    else:
        mids_for_distill = ["model.10", "model.6"]  # original behavior

    # Resolve only the chosen mids (student & teacher)
    for a in mids_for_distill:
        try:
            s_name = resolve_canonical_tap(student, a, cfg.img_size, device=device)
            t_name = resolve_canonical_tap(teacher, a, cfg.img_size, device=device)
            resolved["mid"]["student"][a] = s_name
            resolved["mid"]["teacher"][a] = t_name
        except Exception as e:
            print(f"[Resolve] WARN: could not resolve '{a}': {e}")

    # Build hook layer lists using resolved names
    head_student_layers = [resolved["aux"]["student"]]
    head_teacher_layers = [resolved["aux"]["teacher"]]
    distill_student_layers = [resolved["mid"]["student"][a] for a in mids_for_distill if a in resolved["mid"]["student"]]
    distill_teacher_layers = [resolved["mid"]["teacher"][a] for a in mids_for_distill if a in resolved["mid"]["teacher"]]

    # Hooks
    s_feat_hook = FeatureHooker(student, distill_student_layers, use_regex=False)
    s_head_hook = FeatureHooker(student, head_student_layers,    use_regex=False)
    t_feat_hook = FeatureHooker(teacher, distill_teacher_layers, use_regex=False)
    t_head_hook = FeatureHooker(teacher, head_teacher_layers,    use_regex=False)

    # exact resolved names per anchor
    fusion_name_map: Dict[str, Dict[str, str]] = {}
    fusion_name_map[_safe_name(aux_anchor)] = {
        "anchor": aux_anchor,
        "student": resolved["aux"]["student"],
        "teacher": resolved["aux"]["teacher"],
    }
    for a in mids_for_distill:
        if a in resolved["mid"]["student"] and a in resolved["mid"]["teacher"]:
            fusion_name_map[_safe_name(a)] = {
                "anchor": a,
                "student": resolved["mid"]["student"][a],
                "teacher": resolved["mid"]["teacher"][a],
            }

    aux_key = _safe_name(aux_anchor)
    s_aux_name = fusion_name_map[aux_key]["student"]
    t_aux_name = fusion_name_map[aux_key]["teacher"]

    seg_adapter: Optional[nn.Module] = None
    fusions = nn.ModuleDict()  # lazily created where enabled

    distiller = Distiller(DistillCfg(
        lambda_feat=cfg.lambda_feat,
        lambda_at=cfg.lambda_at,
        lambda_xattn=cfg.lambda_xattn,
        xattn_heads=cfg.xattn.get("heads", 2),
        xattn_dim_head=cfg.xattn.get("dim_head", 16),
        xattn_pool=cfg.xattn.get("pool", 1),
        lambda_stat=getattr(cfg, "lambda_stat", 0.0),
        lambda_roi =getattr(cfg, "lambda_roi",  0.0),
    )).to(device)

    params = list(student.parameters()) + list(distiller.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    # ----- IO (per ablation folder) -----
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch","lam_feat","lam_at","lam_xattn","lam_stat","lam_roi",
                "aux","feat","attn","xattn","stat","roi","fuse10","fuse6","total",
                "dice","iou","ap50","map5095"
            ])

    def _get_fusion_for(fusion_key: str, s_tensor: torch.Tensor, t_tensor: torch.Tensor):
        if fusion_key not in fusions:
            anchor = fusion_name_map[fusion_key]["anchor"]
            d = (cfg.fusion_dims or {}).get(anchor, 128)
            k_tok = (cfg.fusion_tokens or {}).get(anchor, 4)
            fusions[fusion_key] = StructureFusion(
                s_channels=int(s_tensor.shape[1]),
                t_channels=int(t_tensor.shape[1]),
                d=int(d), num_tokens=int(k_tok), use_film=True
            ).to(device)
            optimizer.add_param_group({"params": fusions[fusion_key].parameters(), "lr": cfg.lr})
            print(f"[Fusion:{ablate.name}] created '{fusion_key}' ({anchor})")
        return fusions[fusion_key]

    # track which anchors actually fused in this run (for ckpt metadata)
    used_fusions: List[str] = []

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        student.train()

        # Example schedule (tune to taste)
        if   epoch <= 40:
            lam_feat, lam_attn, lam_xattn = 0.00, 0.00, 0.00
            lam_stat, lam_roi = 1.0, 0.5
        elif epoch <= 80:
            lam_feat, lam_attn, lam_xattn = 0.05, 0.00, 0.00
            lam_stat, lam_roi = 0.6, 0.25
        else:
            lam_feat, lam_attn, lam_xattn = 0.10, 0.05, 0.01
            lam_stat, lam_roi = 0.5, 0.2
        distiller.cfg.lambda_feat  = lam_feat;  cfg.lambda_feat  = lam_feat
        distiller.cfg.lambda_at    = lam_attn;  cfg.lambda_at    = lam_attn
        distiller.cfg.lambda_xattn = lam_xattn; cfg.lambda_xattn = lam_xattn
        distiller.cfg.lambda_stat  = lam_stat;  cfg.lambda_stat  = lam_stat
        distiller.cfg.lambda_roi   = lam_roi;   cfg.lambda_roi   = lam_roi

        pbar = tqdm(dl, desc=f"{ablate.name} epoch {epoch}/{cfg.epochs}")
        totals = {"aux":0.0, "feat":0.0, "attn":0.0, "xattn":0.0, "stat":0.0,"roi":0.0,"fuse10":0.0,"fuse6":0.0,"total":0.0}
        n = 0

        for batch in pbar:
            cam = to_device(batch["cam"], device)
            man = to_device(batch["man"], device)
            mask = to_device(batch["mask"], device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.amp):
                s_feat_hook.clear(); s_head_hook.clear()
                t_feat_hook.clear(); t_head_hook.clear()

                _ = student(cam)
                with torch.no_grad():
                    _ = teacher(man)

                losses: Dict[str, torch.Tensor] = {}
                total = torch.zeros([], device=device)

                # --- AUX head path: fuse or not depending on ablation flag ---
                s23 = s_head_hook.buffers[s_aux_name]
                t23 = t_head_hook.buffers[t_aux_name]
                if seg_adapter is None:
                    Cs = int(s23.shape[1])
                    seg_adapter = AuxMaskHead(in_channels=Cs, num_classes=cfg.num_classes).to(device)
                    optimizer.add_param_group({"params": seg_adapter.parameters(), "lr": cfg.lr})

                if ablate.use_fuse_aux:
                    f23 = _get_fusion_for(aux_key, s23, t23)
                    s23_for_head = f23(s23, t23)
                    if "model.23" not in used_fusions:
                        used_fusions.append("model.23")
                else:
                    s23_for_head = s23

                seg_logits = seg_adapter(s23_for_head, out_hw=mask.shape[-2:])

                if cfg.num_classes == 1 or seg_logits.shape[1] == 1:
                    l_aux = 0.5 * bce_logits_weighted(seg_logits, mask, w_bg=1.2, w_fg=1.0) \
                          + 0.5 * dice_loss(seg_logits, mask)
                    l_aux = l_aux + 0.03 * torch.sigmoid(seg_logits).mean()
                    gate_logits = seg_logits
                else:
                    fg_logits = seg_logits[:, 1:, :, :].amax(dim=1, keepdim=True)
                    l_aux = 0.5 * bce_logits_weighted(fg_logits, mask, w_bg=1.2, w_fg=1.0) \
                          + 0.5 * dice_loss(fg_logits, mask)
                    l_aux = l_aux + 0.03 * torch.sigmoid(fg_logits).mean()
                    gate_logits = fg_logits

                losses["aux"] = cfg.lambda_aux * l_aux
                total = total + losses["aux"]

                # --- Mid-tap fusions (enabled per ablation) + tiny regs ---
                fused_mids = {}
                for a in ["model.10","model.6"]:
                    if (a == "model.10" and not ablate.use_fuse_10) or (a == "model.6" and not ablate.use_fuse_6):
                        continue
                    k = _safe_name(a)
                    if k in fusion_name_map:
                        s_name = fusion_name_map[k]["student"]
                        t_name = fusion_name_map[k]["teacher"]
                        if (s_name in s_feat_hook.buffers) and (t_name in t_feat_hook.buffers):
                            s_mid = s_feat_hook.buffers[s_name]; t_mid = t_feat_hook.buffers[t_name]
                            f_mid = _get_fusion_for(k, s_mid, t_mid)
                            s_mid_fused = f_mid(s_mid, t_mid)
                            fused_mids[a] = s_mid_fused
                            if a not in used_fusions:
                                used_fusions.append(a)
                            lam_reg = (cfg.lambda_fuse_reg or {}).get(a, 0.0)
                            if lam_reg > 0:
                                l_reg = (s_mid_fused - s_mid).pow(2).mean()
                                key_loss = f"fuse{a.split('.')[-1]}"
                                losses[key_loss] = lam_reg * l_reg
                                total = total + losses[key_loss]

                # --- foreground gate for distillation (from aux logits) ---
                with torch.no_grad():
                    p = torch.sigmoid(gate_logits)
                    hard_cover = (p > 0.30).float().mean().item()
                    if hard_cover < 0.02:   gate = p
                    elif hard_cover <= 0.40: gate = (p > 0.10).float()
                    else:                    gate = (p > 0.30).float()

                def _apply_gate(feat_dict):
                    out = {}
                    for k_, v in feat_dict.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 4:
                            g = F.interpolate(gate, size=v.shape[-2:], mode="nearest")
                            out[k_] = v * g
                        else:
                            out[k_] = v
                    return out

                # replace aux tap with fused-or-not; replace mids if fused
                s_feats_mod = dict(s_feat_hook.buffers)
                s_heads_mod = dict(s_head_hook.buffers)
                s_heads_mod[s_aux_name] = s23_for_head
                for a, s_mid_fused in fused_mids.items():
                    s_name = fusion_name_map[_safe_name(a)]["student"]
                    s_feats_mod[s_name] = s_mid_fused

                s_feats_g = _apply_gate(s_feats_mod)
                s_heads_g = _apply_gate(s_heads_mod)

                extra = distiller.compute(
                    s_feats=s_feats_g,
                    t_feats=dict(t_feat_hook.buffers),
                    s_heads=s_heads_g,
                    t_heads=dict(t_head_hook.buffers),
                    cam_mask=mask,
                    man_mask=None
                )
                for k_, v in extra.items():
                    losses[k_] = v
                    total = total + v

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            n += 1; global_step += 1
            totals["total"] += float(total.detach().cpu())
            for k_ in ["aux","feat","attn","xattn","stat","roi","fuse10","fuse6"]:
                if k_ in losses:
                    totals[k_] += float(losses[k_].detach().cpu())
            pbar.set_postfix({k_: f"{totals[k_]/max(1,n):.4f}" for k_ in totals})

        # ----- SAVE -----
        def _sd_to_cpu(sd): return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k,v in sd.items()}
        if (epoch % cfg.save_every) == 0 or epoch == cfg.epochs:
            # record exact layers used + which anchors fused
            resolved_taps = {k: {"anchor": v["anchor"], "student": v["student"], "teacher": v["teacher"]}
                             for k, v in fusion_name_map.items()}
            fusion_meta = {
                "version": {"ultralytics": get_ultralytics_version()},
                "name_map": dict(fusion_name_map),   # stable resolved names
                "dims": (cfg.fusion_dims or {}),
                "tokens": (cfg.fusion_tokens or {}),
                "ablation": {"name": ablate.name,
                             "use_fuse_aux": ablate.use_fuse_aux,
                             "use_fuse_10": ablate.use_fuse_10,
                             "use_fuse_6": ablate.use_fuse_6},
                "used_fusions": list(used_fusions),  # e.g., ["model.23","model.10"]
                "resolved_taps": resolved_taps
            }
            ckpt = {
                "epoch": epoch,
                "student": _sd_to_cpu(student.state_dict()),
                "distiller": _sd_to_cpu(distiller.state_dict()),
                "optimizer": optimizer.state_dict(),
                "cfg": dataclasses.asdict(cfg) if hasattr(dataclasses, "asdict") else None,
                "seg_adapter": (_sd_to_cpu(seg_adapter.state_dict()) if seg_adapter is not None else None),
                "fusions": _sd_to_cpu(fusions.state_dict()),
                "fusion_meta": fusion_meta,
            }
            outp = os.path.join(out_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(ckpt, outp)
            print(f"[Save:{ablate.name}] {outp}")

        # ----- EVAL (optional) -----
        dice=iou=ap50=map5095=0.0
        if cfg.eval_every and (epoch % cfg.eval_every == 0):
            student_sd_cpu = {k: v.detach().cpu() for k,v in student.state_dict().items()}
            seg_adapter_sd = seg_adapter.state_dict() if seg_adapter is not None else None
            fusions_sd = fusions.state_dict() if len(fusions) else None
            fusion_meta_eval = {
                "name_map": dict(fusion_name_map),
                "dims": (cfg.fusion_dims or {}),
                "tokens": (cfg.fusion_tokens or {}),
            }
            student.eval()
            try:
                res = run_eval(
                    cfg, student_sd_cpu, device, out_dir, epoch,
                    seg_adapter_state=seg_adapter_sd,
                    fusions_state=fusions_sd,
                    distiller_state=distiller.state_dict(),
                    fusion_meta=fusion_meta_eval,
                    ablate=ablate
                )
                if res:
                    dice = res.get("dice", 0.0); iou = res.get("iou", 0.0)
                    ap50 = res.get("ap50", 0.0); map5095 = res.get("map5095", 0.0)
            finally:
                student.train()

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                float(distiller.cfg.lambda_feat),
                float(distiller.cfg.lambda_at),
                float(distiller.cfg.lambda_xattn),
                float(distiller.cfg.lambda_stat),
                float(distiller.cfg.lambda_roi),
                totals.get("aux",0.0)/max(1,n),
                totals.get("feat",0.0)/max(1,n),
                totals.get("attn",0.0)/max(1,n),
                totals.get("xattn",0.0)/max(1,n),
                totals.get("stat",0.0)/max(1,n),
                totals.get("roi",0.0)/max(1,n),
                totals.get("fuse10",0.0)/max(1,n),
                totals.get("fuse6",0.0)/max(1,n),
                totals["total"]/max(1,n),
                dice, iou, ap50, map5095
            ])

    # cleanup hooks
    s_feat_hook.remove(); s_head_hook.remove()
    t_feat_hook.remove(); t_head_hook.remove()


# -------------------------
# Main: orchestrate ablations
# -------------------------
def _strip_unknown_cfg_keys(raw_dict, schema_dataclass):
    # drop any YAML keys not in Cfg (e.g., legacy viz keys)
    allowed = {f.name for f in dataclasses.fields(schema_dataclass)}
    return {k: v for k, v in raw_dict.items() if k in allowed}

def main(cfg_path="config.yaml"):
    raw = load_cfg(cfg_path)

    # Fusion defaults
    if "fusion_dims" not in raw:
        raw["fusion_dims"] = {
            raw.get("aux_head_tap", "model.23"): 128,
            "model.10": 96,
            "model.6": 96,
        }
    if "fusion_tokens" not in raw:
        raw["fusion_tokens"] = {
            raw.get("aux_head_tap", "model.23"): 4,
            "model.10": 4,
            "model.6": 4,
        }
    if "lambda_fuse_reg" not in raw:
        raw["lambda_fuse_reg"] = {"model.10": 0.10, "model.6": 0.05}

    # Optional ablation list
    raw.setdefault("ablations", None)

    # Strip unknown YAML keys (avoids TypeError on legacy viz fields)
    raw = _strip_unknown_cfg_keys(raw, Cfg)

    cfg = Cfg(**raw)

    # Decide which ablations to run
    want = set(cfg.ablations) if (cfg.ablations is not None) else {a.name for a in ABLATION_GRID}
    todo = [a for a in ABLATION_GRID if a.name in want]
    if not todo:
        raise ValueError(f"No valid ablation names in cfg.ablations. Valid: {[a.name for a in ABLATION_GRID]}")

    # Run each ablation into its own out_dir/<name>
    for ablate in todo:
        out_dir = os.path.join(cfg.out_dir, ablate.name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Running ablation: {ablate.name} -> {out_dir} ===")
        run_one_training(cfg, ablate, out_dir)



# -------- entry --------
config_path = "/scratch/user/kuocy/IKEAVideo/model_code/distill_train_code/config_ablation_kfold.yaml"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cfg",
        nargs="?",
        default=config_path,
        help="Path to config.yaml (optional; defaults to the hardcoded path above).",
    )
    args = parser.parse_args()
    cfg_path = os.path.expanduser(os.path.expandvars(args.cfg))
    main(cfg_path)
