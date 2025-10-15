# run_dual_aux_batch.py
# Full pipeline: calibrate on VAL (once per ckpt) -> test across multiple test sets
# Saves:
#   - Per-image CSV for each (ckpt, testset) pair
#   - Per-testset summary CSVs that aggregate all ckpts
#
# NOTE: This file is self-contained. It includes your evaluation code and a batch driver.
#       The only external dependency besides standard libs is 'ultralytics', 'numpy',
#       'Pillow', 'pycocotools' (optional), and 'scipy' (optional for some metrics).

from pathlib import Path
import os, sys, re, json, yaml
import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== YOUR UTILS (imported from your project) ========
# You said these are yours. We import from your local file.
# If smart_prior.py is not importable, place it next to this script.
from smart_prior import refine_aux_prior, filter_student_instances_with_prior

# =================== USER CONFIG (global) ===================
# Student / teacher (HPRC)
STUDENT_MODEL = "/scratch/user/kuocy/IKEAVideo/0815Results_complete_labeling/yolo11s-seg_0816_123254_f2/weights/best.pt"
TEACHER_MODEL = "/scratch/user/kuocy/IKEAVideo/0815Results_complete_labeling/yolo11s-seg_0816_024507_f0/weights/best.pt"

# K-fold ablation checkpoints (HPRC)
CKPT_ROOT  = "/scratch/user/kuocy/IKEAVideo/model_code/distill_train_code/train_ablation_kfold_results"
FOLDS      = [3, 4]  # change to subset if needed
ABLATIONS  = ["fuse_aux_only", "fuse_6_aux", "fuse_10_aux", "fuse_6_10_aux"]

def _build_ckpt_map(root, folds, abls):
    ck = {}
    for f in folds:
        for a in abls:
            key  = f"fold{f}/{a}"
            path = os.path.join(root, f"fold{f}", a, "ckpt_epoch_100.pt")
            if not os.path.isfile(path):
                print(f"[WARN] missing ckpt => {path} (will be skipped)")
                continue
            ck[key] = path
    if not ck:
        raise RuntimeError(f"No checkpoints found under {root}.")
    return ck

CKPTS = _build_ckpt_map(CKPT_ROOT, FOLDS, ABLATIONS)

# Base dataset folder (HPRC) — your previous 'mini_2' now lives here.
BASE_YAML_DIR = "/scratch/user/kuocy/IKEAVideo/dataset_code/mini_2"

# “VAL” yaml files used for threshold calibration (unchanged filenames)
CAM_YAML_VAL = os.path.join(BASE_YAML_DIR, "parts_single_cam.yaml")
MAN_YAML_VAL = os.path.join(BASE_YAML_DIR, "parts_single_man.yaml")

# Explicit VAL mask root
EXPLICIT_VAL_MASK_ROOT = os.path.join(BASE_YAML_DIR, "mask_labels", "val", "camera")

# Test sets (same structure, new base path)
TEST_SETS = [
    {
        "name": "test",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "parts_single_cam.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "parts_single_man.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test", "camera"),
    },
    {
        "name": "test_unseen_category",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_cat.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_cat.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_category", "camera"),
    },
    {
        "name": "test_unseen_type_Bench",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_bench.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_bench.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_type_Bench", "camera"),
    },
    {
        "name": "test_unseen_type_Chair",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_chair.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_chair.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_type_Chair", "camera"),
    },
    {
        "name": "test_unseen_type_Desk",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_desk.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_desk.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_type_Desk", "camera"),
    },
    {
        "name": "test_unseen_type_Shelf",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_shelf.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_shelf.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_type_Shelf", "camera"),
    },
    {
        "name": "test_unseen_type_Table",
        "cam_yaml": os.path.join(BASE_YAML_DIR, "cam_unseen_table.yaml"),
        "man_yaml": os.path.join(BASE_YAML_DIR, "man_unseen_table.yaml"),
        "mask_root": os.path.join(BASE_YAML_DIR, "mask_labels", "test_unseen_type_Table", "camera"),
    },


]
# ===========================================================


# Output base
PROJECT = os.path.join(BASE_YAML_DIR, "eval_runs")   # already used later in your script
RUN_NAME = "batch_full5_kfold_ablation"


# Common settings
MASK_SUFFIX = "_rle.json"
IMG_SIZE = 640
DEVICE = "0"  # "0" or "cpu"

# VAL sweeps (IMPORTANT: Aux uses AUX_THR_GRID, not YOLO_CONF_GRID)
YOLO_CONF_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
AUX_THR_GRID   = np.linspace(0.10, 0.70, 13)

# Refinement / filtering knobs
PRIOR_NEAR_THR   = 0.2    # LOGIT space (we pass logits to refine_aux_prior)
PRIOR_FAR_THR    = 0.9    # LOGIT space
PRIOR_NEAR_PX    = 35
PRIOR_FAR_CUT_PX = 60
PRIOR_MIN_AREA   = 300
PRIOR_ERODE_IT   = 0
PRIOR_OPEN_IT    = 1
PRIOR_CLOSE_IT   = 0
CLIP_TO_AUX      = True
COVER_THR        = 0.20
RESCORE_ALPHA    = 0.35

ADD_TEACHER_MISSING = True
PROMOTE_MIN_AREA    = 1400
PROMOTE_MAX_DIST_PX = 30

BOUND_TOL_PX = 2
# ===========================================================

def require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa
    except Exception:
        print("Ultralytics not installed. Run:  pip install ultralytics")
        raise
require_ultralytics()
from ultralytics import YOLO  # noqa

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

# ---------------- utilities ----------------
def _predict_device_arg_from_model(m: nn.Module) -> Union[str, int]:
    try:
        dev = next(m.parameters()).device
    except StopIteration:
        return "cpu"
    if dev.type == "cuda":
        return dev.index if dev.index is not None else 0
    return "cpu"

def _first_4d_tensor(x):
    if isinstance(x, torch.Tensor) and x.ndim == 4:
        return x
    if isinstance(x, (list, tuple)):
        for it in x:
            t = _first_4d_tensor(it)
            if t is not None: return t
    if isinstance(x, dict):
        for _, it in x.items():
            t = _first_4d_tensor(it)
            if t is not None: return t
    return None

# ---------------- feature hooks + modules ----------------
class FeatureHooker:
    def __init__(self, model, layers, use_regex=False):
        self.model = model
        self.layers = layers
        self.use_regex = use_regex
        self.buffers = {}
        self.handles = []
        self._register()
    def _match(self, name):
        if self.use_regex:
            import re as _re
            return any(_re.match(pat, name) for pat in self.layers)
        return name in self.layers
    def _register(self):
        for name, m in self.model.named_modules():
            if self._match(name):
                self.handles.append(m.register_forward_hook(self._make_hook(name)))
    def _make_hook(self, name):
        def hook(_m, _in, out):
            self.buffers[name] = out
        return hook
    def clear(self):
        self.buffers.clear()
    def remove(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles = []

class StructureFusion(nn.Module):
    def __init__(self, s_channels: int, t_channels: int, d: int = 128, num_tokens: int = 4, use_film: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.d = d
        self.use_film = use_film
        self.t_gap = nn.AdaptiveAvgPool2d(1)
        self.t_token = nn.Sequential(nn.Conv2d(t_channels, d * num_tokens, 1, bias=True), nn.ReLU(inplace=True))
        self.q_proj = nn.Conv2d(s_channels, d, 1, bias=True)
        self.out_proj = nn.Conv2d(d, s_channels, 1, bias=True)
        if use_film:
            self.film = nn.Sequential(nn.Conv2d(t_channels, s_channels * 2, 1, bias=True))
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
        out = torch.matmul(attn, v).transpose(1, 2).view(B, self.d, H, W)
        out = self.out_proj(out)
        fused = s_feat + out
        if self.use_film:
            film_out = self.film(t_g)
            need = 2 * Cs
            have = film_out.shape[1]
            if have < need:
                pad = torch.zeros(film_out.shape[0], need - have, *film_out.shape[2:],
                                device=film_out.device, dtype=film_out.dtype)
                film_out = torch.cat([film_out, pad], dim=1)
            elif have > need:
                film_out = film_out[:, :need]
            gamma, beta = torch.split(film_out, [Cs, Cs], dim=1)
            fused = fused * torch.sigmoid(gamma) + beta
        return fused

class AuxMaskHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        mid = max(32, min(256, in_channels))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1, bias=True),
        )
    def forward(self, x: torch.Tensor, out_hw=None):
        y = self.block(x)
        if out_hw is not None:
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y

# ---- channel-compat loaders ----
def _pad_in(w: torch.Tensor, in_dim: int) -> torch.Tensor:
    c = w.shape[1]
    if c == in_dim: return w
    if c > in_dim:  return w[:, :in_dim, ...]
    z = w.new_zeros(w.shape[0], in_dim - c, *w.shape[2:])
    return torch.cat([w, z], dim=1)

def _pad_out(w: torch.Tensor, out_dim: int) -> torch.Tensor:
    o = w.shape[0]
    if o == out_dim: return w
    if o > out_dim:  return w[:out_dim, ...]
    z = w.new_zeros(out_dim - o, *w.shape[1:])
    return torch.cat([w, z], dim=0)

def _pad_bias(b: torch.Tensor, out_dim: int) -> torch.Tensor:
    o = b.shape[0]
    if o == out_dim: return b
    if o > out_dim:  return b[:out_dim]
    z = b.new_zeros(out_dim - o)
    return torch.cat([b, z], dim=0)

def _extract_substate(sd: dict, safe_key: str) -> dict:
    pre = safe_key + "."
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}

def _load_fusion_with_channel_compat(fusion: StructureFusion, fusions_sd: dict, safe_key: str):
    sub = _extract_substate(fusions_sd, safe_key)
    if not sub: return
    qW = fusion.q_proj.weight
    tW = fusion.t_token[0].weight
    oW = fusion.out_proj.weight
    fW = fusion.film[0].weight if hasattr(fusion, "film") else None
    cur_Cs_in = qW.shape[1]
    cur_Ct_in = tW.shape[1]
    cur_Cs_out = oW.shape[0]
    cur_Fout = fW.shape[0] if fW is not None else None
    if "t_token.0.weight" in sub:
        w = sub["t_token.0.weight"]; b = sub.get("t_token.0.bias", None)
        fusion.t_token[0].weight.data.copy_(_pad_in(w, cur_Ct_in))
        if b is not None: fusion.t_token[0].bias.data.copy_(b)
    if "q_proj.weight" in sub:
        fusion.q_proj.weight.data.copy_(_pad_in(sub["q_proj.weight"], cur_Cs_in))
    if "q_proj.bias" in sub:
        fusion.q_proj.bias.data.copy_(sub["q_proj.bias"])
    if "out_proj.weight" in sub:
        fusion.out_proj.weight.data.copy_(_pad_out(sub["out_proj.weight"], cur_Cs_out))
    if "out_proj.bias" in sub:
        fusion.out_proj.bias.data.copy_(_pad_bias(sub["out_proj.bias"], cur_Cs_out))
    if fW is not None:
        if "film.0.weight" in sub:
            w = sub["film.0.weight"]
            w = _pad_in(w, cur_Ct_in)
            w = _pad_out(w, cur_Fout)
            fusion.film[0].weight.data.copy_(w)
        if "film.0.bias" in sub:
            fusion.film[0].bias.data.copy_(_pad_bias(sub["film.0.bias"], cur_Fout))

def _load_seg_head_with_channel_compat(seg: AuxMaskHead, seg_sd: dict):
    if not seg_sd: return
    w0 = seg_sd.get("block.0.weight", None)
    b0 = seg_sd.get("block.0.bias",   None)
    if w0 is not None:
        cur_in = seg.block[0].weight.shape[1]
        seg.block[0].weight.data.copy_(_pad_in(w0, cur_in))
    if b0 is not None:
        seg.block[0].bias.data.copy_(b0)
    for name, param in seg.named_parameters():
        if name in ("block.0.weight", "block.0.bias"): continue
        if name in seg_sd:
            try: param.data.copy_(seg_sd[name])
            except: pass
    for name, buf in seg.named_buffers():
        if name in seg_sd and seg_sd[name].shape == buf.shape:
            buf.copy_(seg_sd[name])

# ---------------- data + metrics ----------------
def _read_yaml_images_root(yaml_path: Path, split_key: str) -> Path:
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", yaml_path.parent)).resolve()
    sp = y.get(split_key)
    if sp is None:
        raise ValueError(f"{yaml_path} has no '{split_key}' split.")
    p = Path(sp[0] if isinstance(sp, (list, tuple)) else sp)
    if not p.is_absolute(): p = (root / p).resolve()
    if not p.exists(): raise FileNotFoundError(f"Split '{split_key}' not found: {p}")
    return p

def _prep_image(path: str, size=640):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    return x

def _resize_bool(mask_np: np.ndarray, out_hw):
    H, W = out_hw
    if mask_np.shape == (H, W):
        return mask_np.astype(bool)
    pil = Image.fromarray((mask_np.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
    return (np.array(pil) > 127)

def _decode_instances_from_json(json_path: Path, mask_suffix: str):
    if mask_utils is None or not json_path.exists():
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []
    items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
    masks = []
    for it in items:
        r = it.get("segmentation", it) if isinstance(it, dict) else None
        if isinstance(r, dict) and "counts" in r and "size" in r:
            rle = {"counts": r["counts"].encode("utf-8") if isinstance(r["counts"], str) else r["counts"],
                   "size": [int(r["size"][0]), int(r["size"][1])]}
            m = mask_utils.decode(rle)
            if m.ndim == 3: m = m[..., 0]
            masks.append((m > 0).astype(np.uint8))
    return masks

def _merge_binary(masks_list):
    merged = None
    for m in masks_list:
        merged = m if merged is None else np.maximum(merged, m)
    return merged

def _pix_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray, bound_tol=2):
    pred = pred_bool.astype(bool); gt = gt_bool.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2*tp + 1e-6) / (2*tp + fp + fn + 1e-6)
    iou  = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec  = (tp + 1e-6) / (tp + fn + 1e-6)
    cov  = pred.mean() * 100.0
    # boundary F (tolerant)
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        se = np.ones((3,3), dtype=bool)
        pred_edge = pred ^ binary_erosion(pred, structure=se, border_value=0)
        gt_edge   = gt   ^ binary_erosion(gt,   structure=se, border_value=0)
        gt_d   = distance_transform_edt(~gt_edge)
        pred_d = distance_transform_edt(~pred_edge)
        pred_match = pred_edge & (gt_d   <= bound_tol)
        gt_match   = gt_edge   & (pred_d <= bound_tol)
        p_b = pred_match.sum() / max(1, pred_edge.sum())
        r_b = gt_match.sum()   / max(1, gt_edge.sum())
        bf  = 0.0 if (p_b + r_b) == 0 else (2 * p_b * r_b) / (p_b + r_b)
    except Exception:
        bf = 0.0
    return dict(Dice=float(dice), IoU=float(iou), Precision=float(prec),
                Recall=float(rec), BoundaryF=float(bf), Coverage=float(cov))

# ---------- EXTRA HELPERS FOR AUX / AP ----------
def _connected_components_bool(mask_bool: np.ndarray) -> list:
    try:
        from scipy.ndimage import label
        lab, n = label(mask_bool.astype(np.uint8))
        comps = []
        for i in range(1, n + 1):
            comp = (lab == i)
            if comp.any():
                comps.append(comp)
        return comps
    except Exception:
        return [mask_bool.astype(bool)]

def _build_instance_preds_from_components(components: list, base_score: float = 1.0):
    return [{"score": float(base_score), "mask": comp.astype(bool)} for comp in components]

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

def _eval_map(preds: List[Dict], gts_by_img: Dict[int, List[np.ndarray]],
              iou_thresholds=np.arange(0.5, 0.96, 0.05)) -> Tuple[float, float]:
    preds = sorted(preds, key=lambda x: -x["score"])
    total_gts = sum(len(v) for v in gts_by_img.values())
    aps = {}
    for thr in iou_thresholds:
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        matched = {img_id: np.zeros(len(gts_by_img[img_id]), dtype=bool) for img_id in gts_by_img.keys()}
        for i, p in enumerate(preds):
            img_id = p["img_id"]
            gt_masks = gts_by_img.get(img_id, [])
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

# ---------------- aux build + logits/prob fetchers ----------------
def build_aux_and_fusion(student_model_path, teacher_model_path, ckpt_path, device):
    y_s = YOLO(student_model_path); y_t = YOLO(teacher_model_path)
    y_s.model.to(device).eval(); y_t.model.to(device).eval()

    ck = torch.load(ckpt_path, map_location="cpu")
    student_sd = ck.get("student", {})
    seg_sd     = ck.get("seg_adapter", {}) or {}
    fusions_sd = ck.get("fusions", {})      or {}
    fusion_meta = ck.get("fusion_meta", {}) or {}
    name_map = fusion_meta.get("name_map", {}) or {}
    dims   = fusion_meta.get("dims", {}) or {}
    tokens = fusion_meta.get("tokens", {}) or {}

    # load distilled student state (for feature tapping used by aux path)
    try:
        y_s.model.load_state_dict(student_sd, strict=False)
    except Exception as e:
        print(f"[WARN] could not load ckpt student weights into student YOLO (aux path): {e}")

    # choose a model.23 tap that matches FiLM out = 2*Cs
    candidates = [(k, v) for k, v in name_map.items()
                  if isinstance(v, dict) and str(v.get("anchor","")).startswith("model.23")]
    if not candidates:
        raise RuntimeError("[ckpt] Cannot locate aux tap (model.23) in fusion_meta.name_map.")

    def _film_out_from_ckpt(k):
        sub = _extract_substate(fusions_sd, k)
        w = sub.get("film.0.weight", None)
        return None if w is None else int(w.shape[0])

    best = None
    for k, v in candidates:
        s_hook_tmp = FeatureHooker(y_s.model, [v["student"]])
        t_hook_tmp = FeatureHooker(y_t.model, [v["teacher"]])
        with torch.no_grad():
            _ = y_s.model(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))
            _ = y_t.model(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))
        s_sample = _first_4d_tensor(s_hook_tmp.buffers.get(v["student"], None))
        t_sample = _first_4d_tensor(t_hook_tmp.buffers.get(v["teacher"], None))
        s_hook_tmp.remove(); t_hook_tmp.remove()
        if s_sample is None or t_sample is None:
            continue
        Cs = int(s_sample.shape[1]); Ct = int(t_sample.shape[1])
        fout = _film_out_from_ckpt(k)
        if fout == 2 * Cs:
            best = (k, v, Cs, Ct)
            break

    if best is None:
        k, v = candidates[0]
        s_hook_tmp = FeatureHooker(y_s.model, [v["student"]])
        t_hook_tmp = FeatureHooker(y_t.model, [v["teacher"]])
        with torch.no_grad():
            _ = y_s.model(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))
            _ = y_t.model(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))
        s_sample = _first_4d_tensor(s_hook_tmp.buffers.get(v["student"], None))
        t_sample = _first_4d_tensor(t_hook_tmp.buffers.get(v["teacher"], None))
        s_hook_tmp.remove(); t_hook_tmp.remove()
        if s_sample is None or t_sample is None:
            raise RuntimeError("Resolved aux taps did not produce 4D tensors.")
        best = (k, v, int(s_sample.shape[1]), int(t_sample.shape[1]))

    aux_key, aux_entry, Cs, Ct = best
    aux_anchor = aux_entry["anchor"]
    s_aux = aux_entry["student"]
    t_aux = aux_entry["teacher"]
    d  = int(dims.get(aux_anchor, 128))
    k_tok = int(tokens.get(aux_anchor, 4))

    fusions = nn.ModuleDict({
        aux_key: StructureFusion(Cs, Ct, d=d, num_tokens=k_tok, use_film=True).to(device).eval()
    })
    _load_fusion_with_channel_compat(fusions[aux_key], fusions_sd, aux_key)

    seg_adapter = AuxMaskHead(in_channels=Cs, num_classes=1).to(device).eval()
    _load_seg_head_with_channel_compat(seg_adapter, seg_sd)

    s_hook = FeatureHooker(y_s.model, [s_aux])
    t_hook = FeatureHooker(y_t.model, [t_aux])

    return dict(
        yolo_student=y_s, yolo_teacher=y_t,
        s_hook=s_hook, t_hook=t_hook,
        aux_key=aux_key, s_aux=s_aux, t_aux=t_aux,
        fusion=fusions[aux_key], seg_head=seg_adapter,
        device=device
    )

def aux_logits_and_prob(aux_bundle, cam_path: Path, man_path: Path, out_hw) -> Tuple[np.ndarray, np.ndarray]:
    device = aux_bundle["device"]
    y_s = aux_bundle["yolo_student"]; y_t = aux_bundle["yolo_teacher"]
    s_hook, t_hook = aux_bundle["s_hook"], aux_bundle["t_hook"]
    s_aux = aux_bundle["s_aux"]; t_aux = aux_bundle["t_aux"]
    fusion = aux_bundle["fusion"]; seg_head = aux_bundle["seg_head"]

    x_cam = _prep_image(str(cam_path), IMG_SIZE).to(device)
    x_man = _prep_image(str(man_path), IMG_SIZE).to(device)

    s_hook.clear(); t_hook.clear()
    with torch.no_grad():
        _ = y_s.model(x_cam)
        _ = y_t.model(x_man)
        s_feat = _first_4d_tensor(s_hook.buffers.get(s_aux, None))
        t_feat = _first_4d_tensor(t_hook.buffers.get(t_aux, None))
        if s_feat is None or t_feat is None:
            return None, None
        s_fused = fusion(s_feat, t_feat)
        logits = seg_head(s_fused, out_hw=out_hw)  # (1,1,H,W)
        logits_np = logits[0,0].cpu().numpy().astype(np.float32)
        prob_np   = torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)
        return logits_np, prob_np

def pure_aux_prob(aux_bundle, cam_path: Path, man_path: Path, out_hw):
    _logits, prob = aux_logits_and_prob(aux_bundle, cam_path, man_path, out_hw)
    return prob

# ---------------- YOLO prediction helpers ----------------
def yolo_union_mask(yolo_model: YOLO, img_path: Path, conf_th: float, img_size: int, out_hw):
    device_arg = _predict_device_arg_from_model(yolo_model.model)
    res = yolo_model.predict(
        source=str(img_path), imgsz=img_size, conf=conf_th, iou=0.6, max_det=100,
        device=device_arg, verbose=False, half=(device_arg != "cpu")
    )[0]
    H0, W0 = out_hw
    if getattr(res, "masks", None) is not None and res.masks is not None:
        pm = res.masks.data.cpu().numpy()
        if pm.ndim == 3 and pm.shape[0] > 0:
            acc = np.zeros((H0, W0), dtype=bool)
            for k in range(pm.shape[0]):
                mk = _resize_bool(pm[k] > 0.5, (H0, W0))
                acc |= mk
            return acc
    return np.zeros((H0, W0), dtype=bool)

def _collect_instances(yolo_model: YOLO, img_path: Path, conf_th: float, img_size: int, out_hw):
    device_arg = _predict_device_arg_from_model(yolo_model.model)
    res = yolo_model.predict(
        source=str(img_path), imgsz=img_size, conf=conf_th, iou=0.6, max_det=100,
        device=device_arg, verbose=False, half=(device_arg != "cpu")
    )[0]
    H0, W0 = out_hw
    masks_bool, scores = [], []
    union = np.zeros((H0, W0), dtype=bool)
    if getattr(res, "masks", None) is not None and res.masks is not None:
        pm = res.masks.data.cpu().numpy()
        sc = res.boxes.conf.cpu().numpy() if getattr(res, "boxes", None) is not None else np.ones(len(pm))
        if pm.ndim == 3 and pm.shape[0] > 0:
            for k in range(pm.shape[0]):
                mk = _resize_bool(pm[k] > 0.5, (H0, W0))
                masks_bool.append(mk)
                scores.append(float(sc[k]))
                union |= mk
    return union, masks_bool, scores

def _normalize_filtered_preds(preds_filt):
    """
    Normalize outputs from filter_student_instances_with_prior into
    a list of (mask_bool, score_float) tuples.

    Accepts any of:
      A) (masks_list, scores_list/ndarray)
      B) list of (mask, score) tuples
      C) list of dicts with keys like {"mask": ..., "score": ...}
    """
    # Case C: list of dicts
    if isinstance(preds_filt, list) and preds_filt and isinstance(preds_filt[0], dict):
        out = []
        for d in preds_filt:
            mk = d.get("mask", d.get("m"))
            sc = d.get("score", d.get("conf", d.get("s", 1.0)))
            if mk is None:
                continue
            try:
                sc = float(sc)
            except Exception:
                # fallback if score missing/bad
                sc = 1.0
            out.append((np.array(mk).astype(bool), sc))
        return out

    # Case A: tuple/list of (masks_list, scores_list)
    if (isinstance(preds_filt, (tuple, list)) and len(preds_filt) == 2 and
        isinstance(preds_filt[0], (list, tuple, np.ndarray)) and
        not (preds_filt and isinstance(preds_filt[0], (tuple, list)) and len(preds_filt[0]) == 2)):
        masks, scores = preds_filt
        return [(np.array(mk).astype(bool), float(sc)) for mk, sc in zip(masks, scores)]

    # Case B: list of (mask, score)
    out = []
    for item in preds_filt:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            mk, sc = item
            try:
                sc = float(sc)
            except Exception:
                sc = 1.0
            out.append((np.array(mk).astype(bool), sc))
        else:
            # Unknown shape; skip safely
            continue
    return out



# ---------------- VAL calibration ----------------
def calibrate_yolo_on_val(yolo_model: YOLO, cam_yaml: str, mask_suffix: str, conf_grid,
                          out_dir: Path, tag: str, val_mask_root_opt: str = ""):
    """
    tag: 'raw' or 'ckpt' (used only for deterministic filename)
    """
    cam_val = _read_yaml_images_root(Path(cam_yaml), "val")
    mask_val = Path(val_mask_root_opt).resolve() if val_mask_root_opt \
        else (Path(cam_val).parents[1] / "mask_labels" / "val" / "camera")
    img_paths = sorted([p for p in cam_val.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not img_paths:
        raise RuntimeError(f"No VAL images found under {cam_val}")

    yolo_rows = []
    for conf in conf_grid:
        mets = []
        for cam_path in img_paths:
            rel = cam_path.relative_to(cam_val)
            mjson = mask_val / rel.parent / (cam_path.stem + mask_suffix)
            gts = _decode_instances_from_json(mjson, mask_suffix)
            if not gts:
                continue
            gt_union = _merge_binary(gts) > 0
            H0, W0 = gt_union.shape
            union = yolo_union_mask(yolo_model, cam_path, conf, IMG_SIZE, (H0, W0))
            mets.append(_pix_metrics(union, gt_union, bound_tol=BOUND_TOL_PX))
        if not mets:
            continue
        mean = {k: float(np.mean([m[k] for m in mets])) for k in mets[0].keys()}
        yolo_rows.append({"conf": conf, **mean})

    calib_dir = out_dir / "_calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    csv_path = calib_dir / f"yolo_{tag}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("conf,Dice,IoU,Precision,Recall,BoundaryF,Coverage\n")
        for r in yolo_rows:
            f.write(f"{r['conf']},{r['Dice']:.6f},{r['IoU']:.6f},{r['Precision']:.6f},"
                    f"{r['Recall']:.6f},{r['BoundaryF']:.6f},{r['Coverage']:.4f}\n")
    if not yolo_rows:
        raise RuntimeError("VAL calibration (YOLO) produced no rows. Check masks and splits.")
    best_conf = max(yolo_rows, key=lambda r: r["Dice"])["conf"]
    print(f"[CALIB] Saved YOLO {tag} grid to: {csv_path} | Best conf={best_conf:.2f}")
    return best_conf


def calibrate_aux_on_val(aux_bundle, cam_yaml: str, man_yaml: str, mask_suffix: str, thr_grid,
                         out_dir: Path, val_mask_root_opt: str = ""):
    cam_val = _read_yaml_images_root(Path(cam_yaml), "val")
    man_val = _read_yaml_images_root(Path(man_yaml), "val")
    mask_val = Path(val_mask_root_opt).resolve() if val_mask_root_opt \
        else (Path(cam_val).parents[1] / "mask_labels" / "val" / "camera")
    img_paths = sorted([p for p in cam_val.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not img_paths:
        raise RuntimeError(f"No VAL images found under {cam_val}")

    aux_rows = []
    for thr in thr_grid:
        mets = []
        for cam_path in img_paths:
            rel = cam_path.relative_to(cam_val)
            man_path = (man_val / rel)
            if not man_path.exists():
                alt = man_val / cam_path.name
                if alt.exists():
                    man_path = alt
                else:
                    continue
            mjson = mask_val / rel.parent / (cam_path.stem + mask_suffix)
            gts = _decode_instances_from_json(mjson, mask_suffix)
            if not gts:
                continue
            gt_union = _merge_binary(gts) > 0
            H0, W0 = gt_union.shape
            prob = pure_aux_prob(aux_bundle, cam_path, man_path, (H0, W0))
            if prob is None:
                continue
            pred = (prob > float(thr))
            mets.append(_pix_metrics(pred, gt_union, bound_tol=BOUND_TOL_PX))
        if not mets:
            continue
        mean = {k: float(np.mean([m[k] for m in mets])) for k in mets[0].keys()}
        aux_rows.append({"thr": float(thr), **mean})

    calib_dir = out_dir / "_calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    aux_csv = calib_dir / "aux.csv"
    with open(aux_csv, "w", encoding="utf-8") as f:
        f.write("thr,Dice,IoU,Precision,Recall,BoundaryF,Coverage\n")
        for r in aux_rows:
            f.write(f"{r['thr']:.4f},{r['Dice']:.6f},{r['IoU']:.6f},{r['Precision']:.6f},"
                    f"{r['Recall']:.6f},{r['BoundaryF']:.6f},{r['Coverage']:.4f}\n")
    if not aux_rows:
        raise RuntimeError("VAL calibration (AUX) produced no rows. Check masks and splits.")
    best_thr = max(aux_rows, key=lambda r: r["Dice"])["thr"]
    print(f"[CALIB] Saved AUX grid to: {aux_csv} | Best thr={best_thr:.2f}")
    return best_thr


# ---------------- TEST evaluation ----------------
def evaluate_on_test(aux_bundle,
                     yolo_student_raw: YOLO, yolo_student_ckpt: YOLO,
                     cam_yaml: str, man_yaml: str, mask_suffix: str,
                     test_mask_root_opt: str,
                     best_conf_raw: float, best_conf_ckpt: float, best_aux_thr: float,
                     out_dir: Path):

    cam_test = _read_yaml_images_root(Path(cam_yaml), "test")
    man_test = _read_yaml_images_root(Path(man_yaml), "test")
    mask_test = Path(test_mask_root_opt).resolve() if test_mask_root_opt else (Path(cam_test).parents[1] / "mask_labels" / "test" / "camera")

    img_paths = sorted([p for p in cam_test.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not img_paths:
        raise RuntimeError(f"No TEST images found under {cam_test}")

    rows = []
    mets_base_raw, mets_base_ckpt, mets_aux, mets_ref_raw, mets_ref_ckpt = [], [], [], [], []

    preds_base_raw_ap, preds_base_ckpt_ap, preds_aux_ap, preds_ref_raw_ap, preds_ref_ckpt_ap = [], [], [], [], []
    gts_by_img_for_ap = {}

    for img_id, cam_path in enumerate(img_paths):
        rel = cam_path.relative_to(cam_test)
        man_path = (man_test / rel)
        if not man_path.exists():
            alt = man_test / cam_path.name
            if alt.exists(): man_path = alt
            else: continue

        mjson = mask_test / rel.parent / (cam_path.stem + mask_suffix)
        gts = _decode_instances_from_json(mjson, mask_suffix)
        if not gts: continue
        gts_bool = [(g > 0).astype(bool) for g in gts]
        gt_union = _merge_binary(gts) > 0
        H0, W0 = gt_union.shape
        gts_by_img_for_ap[img_id] = gts_bool

        union_raw, raw_masks_bool, raw_scores = _collect_instances(yolo_student_raw, cam_path, best_conf_raw, IMG_SIZE, (H0, W0))
        m_base_raw = _pix_metrics(union_raw, gt_union, bound_tol=BOUND_TOL_PX)
        mets_base_raw.append(m_base_raw)
        for mk, sc in zip(raw_masks_bool, raw_scores):
            preds_base_raw_ap.append({"img_id": img_id, "score": float(sc), "mask": mk.astype(bool)})

        union_ckpt, ckpt_masks_bool, ckpt_scores = _collect_instances(yolo_student_ckpt, cam_path, best_conf_ckpt, IMG_SIZE, (H0, W0))
        m_base_ckpt = _pix_metrics(union_ckpt, gt_union, bound_tol=BOUND_TOL_PX)
        mets_base_ckpt.append(m_base_ckpt)
        for mk, sc in zip(ckpt_masks_bool, ckpt_scores):
            preds_base_ckpt_ap.append({"img_id": img_id, "score": float(sc), "mask": mk.astype(bool)})

        logits_np, prob_np = aux_logits_and_prob(aux_bundle, cam_path, man_path, (H0, W0))
        if prob_np is None:
            continue
        aux_bin = (prob_np > float(best_aux_thr))
        m_aux = _pix_metrics(aux_bin, gt_union, bound_tol=BOUND_TOL_PX)
        mets_aux.append(m_aux)
        aux_components = _connected_components_bool(aux_bin)
        for d in _build_instance_preds_from_components(aux_components, base_score=1.0):
            preds_aux_ap.append({"img_id": img_id, "score": d["score"], "mask": d["mask"]})

        aux_prior, _teacher_prob = refine_aux_prior(
            logits_np, union_raw,
            logits_are_probs=False,
            near_thr=PRIOR_NEAR_THR, far_thr=PRIOR_FAR_THR,
            max_near_px=PRIOR_NEAR_PX, far_cut_px=PRIOR_FAR_CUT_PX,
            keep_only_teacher_minus_student=False,
            min_comp_area=PRIOR_MIN_AREA,
            light_erosion_iters=PRIOR_ERODE_IT,
            open_iters=PRIOR_OPEN_IT,
            close_iters=PRIOR_CLOSE_IT
        )

        preds_filt_raw, union_filt_raw = filter_student_instances_with_prior(
            raw_masks_bool, np.array(raw_scores, dtype=np.float32), aux_prior.astype(bool),
            clip_to_prior=CLIP_TO_AUX, cover_thr=COVER_THR, rescore_alpha=RESCORE_ALPHA
        )
        union_ref_raw = union_filt_raw
        if ADD_TEACHER_MISSING:
            try:
                from scipy.ndimage import distance_transform_edt, label
                near_mask = distance_transform_edt(~union_raw) <= PROMOTE_MAX_DIST_PX
                lab, n = label(aux_prior.astype(np.uint8))
                promote = np.zeros_like(aux_prior, dtype=bool)
                for i in range(1, n + 1):
                    comp = (lab == i)
                    if comp.sum() < PROMOTE_MIN_AREA: continue
                    if (comp & near_mask).any():
                        promote |= comp
                union_ref_raw = (union_ref_raw | promote)
            except Exception:
                pass
        m_ref_raw = _pix_metrics(union_ref_raw.astype(bool), gt_union, bound_tol=BOUND_TOL_PX)
        mets_ref_raw.append(m_ref_raw)
        for mk, sc in _normalize_filtered_preds(preds_filt_raw):
            preds_ref_raw_ap.append({"img_id": img_id, "score": float(sc), "mask": mk.astype(bool)})

        preds_filt_ckpt, union_filt_ckpt = filter_student_instances_with_prior(
            ckpt_masks_bool, np.array(ckpt_scores, dtype=np.float32), aux_prior.astype(bool),
            clip_to_prior=CLIP_TO_AUX, cover_thr=COVER_THR, rescore_alpha=RESCORE_ALPHA
        )
        union_ref_ckpt = union_filt_ckpt
        if ADD_TEACHER_MISSING:
            try:
                from scipy.ndimage import distance_transform_edt, label
                near_mask_ck = distance_transform_edt(~union_ckpt) <= PROMOTE_MAX_DIST_PX
                lab, n = label(aux_prior.astype(np.uint8))
                promote_ck = np.zeros_like(aux_prior, dtype=bool)
                for i in range(1, n + 1):
                    comp = (lab == i)
                    if comp.sum() < PROMOTE_MIN_AREA: continue
                    if (comp & near_mask_ck).any():
                        promote_ck |= comp
                union_ref_ckpt = (union_ref_ckpt | promote_ck)
            except Exception:
                pass
        m_ref_ckpt = _pix_metrics(union_ref_ckpt.astype(bool), gt_union, bound_tol=BOUND_TOL_PX)
        mets_ref_ckpt.append(m_ref_ckpt)
        for mk, sc in _normalize_filtered_preds(preds_filt_ckpt):
            preds_ref_ckpt_ap.append({"img_id": img_id, "score": float(sc), "mask": mk.astype(bool)})

        rows.append({
            "image": str(rel),
            "Dice_base_raw": m_base_raw["Dice"], "IoU_base_raw": m_base_raw["IoU"], "Prec_base_raw": m_base_raw["Precision"],
            "Rec_base_raw": m_base_raw["Recall"], "BF_base_raw": m_base_raw["BoundaryF"], "Cov_base_raw": m_base_raw["Coverage"],
            "Dice_base_ckpt": m_base_ckpt["Dice"], "IoU_base_ckpt": m_base_ckpt["IoU"], "Prec_base_ckpt": m_base_ckpt["Precision"],
            "Rec_base_ckpt": m_base_ckpt["Recall"], "BF_base_ckpt": m_base_ckpt["BoundaryF"], "Cov_base_ckpt": m_base_ckpt["Coverage"],
            "Dice_aux": m_aux["Dice"], "IoU_aux": m_aux["IoU"], "Prec_aux": m_aux["Precision"],
            "Rec_aux": m_aux["Recall"], "BF_aux": m_aux["BoundaryF"], "Cov_aux": m_aux["Coverage"],
            "Dice_ref_raw":  m_ref_raw["Dice"],  "IoU_ref_raw":  m_ref_raw["IoU"],  "Prec_ref_raw":  m_ref_raw["Precision"],
            "Rec_ref_raw":  m_ref_raw["Recall"], "BF_ref_raw":  m_ref_raw["BoundaryF"], "Cov_ref_raw":  m_ref_raw["Coverage"],
            "Dice_ref_ckpt": m_ref_ckpt["Dice"], "IoU_ref_ckpt": m_ref_ckpt["IoU"], "Prec_ref_ckpt": m_ref_ckpt["Precision"],
            "Rec_ref_ckpt": m_ref_ckpt["Recall"], "BF_ref_ckpt": m_ref_ckpt["BoundaryF"], "Cov_ref_ckpt": m_ref_ckpt["Coverage"],
        })

    def _mean(mets, key): return float(np.mean([m[key] for m in mets])) if mets else 0.0
    summary_pix = {
        "BaselineRAW":  {k: _mean(mets_base_raw,  k) for k in ["Dice","IoU","Precision","Recall","BoundaryF","Coverage"]},
        "BaselineCKPT": {k: _mean(mets_base_ckpt, k) for k in ["Dice","IoU","Precision","Recall","BoundaryF","Coverage"]},
        "AuxHead":      {k: _mean(mets_aux,       k) for k in ["Dice","IoU","Precision","Recall","BoundaryF","Coverage"]},
        "RefinedRAW":   {k: _mean(mets_ref_raw,   k) for k in ["Dice","IoU","Precision","Recall","BoundaryF","Coverage"]},
        "RefinedCKPT":  {k: _mean(mets_ref_ckpt,  k) for k in ["Dice","IoU","Precision","Recall","BoundaryF","Coverage"]},
    }

    ap50_raw, map_raw = _eval_map(preds_base_raw_ap,  gts_by_img_for_ap)
    ap50_ckp, map_ckp = _eval_map(preds_base_ckpt_ap, gts_by_img_for_ap)
    ap50_aux, map_aux = _eval_map(preds_aux_ap,       gts_by_img_for_ap)
    ap50_rrw, map_rrw = _eval_map(preds_ref_raw_ap,   gts_by_img_for_ap)
    ap50_rck, map_rck = _eval_map(preds_ref_ckpt_ap,  gts_by_img_for_ap)

    summary_ap = {
        "BaselineRAW":  {"AP50": ap50_raw, "mAP50_95": map_raw},
        "BaselineCKPT": {"AP50": ap50_ckp, "mAP50_95": map_ckp},
        "AuxHead":      {"AP50": ap50_aux, "mAP50_95": map_aux},
        "RefinedRAW":   {"AP50": ap50_rrw, "mAP50_95": map_rrw},
        "RefinedCKPT":  {"AP50": ap50_rck, "mAP50_95": map_rck},
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-image CSV
    per_image_csv = out_dir / "test_per_image.csv"
    with open(per_image_csv, "w", encoding="utf-8") as f:
        cols = [
            "image",
            "Dice_base_raw","IoU_base_raw","Prec_base_raw","Rec_base_raw","BF_base_raw","Cov_base_raw",
            "Dice_base_ckpt","IoU_base_ckpt","Prec_base_ckpt","Rec_base_ckpt","BF_base_ckpt","Cov_base_ckpt",
            "Dice_aux","IoU_aux","Prec_aux","Rec_aux","BF_aux","Cov_aux",
            "Dice_ref_raw","IoU_ref_raw","Prec_ref_raw","Rec_ref_raw","BF_ref_raw","Cov_ref_raw",
            "Dice_ref_ckpt","IoU_ref_ckpt","Prec_ref_ckpt","Rec_ref_ckpt","BF_ref_ckpt","Cov_ref_ckpt",
        ]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join([str(r[c]) for c in cols]) + "\n")

    # Pixel summary CSV (with frozen thresholds)
    summary_csv = out_dir / "test_summary_pixel.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("track,YOLO_conf_raw,YOLO_conf_ckpt,AUX_thr,Dice,IoU,Precision,Recall,BoundaryF,Coverage\n")
        def _line(name, blk):
            return f"{name},{best_conf_raw:.2f},{best_conf_ckpt:.2f},{best_aux_thr:.2f}," \
                   f"{blk['Dice']:.6f},{blk['IoU']:.6f},{blk['Precision']:.6f}," \
                   f"{blk['Recall']:.6f},{blk['BoundaryF']:.6f},{blk['Coverage']:.4f}\n"
        for name in ["BaselineRAW","BaselineCKPT","AuxHead","RefinedRAW","RefinedCKPT"]:
            f.write(_line(name, summary_pix[name]))

    # AP summary CSV
    ap_csv = out_dir / "test_summary_ap.csv"
    with open(ap_csv, "w", encoding="utf-8") as f:
        f.write("track,AP50,mAP50_95\n")
        for name in ["BaselineRAW","BaselineCKPT","AuxHead","RefinedRAW","RefinedCKPT"]:
            f.write(f"{name},{summary_ap[name]['AP50']:.6f},{summary_ap[name]['mAP50_95']:.6f}\n")

    print("\n==== TEST (frozen thresholds from VAL) ====")
    print(f"[Thresholds] YOLO_conf_raw = {best_conf_raw:.2f} | YOLO_conf_ckpt = {best_conf_ckpt:.2f} | AUX_THR = {best_aux_thr:.2f}\n")

    def _fmt_pix(block):
        return (f"Dice: {block['Dice']:.4f} | IoU: {block['IoU']:.4f} | "
                f"P: {block['Precision']:.4f} | R: {block['Recall']:.4f} | "
                f"BF: {block['BoundaryF']:.4f} | Cov%: {block['Coverage']:.2f}")
    def _fmt_ap(block):
        return f"AP50: {block['AP50']:.4f} | mAP50-95: {block['mAP50_95']:.4f}"

    for name in ["BaselineRAW","BaselineCKPT","AuxHead","RefinedRAW","RefinedCKPT"]:
        print(f"[{name} PIX]  ", _fmt_pix(summary_pix[name]))
        print(f"[{name} AP ]  ", _fmt_ap(summary_ap[name]))
        print("-"*72)

    print(f"\nSaved: {per_image_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {ap_csv}")

    return dict(pixel=summary_pix, ap=summary_ap,
                per_image_csv=str(per_image_csv),
                pixel_csv=str(summary_csv),
                ap_csv=str(ap_csv))

# ---------------- batch driver ----------------
def main():
    device = torch.device(f"cuda:{DEVICE}" if str(DEVICE).lower() != "cpu" and torch.cuda.is_available() else "cpu")
    base_out = Path(PROJECT).resolve() / RUN_NAME
    base_out.mkdir(parents=True, exist_ok=True)

    # For per-testset aggregation across ckpts
    per_testset_pixel_rows = {ts["name"]: [] for ts in TEST_SETS}
    per_testset_ap_rows    = {ts["name"]: [] for ts in TEST_SETS}

    for ckpt_name, ckpt_path in CKPTS.items():
        print(f"\n================ CKPT: {ckpt_name} ================")
        ckpt_out = base_out / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)

        # 1) Build aux bundle (ckpt student for features + teacher) to produce aux prior
        aux_bundle = build_aux_and_fusion(STUDENT_MODEL, TEACHER_MODEL, ckpt_path, device)

        # 2) Build two predictor students:
        yolo_raw  = YOLO(STUDENT_MODEL)
        yolo_ckpt = YOLO(STUDENT_MODEL)
        try:
            ck = torch.load(ckpt_path, map_location="cpu")
            student_sd = ck.get("student", {})
            yolo_ckpt.model.load_state_dict(student_sd, strict=False)
        except Exception as e:
            print(f"[WARN] could not load ckpt student weights into CKPT predictor: {e}")

        # 3) Calibrate thresholds on VAL (freeze them for all test sets)
        print("Calibrating on VAL...")
        best_conf_raw  = calibrate_yolo_on_val(
            yolo_raw,  CAM_YAML_VAL, MASK_SUFFIX, YOLO_CONF_GRID, ckpt_out,
            tag="raw",  val_mask_root_opt=EXPLICIT_VAL_MASK_ROOT
        )
        best_conf_ckpt = calibrate_yolo_on_val(
            yolo_ckpt, CAM_YAML_VAL, MASK_SUFFIX, YOLO_CONF_GRID, ckpt_out,
            tag="ckpt", val_mask_root_opt=EXPLICIT_VAL_MASK_ROOT
        )
        best_aux_thr   = calibrate_aux_on_val(
            aux_bundle, CAM_YAML_VAL, MAN_YAML_VAL, MASK_SUFFIX, AUX_THR_GRID, ckpt_out,
            val_mask_root_opt=EXPLICIT_VAL_MASK_ROOT
        )

        # 4) Loop over TEST sets
        for ts in TEST_SETS:
            ts_name   = ts["name"]
            cam_yaml  = ts["cam_yaml"]
            man_yaml  = ts["man_yaml"]
            mask_root = ts["mask_root"]

            ts_out = ckpt_out / ts_name
            print(f"\n--- Evaluating CKPT={ckpt_name} on TEST={ts_name} ---")
            res = evaluate_on_test(
                aux_bundle,
                yolo_raw, yolo_ckpt,
                cam_yaml, man_yaml, MASK_SUFFIX,
                mask_root,
                best_conf_raw, best_conf_ckpt, best_aux_thr,
                ts_out
            )

            # Append to per-testset aggregation rows (one line per ckpt)
            pix = res["pixel"]
            ap  = res["ap"]
            per_testset_pixel_rows[ts_name].append({
                "ckpt": ckpt_name,
                "YOLO_conf_raw":  f"{best_conf_raw:.2f}",
                "YOLO_conf_ckpt": f"{best_conf_ckpt:.2f}",
                "AUX_thr":        f"{best_aux_thr:.2f}",
                "BaselineRAW_Dice":  pix["BaselineRAW"]["Dice"],
                "BaselineCKPT_Dice": pix["BaselineCKPT"]["Dice"],
                "AuxHead_Dice":      pix["AuxHead"]["Dice"],
                "RefinedRAW_Dice":   pix["RefinedRAW"]["Dice"],
                "RefinedCKPT_Dice":  pix["RefinedCKPT"]["Dice"],
                "BaselineRAW_IoU":   pix["BaselineRAW"]["IoU"],
                "BaselineCKPT_IoU":  pix["BaselineCKPT"]["IoU"],
                "AuxHead_IoU":       pix["AuxHead"]["IoU"],
                "RefinedRAW_IoU":    pix["RefinedRAW"]["IoU"],
                "RefinedCKPT_IoU":   pix["RefinedCKPT"]["IoU"],
            })
            per_testset_ap_rows[ts_name].append({
                "ckpt": ckpt_name,
                "AP50_BaselineRAW":  ap["BaselineRAW"]["AP50"],
                "mAP50_95_BaselineRAW": ap["BaselineRAW"]["mAP50_95"],
                "AP50_BaselineCKPT": ap["BaselineCKPT"]["AP50"],
                "mAP50_95_BaselineCKPT": ap["BaselineCKPT"]["mAP50_95"],
                "AP50_AuxHead":      ap["AuxHead"]["AP50"],
                "mAP50_95_AuxHead":  ap["AuxHead"]["mAP50_95"],
                "AP50_RefinedRAW":   ap["RefinedRAW"]["AP50"],
                "mAP50_95_RefinedRAW": ap["RefinedRAW"]["mAP50_95"],
                "AP50_RefinedCKPT":  ap["RefinedCKPT"]["AP50"],
                "mAP50_95_RefinedCKPT": ap["RefinedCKPT"]["mAP50_95"],
            })

        # clean hooks for this ckpt
        aux_bundle["s_hook"].remove()
        aux_bundle["t_hook"].remove()

    # 5) Write per-testset aggregated summaries (all ckpts)
    for ts in TEST_SETS:
        ts_name = ts["name"]
        agg_dir = base_out / "aggregates" / ts_name
        agg_dir.mkdir(parents=True, exist_ok=True)

        # ----- ORDER rows as: fold0..4 per ablation, ablation order = ABLATIONS -----
        ordered_keys = []
        for abl in ABLATIONS:
            for f in FOLDS:
                key = f"fold{f}/{abl}"
                if any(r["ckpt"] == key for r in per_testset_pixel_rows[ts_name]):
                    ordered_keys.append(key)

        # ----- Write PIXEL aggregate with the requested ordering -----
        pixel_csv = agg_dir / "all_models_summary_pixel.csv"
        pixel_cols = [
            "ckpt","YOLO_conf_raw","YOLO_conf_ckpt","AUX_thr",
            "BaselineRAW_Dice","BaselineCKPT_Dice","AuxHead_Dice","RefinedRAW_Dice","RefinedCKPT_Dice",
            "BaselineRAW_IoU","BaselineCKPT_IoU","AuxHead_IoU","RefinedRAW_IoU","RefinedCKPT_IoU",
        ]
        with open(pixel_csv, "w", encoding="utf-8") as f:
            f.write(",".join(pixel_cols) + "\n")
            for key in ordered_keys:
                row = next(r for r in per_testset_pixel_rows[ts_name] if r["ckpt"] == key)
                f.write(",".join([str(row[c]) for c in pixel_cols]) + "\n")

        # ----- Write AP aggregate with the requested ordering -----
        ap_csv = agg_dir / "all_models_summary_ap.csv"
        ap_cols = [
            "ckpt",
            "AP50_BaselineRAW","mAP50_95_BaselineRAW",
            "AP50_BaselineCKPT","mAP50_95_BaselineCKPT",
            "AP50_AuxHead","mAP50_95_AuxHead",
            "AP50_RefinedRAW","mAP50_95_RefinedRAW",
            "AP50_RefinedCKPT","mAP50_95_RefinedCKPT",
        ]
        with open(ap_csv, "w", encoding="utf-8") as f:
            f.write(",".join(ap_cols) + "\n")
            for key in ordered_keys:
                row = next(r for r in per_testset_ap_rows[ts_name] if r["ckpt"] == key)
                f.write(",".join([str(row[c]) for c in ap_cols]) + "\n")

        print(f"[Aggregate saved] {pixel_csv}")
        print(f"[Aggregate saved] {ap_csv}")

        # ===== NEW: per-ablation averages (mean ± std) across folds =====
        import numpy as _np

        # Helper: collect rows by ablation
        def _rows_for_ablation_pixel(abl):
            return [r for r in per_testset_pixel_rows[ts_name] if r["ckpt"].endswith(f"/{abl}")]
        def _rows_for_ablation_ap(abl):
            return [r for r in per_testset_ap_rows[ts_name] if r["ckpt"].endswith(f"/{abl}")]

        # ---- Pixel means & stds across folds ----
        pixel_mean_csv = agg_dir / "by_ablation_fold_mean_pixel.csv"
        pixel_mean_cols = [
            "ablation","n_folds",
            "YOLO_conf_raw_mean","YOLO_conf_ckpt_mean","AUX_thr_mean",
            "BaselineRAW_Dice_mean","BaselineRAW_Dice_std",
            "BaselineCKPT_Dice_mean","BaselineCKPT_Dice_std",
            "AuxHead_Dice_mean","AuxHead_Dice_std",
            "RefinedRAW_Dice_mean","RefinedRAW_Dice_std",
            "RefinedCKPT_Dice_mean","RefinedCKPT_Dice_std",
            "BaselineRAW_IoU_mean","BaselineRAW_IoU_std",
            "BaselineCKPT_IoU_mean","BaselineCKPT_IoU_std",
            "AuxHead_IoU_mean","AuxHead_IoU_std",
            "RefinedRAW_IoU_mean","RefinedRAW_IoU_std",
            "RefinedCKPT_IoU_mean","RefinedCKPT_IoU_std",
        ]
        with open(pixel_mean_csv, "w", encoding="utf-8") as f:
            f.write(",".join(pixel_mean_cols) + "\n")
            for abl in ABLATIONS:
                rows = _rows_for_ablation_pixel(abl)
                if not rows:
                    continue
                n = len(rows)
                def vals(key): return _np.array([float(r[key]) for r in rows], dtype=float)
                def ms(key):  a = vals(key); return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)

                y_raw_m = _np.array([float(r["YOLO_conf_raw"])  for r in rows]).mean()
                y_ckp_m = _np.array([float(r["YOLO_conf_ckpt"]) for r in rows]).mean()
                aux_m   = _np.array([float(r["AUX_thr"])        for r in rows]).mean()

                def mstd(prefix, metric):
                    m, s = ms(f"{prefix}_{metric}")
                    return [f"{m:.6f}", f"{s:.6f}"]

                line = [
                    abl, str(n),
                    f"{y_raw_m:.4f}", f"{y_ckp_m:.4f}", f"{aux_m:.4f}",
                    *mstd("BaselineRAW","Dice"),
                    *mstd("BaselineCKPT","Dice"),
                    *mstd("AuxHead","Dice"),
                    *mstd("RefinedRAW","Dice"),
                    *mstd("RefinedCKPT","Dice"),
                    *mstd("BaselineRAW","IoU"),
                    *mstd("BaselineCKPT","IoU"),
                    *mstd("AuxHead","IoU"),
                    *mstd("RefinedRAW","IoU"),
                    *mstd("RefinedCKPT","IoU"),
                ]
                f.write(",".join(map(str, line)) + "\n")

        # ---- AP means & stds across folds ----
        ap_mean_csv = agg_dir / "by_ablation_fold_mean_ap.csv"
        ap_mean_cols = [
            "ablation","n_folds",
            "AP50_BaselineRAW_mean","AP50_BaselineRAW_std","mAP50_95_BaselineRAW_mean","mAP50_95_BaselineRAW_std",
            "AP50_BaselineCKPT_mean","AP50_BaselineCKPT_std","mAP50_95_BaselineCKPT_mean","mAP50_95_BaselineCKPT_std",
            "AP50_AuxHead_mean","AP50_AuxHead_std","mAP50_95_AuxHead_mean","mAP50_95_AuxHead_std",
            "AP50_RefinedRAW_mean","AP50_RefinedRAW_std","mAP50_95_RefinedRAW_mean","mAP50_95_RefinedRAW_std",
            "AP50_RefinedCKPT_mean","AP50_RefinedCKPT_std","mAP50_95_RefinedCKPT_mean","mAP50_95_RefinedCKPT_std",
        ]
        with open(ap_mean_csv, "w", encoding="utf-8") as f:
            f.write(",".join(ap_mean_cols) + "\n")
            for abl in ABLATIONS:
                rows = _rows_for_ablation_ap(abl)
                if not rows:
                    continue
                n = len(rows)
                def mstd(key):
                    a = _np.array([float(r[key]) for r in rows], dtype=float)
                    m = a.mean()
                    s = a.std(ddof=1) if len(a) > 1 else 0.0
                    return f"{m:.6f}", f"{s:.6f}"

                line = [
                    abl, str(n),
                    *mstd("AP50_BaselineRAW"), *mstd("mAP50_95_BaselineRAW"),
                    *mstd("AP50_BaselineCKPT"), *mstd("mAP50_95_BaselineCKPT"),
                    *mstd("AP50_AuxHead"),      *mstd("mAP50_95_AuxHead"),
                    *mstd("AP50_RefinedRAW"),   *mstd("mAP50_95_RefinedRAW"),
                    *mstd("AP50_RefinedCKPT"),  *mstd("mAP50_95_RefinedCKPT"),
                ]
                f.write(",".join(map(str, line)) + "\n")

        print(f"[Per-ablation means saved] {pixel_mean_csv}")
        print(f"[Per-ablation means saved] {ap_mean_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
