# test_dual_aux_family_run_test_only_metrics.py
# TEST-only run with fixed thresholds + terminal logging for progress, params, MACs, FPS/latency, and memory.

from pathlib import Path
import os, sys, re, json, yaml, time, datetime
import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import psutil  # pip install psutil
try:
    from thop import profile as thop_profile  # pip install thop
except Exception:
    thop_profile = None

from smart_prior import refine_aux_prior, filter_student_instances_with_prior

# ================= USER CONFIG =================
STUDENT_MODEL = # file path (model .pt pretrained on camera data )
TEACHER_MODEL = # file path (model .pt pretrained on manual data )

# pick the aux-on ablation ckpt you’re testing (fuse_aux_only / fuse_6_aux / fuse_10_aux / fuse_6_10_aux)
CKPT_PATH     = # file path (model .pt trained on dual data )

CAM_YAML      = # file path (yaml file include the file path to test data camera image and label)
MAN_YAML      = # file path (yaml file include the file path to test data manual image and label)

MASK_SUFFIX             = "_rle.json"
EXPLICIT_TEST_MASK_ROOT = # file path (test data camera label)

IMG_SIZE      = 640
DEVICE        = "0"   # "0" or "cpu"

# ---- Fixed thresholds (no validation) ----
YOLO_CONF_RAW  = 0.20
YOLO_CONF_CKPT = 0.20
AUX_THR        = 0.40

# Refinement / filtering knobs
PRIOR_NEAR_THR   = 0.2    # LOGIT space
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

# Teacher-only promotion (optional)
ADD_TEACHER_MISSING = True
PROMOTE_MIN_AREA    = 1400
PROMOTE_MAX_DIST_PX = 30

# Boundary F tolerance (pixels at original size)
BOUND_TOL_PX = 2
# ===============================================

def module_device(module: torch.nn.Module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def try_flops(module: torch.nn.Module, input_shape=(1,3,640,640)):
    if thop_profile is None:
        return None, None
    try:
        dev = module_device(module)
        dummy = torch.zeros(*input_shape, device=dev)
        module.eval()
        with torch.no_grad():
            macs, params = thop_profile(module, inputs=(dummy,), verbose=False)
        return macs, params  # MACs
    except Exception:
        return None, None


def ts():
    return f"[{datetime.datetime.now().strftime('%H:%M:%S')}] INFO:"

def human_bytes(n):
    for unit in ['B','KiB','MiB','GiB','TiB']:
        if n < 1024.0:
            return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"

def count_params_and_size(module: torch.nn.Module):
    params = sum(p.numel() for p in module.parameters())
    dtypes = {p.dtype for p in module.parameters()}
    itemsize = max(torch.finfo(dt).bits for dt in dtypes) // 8 if dtypes else 4
    size_bytes = params * itemsize
    return params, size_bytes

def try_flops(module: torch.nn.Module, input_shape=(1,3,640,640), device='cpu'):
    if thop_profile is None:
        return None, None
    try:
        dummy = torch.zeros(*input_shape, device=device)
        module.eval()
        with torch.no_grad():
            macs, params = thop_profile(module, inputs=(dummy,), verbose=False)
        return macs, params  # MACs
    except Exception:
        return None, None

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
                Recall=float(rec), BoundaryF=float(bf))

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

# ---------------- feature hooks + fusion/seg ----------------
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

    # load distilled student for aux path feature taps
    try:
        y_s.model.load_state_dict(student_sd, strict=False)
    except Exception as e:
        print(f"{ts()} [WARN] could not load ckpt student weights into student YOLO (aux path): {e}")

    # choose aux tap for model.23
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
        s_aux=s_aux, t_aux=t_aux,
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

def yolo_union_mask(yolo_model: YOLO, img_path: Path, conf_th: float, img_size: int, out_hw):
    device_arg = _predict_device_arg_from_model(yolo_model.model)
    res = yolo_model.predict(
        source=str(img_path), imgsz=img_size, conf=conf_th,
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
        source=str(img_path), imgsz=img_size, conf=conf_th,
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

class AvgTimer:
    def __init__(self):
        self.t = 0.0
        self.n = 0
    def add(self, dt):
        self.t += dt
        self.n += 1
    @property
    def avg(self):
        return (self.t / max(1,self.n))
    @property
    def fps(self):
        return (self.n / self.t) if self.t > 0 else 0.0

def normalize_filtered_preds(preds_filt):
    """
    Normalize various prediction container shapes into a list of (mask_bool, score_float).

    Supported inputs:
      - {"masks": [...], "scores": [...]}  (or keys "mask"/"score", "segments"/"confs")
      - [{"mask": ..., "score": ...}, ...]
      - (masks_list, scores_list)
      - [(mask, score), ...]
    Returns:
      List[Tuple[np.ndarray(bool HxW), float]]
    """
    out = []
    if preds_filt is None:
        return out

    # Case A: dict of arrays/lists
    if isinstance(preds_filt, dict):
        # common key variants
        masks = preds_filt.get("masks", preds_filt.get("mask", preds_filt.get("segments")))
        scores = preds_filt.get("scores", preds_filt.get("score", preds_filt.get("confs")))
        if masks is not None and scores is not None:
            for mk, sc in zip(masks, scores):
                out.append((np.array(mk).astype(bool), float(sc)))
            return out
        # sometimes wrapped under 'preds'
        preds_list = preds_filt.get("preds")
        if isinstance(preds_list, (list, tuple)):
            return normalize_filtered_preds(preds_list)
        return out  # unknown dict shape

    # Case B: list of dicts [{"mask":..,"score":..}, ...]
    if isinstance(preds_filt, (list, tuple)) and preds_filt and isinstance(preds_filt[0], dict):
        for d in preds_filt:
            mk = d.get("mask", d.get("m", d.get("segment")))
            sc = d.get("score", d.get("conf", 1.0))
            if mk is None:
                continue
            out.append((np.array(mk).astype(bool), float(sc)))
        return out

    # Case C: tuple/list of (masks_list, scores_list)
    if isinstance(preds_filt, (tuple, list)) and len(preds_filt) == 2 and \
       isinstance(preds_filt[0], (list, tuple, np.ndarray)):
        masks, scores = preds_filt
        for mk, sc in zip(masks, scores):
            out.append((np.array(mk).astype(bool), float(sc)))
        return out

    # Case D: list of (mask, score) tuples
    if isinstance(preds_filt, (list, tuple)) and preds_filt and \
       isinstance(preds_filt[0], (tuple, list)) and len(preds_filt[0]) == 2:
        for mk, sc in preds_filt:
            out.append((np.array(mk).astype(bool), float(sc)))
        return out

    # Fallback: unknown type/shape
    return out


def evaluate_on_test(aux_bundle,
                     yolo_student_raw: YOLO, yolo_student_ckpt: YOLO,
                     cam_yaml: str, man_yaml: str, mask_suffix: str,
                     test_mask_root_opt: str,
                     best_conf_raw: float, best_conf_ckpt: float, best_aux_thr: float):

    # Prep splits
    cam_test = _read_yaml_images_root(Path(cam_yaml), "test")
    man_test = _read_yaml_images_root(Path(man_yaml), "test")
    mask_test = Path(test_mask_root_opt).resolve() if test_mask_root_opt else (Path(cam_test).parents[1] / "mask_labels" / "test" / "camera")

    img_paths = sorted([p for p in cam_test.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not img_paths:
        raise RuntimeError(f"No TEST images found under {cam_test}")

    # Logging header
    print(f"{ts()} == TEST evaluation: AuxHead + Refined filters from YOLO ==")
    print(f"{ts()} TEST images found: {len(img_paths)} under {cam_test}")

    # Timers and memory baselines
    start_wall = time.time()
    yolo_raw_timer   = AvgTimer()
    yolo_ckpt_timer  = AvgTimer()
    aux_timer        = AvgTimer()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(aux_bundle["device"])
    proc = psutil.Process(os.getpid())
    cpu_mem_start = proc.memory_info().rss

    mets_base_raw, mets_base_ckpt, mets_aux, mets_ref_raw, mets_ref_ckpt = [], [], [], [], []
    preds_base_raw_ap, preds_base_ckpt_ap, preds_aux_ap, preds_ref_raw_ap, preds_ref_ckpt_ap = [], [], [], [], []
    gts_by_img_for_ap = {}

    N = len(img_paths)
    for idx, cam_path in enumerate(img_paths, start=1):
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
        gts_by_img_for_ap[idx] = gts_bool

        # (1) Baseline RAW
        t0 = time.time()
        union_raw, raw_masks_bool, raw_scores = _collect_instances(yolo_student_raw, cam_path, best_conf_raw, IMG_SIZE, (H0, W0))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        yolo_raw_timer.add(time.time() - t0)
        m_base_raw = _pix_metrics(union_raw, gt_union, bound_tol=BOUND_TOL_PX)
        mets_base_raw.append(m_base_raw)
        for mk, sc in zip(raw_masks_bool, raw_scores):
            preds_base_raw_ap.append({"img_id": idx, "score": float(sc), "mask": mk.astype(bool)})

        # (2) Baseline CKPT
        t0 = time.time()
        union_ckpt, ckpt_masks_bool, ckpt_scores = _collect_instances(yolo_student_ckpt, cam_path, best_conf_ckpt, IMG_SIZE, (H0, W0))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        yolo_ckpt_timer.add(time.time() - t0)
        m_base_ckpt = _pix_metrics(union_ckpt, gt_union, bound_tol=BOUND_TOL_PX)
        mets_base_ckpt.append(m_base_ckpt)
        for mk, sc in zip(ckpt_masks_bool, ckpt_scores):
            preds_base_ckpt_ap.append({"img_id": idx, "score": float(sc), "mask": mk.astype(bool)})

        # (3) AuxHead (pixel + instances from components)
        t0 = time.time()
        logits_np, prob_np = aux_logits_and_prob(aux_bundle, cam_path, man_path, (H0, W0))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        aux_timer.add(time.time() - t0)
        if prob_np is None:
            continue
        aux_bin = (prob_np > float(best_aux_thr))
        m_aux = _pix_metrics(aux_bin, gt_union, bound_tol=BOUND_TOL_PX)
        mets_aux.append(m_aux)
        aux_components = _connected_components_bool(aux_bin)
        for d in _build_instance_preds_from_components(aux_components, base_score=1.0):
            preds_aux_ap.append({"img_id": idx, "score": d["score"], "mask": d["mask"]})

        # Build refined aux prior (logit space) once (relative to RAW union)
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

        # (4) RefinedRAW: filter RAW student instances
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
        for mk, sc in normalize_filtered_preds(preds_filt_raw):
            preds_ref_raw_ap.append({"img_id": idx, "score": float(sc), "mask": mk.astype(bool)})

        # (5) RefinedCKPT: filter CKPT student instances
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
        for mk, sc in normalize_filtered_preds(preds_filt_ckpt):
            preds_ref_ckpt_ap.append({"img_id": idx, "score": float(sc), "mask": mk.astype(bool)})

        # periodic progress log (every 25 images)
        if (idx % 25) == 0 or idx == N:
            pct = int((idx / N) * 100)
            elapsed = time.time() - start_wall
            print(f"TEST: {idx:4d}/{N:4d} [{elapsed:06.2f}s] {ts()}   processed {idx}/{N} images ({pct}%)")

    # Aggregate pixel-wise means
    def _mean(mets, key): return float(np.mean([m[key] for m in mets])) if mets else 0.0
    summary_pix = {
        "BaselineRAW":  {k: _mean(mets_base_raw,  k) for k in ["Dice","IoU","Precision","Recall","BoundaryF"]},
        "BaselineCKPT": {k: _mean(mets_base_ckpt, k) for k in ["Dice","IoU","Precision","Recall","BoundaryF"]},
        "AuxHead":      {k: _mean(mets_aux,       k) for k in ["Dice","IoU","Precision","Recall","BoundaryF"]},
        "RefinedRAW":   {k: _mean(mets_ref_raw,   k) for k in ["Dice","IoU","Precision","Recall","BoundaryF"]},
        "RefinedCKPT":  {k: _mean(mets_ref_ckpt,  k) for k in ["Dice","IoU","Precision","Recall","BoundaryF"]},
    }

    # AP/mAP instance metrics
    ap50_raw, map_raw = _eval_map(preds_base_raw_ap,  gts_by_img_for_ap)
    ap50_ckp, map_ckp = _eval_map(preds_base_ckpt_ap, gts_by_img_for_ap)
    ap50_aux, map_aux = _eval_map(preds_aux_ap,       gts_by_img_for_ap)
    ap50_rrw, map_rrw = _eval_map(preds_ref_raw_ap,   gts_by_img_for_ap)
    ap50_rck, map_rck = _eval_map(preds_ref_ckpt_ap,  gts_by_img_for_ap)

    # Pretty print results
    def _fmt_pix(block):
        return (f"Dice {block['Dice']:.4f} | IoU {block['IoU']:.4f} | "
                f"P {block['Precision']:.4f} | R {block['Recall']:.4f} | BF {block['BoundaryF']:.4f}")
    def _fmt_ap(ap, mp): return f"AP50 {ap:.4f} | mAP50:95 {mp:.4f}"

    print()
    print(f"{ts()} [THR] YOLO_raw={best_conf_raw:.2f} | YOLO_ckpt={best_conf_ckpt:.2f} | AUX={best_aux_thr:.2f}\n")
    for name in ["BaselineRAW","BaselineCKPT","AuxHead","RefinedRAW","RefinedCKPT"]:
        print(f"{ts()} [{name}] {_fmt_pix(summary_pix[name])}")
        if name == "BaselineRAW":   print(f"{ts()} [{name}] {_fmt_ap(ap50_raw, map_raw)}")
        if name == "BaselineCKPT":  print(f"{ts()} [{name}] {_fmt_ap(ap50_ckp, map_ckp)}")
        if name == "AuxHead":       print(f"{ts()} [{name}] {_fmt_ap(ap50_aux, map_aux)}")
        if name == "RefinedRAW":    print(f"{ts()} [{name}] {_fmt_ap(ap50_rrw, map_rrw)}")
        if name == "RefinedCKPT":   print(f"{ts()} [{name}] {_fmt_ap(ap50_rck, map_rck)}")
        print("-"*72)

    # Timing + Memory
    elapsed = time.time() - start_wall
    overall_fps = (len(img_paths) / elapsed) if elapsed > 0 else 0.0

    gpu_peak_alloc = torch.cuda.max_memory_allocated(aux_bundle["device"]) if torch.cuda.is_available() else 0
    gpu_peak_res   = torch.cuda.max_memory_reserved(aux_bundle["device"]) if torch.cuda.is_available() else 0
    cpu_mem_end    = psutil.Process(os.getpid()).memory_info().rss

    print(f"{ts()} === Inference Timing ===")
    print(f"{ts()} Overall elapsed: {elapsed:.2f}s | Overall FPS: {overall_fps:.2f} img/s")
    print(f"{ts()} YOLO_raw  avg: {yolo_raw_timer.avg*1000:.2f} ms/img | FPS: {yolo_raw_timer.fps:.2f}")
    print(f"{ts()} YOLO_ckpt avg: {yolo_ckpt_timer.avg*1000:.2f} ms/img | FPS: {yolo_ckpt_timer.fps:.2f}")
    print(f"{ts()} AuxHead   avg: {aux_timer.avg*1000:.2f} ms/img | FPS: {aux_timer.fps:.2f}")
    print()
    print(f"{ts()} === Memory ===")
    print(f"{ts()} CPU RSS start: {human_bytes(cpu_mem_start)} -> end: {human_bytes(cpu_mem_end)} (Δ {human_bytes(max(0, cpu_mem_end-cpu_mem_start))})")
    if torch.cuda.is_available():
        print(f"{ts()} GPU peak allocated: {human_bytes(gpu_peak_alloc)} | peak reserved: {human_bytes(gpu_peak_res)}")
    else:
        print(f"{ts()} GPU: not available")
    print("-"*72)

def main():
    device = torch.device(f"cuda:{DEVICE}" if str(DEVICE).lower() != "cpu" and torch.cuda.is_available() else "cpu")

    # 1) Build aux bundle (ckpt student for features + teacher) to produce aux prior
    aux_bundle = build_aux_and_fusion(STUDENT_MODEL, TEACHER_MODEL, CKPT_PATH, device)

    # 2) Build two predictor students:
    #    - RAW:   pure best.pt (no ckpt load) -> BaselineRAW
    #    - CKPT:  best.pt + load student ckpt -> BaselineCKPT
    yolo_raw  = YOLO(STUDENT_MODEL)
    yolo_ckpt = YOLO(STUDENT_MODEL)
    try:
        ck = torch.load(CKPT_PATH, map_location="cpu")
        student_sd = ck.get("student", {})
        yolo_ckpt.model.load_state_dict(student_sd, strict=False)
    except Exception as e:
        print(f"{ts()} [WARN] could not load ckpt student weights into CKPT predictor: {e}")

    # ---- Model Stats (Params / MACs / size) ----
    print(f"{ts()} === Model Stats ===")
    student_core = yolo_raw.model
    teacher_core = aux_bundle["yolo_teacher"].model
    fusion_core  = aux_bundle["fusion"]
    seg_head     = aux_bundle["seg_head"]

    for name, m in [("Student(YOLO)", student_core),
                ("Teacher(YOLO)", teacher_core),
                ("Fusion",        fusion_core),
                ("AuxMaskHead",   seg_head)]:
        params, size_b = count_params_and_size(m)
        macs, _ = try_flops(m, input_shape=(1,3,IMG_SIZE,IMG_SIZE))  # <-- no .device access
        macs_str = f"{macs/1e9:.2f} GMac" if macs is not None else "n/a"
        print(f"{ts()} {name:14s} | Params: {params/1e6:.2f} M | Weights: {human_bytes(size_b)} | MACs@{IMG_SIZE}: {macs_str}")

    total_params = sum(count_params_and_size(m)[0] for m in [student_core, teacher_core, fusion_core, seg_head])
    print(f"{ts()} Total params (all parts): {total_params/1e6:.2f} M")
    print("-"*72)

    # 3) Evaluate on TEST with fixed thresholds (no VAL)
    test_mask_root = EXPLICIT_TEST_MASK_ROOT if EXPLICIT_TEST_MASK_ROOT else ""
    evaluate_on_test(
        aux_bundle,
        yolo_raw, yolo_ckpt,
        CAM_YAML, MAN_YAML, MASK_SUFFIX,
        test_mask_root,
        YOLO_CONF_RAW, YOLO_CONF_CKPT, AUX_THR
    )

    # 4) Clean hooks
    aux_bundle["s_hook"].remove()
    aux_bundle["t_hook"].remove()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{ts()} [ERROR] {e}")
        sys.exit(1)

