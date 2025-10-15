# test_dual_fusion_only.py — dual-input fusion eval with channel-compat loader
# save the overlay masks images for visualization (test data only)
from pathlib import Path
import sys, yaml, json, numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
# new: smart teacher prior & filtering
from smart_prior import refine_aux_prior, filter_student_instances_with_prior

# ========== USER EDITABLE ==========
STUDENT_MODEL = # file path (model .pt pretrained on camera data )
TEACHER_MODEL = # file path (model .pt pretrained on manual data )
CKPT_PATH     = # file path (model .pt trained on dual input )
IMG_SIZE      = 640
DEVICE        = "0"                           # "0" | "cpu"
PROJECT       = # file path (output directory )
RUN_NAME      = "model.6_10_23"

AUX_HEAD_TAP  = "model.23"
NUM_CLASSES   = 1
CONF_TH       = 0.2 #0.5
CAM_YAML      = # file path (yaml file include the file path of validation and test data, camera images and labels)
MAN_YAML      = # file path (yaml file include the file path of validation and test data, manual images and labels)

MASK_ROOT_CAMERA = # file path
MASK_SUFFIX      = "_rle.json"

# ---- AUX overlay thresholding (for the blue mask) ----
AUX_THRESH = 0.4                     # single operating point for the aux head overlay/metrics
AUX_SWEEP  = None  # e.g., np.linspace(0.05, 0.95, 19) to report best-per-image Dice
# ===================================
# ---- Teacher prior refinement & filtering knobs ----
PRIOR_NEAR_THR   = 0.2
PRIOR_FAR_THR    = 0.9
PRIOR_NEAR_PX    = 35
PRIOR_FAR_CUT_PX = 60
PRIOR_MIN_AREA   = 300
PRIOR_ERODE_IT   = 0
PRIOR_OPEN_IT    = 1
PRIOR_CLOSE_IT   = 0
CLIP_TO_AUX      = True
COVER_THR        = 0.2
RESCORE_ALPHA    = 0.35

# near the top (after imports)
DEBUG_SAVE = False

def _save_gray(path, arr01):
    try:
        import cv2
        cv2.imwrite(str(path), (np.clip(arr01,0,1)*255).astype(np.uint8))
    except Exception:
        from PIL import Image
        Image.fromarray((np.clip(arr01,0,1)*255).astype(np.uint8)).save(path)

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

# ---------------- helpers ----------------
def _safe_name(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]+', '_', name)

def _resolve_images_root(yaml_path: Path) -> Path:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    split = cfg.get("test", cfg.get("val"))
    if split is None:
        raise ValueError("data.yaml has neither 'test' nor 'val'.")
    p = Path(split[0] if isinstance(split, (list, tuple)) else split)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Resolved split folder not found: {p}")
    return p

def _first_4d_tensor(x):
    if isinstance(x, torch.Tensor) and x.ndim == 4:
        return x
    if isinstance(x, (list, tuple)):
        for it in x:
            t = _first_4d_tensor(it)
            if t is not None:
                return t
    if isinstance(x, dict):
        for _, it in x.items():
            t = _first_4d_tensor(it)
            if t is not None:
                return t
    return None

def _decode_instances_from_json(json_path: Path):
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
            if m.ndim == 3:
                m = m[..., 0]
            masks.append((m > 0).astype(np.uint8))
    return masks

def _merge_binary(masks_list):
    merged = None
    for m in masks_list:
        merged = m if merged is None else np.maximum(merged, m)
    return merged

def _resize_bool(mask_np: np.ndarray, out_hw):
    H, W = out_hw
    if mask_np.shape == (H, W):
        return mask_np.astype(bool)
    pil = Image.fromarray((mask_np.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
    return (np.array(pil) > 127)

def _overlay_filled(img_uint8: np.ndarray, mask_bool: np.ndarray, color=(255,0,0), alpha=0.45):
    out = img_uint8.copy()
    overlay = np.zeros_like(out)
    overlay[mask_bool] = np.array(color, dtype=np.uint8)
    out = (overlay * alpha + out * (1.0 - alpha)).astype(np.uint8)
    out[~mask_bool] = img_uint8[~mask_bool]
    return out

def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    r = np.linspace(0, 1, 101)
    p_env = np.maximum.accumulate(precisions[::-1])[::-1]
    vals = []
    for rv in r:
        inds = np.where(recalls >= rv)[0]
        vals.append(p_env[inds[0]] if inds.size > 0 else 0.0)
    return float(np.mean(vals))

def _eval_map(preds, gts, iou_thresholds=np.arange(0.5, 0.96, 0.05)):
    preds = sorted(preds, key=lambda x: -x["score"])
    total_gts = sum(len(v) for v in gts.values())
    aps = {}
    for thr in iou_thresholds:
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        matched = {img_id: np.zeros(len(gts[img_id]), dtype=bool) for img_id in gts}
        for i, p in enumerate(preds):
            img_id = p["img_id"]
            gt_masks = gts.get(img_id, [])
            if len(gt_masks) == 0:
                fp[i] = 1.0
                continue
            ious = []
            for j, g in enumerate(gt_masks):
                if matched[img_id][j]:
                    ious.append(-1.0)
                else:
                    inter = np.logical_and(p["mask"], g).sum()
                    uni = np.logical_or(p["mask"], g).sum()
                    ious.append(0.0 if uni == 0 else float(inter) / float(uni))
            j_best = int(np.argmax(ious)) if ious else -1
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

def _dice_iou_bool(pred_bool: np.ndarray, gt_bool: np.ndarray, eps=1e-6):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    p_sum = pred_bool.sum()
    g_sum = gt_bool.sum()
    union = p_sum + g_sum - inter
    dice = (2 * inter + eps) / (p_sum + g_sum + eps)
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)

# ---------------- model bits ----------------
class StructureFusion(nn.Module):
    def __init__(self, s_channels: int, t_channels: int, d: int = 128, num_tokens: int = 4, use_film: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.d = d
        self.use_film = use_film
        self.t_gap = nn.AdaptiveAvgPool2d(1)
               # Ct -> d*K
        self.t_token = nn.Sequential(nn.Conv2d(t_channels, d * num_tokens, 1, bias=True), nn.ReLU(inplace=True))
        # Cs -> d
        self.q_proj = nn.Conv2d(s_channels, d, 1, bias=True)
        # d -> Cs
        self.out_proj = nn.Conv2d(d, s_channels, 1, bias=True)
        if use_film:
            # Ct -> 2*Cs
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
            gamma_beta = self.film(t_g)
            gamma, beta = gamma_beta.split(Cs, dim=1)
            fused = fused * torch.sigmoid(gamma) + beta
        return fused

class AuxMaskHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        mid = max(32, min(256, in_channels))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1, bias=True),
        )
    def forward(self, x: torch.Tensor, out_hw=None):
        y = self.block(x)
        if out_hw is not None:
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y

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
                h = m.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

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

def _prep_image(path: str, size=640):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return x

# ---------- weight-compat utilities ----------
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
    """Load a single fusion block, adapting in/out channels if they differ."""
    sub = _extract_substate(fusions_sd, safe_key)
    if not sub:
        return  # nothing saved for this key; run with fresh fusion

    # Current shapes
    qW = fusion.q_proj.weight
    tW = fusion.t_token[0].weight
    oW = fusion.out_proj.weight
    fW = fusion.film[0].weight if hasattr(fusion, "film") else None

    cur_Cs_in = qW.shape[1]
    cur_Ct_in = tW.shape[1]
    cur_Cs_out = oW.shape[0]
    cur_Film_out = fW.shape[0] if fW is not None else None
    cur_Film_in  = fW.shape[1] if fW is not None else None

    # Bring checkpoint tensors to expected dims
    adj = {}
    # t_token.0 (Conv2d in token generator)
    if "t_token.0.weight" in sub:
        w = sub["t_token.0.weight"]; b = sub.get("t_token.0.bias", None)
        adj["t_token.0.weight"] = _pad_in(w, cur_Ct_in)
        if b is not None: adj["t_token.0.bias"] = b  # out channels unchanged (d*K)
    # q_proj
    if "q_proj.weight" in sub:
        adj["q_proj.weight"] = _pad_in(sub["q_proj.weight"], cur_Cs_in)
        if "q_proj.bias" in sub: adj["q_proj.bias"] = sub["q_proj.bias"]
    # out_proj
    if "out_proj.weight" in sub:
        adj["out_proj.weight"] = _pad_out(sub["out_proj.weight"], cur_Cs_out)
    if "out_proj.bias" in sub:
        adj["out_proj.bias"] = _pad_bias(sub["out_proj.bias"], cur_Cs_out)
    # film
    if fW is not None:
        if "film.0.weight" in sub:
            w = sub["film.0.weight"]
            w = _pad_in(w, cur_Ct_in)      # pad/slice teacher-in channels
            w = _pad_out(w, cur_Film_out)  # pad/slice 2*Cs
            adj["film.0.weight"] = w
        if "film.0.bias" in sub:
            adj["film.0.bias"] = _pad_bias(sub["film.0.bias"], cur_Film_out)

    # Load adapted subset
    fusion.load_state_dict(adj, strict=False)

def _load_seg_head_with_channel_compat(seg: AuxMaskHead, seg_sd: dict):
    """Only the first conv depends on input channels; adapt it."""
    if not seg_sd:
        return
    # map into submodule names under .block
    w0 = seg_sd.get("block.0.weight", None)
    b0 = seg_sd.get("block.0.bias",   None)
    if w0 is not None:
        cur_in = seg.block[0].weight.shape[1]
        seg.block[0].weight.data.copy_(_pad_in(w0, cur_in))
    if b0 is not None:
        seg.block[0].bias.data.copy_(b0)
    # the rest load as-is
    for name, param in seg.named_parameters():
        if name in ("block.0.weight", "block.0.bias"):
            continue
        sd_name = name
        if sd_name in seg_sd:
            try:
                param.data.copy_(seg_sd[sd_name])
            except Exception:
                pass
    for name, buf in seg.named_buffers():
        if name in seg_sd and seg_sd[name].shape == buf.shape:
            buf.copy_(seg_sd[name])


# ---------------- main ----------------
def main():
    if mask_utils is None:
        print("[WARN] pycocotools not installed — Dice/IoU and mAP will be skipped.")
    device = torch.device(f"cuda:{DEVICE}" if str(DEVICE).lower() != "cpu" and torch.cuda.is_available() else "cpu")

    cam_root = _resolve_images_root(Path(CAM_YAML))
    man_root = _resolve_images_root(Path(MAN_YAML))
    mask_root = Path(MASK_ROOT_CAMERA).resolve() if MASK_ROOT_CAMERA else None

    out_dir = Path(PROJECT).resolve() / RUN_NAME
    (out_dir / "overlays_union").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays_auxfusion").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays_auxfusion_refined").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays_union_filtered").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays_promoted").mkdir(parents=True, exist_ok=True)

    # Models
    student = YOLO(STUDENT_MODEL).model.to(device).eval()
    teacher = YOLO(TEACHER_MODEL).model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Load ckpt bits
    ck = torch.load(CKPT_PATH, map_location="cpu")
    seg_sd     = ck.get("seg_adapter", None) or {}
    fusions_sd = ck.get("fusions", None) or {}
    fusion_meta = ck.get("fusion_meta", {}) or {}
    name_map = fusion_meta.get("name_map", {}) or {}
    dims   = fusion_meta.get("dims", {}) or {}
    tokens = fusion_meta.get("tokens", {}) or {}

    # Resolve exact trained taps from ckpt (preferred), else fall back to AUX_HEAD_TAP
    safe_key = re.sub(r'[^0-9A-Za-z_]+', '_', AUX_HEAD_TAP)
    if safe_key in name_map and "student" in name_map[safe_key] and "teacher" in name_map[safe_key]:
        S_TAP = name_map[safe_key]["student"]
        T_TAP = name_map[safe_key]["teacher"]
    else:
        # Fallback (last resort)
        S_TAP = T_TAP = AUX_HEAD_TAP

    # Hooks to discover head channels
    s_head_hook = FeatureHooker(student, [S_TAP], use_regex=False)
    t_head_hook = FeatureHooker(teacher, [T_TAP], use_regex=False)

    with torch.no_grad():
        _ = student(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))
        _ = teacher(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device))

    # pull the actual tensors from the resolved taps
    s_sample = _first_4d_tensor(s_head_hook.buffers.get(S_TAP, None))
    t_sample = _first_4d_tensor(t_head_hook.buffers.get(T_TAP, None))
    if s_sample is None or t_sample is None:
        raise RuntimeError(f"Resolved taps produced no 4D tensors. student='{S_TAP}' teacher='{T_TAP}'")

    Cs = int(s_sample.shape[1])
    Ct = int(t_sample.shape[1])
    d  = int(dims.get(AUX_HEAD_TAP, 128))
    k  = int(tokens.get(AUX_HEAD_TAP, 4))

    # Build fusion & seg head with CURRENT channel counts
    fusions = nn.ModuleDict()
    safe_key = _safe_name(AUX_HEAD_TAP)
    fusions[safe_key] = StructureFusion(Cs, Ct, d=d, num_tokens=k, use_film=True).to(device).eval()
    # Load saved weights with channel adaptation
    _load_fusion_with_channel_compat(fusions[safe_key], fusions_sd, safe_key)

    seg_adapter = AuxMaskHead(in_channels=Cs, num_classes=NUM_CLASSES).to(device).eval()
    _load_seg_head_with_channel_compat(seg_adapter, seg_sd)
    print(f"[SANITY] student tap='{S_TAP}' C={Cs} | teacher tap='{T_TAP}' C={Ct}")

    # For baseline instance predictions (same criteria as baseline file)
    yolo_pred = YOLO(STUDENT_MODEL)

    # Iterate images
    img_paths = sorted([p for p in cam_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    dice_list, iou_list = [], []

    # teacher aux head (fixed 0.82) stats
    aux_dice_list, aux_iou_list = [], []

    # teacher refined prior stats
    aux_refined_dice_list, aux_refined_iou_list = [], []

    # aux sweep-best stats
    sweep_best_dice_list, sweep_best_thr_list = [], []

    # student filtered-by-prior stats
    dice_list_filt, iou_list_filt = [], []
    predictions = []
    predictions_filt = []   # for optional mAP on filtered preds
    gt_instances_by_img = {}

    with torch.no_grad():
        for img_id, cam_path in enumerate(img_paths):
            rel = cam_path.relative_to(cam_root)
            man_path = (man_root / rel)
            if not man_path.exists():
                alt = man_root / cam_path.name
                if alt.exists():
                    man_path = alt
                else:
                    print(f"[SKIP] No paired manual image for: {cam_path}")
                    continue

            # Load GT
            if mask_utils is None or not MASK_ROOT_CAMERA:
                continue
            mjson = Path(MASK_ROOT_CAMERA).resolve() / rel.parent / (cam_path.stem + MASK_SUFFIX)
            gt_instances = _decode_instances_from_json(mjson)
            if not gt_instances:
                continue
            gt_instances_bool = [(g > 0).astype(bool) for g in gt_instances]
            gt_instances_by_img[img_id] = gt_instances_bool
            gt_union = _merge_binary(gt_instances)
            H0, W0 = gt_union.shape
            gt_bool = (gt_union > 0)

            # Baseline student instance predictions (mAP + union Dice/IoU)
            res = yolo_pred.predict(
                source=str(cam_path), imgsz=IMG_SIZE, conf=CONF_TH,
                device=DEVICE, verbose=False, half=(str(DEVICE).lower() != "cpu")
            )[0]

            if getattr(res, "masks", None) is not None and res.masks is not None:
                pmasks = res.masks.data.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy() if getattr(res, "boxes", None) is not None else np.ones(len(pmasks))
                if pmasks.ndim == 3 and pmasks.shape[0] > 0:
                    acc_union = np.zeros((H0, W0), dtype=bool)
                    for kidx in range(pmasks.shape[0]):
                        mk = pmasks[kidx] > 0.5
                        mk = _resize_bool(mk, (H0, W0))
                        acc_union |= mk
                        predictions.append({"img_id": img_id, "score": float(scores[kidx]), "mask": mk})
                    pred_union = acc_union
                else:
                    pred_union = np.zeros((H0, W0), dtype=bool)
            else:
                pred_union = np.zeros((H0, W0), dtype=bool)

            # Dice/IoU vs GT union (for the red union baseline)
            inter = np.logical_and(pred_union, gt_bool).sum()
            p_sum, t_sum = pred_union.sum(), gt_bool.sum()
            union = p_sum + t_sum - inter
            dice = (2*inter + 1e-6) / (p_sum + t_sum + 1e-6)
            iou  = (inter + 1e-6) / (union + 1e-6)
            dice_list.append(float(dice)); iou_list.append(float(iou))

            # Save baseline union overlay (red)
            rgb = np.array(Image.open(cam_path).convert("RGB"))
            
            ov_union  = _overlay_filled(rgb, pred_union, color=(255,0,0), alpha=0.45)
            Image.fromarray(ov_union).save(out_dir / "overlays_union" / f"{cam_path.stem}.png")

            # ---------- Dual-input aux fusion (teacher-guided) ----------
            x_cam = _prep_image(str(cam_path), IMG_SIZE).to(device)
            x_man = _prep_image(str(man_path), IMG_SIZE).to(device)

            s_head_hook.clear(); t_head_hook.clear()
            _ = student(x_cam); _ = teacher(x_man)
            s_aux = _first_4d_tensor(s_head_hook.buffers.get(S_TAP, None))
            t_aux = _first_4d_tensor(t_head_hook.buffers.get(T_TAP, None))
            if s_aux is None or t_aux is None:
                print(f"[WARN] {cam_path.name}: aux tap present but no 4D tensor; skipping aux overlay/prior.")
                continue

            s_aux_fused = fusions[safe_key](s_aux, t_aux)
            pred_logits = seg_adapter(s_aux_fused, out_hw=(H0, W0))  # (1,1,H,W) logits for NUM_CLASSES==1
            if NUM_CLASSES == 1 or pred_logits.shape[1] == 1:
                teacher_logits = pred_logits[0,0].detach().cpu().numpy()  # keep logits for refine
            else:
                fg_logits = pred_logits[:, 1:, :, :].amax(dim=1, keepdim=True)
                teacher_logits = fg_logits[0,0].detach().cpu().numpy()

            # === Build refined prior (adaptive + proximity + area + erosion + teacher-only) ===
            # NOTE: This returns both the refined prior (boolean) AND the teacher prob map (sigmoid),
            # so we don't need to sigmoid the logits ourselves afterwards.
            aux_prior, teacher_prob = refine_aux_prior(
                teacher_logits, pred_union,
                logits_are_probs=False,
                near_thr=PRIOR_NEAR_THR, far_thr=PRIOR_FAR_THR,
                max_near_px=PRIOR_NEAR_PX, far_cut_px=PRIOR_FAR_CUT_PX,
                keep_only_teacher_minus_student=False,
                min_comp_area=PRIOR_MIN_AREA,
                light_erosion_iters=PRIOR_ERODE_IT,
                open_iters=PRIOR_OPEN_IT,
                close_iters=PRIOR_CLOSE_IT
            )
            
            if DEBUG_SAVE:
                # coverage + basic sanity
                cov = float(aux_prior.mean())
                tp_min, tp_max = float(teacher_prob.min()), float(teacher_prob.max())
                if cov > 0.9 or cov < 0.001:
                    print(f"[DEBUG] {cam_path.name}: prior coverage={cov:.3f}  teacher_prob[min,max]=({tp_min:.3f},{tp_max:.3f})")
                _save_gray(out_dir / "debug" / f"{cam_path.stem}_tprob.png", teacher_prob)
                _save_gray(out_dir / "debug" / f"{cam_path.stem}_prior.png", aux_prior.astype(np.float32))

            # --- (A) Teacher aux head @ fixed thr (blue, unrefined) ---
            aux_mask_fixed = (teacher_prob > float(AUX_THRESH))
            ad_fix, ai_fix = _dice_iou_bool(aux_mask_fixed, gt_bool)
            aux_dice_list.append(ad_fix); aux_iou_list.append(ai_fix)

            
            # save fixed overlay (blue, a bit lighter)
            ov_aux_fixed = _overlay_filled(rgb, aux_mask_fixed, color=(0,0,255), alpha=0.30)
            Image.fromarray(ov_aux_fixed).save(out_dir / "overlays_auxfusion" / f"{cam_path.stem}.png")

            # --- (B) Refined prior from teacher + student union (blue, refined) ---
            ad_ref, ai_ref = _dice_iou_bool(aux_prior, gt_bool)
            aux_refined_dice_list.append(ad_ref); aux_refined_iou_list.append(ai_ref)


            # save refined overlay (blue, stronger)
            ov_aux_refined = _overlay_filled(rgb, aux_prior, color=(0,0,255), alpha=0.45)
            Image.fromarray(ov_aux_refined).save(out_dir / "overlays_auxfusion_refined" / f"{cam_path.stem}.png")

            # --- (C) Sweep-best on teacher prob (not refined) ---
            if AUX_SWEEP is not None:
                best_d = -1.0
                best_t = None
                for thr in AUX_SWEEP:
                    m = (teacher_prob > float(thr))
                    d, _ = _dice_iou_bool(m, gt_bool)
                    if d > best_d:
                        best_d, best_t = d, float(thr)
                sweep_best_dice_list.append(best_d)
                sweep_best_thr_list.append(best_t)


            # --- (D) Filter student instances using refined prior (green) ---
            pmasks_bool = []
            scores = []
            if getattr(res, "masks", None) is not None and res.masks is not None:
                pm = res.masks.data.cpu().numpy()
                sc = res.boxes.conf.cpu().numpy() if getattr(res, "boxes", None) is not None else np.ones(len(pm))
                if pm.ndim == 3 and pm.shape[0] > 0:
                    for kidx in range(pm.shape[0]):
                        mk = pm[kidx] > 0.5
                        mk = _resize_bool(mk, (H0, W0))
                        pmasks_bool.append(mk)
                        scores.append(float(sc[kidx]))

            preds_filt, union_filt = filter_student_instances_with_prior(
                pmasks_bool, np.array(scores, dtype=np.float32), aux_prior,
                clip_to_prior=CLIP_TO_AUX,
                cover_thr=COVER_THR,
                rescore_alpha=RESCORE_ALPHA
            )
            # --- promote teacher-only components into the final union ---
            ADD_TEACHER_MISSING   = True
            PROMOTE_MIN_AREA      = 1400    # px
            PROMOTE_MAX_DIST_PX   = 30     # distance to student union (pre-filter) to allow adding

            if ADD_TEACHER_MISSING:
                # distance from any student pixel (use pred_union, i.e., the red union)
                from scipy.ndimage import distance_transform_edt, label
                near_mask = distance_transform_edt(~pred_union) <= PROMOTE_MAX_DIST_PX

                # connected components on teacher-only prior
                lab, n = label(aux_prior.astype(np.uint8))
                promote = np.zeros_like(aux_prior, dtype=bool)
                for i in range(1, n+1):
                    comp = (lab == i)
                    if comp.sum() < PROMOTE_MIN_AREA:
                        continue
                    # only add components that are near the student union
                    if (comp & near_mask).any():
                        promote |= comp


                if promote.any():
                    ov_promote = _overlay_filled(rgb, promote, color=(255,255,0), alpha=0.55)  # yellow add-ins
                    Image.fromarray(ov_promote).save(out_dir / "overlays_promoted" / f"{cam_path.stem}.png")

                # merge into the filtered union
                union_filt = union_filt | promote

            # union Dice/IoU of filtered student vs GT
            d_f, i_f = _dice_iou_bool(union_filt, gt_bool)
            dice_list_filt.append(d_f); iou_list_filt.append(i_f)

            # optional: include filtered preds for a second mAP (student filtered)
            for p in preds_filt:
                predictions_filt.append({"img_id": img_id, "score": float(p["score"]), "mask": p["mask"]})


            # save filtered union (green)
            ov_union_filt = _overlay_filled(rgb, union_filt, color=(0,255,0), alpha=0.45)
            Image.fromarray(ov_union_filt).save(out_dir / "overlays_union_filtered" / f"{cam_path.stem}.png")



    # Metrics summary (same criteria/thresholds as baseline)
    # Student-only mAP (baseline)
    if predictions and gt_instances_by_img:
        ap50, map5095 = _eval_map(predictions, gt_instances_by_img)
    else:
        ap50 = map5095 = 0.0

    # Student-only Dice/IoU (union)
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    mean_iou  = float(np.mean(iou_list))  if iou_list  else 0.0

    # Teacher aux @ fixed thr (0.82) stats
    aux_mean_dice = float(np.mean(aux_dice_list)) if aux_dice_list else 0.0
    aux_mean_iou  = float(np.mean(aux_iou_list))  if aux_iou_list  else 0.0

    # Teacher refined prior stats
    aux_refined_mean_dice = float(np.mean(aux_refined_dice_list)) if aux_refined_dice_list else 0.0
    aux_refined_mean_iou  = float(np.mean(aux_refined_iou_list))  if aux_refined_iou_list  else 0.0

    # Sweep-best (teacher prob only)
    if AUX_SWEEP is not None and sweep_best_dice_list:
        best_mean_dice = float(np.mean(sweep_best_dice_list))
        best_mean_thr  = float(np.mean(sweep_best_thr_list))
    else:
        best_mean_dice = 0.0
        best_mean_thr  = 0.0

    # Student filtered-by-prior Dice/IoU
    mean_dice_filt = float(np.mean(dice_list_filt)) if dice_list_filt else 0.0
    mean_iou_filt  = float(np.mean(iou_list_filt))  if iou_list_filt  else 0.0

    # Optional: mAP for filtered predictions
    if predictions_filt and gt_instances_by_img:
        ap50_f, map5095_f = _eval_map(predictions_filt, gt_instances_by_img)
    else:
        ap50_f = map5095_f = 0.0

    print("\n==== TEST SUMMARY ====")
    print("[Student-only]")
    print(f"  Dice (union)              : {mean_dice:.4f}")
    print(f"  IoU  (union)              : {mean_iou:.4f}")
    print(f"  mAP@50 (instances)        : {ap50:.4f}")
    print(f"  mAP@50-95                 : {map5095:.4f}")

    print("\n[Teacher aux head]")
    print(f"  Fixed thr {AUX_THRESH:.2f} → Dice: {aux_mean_dice:.4f}  IoU: {aux_mean_iou:.4f}")
    print(f"  Sweep-best (per-image)     → Dice: {best_mean_dice:.4f}  mean thr: {best_mean_thr:.2f}")
    print(f"  Refined prior (adaptive)   → Dice: {aux_refined_mean_dice:.4f}  IoU: {aux_refined_mean_iou:.4f}")

    print("\n[Student filtered by refined prior]")
    print(f"  Dice (union filtered)      : {mean_dice_filt:.4f}")
    print(f"  IoU  (union filtered)      : {mean_iou_filt:.4f}")
    print(f"  mAP@50 (filtered)          : {ap50_f:.4f}")
    print(f"  mAP@50-95 (filtered)       : {map5095_f:.4f}")

    print(f"\nOverlays (student union)           : {out_dir/'overlays_union'}")
    print(f"Overlays (teacher aux @ {AUX_THRESH:.2f}) : {out_dir/'overlays_auxfusion'}")
    print(f"Overlays (teacher refined prior)   : {out_dir/'overlays_auxfusion_refined'}")
    print(f"Overlays (student filtered)        : {out_dir/'overlays_union_filtered'}")


    s_head_hook.remove(); t_head_hook.remove()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)

