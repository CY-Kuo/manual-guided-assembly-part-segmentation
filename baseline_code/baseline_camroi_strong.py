import os, sys, time, json
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

# =================== USER EDITS ===================
STUDENT_MODEL = r"C:\Users\djkim\Documents\Yolo11-seg_train_results\0815Results_complete_labeling\runs\segment\grid\scratch\user\kuocy\IKEAVideo\model_weights\yolo11s-seg_0816_123254_f2\weights\best.pt"
CAM_YAML      = r"C:\Users\djkim\Documents\mini_2\parts_single_cam.yaml"

MASK_SUFFIX             = "_rle.json"
EXPLICIT_VAL_MASK_ROOT  = r"C:\Users\djkim\Documents\mini_2\mask_labels\val\camera"
EXPLICIT_TEST_MASK_ROOT = r"C:\Users\djkim\Documents\mini_2\mask_labels\test\camera"

PROJECT       = r"C:\Users\djkim\Documents\mini_2\eval_runs"
RUN_NAME      = "baseline_camroi_strong"
# ==================================================

IMG_SIZE      = 640
DEVICE        = "0"  # "0" for first GPU, or "cpu"
YOLO_CONF_FIXED = 0.20

# ---- CAM-ROI grids (soft multiply) ----
CAM_THRESH_GRID   = [0.15, 0.25, 0.35]
ROI_WEIGHT_GRID   = [0.25, 0.35, 0.50]

# Optional: clip YOLO mask (soft) to its own box before ROI gating
CLIP_TO_BOX = True
BOX_PAD = 2  # pixels of padding around the box when clipping (0–2 is sensible)

BOUNDARY_TOL = 2
SAVE_OVERLAYS = True
OVERLAY_SUBDIR = "overlays_camroi_test"

# ----------------- deps -----------------
from ultralytics import YOLO
from pycocotools import mask as c_mask
import yaml

# ----------------- utils -----------------
def _progress(msg):
    print(time.strftime("[%H:%M:%S] "), msg, flush=True)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def read_yaml_images_root(yaml_path: Path, split_key: str) -> Path:
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

def imread_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def save_overlay(out_path, rgb_img, masks, color=(0,255,0), alpha=0.45, boxes=None):
    vis = rgb_img.copy()
    overlay = vis.copy()
    for m in masks:
        overlay[m > 0.5] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0)
    if boxes is not None:
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
    imwrite_rgb(out_path, vis)

def _write_header_once(path, header):
    path = Path(path)
    if not path.exists():
        with open(path, "w", newline="") as f:
            f.write(header)

# ----------------- GT & metrics -----------------
def decode_rle_obj(rle_obj):
    rle = rle_obj
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
    m = c_mask.decode(rle)
    if m.ndim == 3: m = m[..., 0]
    return (m > 0).astype(np.uint8)

def load_gt_instances_from_rle_json(json_path: Path):
    if not json_path.exists():
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segmentation" in data:
        seg = data["segmentation"]
        rles = seg if isinstance(seg, list) else [seg]
    elif isinstance(data, dict) and "counts" in data and "size" in data:
        rles = [data]
    elif isinstance(data, list):
        rles = data
    else:
        return []
    return [decode_rle_obj(r) for r in rles]

def merge_binary(masks_list):
    merged = None
    for m in masks_list:
        merged = m if merged is None else np.maximum(merged, m)
    return (merged > 0) if merged is not None else None

def pixel_metrics(gt, pr):
    gt = (gt > 0).astype(np.uint8)
    pr = (pr > 0).astype(np.uint8)
    tp = np.logical_and(gt==1, pr==1).sum()
    fp = np.logical_and(gt==0, pr==1).sum()
    fn = np.logical_and(gt==1, pr==0).sum()
    dice = (2*tp + 1e-9) / (2*tp + fp + fn + 1e-9)
    iou  = (tp + 1e-9)  / (tp + fp + fn + 1e-9)
    prec = (tp + 1e-9)  / (tp + fp + 1e-9)
    rec  = (tp + 1e-9)  / (tp + fn + 1e-9)
    return dice, iou, prec, rec

def boundary_f(gt, pr, tol=2):
    from skimage.morphology import binary_dilation, disk
    gt = (gt > 0).astype(np.uint8); pr = (pr > 0).astype(np.uint8)
    er = np.ones((3,3), np.uint8)
    b_gt = (gt ^ cv2.erode(gt, er, 1)) > 0
    b_pr = (pr ^ cv2.erode(pr, er, 1)) > 0
    if tol > 0:
        se = disk(tol)
        b_gt_d = binary_dilation(b_gt, footprint=se)
        b_pr_d = binary_dilation(b_pr, footprint=se)
    else:
        b_gt_d, b_pr_d = b_gt, b_pr
    tp_p = (b_pr & b_gt_d).sum(); pp = b_pr.sum()
    tp_r = (b_gt & b_pr_d).sum(); rp = b_gt.sum()
    prec = tp_p / (pp + 1e-9); rec = tp_r / (rp + 1e-9)
    return 0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec+1e-9)

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-9)

def eval_map(preds, gts_by_img, iou_thresholds=np.arange(0.5,0.96,0.05)):
    preds = sorted(preds, key=lambda d: -d["score"])
    total_gts = sum(len(v) for v in gts_by_img.values())
    aps = {}
    for thr in iou_thresholds:
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts_by_img.items()}
        for i, p in enumerate(preds):
            imgid = p["image_id"]; pm = p["mask"]
            cand = gts_by_img.get(imgid, [])
            best, j_best = 0.0, -1
            for j, gm in enumerate(cand):
                if matched[imgid][j]: continue
                iou = mask_iou(pm, gm)
                if iou > best:
                    best, j_best = iou, j
            if j_best >= 0 and best >= thr:
                tp[i] = 1.0; matched[imgid][j_best] = True
            else:
                fp[i] = 1.0
        ctp = np.cumsum(tp); cfp = np.cumsum(fp)
        rec = ctp / max(1, total_gts)
        prec = ctp / np.maximum(1, ctp + cfp)
        rgrid = np.linspace(0,1,101)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for k in range(mpre.size-2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k+1])
        ap = 0.0
        for r in rgrid:
            ap += mpre[np.searchsorted(mrec, r, side="right") - 1]
        aps[thr] = ap / 101.0
    return float(aps.get(0.5, 0.0)), float(np.mean(list(aps.values()))) if aps else 0.0

# ----------------- inference helpers -----------------
def resize_float(m, out_hw):
    H, W = out_hw
    if m.shape == (H, W):
        return m.astype(np.float32)
    pil = Image.fromarray((m * 255.0).astype(np.uint8))
    pil = pil.resize((W, H), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0

def collect_instances(model, img_path: Path, conf_th: float, imgsz: int, out_hw):
    res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf_th,
                        device=DEVICE, verbose=False, half=(DEVICE!="cpu"))[0]
    H0, W0 = out_hw
    masks_float, scores, classes, boxes = [], [], [], None
    if res.masks is not None:
        pm = res.masks.data.detach().cpu().numpy().astype(np.float32)  # soft [0,1]
        if res.boxes is not None and len(res.boxes) > 0:
            sc = res.boxes.conf.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(int)
            boxes = res.boxes.xyxy.cpu().numpy()
        else:
            sc = np.ones(len(pm))
            cl = np.zeros(len(pm), dtype=int)
            boxes = None
        for k in range(pm.shape[0]):
            mk = resize_float(pm[k], (H0, W0))
            masks_float.append(mk)
            scores.append(float(sc[k]))
            classes.append(int(cl[k]))
    return masks_float, scores, classes, boxes

def masks_to_union(masks_bin, H, W):
    if not masks_bin: return np.zeros((H,W), dtype=np.uint8)
    acc = np.zeros((H,W), dtype=np.uint8)
    for m in masks_bin: acc |= m.astype(np.uint8)
    return acc

# ----------------- CAM-ROI (CV2 saliency, per-box) -----------------
def _cv2_saliency(rgb):
    """Return [0,1] edge/saliency map same size as rgb."""
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax - mmin < 1e-9:
        return np.zeros((H, W), dtype=np.float32)
    m = (mag - mmin) / (mmax - mmin)
    return cv2.GaussianBlur(m, (0,0), 1.0).astype(np.float32)

def per_box_roi(rgb, boxes, H, W):
    """Compute a per-instance saliency ROI map list aligned with boxes; each ROI is full image sized, zeros outside the box."""
    rois = []
    if boxes is None:
        base = _cv2_saliency(rgb)
        return [base]
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W-1,x2), min(H-1,y2)
        if x2 <= x1 or y2 <= y1:
            rois.append(np.zeros((H,W), dtype=np.float32)); continue
        crop = rgb[y1:y2+1, x1:x2+1]
        r = _cv2_saliency(crop)
        r = cv2.resize(r, (x2-x1+1, y2-y1+1), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((H,W), dtype=np.float32)
        full[y1:y2+1, x1:x2+1] = r
        rois.append(full)
    return rois

def soft_gate(roi_soft, thr):
    """Smooth threshold: map [0,1] -> [0,1] where <thr becomes 0, >=thr ramps to 1."""
    return np.clip((roi_soft - thr) / (1.0 - thr + 1e-6), 0.0, 1.0).astype(np.float32)

def clip_prob_to_box(prob: np.ndarray, box, pad=0):
    """Soft clip: zero probability outside the (padded) box."""
    h, w = prob.shape
    if box is None:
        return prob
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    clipped = np.zeros_like(prob, dtype=np.float32)
    if x2 >= x1 and y2 >= y1:
        clipped[y1:y2+1, x1:x2+1] = prob[y1:y2+1, x1:x2+1]
    return clipped

def refine_with_camroi(rgb, masks_float, boxes, thr, roi_weight, H, W):
    """Refine list of soft masks with per-instance ROI gate, then threshold."""
    if not masks_float: return []
    rois = per_box_roi(rgb, boxes, H, W)
    refined = []
    for j, m in enumerate(masks_float):
        # Optional: clip each mask to its own box before ROI blending
        if CLIP_TO_BOX and boxes is not None and j < len(boxes):
            m = clip_prob_to_box(m, boxes[j], BOX_PAD)

        roi_soft = rois[j if j < len(rois) else -1]
        gate = soft_gate(roi_soft, thr)  # [0,1]
        x = np.clip(m*(1.0-roi_weight) + m*roi_weight*gate, 0.0, 1.0)
        x = cv2.GaussianBlur(x, (0,0), 0.8)
        refined.append((x >= 0.5).astype(np.uint8))
    return refined

# ----------------- CALIBRATION: CAM-ROI on VAL -----------------
def val_calibrate_camroi(yolo_model: YOLO, cam_yaml, val_mask_root, mask_suffix, out_dir):
    out_dir = Path(out_dir)
    cam_val = read_yaml_images_root(Path(cam_yaml), "val")
    imgs = sorted([p for p in cam_val.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])

    _progress(f"[CAM-ROI] Preparing cache for {len(imgs)} val images ...")
    cache = []
    for ip in imgs:
        rel = ip.relative_to(cam_val)
        gt_path = Path(val_mask_root) / rel.parent / (ip.stem + mask_suffix)
        gts = load_gt_instances_from_rle_json(gt_path)
        if not gts: continue
        gt_union = merge_binary(gts)
        H,W = gt_union.shape
        rgb = imread_rgb(ip).astype(np.uint8)
        masks, scores, classes, boxes = collect_instances(yolo_model, ip, YOLO_CONF_FIXED, IMG_SIZE, (H,W))
        cache.append((ip, rgb, gt_union, H, W, masks, boxes))

    total = len(CAM_THRESH_GRID) * len(ROI_WEIGHT_GRID)
    _progress(f"[CAM-ROI] Cached {len(cache)} images. Running grid of {total} combos.")
    csv_path = Path(out_dir)/"val_calib_camroi.csv"
    _write_header_once(csv_path, "thr,roi_w,Dice,IoU,Precision,Recall,BoundaryF\n")

    rows = []
    done = 0
    for thr in CAM_THRESH_GRID:
        for roi_w in ROI_WEIGHT_GRID:
            t0 = time.time()
            mets = []
            for (ip, rgb, gt_union, H, W, masks, boxes) in cache:
                if not masks:
                    pr = np.zeros_like(gt_union)
                    d,i,p,r = pixel_metrics(gt_union, pr); bf = boundary_f(gt_union, pr, BOUNDARY_TOL)
                    mets.append((d,i,p,r,bf)); continue
                refined = refine_with_camroi(rgb, masks, boxes, thr, roi_w, H, W)
                union = masks_to_union(refined, H, W)
                d,i,p,r = pixel_metrics(gt_union, union); bf = boundary_f(gt_union, union, BOUNDARY_TOL)
                mets.append((d,i,p,r,bf))
            mean = np.mean(np.array(mets), axis=0) if mets else np.zeros(5)
            row = {"thr":thr,"roi_w":roi_w,"Dice":mean[0],"IoU":mean[1],"Precision":mean[2],"Recall":mean[3],"BoundaryF":mean[4]}
            rows.append(row)
            with open(csv_path,"a") as f:
                f.write(f"{row['thr']:.4f},{row['roi_w']:.2f},{row['Dice']:.6f},{row['IoU']:.6f},{row['Precision']:.6f},{row['Recall']:.6f},{row['BoundaryF']:.6f}\n")
            done += 1
            _progress(f"[CAM-ROI] {done}/{total} thr={thr:.2f} roi_w={roi_w:.2f} in {time.time()-t0:.1f}s; Dice={row['Dice']:.4f}")

    if not rows: raise RuntimeError("CAM-ROI calibration found no rows.")
    best = max(rows, key=lambda r: r["Dice"])
    _progress(f"[CAM-ROI] Best → thr={best['thr']}, roi_w={best['roi_w']}, Dice={best['Dice']:.4f}")
    return best

def load_best_camroi_from_csv(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists(): return None
    best = None
    with open(csv_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines[1:]:
        thr_s, rw_s, dice_s, *_ = ln.split(",")
        row = {"thr": float(thr_s), "roi_w": float(rw_s), "Dice": float(dice_s)}
        if (best is None) or (row["Dice"] > best["Dice"]):
            best = row
    return best

# ----------------- TEST: CAM-ROI on TEST -----------------
def evaluate_test_camroi(yolo_model: YOLO, cam_yaml, test_mask_root, mask_suffix, params, out_dir):
    out_dir = Path(out_dir)
    cam_test = read_yaml_images_root(Path(cam_yaml), "test")
    imgs = sorted([p for p in cam_test.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])

    per_csv = Path(out_dir)/"test_per_image_BaselineCAMROI.csv"
    _write_header_once(per_csv, "image,Dice,IoU,Precision,Recall,BoundaryF\n")

    preds_ap = []
    gts_by_img = {}
    D=[]; J=[]; P=[]; R=[]; BF=[]

    if SAVE_OVERLAYS:
        ov_dir = Path(out_dir) / OVERLAY_SUBDIR
        ensure_dir(ov_dir)

    _progress(f"[BaselineCAMROI] Evaluating {len(imgs)} test images ...")
    for img_id, ip in enumerate(imgs):
        rel = ip.relative_to(cam_test)
        gt_path = Path(test_mask_root) / rel.parent / (ip.stem + mask_suffix)
        gts = load_gt_instances_from_rle_json(gt_path)
        if not gts: continue
        gt_union = merge_binary(gts)
        H,W = gt_union.shape
        rgb = imread_rgb(ip).astype(np.uint8)

        masks, scores, classes, boxes = collect_instances(yolo_model, ip, YOLO_CONF_FIXED, IMG_SIZE, (H, W))
        refined = refine_with_camroi(rgb, masks, boxes, params["thr"], params["roi_w"], H, W)

        if SAVE_OVERLAYS:
            ov_path = Path(out_dir) / OVERLAY_SUBDIR / f"BaselineCAMROI_{ip.stem}_vis.png"
            save_overlay(ov_path, rgb, refined, color=(0,255,0))

        union = masks_to_union(refined, H, W)
        d,i,p,r = pixel_metrics(gt_union, union); bf = boundary_f(gt_union, union, BOUNDARY_TOL)

        with open(per_csv,"a") as f:
            f.write(f"{str(rel)},{d:.6f},{i:.6f},{p:.6f},{r:.6f},{bf:.6f}\n")

        D.append(d); J.append(i); P.append(p); R.append(r); BF.append(bf)

        # class-agnostic AP (scores from YOLO boxes)
        for j, m in enumerate(refined):
            sc = float(scores[j] if j < len(scores) else 0.5)
            preds_ap.append({"image_id": img_id, "mask": m.astype(bool), "score": sc})
        gts_by_img[img_id] = [(g>0).astype(bool) for g in gts]

        if (img_id+1) % 25 == 0:
            _progress(f"[BaselineCAMROI] {img_id+1}/{len(imgs)} images processed.")

    D=np.array(D); J=np.array(J); P=np.array(P); R=np.array(R); BF=np.array(BF)
    pix_summary = dict(Dice=float(D.mean() if len(D) else 0.0),
                       IoU=float(J.mean() if len(J) else 0.0),
                       Precision=float(P.mean() if len(P) else 0.0),
                       Recall=float(R.mean() if len(R) else 0.0),
                       BoundaryF=float(BF.mean() if len(BF) else 0.0))

    ap50, map_5095 = eval_map(preds_ap, gts_by_img)

    sum_csv = Path(out_dir)/"test_summary_BaselineCAMROI.csv"
    with open(sum_csv,"w") as f:
        f.write("method,YOLO_conf,thr,roi_w,Dice,IoU,Precision,Recall,BoundaryF,AP50,mAP50_95\n")
        f.write(f"BaselineCAMROI,{YOLO_CONF_FIXED:.2f},{params['thr']:.2f},{params['roi_w']:.2f},"
                f"{pix_summary['Dice']:.6f},{pix_summary['IoU']:.6f},{pix_summary['Precision']:.6f},{pix_summary['Recall']:.6f},{pix_summary['BoundaryF']:.6f},"
                f"{ap50:.6f},{map_5095:.6f}\n")

    _progress(f"[BaselineCAMROI] DONE. Dice {pix_summary['Dice']:.4f} | IoU {pix_summary['IoU']:.4f} | P {pix_summary['Precision']:.4f} | R {pix_summary['Recall']:.4f} | BF {pix_summary['BoundaryF']:.4f}")
    _progress(f"[BaselineCAMROI] AP50 {ap50:.4f} | mAP50:95 {map_5095:.4f}")
    _progress(f"[BaselineCAMROI] Wrote per-image → {per_csv}")
    _progress(f"[BaselineCAMROI] Wrote summary   → {sum_csv}")

# ----------------- main -----------------
def main():
    device = torch.device(f"cuda:{DEVICE}" if str(DEVICE).lower()!="cpu" and torch.cuda.is_available() else "cpu")
    out_dir = Path(PROJECT)/RUN_NAME
    ensure_dir(out_dir)

    model = YOLO(STUDENT_MODEL)
    model.to(device)
    model.model.eval()
    torch.backends.cudnn.benchmark = True

    # Load best CAM-ROI from CSV if present, else calibrate
    cam_csv = Path(out_dir)/"val_calib_camroi.csv"
    best_cam = load_best_camroi_from_csv(cam_csv)
    if best_cam is None:
        _progress("Calibrating CAM-ROI on VAL ...")
        best_cam = val_calibrate_camroi(model, CAM_YAML, EXPLICIT_VAL_MASK_ROOT, MASK_SUFFIX, out_dir)
        _progress(f"Best CAM-ROI: thr={best_cam['thr']}, roi_w={best_cam['roi_w']}")
    else:
        _progress(f"[CAM-ROI] Loaded best from CSV: thr={best_cam['thr']}, roi_w={best_cam['roi_w']}, Dice={best_cam.get('Dice','n/a')}")

    # TEST with best (thr, roi_w)
    _progress("Evaluating BaselineCAMROI on TEST ...")
    evaluate_test_camroi(model, CAM_YAML, EXPLICIT_TEST_MASK_ROOT, MASK_SUFFIX,
                         params={"thr": float(best_cam["thr"]), "roi_w": float(best_cam["roi_w"])},
                         out_dir=out_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
