# manual_roi_to_sam.py
import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import yaml
from tqdm.auto import tqdm
import logging

# ---------------- logging ----------------
def setup_logging(level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=[logging.StreamHandler(sys.stdout)])
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
setup_logging()
log = logging.getLogger("ManualROI→SAM")

# ======= Your baseline-like setup (mirrors your YOLO→SAM style) =======
YOLO_CONF = 0.20        # (unused here; kept for header symmetry)
IMG_SIZE  = 640         # (unused here)
DEVICE    = "0"         # "0" or "cpu"
LOG_EVERY = 25

# ======= Paths you edit =======
CAM_YAML       = r"C:\Users\djkim\Documents\mini_2\parts_single_cam.yaml"
MASK_SUFFIX    = "_rle.json"
VAL_MASK_ROOT  = r"C:\Users\djkim\Documents\mini_2\mask_labels\val\camera"   # (unused here)
TEST_MASK_ROOT = r"C:\Users\djkim\Documents\mini_2\mask_labels\test\camera"
PROJECT        = r"C:\Users\djkim\Documents\mini_2\eval_runs"
RUN_NAME       = "baseline_manualROI_SAM_strong"

# Where your manual ROI JSONs live (note: images\camera\test\*.json)
ROI_ROOT       = r"C:\Users\djkim\Documents\mini_2\images\camera"
ROI_SUFFIX     = "_roi.json"

# ======= SAM (optional; fallback to GrabCut if missing) =======
SAM_CHECKPOINT = r"C:\Users\djkim\IKEAVideo\yolo_code\sam_vit_h_4b8939.pth"          # e.g., r"C:\models\sam_vit_h.pth" ; leave empty to use fallback
SAM_MODEL_TYPE = "vit_h"      # vit_h/vit_l/vit_b

# ======= Box robustness =======
BOX_PAD = 2                   # expand ROI boxes by this many pixels on each side when clipping mask
SAVE_OVERLAYS_TEST = True
OVERLAY_SUBDIR = "overlays_manualroi_sam_test"

BOUNDARY_TOL = 2

# ======= YAML helpers =======
def read_yaml_images_root(yaml_path: Path, split_key: str) -> Path:
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", yaml_path.parent)).resolve()
    sp = y.get(split_key)
    if sp is None:
        raise ValueError(f"{yaml_path} has no '{split_key}' split.")
    p = Path(sp[0] if isinstance(sp,(list,tuple)) else sp)
    if not p.is_absolute(): p = (root / p).resolve()
    if not p.exists(): raise FileNotFoundError(f"Split '{split_key}' not found: {p}")
    return p

# ======= GT (RLE JSON) =======
from pycocotools import mask as c_mask
def decode_rle_obj(rle_obj):
    rle = rle_obj
    counts = rle.get("counts")
    if isinstance(counts, str): rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
    m = c_mask.decode(rle)
    if m.ndim == 3: m = m[...,0]
    return (m > 0).astype(np.uint8)

def load_gt_instances_from_rle_json(json_path: Path):
    if not json_path.exists(): return []
    with open(json_path, "r") as f: data = json.load(f)
    if isinstance(data, dict) and "segmentation" in data:
        seg = data["segmentation"]; rles = seg if isinstance(seg, list) else [seg]
    elif isinstance(data, dict) and "counts" in data and "size" in data:
        rles = [data]
    elif isinstance(data, list):
        rles = data
    else:
        return []
    return [decode_rle_obj(r) for r in rles]

# ======= ROI loader =======
def roi_path_for_image(img_path: Path, test_images_root: Path) -> Path:
    """
    Map: C:\...\images\test\camera\sub\img.jpg  ->
         C:\...\images\camera\test\sub\img_roi.json
    """
    rel = img_path.relative_to(test_images_root)  # e.g., camera\sub\img.jpg
    # swap 'camera\...' under images\test to ROI_ROOT\test\...
    return Path(ROI_ROOT) / "test" / rel.parent / (img_path.stem + ROI_SUFFIX)

def load_roi_boxes(roi_path: Path):
    """
    Accepts:
      - {"type":"box","box":[x1,y1,x2,y2]}
      - [{"type":"box","box":[...]} , ...]
    Returns: list of [x1,y1,x2,y2] (float)
    """
    if not roi_path.exists(): return []
    with open(roi_path, "r") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    boxes=[]
    for it in items:
        if isinstance(it, dict) and it.get("type","box")=="box" and "box" in it and len(it["box"])==4:
            b = [float(x) for x in it["box"]]
            boxes.append(b)
    return boxes

# ======= Util =======
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def imread_rgb(p): return cv2.cvtColor(cv2.imread(str(p),cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def imwrite_rgb(p, rgb): cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def resize_bool(m, out_hw):
    H,W = out_hw
    if m.shape==(H,W): return (m>0)
    pil = Image.fromarray((m.astype(np.uint8)*255)).resize((W,H), Image.NEAREST)
    return (np.array(pil)>127)

def clip_mask_to_box(mask_bool: np.ndarray, box, pad=0):
    """Zero out mask outside padded box (after resizing to GT size)."""
    h,w = mask_bool.shape
    x1,y1,x2,y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    out = np.zeros_like(mask_bool, dtype=bool)
    if x2 >= x1 and y2 >= y1:
        out[y1:y2+1, x1:x2+1] = mask_bool[y1:y2+1, x1:x2+1]
    return out

def save_overlay(path, rgb, masks, color=(0,255,0), alpha=0.45, boxes=None):
    vis=rgb.copy(); overlay=vis.copy()
    for m in masks: overlay[m>0.5]=color
    vis=cv2.addWeighted(overlay,alpha,vis,1-alpha,0)
    if boxes is not None:
        for (x1,y1,x2,y2) in boxes: cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
    imwrite_rgb(path, vis)

# ======= Metrics =======
def pixel_metrics(gt, pr):
    gt=(gt>0).astype(np.uint8); pr=(pr>0).astype(np.uint8)
    tp=np.logical_and(gt==1, pr==1).sum()
    fp=np.logical_and(gt==0, pr==1).sum()
    fn=np.logical_and(gt==1, pr==0).sum()
    dice=(2*tp+1e-9)/(2*tp+fp+fn+1e-9)
    iou =(tp+1e-9)/(tp+fp+fn+1e-9)
    prec=(tp+1e-9)/(tp+fp+1e-9)
    rec =(tp+1e-9)/(tp+fn+1e-9)
    return dice,iou,prec,rec

def _disk_kernel(r):
    k=2*r+1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))

def boundary_f(gt, pr, tol=2):
    gt=(gt>0).astype(np.uint8); pr=(pr>0).astype(np.uint8)
    k3=np.ones((3,3),np.uint8)
    b_gt=cv2.subtract(gt, cv2.erode(gt,k3,1)).astype(bool)
    b_pr=cv2.subtract(pr, cv2.erode(pr,k3,1)).astype(bool)
    if tol>0:
        ker=_disk_kernel(tol)
        b_gt_d=cv2.dilate(b_gt.astype(np.uint8),ker,1).astype(bool)
        b_pr_d=cv2.dilate(b_pr.astype(np.uint8),ker,1).astype(bool)
    else:
        b_gt_d,b_pr_d=b_gt,b_pr
    tp_p=np.logical_and(b_pr,b_gt_d).sum(); pp=b_pr.sum()
    tp_r=np.logical_and(b_gt,b_pr_d).sum(); rp=b_gt.sum()
    prec=tp_p/(pp+1e-9); rec=tp_r/(rp+1e-9)
    return 0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec+1e-9)

def mask_iou(a,b):
    inter=np.logical_and(a,b).sum()
    union=np.logical_or(a,b).sum()
    return inter/(union+1e-9)

def eval_map(preds, gts_by_img, iou_thresholds=np.arange(0.5,0.96,0.05)):
    preds=sorted(preds,key=lambda d:-d["score"])
    total_gts=sum(len(v) for v in gts_by_img.values())
    aps={}
    for thr in iou_thresholds:
        tp=np.zeros(len(preds),dtype=np.float32)
        fp=np.zeros(len(preds),dtype=np.float32)
        matched={k:np.zeros(len(v),dtype=bool) for k,v in gts_by_img.items()}
        for i,p in enumerate(preds):
            imgid=p["image_id"]; pm=p["mask"]
            cand=gts_by_img.get(imgid,[])
            best,jbest=0.0,-1
            for j,gm in enumerate(cand):
                if matched[imgid][j]: continue
                iou=mask_iou(pm,gm)
                if iou>best: best=iou; jbest=j
            if jbest>=0 and best>=thr:
                tp[i]=1.0; matched[imgid][jbest]=True
            else:
                fp[i]=1.0
        ctp=np.cumsum(tp); cfp=np.cumsum(fp)
        rec=ctp/max(1,total_gts); prec=ctp/np.maximum(1,ctp+cfp)
        mrec=np.concatenate(([0.0],rec,[1.0])); mpre=np.concatenate(([0.0],prec,[0.0]))
        for k in range(mpre.size-2,-1,-1): mpre[k]=max(mpre[k],mpre[k+1])
        rgrid=np.linspace(0,1,101); ap=0.0
        for r in rgrid: ap+= mpre[np.searchsorted(mrec,r,side="right")-1]
        aps[thr]=ap/101.0
    return float(aps.get(0.5,0.0)), float(np.mean(list(aps.values()))) if aps else 0.0

# ======= SAM predictor (or fallback) =======
try:
    from segment_anything import sam_model_registry, SamPredictor
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

class BoxPromptSegmenter:
    def __init__(self, sam_ckpt:str, sam_model_type:str, device:str):
        self.device = torch.device(f"cuda:{device}" if device!="cpu" and torch.cuda.is_available() else "cpu")
        self.ok = False
        if _HAS_SAM and sam_ckpt and os.path.isfile(sam_ckpt):
            log.info(f"Loading SAM checkpoint: {sam_ckpt} ({sam_model_type})")
            sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(self.device)
            self.pred = SamPredictor(sam)
            self.ok = True
        else:
            if _HAS_SAM and not sam_ckpt:
                log.warning("SAM installed but no checkpoint provided — using GrabCut fallback.")
            elif not _HAS_SAM:
                log.warning("segment-anything not available — using GrabCut fallback.")

    @torch.no_grad()
    def predict_masks(self, rgb, boxes_xyxy):
        H,W = rgb.shape[:2]
        if boxes_xyxy is None or len(boxes_xyxy)==0: return []
        if self.ok:
            self.pred.set_image(rgb)
            out=[]
            for b in boxes_xyxy:
                box = np.array(b, dtype=np.float32)
                m,_,_ = self.pred.predict(box=box[None,:], multimask_output=False)
                out.append(m[0].astype(np.uint8))
            return out
        # fallback: GrabCut seeded by the box
        masks=[]
        for (x1,y1,x2,y2) in np.asarray(boxes_xyxy).astype(int):
            x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
            if x2<=x1 or y2<=y1: continue
            grab = np.zeros((H,W), np.uint8)
            rect = (x1,y1,x2-x1,y2-y1)
            bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
            try:
                cv2.grabCut(rgb, grab, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                m = ((grab==cv2.GC_FGD) | (grab==cv2.GC_PR_FGD)).astype(np.uint8)
            except Exception as e:
                log.error(f"GrabCut failed on box {rect}: {e}")
                m = np.zeros((H,W), np.uint8)
            masks.append(m)
        return masks

# ======= Build GT instance list for AP =======
def build_gt_instances_per_image(cam_test_root: Path, imgs, mask_test_root: Path):
    gts_by_img={}
    for idx, ip in enumerate(imgs):
        gt_path = mask_test_root / ip.relative_to(cam_test_root).parent / (ip.stem + MASK_SUFFIX)
        gs = load_gt_instances_from_rle_json(gt_path)
        gts_by_img[idx] = [(g>0).astype(bool) for g in gs]
    return gts_by_img

# ======= Main evaluation =======
def evaluate_manual_roi_sam(cam_yaml, out_dir):
    out_dir = Path(out_dir); ensure_dir(out_dir)
    if SAVE_OVERLAYS_TEST: ensure_dir(out_dir/OVERLAY_SUBDIR)

    cam_test = read_yaml_images_root(Path(cam_yaml), "test")
    imgs = sorted([p for p in cam_test.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    log.info(f"TEST images found: {len(imgs)} under {cam_test}")

    segger = BoxPromptSegmenter(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)

    per_rows=[]; preds_ap=[]

    gts_by_img = build_gt_instances_per_image(cam_test, imgs, Path(TEST_MASK_ROOT))

    for img_id, ip in enumerate(tqdm(imgs, desc="ManualROI→SAM TEST", unit="img")):
        try:
            gt_path = Path(TEST_MASK_ROOT)/ip.relative_to(cam_test).parent/(ip.stem+MASK_SUFFIX)
            gts = load_gt_instances_from_rle_json(gt_path)
            if not gts: 
                continue
            gt_union = np.zeros_like(gts[0], dtype=np.uint8)
            for g in gts: gt_union |= (g>0).astype(np.uint8)
            H,W = gt_union.shape

            # find ROI JSON
            roi_file = roi_path_for_image(ip, cam_test)
            boxes = load_roi_boxes(roi_file)
            if not boxes:
                log.warning(f"[SKIP] No ROI boxes: {roi_file}")
                continue

            # run SAM / fallback
            rgb = imread_rgb(ip)
            masks = segger.predict_masks(rgb, boxes)

            # union with clip-to-box (with padding)
            union = np.zeros((H,W), np.uint8)
            vis_masks=[]
            for j,m in enumerate(masks):
                mb = resize_bool(m, (H,W)).astype(bool)
                # clip to (padded) ROI box
                box_j = boxes[j if j < len(boxes) else -1]
                mb = clip_mask_to_box(mb, box_j, pad=BOX_PAD)
                union |= mb.astype(np.uint8)
                vis_masks.append(mb.astype(np.uint8))
                # AP: manual ROI => give high score (1.0)
                preds_ap.append({"image_id": img_id, "mask": mb, "score": 1.0})

            d,i,p,r = pixel_metrics(gt_union, union); bf=boundary_f(gt_union, union, BOUNDARY_TOL)
            per_rows.append({"image": str(ip.relative_to(cam_test)), "Dice":d,"IoU":i,"Precision":p,"Recall":r,"BoundaryF":bf})

            if SAVE_OVERLAYS_TEST:
                ov = out_dir/OVERLAY_SUBDIR/f"SAM_ROI_{ip.stem}_vis.png"
                save_overlay(ov, rgb, vis_masks, boxes=boxes)

            if (img_id+1) % LOG_EVERY == 0:
                log.info(f"  processed {img_id+1}/{len(imgs)} images")
        except Exception as e:
            log.exception(f"[TEST] Error on {ip}: {e}")

        if (img_id+1) % 50 == 0:
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # summaries
    D=np.array([r["Dice"] for r in per_rows]); J=np.array([r["IoU"] for r in per_rows])
    P=np.array([r["Precision"] for r in per_rows]); R=np.array([r["Recall"] for r in per_rows]); BF=np.array([r["BoundaryF"] for r in per_rows])
    pix=dict(Dice=float(D.mean()) if D.size else 0.0,
             IoU=float(J.mean()) if J.size else 0.0,
             Precision=float(P.mean()) if P.size else 0.0,
             Recall=float(R.mean()) if R.size else 0.0,
             BoundaryF=float(BF.mean()) if BF.size else 0.0)

    ap50,map95 = eval_map(preds_ap, gts_by_img)

    # CSVs
    per_csv=Path(out_dir)/"test_per_image_ManualROI_SAM.csv"
    with open(per_csv,"w") as f:
        f.write("image,Dice,IoU,Precision,Recall,BoundaryF\n")
        for r in per_rows:
            f.write(f"{r['image']},{r['Dice']:.6f},{r['IoU']:.6f},{r['Precision']:.6f},{r['Recall']:.6f},{r['BoundaryF']:.6f}\n")

    sum_csv=Path(out_dir)/"test_summary_ManualROI_SAM.csv"
    with open(sum_csv,"w") as f:
        f.write("method,YOLO_conf,Dice,IoU,Precision,Recall,BoundaryF,AP50,mAP50_95\n")
        f.write(f"ManualROI_SAM,{YOLO_CONF:.2f},{pix['Dice']:.6f},{pix['IoU']:.6f},{pix['Precision']:.6f},{pix['Recall']:.6f},{pix['BoundaryF']:.6f},{ap50:.6f},{map95:.6f}\n")

    log.info(f"[ManualROI→SAM] Dice {pix['Dice']:.4f} | IoU {pix['IoU']:.4f} | P {pix['Precision']:.4f} | R {pix['Recall']:.4f} | BF {pix['BoundaryF']:.4f}")
    log.info(f"[ManualROI→SAM] AP50 {ap50:.4f} | mAP50:95 {map95:.4f}")
    log.info(f"Saved per-image → {per_csv}")
    log.info(f"Saved summary  → {sum_csv}")

# ======= main =======
def main():
    start = time.time()
    log.info("========== Manual ROI → SAM (clip-to-box) baseline ==========")
    log.info(f"Run name: {RUN_NAME}")
    log.info(f"Device: {'CUDA' if (DEVICE!='cpu' and torch.cuda.is_available()) else 'CPU'} | YOLO conf (header only): {YOLO_CONF:.2f}")
    if torch.cuda.is_available():
        log.info(f"CUDA devices: {torch.cuda.device_count()} | Current: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

    out_dir = Path(PROJECT)/RUN_NAME; out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_manual_roi_sam(CAM_YAML, out_dir)
    log.info(f"Total runtime: {time.time()-start:.1f}s")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"FATAL: {e}")
        sys.exit(1)
