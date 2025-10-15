import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import yaml
import logging
from datetime import datetime
from tqdm.auto import tqdm

# ---------------- logging setup ----------------
def setup_logging(level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=[logging.StreamHandler(sys.stdout)])
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

setup_logging()
log = logging.getLogger("SAM-BoxFromYOLO")

# ======= Fixed YOLO + general settings =======
YOLO_CONF = 0.20
IMG_SIZE  = 640
DEVICE    = "0"  # "0" or "cpu"
LOG_EVERY = 25   # images

# ======= Paths you edit =======
STUDENT_MODEL = r"C:\Users\djkim\Documents\Yolo11-seg_train_results\0815Results_complete_labeling\runs\segment\grid\scratch\user\kuocy\IKEAVideo\model_weights\yolo11s-seg_0816_123254_f2\weights\best.pt"
CAM_YAML      = r"C:\Users\djkim\Documents\mini_2\parts_single_cam.yaml"
MASK_SUFFIX   = "_rle.json"
VAL_MASK_ROOT  = r"C:\Users\djkim\Documents\mini_2\mask_labels\val\camera"
TEST_MASK_ROOT = r"C:\Users\djkim\Documents\mini_2\mask_labels\test\camera"
PROJECT       = r"C:\Users\djkim\Documents\mini_2\eval_runs"
RUN_NAME      = "baseline_SAM_box_from_YOLO_strong"

# ======= SAM (optional; fallback to GrabCut if missing) =======
SAM_CHECKPOINT = r"C:\Users\djkim\IKEAVideo\yolo_code\sam_vit_h_4b8939.pth"          # e.g., r"C:\models\sam_vit_h.pth" ; leave empty to use fallback
SAM_MODEL_TYPE = "vit_h"      # vit_h/vit_l/vit_b

# ======= Box robustness =======
BOX_PAD = 2                   # expand YOLO boxes by this many pixels on each side
SAVE_OVERLAYS_TEST = True
OVERLAY_SUBDIR = "overlays_sam_test"

# ======= Imports =======
from ultralytics import YOLO
try:
    from segment_anything import sam_model_registry, SamPredictor
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

# ======= YAML helpers =======
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

def merge_binary(masks_list):
    merged = None
    for m in masks_list: merged = m if merged is None else np.maximum(merged, m)
    return (merged > 0) if merged is not None else None

# ======= Metrics =======
def pixel_metrics(gt, pr):
    gt = (gt > 0).astype(np.uint8); pr = (pr > 0).astype(np.uint8)
    tp = np.logical_and(gt==1, pr==1).sum()
    fp = np.logical_and(gt==0, pr==1).sum()
    fn = np.logical_and(gt==1, pr==0).sum()
    dice = (2*tp + 1e-9)/(2*tp+fp+fn+1e-9)
    iou  = (tp + 1e-9)/(tp+fp+fn+1e-9)
    prec = (tp + 1e-9)/(tp+fp+1e-9)
    rec  = (tp + 1e-9)/(tp+fn+1e-9)
    return dice, iou, prec, rec

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

# Instance AP (class-agnostic)
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

# ======= Util =======
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def imread_rgb(p): 
    bgr = cv2.imread(str(p),cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def imwrite_rgb(p, rgb): cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def resize_bool(m, out_hw):
    H,W = out_hw
    if m.shape==(H,W): return m.astype(bool)
    pil=Image.fromarray((m.astype(np.uint8)*255)).resize((W,H), Image.NEAREST)
    return (np.array(pil)>127)

def expand_and_clip_box(box, H, W, pad=0):
    x1,y1,x2,y2 = map(int, box)
    if pad>0:
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def collect_yolo_boxes(model: YOLO, img_path: Path, conf: float):
    r = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=conf, device=DEVICE, verbose=False, half=(DEVICE!="cpu"))[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
    scores= r.boxes.conf.cpu().numpy()   if r.boxes is not None else np.zeros((0,))
    classes=r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,),dtype=int)
    return boxes, scores, classes

def save_overlay(path, rgb, masks, color=(0,255,0), alpha=0.45, boxes=None):
    vis=rgb.copy(); overlay=vis.copy()
    for m in masks: overlay[m>0.5]=color
    vis=cv2.addWeighted(overlay,alpha,vis,1-alpha,0)
    if boxes is not None:
        for (x1,y1,x2,y2) in boxes: cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
    imwrite_rgb(path, vis)

# ======= SAM predictor (or fallback) =======
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
            if _HAS_SAM:
                log.warning("SAM installed but no checkpoint provided — using GrabCut fallback.")
            else:
                log.warning("segment-anything not available — using GrabCut fallback.")

    @torch.no_grad()
    def predict_masks(self, rgb, boxes_xyxy, box_pad=0):
        H,W = rgb.shape[:2]
        if boxes_xyxy.size == 0:
            return []
        if self.ok:
            self.pred.set_image(rgb)
            masks_out=[]
            for b in boxes_xyxy:
                eb = expand_and_clip_box(b, H, W, pad=box_pad)
                if eb is None: 
                    continue
                m,_,_ = self.pred.predict(box=eb[None,:], multimask_output=False)
                masks_out.append(m[0].astype(np.uint8))
            return masks_out
        # --- fallback: simple grabcut per expanded box ---
        masks=[]
        for b in boxes_xyxy.astype(int):
            eb = expand_and_clip_box(b, H, W, pad=box_pad)
            if eb is None: 
                continue
            x1,y1,x2,y2 = map(int, eb)
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

# ======= VAL calibration (tiny threshold sweep, cached) =======
def load_best_sam_from_csv(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists(): 
        return None
    best = None
    with open(csv_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines[1:]:
        thr_s, dice_s, *_ = ln.split(",")
        row = {"thr": float(thr_s), "Dice": float(dice_s)}
        if (best is None) or (row["Dice"] > best["Dice"]):
            best = row
    return best

def val_calibrate_sam(yolo_model: YOLO, cam_yaml, val_mask_root, mask_suffix, segger: BoxPromptSegmenter, out_dir):
    csv_path = Path(out_dir)/"val_calib_sam.csv"
    cached = load_best_sam_from_csv(csv_path)
    if cached is not None:
        log.info(f"[VAL] Loaded SAM best from CSV: thr={cached['thr']:.2f} (Dice={cached['Dice']:.4f})")
        return cached

    log.info("== VAL calibration: SAM threshold sweep ==")
    t0 = time.time()
    cam_val = read_yaml_images_root(Path(cam_yaml), "val")
    imgs = sorted([p for p in cam_val.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    log.info(f"VAL images found: {len(imgs)} under {cam_val}")
    if not imgs: raise RuntimeError("No VAL images.")
    thr_grid = [0.35, 0.50]  # simple, robust grid
    rows=[]

    with open(csv_path,"w") as f:
        f.write("thr,Dice,IoU,Precision,Recall,BoundaryF\n")

    for thr in thr_grid:
        mets=[]
        log.info(f"  - Testing SAM threshold = {thr:.2f}")
        for i, ip in enumerate(tqdm(imgs, desc=f"VAL thr={thr:.2f}", unit="img")):
            try:
                gt_path = Path(val_mask_root)/ip.relative_to(cam_val).parent/(ip.stem+mask_suffix)
                gts = load_gt_instances_from_rle_json(gt_path)
                if not gts: 
                    continue
                gt_union = merge_binary(gts); H,W = gt_union.shape
                rgb = imread_rgb(ip)
                boxes, scores, _ = collect_yolo_boxes(yolo_model, ip, YOLO_CONF)
                masks = segger.predict_masks(rgb, boxes, box_pad=BOX_PAD)
                union = np.zeros((H,W), np.uint8)
                for m in masks:
                    union |= resize_bool(m>=thr, (H,W)).astype(np.uint8)
                d,i,p,r = pixel_metrics(gt_union, union); bf=boundary_f(gt_union, union, 2)
                mets.append((d,i,p,r,bf))
                if (i+1) % LOG_EVERY == 0:
                    log.info(f"    processed {i+1}/{len(imgs)} images (thr={thr:.2f})")
            except Exception as e:
                log.exception(f"[VAL] Error on {ip}: {e}")
        if mets:
            mean=np.mean(np.array(mets),axis=0)
            row = {"thr":thr,"Dice":mean[0],"IoU":mean[1],"Precision":mean[2],"Recall":mean[3],"BoundaryF":mean[4]}
            rows.append(row)
            with open(csv_path,"a") as f:
                f.write(f"{row['thr']:.4f},{row['Dice']:.6f},{row['IoU']:.6f},{row['Precision']:.6f},{row['Recall']:.6f},{row['BoundaryF']:.6f}\n")
            log.info(f"  -> thr={thr:.2f} | Dice {mean[0]:.4f} IoU {mean[1]:.4f} P {mean[2]:.4f} R {mean[3]:.4f} BF {mean[4]:.4f}")

    best = max(rows, key=lambda r:r["Dice"]) if rows else {"thr":0.5, "Dice":0.0}
    log.info(f"Best SAM thr on VAL: {best['thr']:.2f} (saved {csv_path}) [{time.time()-t0:.1f}s]")
    return best

# ======= TEST evaluation =======
def evaluate_test_sam(yolo_model: YOLO, cam_yaml, test_mask_root, mask_suffix, segger: BoxPromptSegmenter, params, out_dir):
    log.info("== TEST evaluation: SAM box prompts from YOLO ==")
    t0 = time.time()
    cam_test = read_yaml_images_root(Path(cam_yaml), "test")
    imgs = sorted([p for p in cam_test.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    log.info(f"TEST images found: {len(imgs)} under {cam_test}")
    per_rows=[]; preds_ap=[]; gts_by_img={}

    if SAVE_OVERLAYS_TEST:
        ov_dir = Path(out_dir)/OVERLAY_SUBDIR
        ensure_dir(ov_dir)

    for img_id, ip in enumerate(tqdm(imgs, desc="TEST", unit="img")):
        try:
            gt_path = Path(test_mask_root)/ip.relative_to(cam_test).parent/(ip.stem+mask_suffix)
            gts = load_gt_instances_from_rle_json(gt_path)
            if not gts: 
                continue
            gt_union = merge_binary(gts); H,W=gt_union.shape
            gts_by_img[img_id] = [(g > 0).astype(bool) for g in gts]  # for AP
            rgb = imread_rgb(ip)
            boxes, scores, _ = collect_yolo_boxes(yolo_model, ip, YOLO_CONF)
            masks = segger.predict_masks(rgb, boxes, box_pad=BOX_PAD)

            # union + per-instance AP
            union=np.zeros((H,W),np.uint8)
            for j,m in enumerate(masks):
                mb = resize_bool(m>=params["thr"], (H,W)).astype(np.uint8)
                union |= mb
                preds_ap.append({"image_id": img_id, "mask": mb.astype(bool), "score": float(scores[j] if j<len(scores) else 0.5)})

            d,i,p,r = pixel_metrics(gt_union, union); bf=boundary_f(gt_union, union, 2)
            per_rows.append({"image": str(ip.relative_to(cam_test)), "Dice":d,"IoU":i,"Precision":p,"Recall":r,"BoundaryF":bf})

            if SAVE_OVERLAYS_TEST:
                ov = Path(out_dir)/OVERLAY_SUBDIR/f"SAM_{ip.stem}_vis.png"
                vis=rgb.copy(); overlay=vis.copy()
                for m in masks: overlay[resize_bool(m,(H,W))>0]= (0,255,0)
                vis=cv2.addWeighted(overlay,0.45,vis,0.55,0); cv2.imwrite(str(ov), cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))

            if (img_id+1) % LOG_EVERY == 0:
                log.info(f"  processed {img_id+1}/{len(imgs)} images")

        except Exception as e:
            log.exception(f"[TEST] Error on {ip}: {e}")

        if (img_id+1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    per_csv=Path(out_dir)/"test_per_image_SAM.csv"
    with open(per_csv,"w") as f:
        f.write("image,Dice,IoU,Precision,Recall,BoundaryF\n")
        for r in per_rows:
            f.write(f"{r['image']},{r['Dice']:.6f},{r['IoU']:.6f},{r['Precision']:.6f},{r['Recall']:.6f},{r['BoundaryF']:.6f}\n")

    sum_csv=Path(out_dir)/"test_summary_SAM.csv"
    with open(sum_csv,"w") as f:
        f.write("method,YOLO_conf,thr,Dice,IoU,Precision,Recall,BoundaryF,AP50,mAP50_95\n")
        f.write(f"SAM_box_from_YOLO,{YOLO_CONF:.2f},{params['thr']:.2f},{pix['Dice']:.6f},{pix['IoU']:.6f},{pix['Precision']:.6f},{pix['Recall']:.6f},{pix['BoundaryF']:.6f},{ap50:.6f},{map95:.6f}\n")

    log.info(f"[SAM] YOLO conf = {YOLO_CONF:.2f} | thr={params['thr']:.2f}")
    log.info(f"[SAM] Dice {pix['Dice']:.4f} | IoU {pix['IoU']:.4f} | P {pix['Precision']:.4f} | R {pix['Recall']:.4f} | BF {pix['BoundaryF']:.4f}")
    log.info(f"[SAM] AP50 {ap50:.4f} | mAP50:95 {map95:.4f}")
    log.info(f"Saved per-image → {per_csv}")
    log.info(f"Saved summary  → {sum_csv}")
    log.info(f"TEST done in {time.time()-t0:.1f}s")

# ======= main =======
def main():
    start = time.time()
    log.info("========== SAM (box-prompt from YOLO) baseline ==========")
    log.info(f"Run name: {RUN_NAME}")
    log.info(f"Device: {'CUDA' if (DEVICE!='cpu' and torch.cuda.is_available()) else 'CPU'} | YOLO conf: {YOLO_CONF:.2f}")
    if torch.cuda.is_available():
        log.info(f"CUDA devices: {torch.cuda.device_count()} | Current: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

    out_dir = Path(PROJECT)/RUN_NAME; out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading YOLO student model...")
    model = YOLO(STUDENT_MODEL)
    # Perf knobs like the other baselines
    model.model.eval()
    torch.backends.cudnn.benchmark = True
    log.info("YOLO loaded.")

    segger = BoxPromptSegmenter(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)

    # VAL calib (cached)
    best = val_calibrate_sam(model, CAM_YAML, VAL_MASK_ROOT, MASK_SUFFIX, segger, out_dir)

    # TEST
    evaluate_test_sam(model, CAM_YAML, TEST_MASK_ROOT, MASK_SUFFIX, segger, best, out_dir)
    log.info(f"Total runtime: {time.time()-start:.1f}s")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"FATAL: {e}")
        sys.exit(1)
