# yolo_dataset_builder.py
import os, cv2, random, json, numpy as np
from pathlib import Path
from pycocotools import mask as maskUtils
from IKEAVideo.dataloader.assembly_video import load_video, load_frame

# -------------------- validity helpers --------------------

def rle_valid(r):
    return isinstance(r, dict) and r.get("size") and r.get("counts")

def list_all_valid(rle_list):
    return all(rle_valid(r) for r in rle_list)

def extrinsic_valid(mat):
    if not (isinstance(mat, list) and len(mat) == 4):
        return False
    for row in mat:
        if not (isinstance(row, list) and len(row) == 4):
            return False
        if not all(isinstance(val, (int, float)) for val in row):
            return False
    return True

def list_all_extrinsics_valid(extrinsics):
    return all(extrinsic_valid(m) for m in extrinsics)

# -------------------- mask / polygon helpers --------------------

def count_ids(part_uid):
    """Count how many IDs are in a uid like '0,3,5' (or list/tuple)."""
    if isinstance(part_uid, (list, tuple, set)):
        return len(part_uid)
    s = str(part_uid).strip()
    if not s:
        return 0
    return len([x for x in s.replace(' ', '').split(',') if x != ''])

def class_from_uid(part_uid):
    """0 = single part, 1 = subassembly (>=2 parts)."""
    return 1 if count_ids(part_uid) >= 2 else 0

def rle_to_mask_u8(rle, target_hw=None):
    """Decode COCO RLE → uint8 mask in {0,255}. Optional resize to (H,W)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    m = maskUtils.decode({"size": rle["size"], "counts": counts})  # 0/1
    if target_hw is not None:
        H, W = target_hw
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m.astype(np.uint8) * 255)

def contours_to_rows(mask_u8, W, H, class_id,
                     keep_all=True,
                     min_area_px=2,
                     chain_mode=cv2.CHAIN_APPROX_NONE):
    """
    Convert a binary mask (0/255) to YOLO polygon row(s).
    - keep_all=True: return one polygon per external component.
    - Holes are not exported (YOLO txt doesn't support them).
    """
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, chain_mode)
    rows = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        poly = (c[:, 0, :] / [W, H]).reshape(-1)
        rows.append(f"{class_id} " + " ".join(f"{p:.6f}" for p in poly))
        #if not keep_all:
            #break
    return rows

def save_rle_list_json(path, rle_list):
    norm = []
    for r in rle_list:
        counts = r["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        norm.append({"size": r["size"], "counts": counts})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(norm))
# -------------------- manual PDF crop --------------------

def vertical_zone_crop(img, mask_list, pad=0):
    """
    img       : HxWx3
    mask_list : list of HxW (0/1 or 0/255)
    returns   : crop_img, crop_masks, ymin, ymax
    """
    H, W = img.shape[:2]
    if len(mask_list) == 0:
        return img.copy(), [], 0, H

    # build union mask in {0,1}
    union = np.zeros((H, W), dtype=np.uint8)
    for m in mask_list:
        union |= (m > 0).astype(np.uint8)

    ys, xs = np.where(union > 0)
    if ys.size == 0:
        return img.copy(), [], 0, H

    y0, y1 = ys.min()-pad, ys.max()+pad
    x0, x1 = xs.min()-pad, xs.max()+pad
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(H-1, y1), min(W-1, x1)
    cy = (y0 + y1) / 2

    if cy < H/2 - 50:
        ymin, ymax = 0, H//2
    elif cy > H/2 + 50:
        ymin, ymax = H//2, H
    else:
        ymin, ymax = 0, H

    crop_img   = img[ymin:ymax, :].copy()
    crop_masks = [ (m[ymin:ymax, :].copy()).astype(np.uint8) for m in mask_list ]
    return crop_img, crop_masks, ymin, ymax

# -------------------- split util --------------------

def first_type_per_category(keyframes, skip_cat="Misc"):
    first_seen = {}
    for f in keyframes:
        cat, name = f.get("category"), f.get("name")
        if cat == skip_cat:
            continue
        if cat not in first_seen:
            first_seen[cat] = name
    return first_seen

# -------------------- main builder --------------------

def build(mini_root, keyframes, video_dir, split_ratio_1=0.8, split_ratio_2=0.5):
    """
    keyframes: list of dicts from KeyframeDataset
    mini_root: Path where mini/images + mini/labels + mini/parts_single.yaml live
    """
    mini_root = Path(mini_root)
    unseen_type = first_type_per_category(keyframes)

    split_buckets = {
        "test_unseen_category": [],
        **{f"test_unseen_type_{c}": [] for c in unseen_type},
        "valid": [],
        "na":   [],
        "train":[],
        "val" :[],
        "test":[]
    }

    # filter/route frames
    for f in keyframes:
        cam_masks   = f.get("mask")
        man_section = f.get("manual", {})
        man_masks   = man_section.get("mask")

        if not cam_masks or not man_masks:
            split_buckets["na"].append(f)
            continue

        if not (list_all_valid(cam_masks) and
                list_all_valid(man_masks) and
                list_all_extrinsics_valid(f.get("extrinsics", []))):
            split_buckets["na"].append(f)
            continue

        cat, name = f["category"], f["name"]
        if cat == "Misc":
            split_buckets["test_unseen_category"].append(f)
        elif unseen_type.get(cat) == name:
            split_buckets[f"test_unseen_type_{cat}"].append(f)
        else:
            split_buckets["valid"].append(f)

    valid = split_buckets["valid"]
    random.shuffle(valid)
    cut  = int(len(valid) * split_ratio_1)
    cut2 = int(len(valid[cut:]) * split_ratio_2)

    split_buckets["train"] = valid[:cut]
    split_buckets["val"]   = valid[cut : cut + cut2]
    split_buckets["test"]  = valid[cut + cut2:]
    split_buckets.pop("valid", None)

    # mkdirs
    all_splits = list(split_buckets.keys())
    uidmap = {s: {} for s in all_splits}

    for split in all_splits:
        for side in ("camera", "manual"):
            (mini_root / "images" / split / side).mkdir(parents=True, exist_ok=True)
            (mini_root / "labels" / split / side).mkdir(parents=True, exist_ok=True)
            (mini_root / "mask_labels" / split / side).mkdir(parents=True, exist_ok=True)
    # ------------- dump images & labels -------------
    def process(frames, split):
        for idx, f in enumerate(frames):
            name = f"{idx:04d}"
            cam_uid_order, man_uid_order = [], []

            # ---------- camera ----------
            print("NOW AT", f["category"], f["name"], f.get('frame_id'))
            vid  = load_video(video_dir, f["category"], f["name"], f["video_url"])
            try:
                cam_img = load_frame(vid, f["frame_time"])   # BGR
            except Exception:
                continue

            h, w = cam_img.shape[:2]
            cv2.imwrite(str(mini_root / f"images/{split}/camera/{name}.jpg"), cam_img)

            cam_rows = []
            for part_uid, rle in zip(f.get('frame_parts', []), f.get("mask", [])):
                cls     = class_from_uid(part_uid)
                mask_u8 = rle_to_mask_u8(rle, target_hw=(h, w))  # 0/255
                rows    = contours_to_rows(mask_u8, w, h, cls,
                                           keep_all=False,
                                           min_area_px=2,
                                           chain_mode=cv2.CHAIN_APPROX_NONE)
                if rows:
                    cam_rows.extend(rows)
                    cam_uid_order.append(part_uid)  # note: same uid may repeat if multiple components

            (mini_root / f"labels/{split}/camera/{name}.txt").write_text("\n".join(cam_rows))
            save_rle_list_json(mini_root / f"mask_labels/{split}/camera/{name}_rle.json", f.get("mask", []))
            # ---------- manual ----------
            # Expect f["pdf_img"] to be a PIL image or RGB ndarray
            pdf_img = f["pdf_img"]
            if hasattr(pdf_img, "mode"):  # PIL.Image
                page_np = cv2.cvtColor(np.array(pdf_img), cv2.COLOR_RGB2BGR)
            else:
                page_np = pdf_img.copy()  # assume already BGR or RGB (either is fine for polygons)

            man_img = page_np
            page_H, page_W = man_img.shape[:2]

            man_masks = []
            for rle in f['manual']['mask']:
                mask1 = rle_to_mask_u8(rle, target_hw=(page_H, page_W))  # 0/255
                man_masks.append(mask1)

            crop_img, crop_masks, ymin, ymax = vertical_zone_crop(man_img, man_masks, pad=20)
            h2, w2 = crop_img.shape[:2]
            cv2.imwrite(str(mini_root / f"images/{split}/manual/{name}.jpg"), crop_img)

            man_rows = []
            cropped_rles = []
            for part_uid, c_mask in zip(f['manual']['parts'], crop_masks):
                cls     = class_from_uid(part_uid)
                mask_u8 = (c_mask > 0).astype(np.uint8) * 255
                rows    = contours_to_rows(mask_u8, w2, h2, cls,
                                           keep_all=False,
                                           min_area_px=2,
                                           chain_mode=cv2.CHAIN_APPROX_NONE)
                if rows:
                    man_rows.extend(rows)
                    man_uid_order.append(part_uid)
                rle_crop = maskUtils.encode(np.asfortranarray(c_mask.astype(np.uint8)))
                if isinstance(rle_crop["counts"], bytes):
                    rle_crop["counts"] = rle_crop["counts"].decode("ascii")
                cropped_rles.append(rle_crop)

            (mini_root / f"labels/{split}/manual/{name}.txt").write_text("\n".join(man_rows))
            save_rle_list_json(mini_root / f"mask_labels/{split}/manual/{name}_rle.json", cropped_rles)

            # record mapping
            info = {
                'image_filename': name,
                'frame_id': f.get('frame_id'),
                'category': f.get('category'),
                'name': f.get('name'),
                'video_id': f.get('video_url'),
                'camera': cam_uid_order,
                'manual': man_uid_order,
                'paths': {
                    'cam_img': f"images/{split}/camera/{name}.jpg",
                    'man_img': f"images/{split}/manual/{name}.jpg",
                    'cam_txt': f"labels/{split}/camera/{name}.txt",
                    'man_txt': f"labels/{split}/manual/{name}.txt",
                    'cam_rle': f"mask_labels/{split}/camera/{name}_rle.json",
                    'man_rle': f"mask_labels/{split}/manual/{name}_rle.json",
                }
            }
            uidmap[split][name] = info

    for split, frames in split_buckets.items():
        if split in ("na",):
            continue
        process(frames, split)

    (mini_root / "uid_map.json").write_text(json.dumps(uidmap, indent=2))

    # dataset YAML (2 classes: part, subparts)
    yaml = f"""\
path: {mini_root.resolve()}
train: images/train
val:   images/val
test:  images/test
names:
  0: part
  1: subparts
"""
    (mini_root / "parts_single.yaml").write_text(yaml)
    print("YOLO dataset built:", mini_root)
