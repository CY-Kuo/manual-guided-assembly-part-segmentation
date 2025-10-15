# train_kfold_use_existing_all.py
# Run existing folds (fold0..foldN) with the same ablation trainer,
# but FIX camera_root to the camera leaf to avoid double-counting manual images.

import os, dataclasses, time, json, random
from pathlib import Path
from typing import Optional, List, Tuple

import torch, numpy as np

from train_ablation import (
    Cfg, ABLATION_GRID, AblationFlags, load_cfg, _strip_unknown_cfg_keys, run_one_training
)

# ---- USER DEFAULTS ----
DEFAULT_CONFIG_PATH = "/scratch/user/kuocy/IKEAVideo/model_code/distill_train_code/config_ablation_kfold.yaml"
DEFAULT_KFOLD_ROOT  = "/scratch/user/kuocy/IKEAVideo/model_code/distill_train_code/train_ablation_kfold_results/_kfold_temp"

def _seed_all(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _pick_ablations(cfg_ablations: Optional[List[str]], only_ablation: Optional[str]) -> List[AblationFlags]:
    if only_ablation: want = {only_ablation}
    else:             want = set(cfg_ablations) if cfg_ablations is not None else {a.name for a in ABLATION_GRID}
    abls = [a for a in ABLATION_GRID if a.name in want]
    if not abls: raise ValueError(f"No valid ablation names. Valid: {[a.name for a in ABLATION_GRID]}")
    return abls

def _fold_dirs(kfold_root: Path, only_folds: Optional[str]) -> List[Path]:
    if only_folds:
        idxs = []
        for tok in only_folds.split(","):
            tok = tok.strip()
            if not tok: continue
            if "-" in tok:
                a, b = map(int, tok.split("-", 1))
                idxs.extend(range(min(a,b), max(a,b)+1))
            else:
                idxs.append(int(tok))
        folds = [kfold_root / f"fold{i}" for i in idxs]
    else:
        folds = sorted([p for p in kfold_root.iterdir() if p.is_dir() and p.name.lower().startswith("fold")])
    if not folds: raise FileNotFoundError(f"No fold* directories found under {kfold_root}")
    return folds

def _make_fold_cfg(base_cfg: Cfg, fold_dir: Path) -> Cfg:
    train_root = fold_dir / "train"
    val_root   = fold_dir / "val"
    if not train_root.exists(): raise FileNotFoundError(f"train split not found: {train_root}")
    if not val_root.exists():   raise FileNotFoundError(f"val split not found: {val_root}")

    cam_dir_name = Path(base_cfg.images_camera_dirname).name   # "camera"
    man_dir_name = Path(base_cfg.images_manual_dirname).name   # "manual"

    # ✅ FIX: point camera_root at the **camera leaf** to prevent manual images from being enumerated.
    return dataclasses.replace(
        base_cfg,
        camera_root=str(train_root / "images" / cam_dir_name),
        mask_root_camera=str(train_root / "mask_labels" / cam_dir_name),
        mask_root_manual=str(train_root / "mask_labels" / man_dir_name),
        val_camera_root=str(val_root / "images" / cam_dir_name),
        val_mask_root_camera=str(val_root / "mask_labels" / cam_dir_name),
        out_dir=str(base_cfg.out_dir)
    )

# ---- diagnostics ----
def _count_images(root: Optional[Path], exts: List[str]) -> int:
    if root is None or not root.exists(): return 0
    ex = {e.lower() for e in exts}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ex)

def _count_json(root: Optional[Path], suffix: str) -> int:
    if root is None or not root.exists(): return 0
    sfx = suffix.lower()
    return sum(1 for p in root.rglob("*")
               if p.is_file()
               and p.suffix.lower() in {".json",".yml",".yaml"}
               and p.name.lower().endswith(sfx))

def _debug_print_trainval_fs(cfg: Cfg):
    cam_dir = Path(cfg.images_camera_dirname).name
    man_dir = Path(cfg.images_manual_dirname).name
    img_exts = cfg.img_exts
    mask_suffix = cfg.mask_suffix

    cam_train    = Path(cfg.camera_root).resolve()                           # ✅ camera leaf
    train_images = cam_train.parent                                          # images/
    man_train    = train_images / man_dir
    mcam_train   = Path(cfg.mask_root_camera).resolve()
    mman_train   = Path(cfg.mask_root_manual).resolve()

    val_cam    = Path(cfg.val_camera_root).resolve() if cfg.val_camera_root else None
    val_mcam   = Path(cfg.val_mask_root_camera).resolve() if cfg.val_mask_root_camera else None

    n_cam_tr  = _count_images(cam_train, img_exts)
    n_man_tr  = _count_images(man_train, img_exts)
    n_mcam_tr = _count_json(mcam_train, mask_suffix)
    n_mman_tr = _count_json(mman_train, mask_suffix)
    n_cam_va  = _count_images(val_cam, img_exts) if val_cam else 0
    n_mcam_va = _count_json(val_mcam, mask_suffix) if val_mcam else 0

    def S(p): return "OK" if p and p.exists() else "MISSING"

    print("\n[DEBUG/TRAIN] ==== TRAIN FILESYSTEM CHECK ====")
    print(f"  train images base      : {str(train_images):<76} [{S(train_images)}]")
    print(f"  images/camera (root)   : {str(cam_train):<76} [{S(cam_train)}]  (imgs={n_cam_tr})")
    print(f"  images/manual (paired) : {str(man_train):<76} [{S(man_train)}]  (imgs={n_man_tr})")
    print(f"  mask_labels/camera     : {str(mcam_train):<76} [{S(mcam_train)}]   (json={n_mcam_tr})")
    print(f"  mask_labels/manual     : {str(mman_train):<76} [{S(mman_train)}]   (json={n_mman_tr})")

    # sample pair
    sample = None
    if cam_train.exists():
        ex = {e.lower() for e in img_exts}
        for p in sorted(cam_train.rglob("*")):
            if p.is_file() and p.suffix.lower() in ex:
                sample = p; break
    if sample is not None:
        rel = sample.relative_to(cam_train)
        man_path = man_train / rel
        mjson = mcam_train / rel.parent / (sample.stem + mask_suffix)
        print("  Sample mapping:")
        print(f"    cam  -> {sample}")
        print(f"    man  -> {man_path}  [{'OK' if man_path.exists() else 'MISSING'}]")
        print(f"    mask -> {mjson}  [{'OK' if mjson.exists() else 'MISSING'}]")
    else:
        print("  Samples: (no camera images found to demo)")
    print("============================================\n")

    if val_cam:
        print("[DEBUG/VAL] ==== VAL FILESYSTEM CHECK ====")
        print(f"  val images/camera      : {str(val_cam):<76} [{'OK' if val_cam.exists() else 'MISSING'}]  (imgs={n_cam_va})")
        print(f"  val mask_labels/cam    : {str(val_mcam):<76} [{'OK' if val_mcam and val_mcam.exists() else 'MISSING'}]   (json={n_mcam_va})")
        print("==========================================\n")

def _eval_feasibility_scan(cfg: Cfg) -> Tuple[int,int,int,int,bool]:
    try:
        from pycocotools import mask as mask_utils  # noqa
        has_pycoco = True
    except Exception:
        has_pycoco = False
    cam_root = Path(cfg.val_camera_root) if cfg.val_camera_root else None
    mroot    = Path(cfg.val_mask_root_camera) if cfg.val_mask_root_camera else None
    if not cam_root or not cam_root.exists():
        print("[EVAL-SCAN] No val_camera_root or path missing.")
        return 0,0,0,0,has_pycoco
    ex = {e.lower() for e in cfg.img_exts}
    cams = sorted([p for p in cam_root.rglob("*") if p.is_file() and p.suffix.lower() in ex])
    n_imgs = len(cams); n_with_mask = n_with_pair = n_decodable = 0
    man_dir = Path(cfg.images_manual_dirname).name
    man_root = cam_root.parent / man_dir
    for p in cams:
        rel = p.relative_to(cam_root)
        man = man_root / rel
        mjs = (mroot / rel.parent / (p.stem + cfg.mask_suffix)) if mroot else None
        has_mask = (mjs is not None and mjs.exists())
        has_pair = man.exists()
        if has_mask: n_with_mask += 1
        if has_pair: n_with_pair += 1
        if has_mask and has_pycoco:
            try:
                with open(mjs, "r") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
                from pycocotools import mask as mask_utils
                for it in items:
                    r = it.get("segmentation", it)
                    if isinstance(r, dict) and "counts" in r and "size" in r:
                        rle = {"counts": r["counts"].encode("utf-8") if isinstance(r["counts"], str) else r["counts"],
                               "size": [int(r["size"][0]), int(r["size"][1])]}
                        if mask_utils.decode(rle) is not None:
                            n_decodable += 1; break
            except Exception:
                pass
    print(f"[EVAL-SCAN] imgs={n_imgs}  with_mask={n_with_mask}  with_pair={n_with_pair}  decodable_masks={n_decodable}  pycocotools={'yes' if has_pycoco else 'NO'}")
    return n_imgs, n_with_mask, n_with_pair, n_decodable, has_pycoco

def main_use_existing_all(cfg_path: str,
                          kfold_root: str,
                          out_dir_override: Optional[str] = None,
                          only_ablation: Optional[str] = None,
                          only_folds: Optional[str] = None):
    _seed_all(0)

    raw = load_cfg(cfg_path)
    raw.setdefault("fusion_dims",   {raw.get("aux_head_tap", "model.23"): 128, "model.10": 96, "model.6": 96})
    raw.setdefault("fusion_tokens", {raw.get("aux_head_tap", "model.23"): 4,   "model.10": 4,   "model.6": 4})
    raw.setdefault("lambda_fuse_reg", {"model.10": 0.10, "model.6": 0.05})
    raw = _strip_unknown_cfg_keys(raw, Cfg)
    base_cfg = Cfg(**raw)

    kroot = Path(os.path.expanduser(os.path.expandvars(kfold_root))).resolve()
    folds = _fold_dirs(kroot, only_folds)
    ablations = _pick_ablations(base_cfg.ablations, only_ablation)

    run_tag = time.strftime("%m%d_%H%M%S")
    base_out = Path(out_dir_override or base_cfg.out_dir).resolve() / f"kfold_direct_all_{run_tag}"
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\n========== USING EXISTING KFOLD ROOT: {kroot} ==========")
    print(f"Ablations to run: {[a.name for a in ablations]}")
    print(f"Output base     : {base_out}")
    print(f"eval_every      : {getattr(base_cfg, 'eval_every', 0)}")
    print(f"lambda_fuse_reg : {getattr(base_cfg, 'lambda_fuse_reg', None)}")

    for fold_dir in folds:
        fold_name = fold_dir.name
        print(f"\n---------- {fold_name}: preparing config ----------")
        fold_cfg = _make_fold_cfg(base_cfg, fold_dir)

        print("[CHECK] train camera root =", Path(fold_cfg.camera_root))
        print("[CHECK] val   camera root =", fold_cfg.val_camera_root)
        print("[CHECK] val   masks root  =", fold_cfg.val_mask_root_camera)
        _debug_print_trainval_fs(fold_cfg)
        _eval_feasibility_scan(fold_cfg)

        fold_out = base_out / fold_name
        fold_out.mkdir(parents=True, exist_ok=True)

        for ablate in ablations:
            ab_dir = fold_out / ablate.name
            ab_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Running ablation: {ablate.name} -> {ab_dir} ===")
            try:
                run_one_training(fold_cfg, ablate, str(ab_dir))
            except Exception as e:
                print(f"[ERROR] {fold_name}/{ablate.name} failed: {e}")

    print("\n✅ All folds finished.")

# ---- entry ----
if __name__ == "__main__":
    cfg_path      = DEFAULT_CONFIG_PATH
    kfold_root    = DEFAULT_KFOLD_ROOT
    out_dir       = None
    only_ablation = None
    only_folds    = "3,4"

    cfg_path   = os.path.expanduser(os.path.expandvars(cfg_path))
    kfold_root = os.path.expanduser(os.path.expandvars(kfold_root))
    main_use_existing_all(cfg_path, kfold_root, out_dir, only_ablation, only_folds)
