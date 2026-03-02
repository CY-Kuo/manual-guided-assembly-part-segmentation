#!/usr/bin/env python3
# scripts/run_inference_baseline.py
# Unified inference entrypoint for all baselines. Load weights and produce segmentation outputs + metrics.

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_ablation import (
    _prep_image,
    _paired_manual_path,
    _decode_instances_from_json,
    _dice_iou,
    _eval_map,
    _resize_bool_mask,
)
from feature_hooks import FeatureHooker
from aux_heads import AuxMaskHead
from train_ablation import resolve_canonical_tap
from test_ablation_5tracks_multi import _pix_metrics

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

try:
    import yaml
except Exception:
    yaml = None


def _load_state_dict_matching_shapes(model, state_dict: dict, tag: str = "") -> None:
    """Load state_dict into model only for keys with matching shapes; skip mismatches (e.g. 1-class vs 80-class head)."""
    model_sd = model.state_dict()
    loaded = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
    missing = set(state_dict.keys()) - set(loaded.keys())
    if missing:
        print(f"[{tag}] Skipping {len(missing)} checkpoint keys (shape mismatch): {list(missing)[:4]}{'...' if len(missing) > 4 else ''}")
    model.load_state_dict(loaded, strict=False)


def _resolve_student_model_path(path_from_config: Optional[str]) -> str:
    """Use config path if it exists on disk; otherwise use Ultralytics built-in (same arch as training)."""
    if path_from_config and path_from_config.strip():
        p = Path(path_from_config.strip())
        if p.is_file():
            return str(p.resolve())
        # Strip leading backslashes that can appear when paths were saved from other OS
        p_clean = Path(str(p).lstrip("\\").lstrip("/"))
        if p_clean.is_file():
            return str(p_clean.resolve())
    return "yolo11s-seg.pt"  # Ultralytics built-in; checkpoint weights are loaded on top


def _read_yaml_images_root(yaml_path: Path, split_key: str) -> Path:
    """Read YAML and return the image root path for split_key (e.g. 'test').
    Resolves relative paths against the YAML file's parent directory, so the YAML's
    'path' field is ignored. This lets the same *_unseen_*.yaml (e.g. with path:
    /scratch/user/kuocy/...) work on Harvard when the YAML lives under the Harvard
    mini_2 directory."""
    if yaml is None:
        raise FileNotFoundError("PyYAML required for YAML-based test paths")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Use YAML's parent as root so path: /scratch/user/kuocy/... is ignored on Harvard
    root = yaml_path.parent.resolve()
    sp = data.get(split_key)
    if sp is None:
        raise ValueError(f"{yaml_path} has no '{split_key}' split.")
    p = Path(sp[0] if isinstance(sp, (list, tuple)) else sp)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Split '{split_key}' not found: {p}")
    return p


def _read_test_image_paths(cam_yaml: str, man_yaml: str) -> Tuple[List[Path], List[Path], Path, Path]:
    """Infer test camera/manual dirs from YAML paths. Supports unseen splits via YAML 'test' key. Return (cam_paths, man_paths, yaml_dir, cam_dir)."""
    yaml_dir = Path(cam_yaml).resolve().parent
    cam_dir = yaml_dir / "images" / "test" / "camera"
    man_dir = yaml_dir / "images" / "test" / "manual"
    if not cam_dir.exists():
        cam_dir = yaml_dir / "test" / "camera"
        man_dir = yaml_dir / "test" / "manual"
    if not cam_dir.exists() and yaml is not None:
        try:
            cam_dir = _read_yaml_images_root(Path(cam_yaml), "test")
            man_dir = _read_yaml_images_root(Path(man_yaml), "test")
        except Exception:
            pass
    if not cam_dir.exists():
        raise FileNotFoundError(f"Test camera dir not found: {cam_dir}")
    exts = {".jpg", ".jpeg", ".png"}
    cam_paths = sorted([p for p in cam_dir.rglob("*") if p.suffix.lower() in exts])
    man_paths = []
    for cp in cam_paths:
        mp = _paired_manual_path(cp, cam_dir, "manual")
        if mp is None:
            mp = man_dir / cp.name
        man_paths.append(mp)
    return cam_paths, man_paths, yaml_dir, cam_dir


def run_inference_early_fusion(
    ckpt_path: str,
    config: Dict,
    test_cam_yaml: str,
    test_man_yaml: str,
    test_mask_root: Optional[str],
    out_dir: str,
    device: torch.device,
    save_masks: bool = True,
    save_visualizations: bool = False,
    img_size: int = 640,
    mask_suffix: str = "_rle.json",
) -> Dict[str, float]:
    from ultralytics import YOLO
    from baselines.model_early_fusion import patch_yolo_first_conv_6ch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state_dict") or ckpt.get("student", {})
    seg_state = ckpt.get("seg_adapter_state_dict") or ckpt.get("seg_adapter", {})
    cfg_resolved = ckpt.get("config_resolved", config)
    img_size = cfg_resolved.get("img_size", img_size)
    num_classes = cfg_resolved.get("num_classes", 1)
    student_model = _resolve_student_model_path(
        cfg_resolved.get("student_model") or config.get("student_model")
    )
    aux_head_tap = cfg_resolved.get("aux_head_tap", "model.23")

    cam_paths, man_paths, yaml_dir, cam_dir = _read_test_image_paths(test_cam_yaml, test_man_yaml)
    mask_root = Path(test_mask_root) if test_mask_root else (yaml_dir / "mask_labels" / "test" / "camera")
    if not mask_root.exists():
        mask_root = yaml_dir / "mask_labels" / "test" / "camera"

    _tmp = YOLO(student_model).model.to(device)
    s_aux = resolve_canonical_tap(_tmp, aux_head_tap, img_size, prefer=("proto",), device=device)
    del _tmp
    model = YOLO(student_model).model.to(device)
    patch_yolo_first_conv_6ch(model, init_from_rgb=True)
    _load_state_dict_matching_shapes(model, model_state, "Early fusion")
    model.eval()
    head_hook = FeatureHooker(model, [s_aux], use_regex=False)
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, img_size, img_size, device=device))
    Cs = int(head_hook.buffers[s_aux].shape[1])
    seg_adapter = AuxMaskHead(in_channels=Cs, num_classes=num_classes).to(device)
    seg_adapter.load_state_dict(seg_state, strict=True)
    seg_adapter.eval()
    head_hook.clear()

    os.makedirs(out_dir, exist_ok=True)
    pred_masks_dir = os.path.join(out_dir, "pred_masks")
    vis_dir = os.path.join(out_dir, "vis")
    if save_masks:
        os.makedirs(pred_masks_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)

    dice_list, iou_list = [], []
    prec_list, rec_list, bf_list = [], [], []
    gt_instances_by_img: Dict[int, List[np.ndarray]] = {}
    predictions: List[Dict] = []

    for img_id, (cam_path, man_path) in enumerate(zip(cam_paths, man_paths)):
        if not man_path.exists():
            continue
        try:
            rel = cam_path.relative_to(cam_dir)
        except Exception:
            rel = Path(cam_path.name)
        mjson = mask_root / rel.parent / (rel.stem + mask_suffix)
        if not mjson.exists():
            continue
        try:
            with open(mjson, "r") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
            merged = None
            for it in items:
                r = it.get("segmentation", it)
                if isinstance(r, dict) and "counts" in r and "size" in r and mask_utils:
                    rle = {"counts": r["counts"].encode("utf-8") if isinstance(r["counts"], str) else r["counts"], "size": [int(r["size"][0]), int(r["size"][1])]}
                    m = mask_utils.decode(rle)
                    if m.ndim == 3:
                        m = m[..., 0]
                    m = (m > 0).astype(np.uint8)
                    merged = m if merged is None else np.maximum(merged, m)
            if merged is None:
                continue
            H0, W0 = merged.shape
        except Exception:
            continue
        gt_instances = _decode_instances_from_json(str(mjson))
        gt_instances_by_img[img_id] = [(g > 0).astype(bool) for g in gt_instances]
        x_cam = _prep_image(str(cam_path), img_size)
        x_man = _prep_image(str(man_path), img_size)
        x_6ch = torch.cat([x_cam, x_man], dim=1).to(device)

        head_hook.clear()
        with torch.no_grad():
            _ = model(x_6ch)
        feat = head_hook.buffers[s_aux]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_logits = seg_adapter(feat, out_hw=(H0, W0))
        if num_classes == 1 or pred_logits.shape[1] == 1:
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        else:
            fg = pred_logits[:, 1:, :, :].amax(dim=1, keepdim=True)
            pred_bin = (torch.sigmoid(fg) > 0.5).float()
        pred_np = (pred_bin[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        d, j = _dice_iou(pred_bin, torch.from_numpy(merged.astype(np.float32)).view(1, 1, H0, W0).to(device))
        dice_list.append(d)
        iou_list.append(j)

        pm = _pix_metrics(pred_np.astype(bool), merged.astype(bool))
        prec_list.append(pm["Precision"])
        rec_list.append(pm["Recall"])
        bf_list.append(pm["BoundaryF"])
        if save_masks:
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(os.path.join(pred_masks_dir, f"{cam_path.stem}.png"))
        if save_visualizations:
            cam_img = np.array(Image.open(cam_path).convert("RGB"))
            overlay = cam_img.copy()
            pred_vis = _resize_bool_mask(pred_np, (cam_img.shape[0], cam_img.shape[1]))
            overlay[pred_vis] = [0, 255, 0]
            blend = (cam_img * 0.5 + overlay * 0.5).astype(np.uint8)
            Image.fromarray(blend).save(os.path.join(vis_dir, f"{cam_path.stem}.png"))
        try:
            import cv2
            num, labels = cv2.connectedComponents(pred_np, connectivity=8)
            for idx in range(1, num):
                comp = (labels == idx).astype(bool)
                if comp.sum() < 2:
                    continue
                comp_resized = _resize_bool_mask(comp.astype(np.uint8), (H0, W0))
                predictions.append({"img_id": img_id, "score": 0.9, "mask": comp_resized})
            if num <= 1:
                predictions.append({"img_id": img_id, "score": 0.0, "mask": np.zeros((H0, W0), dtype=bool)})
        except Exception:
            predictions.append({"img_id": img_id, "score": 0.0, "mask": np.zeros((H0, W0), dtype=bool)})

    head_hook.remove()
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec = float(np.mean(rec_list)) if rec_list else 0.0
    mean_bf = float(np.mean(bf_list)) if bf_list else 0.0
    ap50, map_5095 = _eval_map(predictions, gt_instances_by_img)

    metrics = {
        "dice": mean_dice,
        "iou": mean_iou,
        "precision": mean_prec,
        "recall": mean_rec,
        "boundaryF": mean_bf,
        "ap50": ap50,
        "map5095": map_5095,
    }
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    pred_manifest = {"ckpt_path": ckpt_path, "baseline_type": "early_fusion", "config": cfg_resolved, "split": "test"}
    with open(os.path.join(out_dir, "pred_manifest.json"), "w") as f:
        json.dump(pred_manifest, f, indent=2)
    return metrics


def run_inference_direct_fusion(
    ckpt_path: str,
    config: Dict,
    test_cam_yaml: str,
    test_man_yaml: str,
    test_mask_root: Optional[str],
    out_dir: str,
    device: torch.device,
    save_masks: bool = True,
    save_visualizations: bool = False,
) -> Dict[str, float]:
    from baselines.direct_fusion_baseline import DirectFusionBaseline
    cam_paths, man_paths, yaml_dir, cam_dir = _read_test_image_paths(test_cam_yaml, test_man_yaml)
    mask_root = Path(test_mask_root) if test_mask_root else (yaml_dir / "mask_labels" / "test" / "camera")
    if not mask_root.exists():
        mask_root = yaml_dir / "mask_labels" / "test" / "camera"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config_resolved", config)
    student_model_path = _resolve_student_model_path(cfg.get("student_model"))
    net = DirectFusionBaseline(
        student_model_path=student_model_path,
        img_size=cfg.get("img_size", 640),
        aux_head_tap=cfg.get("aux_head_tap", "model.23"),
        fusion_taps=cfg.get("fusion_taps"),
        fusion_dims=cfg.get("fusion_dims"),
        fusion_tokens=cfg.get("fusion_tokens"),
        num_classes=cfg.get("num_classes", 1),
        device=device,
    )
    if "camera_branch" in ckpt:
        _load_state_dict_matching_shapes(net.camera_branch, ckpt["camera_branch"], "Direct fusion camera_branch")
        _load_state_dict_matching_shapes(net.manual_branch, ckpt["manual_branch"], "Direct fusion manual_branch")
        net.fusions.load_state_dict(ckpt["fusions"], strict=False)
        net.seg_adapter.load_state_dict(ckpt["seg_adapter"], strict=True)
    else:
        net.load_state_dict(ckpt, strict=False)
    net.to(device).eval()
    os.makedirs(out_dir, exist_ok=True)
    pred_masks_dir = os.path.join(out_dir, "pred_masks")
    vis_dir = os.path.join(out_dir, "vis")
    if save_masks:
        os.makedirs(pred_masks_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    mask_suffix = cfg.get("mask_suffix", "_rle.json")
    num_classes = cfg.get("num_classes", 1)
    img_size = cfg.get("img_size", 640)
    amp = cfg.get("amp", True)
    dice_list, iou_list = [], []
    prec_list, rec_list, bf_list = [], [], []
    gt_instances_by_img: Dict[int, List] = {}
    predictions: List[Dict] = []
    for img_id, (cam_path, man_path) in enumerate(zip(cam_paths, man_paths)):
        if not man_path.exists():
            continue
        try:
            rel = cam_path.relative_to(cam_dir)
        except Exception:
            rel = Path(cam_path.name)
        mjson = mask_root / rel.parent / (rel.stem + mask_suffix)
        if not mjson.exists():
            continue
        try:
            with open(mjson, "r") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else (data.get("rles", [data]) if isinstance(data, dict) else [])
            merged = None
            for it in items:
                r = it.get("segmentation", it)
                if isinstance(r, dict) and "counts" in r and "size" in r and mask_utils:
                    rle = {"counts": r["counts"].encode("utf-8") if isinstance(r["counts"], str) else r["counts"], "size": [int(r["size"][0]), int(r["size"][1])]}
                    m = mask_utils.decode(rle)
                    if m.ndim == 3:
                        m = m[..., 0]
                    m = (m > 0).astype(np.uint8)
                    merged = m if merged is None else np.maximum(merged, m)
            if merged is None:
                continue
            H0, W0 = merged.shape
        except Exception:
            continue
        gt_instances = _decode_instances_from_json(str(mjson))
        gt_instances_by_img[img_id] = [(g > 0).astype(bool) for g in gt_instances]
        x_cam = _prep_image(str(cam_path), img_size).to(device)
        x_man = _prep_image(str(man_path), img_size).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred_logits = net(cam=x_cam, man=x_man, out_hw=(H0, W0))
        if num_classes == 1 or pred_logits.shape[1] == 1:
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        else:
            fg = pred_logits[:, 1:, :, :].amax(dim=1, keepdim=True)
            pred_bin = (torch.sigmoid(fg) > 0.5).float()
        pred_np = (pred_bin[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        d, j = _dice_iou(pred_bin, torch.from_numpy(merged.astype(np.float32)).view(1, 1, H0, W0).to(device))
        dice_list.append(d)
        iou_list.append(j)

        pm = _pix_metrics(pred_np.astype(bool), merged.astype(bool))
        prec_list.append(pm["Precision"])
        rec_list.append(pm["Recall"])
        bf_list.append(pm["BoundaryF"])
        if save_masks:
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(os.path.join(pred_masks_dir, f"{cam_path.stem}.png"))
        if save_visualizations:
            cam_img = np.array(Image.open(cam_path).convert("RGB"))
            overlay = cam_img.copy()
            pred_vis = _resize_bool_mask(pred_np, (cam_img.shape[0], cam_img.shape[1]))
            overlay[pred_vis] = [0, 255, 0]
            blend = (cam_img * 0.5 + overlay * 0.5).astype(np.uint8)
            Image.fromarray(blend).save(os.path.join(vis_dir, f"{cam_path.stem}.png"))
        try:
            import cv2
            num, labels = cv2.connectedComponents(pred_np, connectivity=8)
            for idx in range(1, num):
                comp = (labels == idx).astype(bool)
                if comp.sum() < 2:
                    continue
                comp_resized = _resize_bool_mask(comp.astype(np.uint8), (H0, W0))
                predictions.append({"img_id": img_id, "score": 0.9, "mask": comp_resized})
            if num <= 1:
                predictions.append({"img_id": img_id, "score": 0.0, "mask": np.zeros((H0, W0), dtype=bool)})
        except Exception:
            predictions.append({"img_id": img_id, "score": 0.0, "mask": np.zeros((H0, W0), dtype=bool)})
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec = float(np.mean(rec_list)) if rec_list else 0.0
    mean_bf = float(np.mean(bf_list)) if bf_list else 0.0
    ap50, map_5095 = _eval_map(predictions, gt_instances_by_img)
    metrics = {
        "dice": mean_dice,
        "iou": mean_iou,
        "precision": mean_prec,
        "recall": mean_rec,
        "boundaryF": mean_bf,
        "ap50": ap50,
        "map5095": map_5095,
    }
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    with open(os.path.join(out_dir, "pred_manifest.json"), "w") as f:
        json.dump({"ckpt_path": ckpt_path, "baseline_type": "direct_fusion_e2e", "config": cfg, "split": "test"}, f, indent=2)
    return metrics


def run_inference_sota_mask2former(
    ckpt_path: str,
    config: Dict,
    test_cam_yaml: str,
    test_man_yaml: str,
    test_mask_root: Optional[str],
    out_dir: str,
    device: torch.device,
    save_masks: bool = True,
    save_visualizations: bool = False,
) -> Dict[str, float]:
    """Run SOTA (Mask2Former/detectron2) inference. Requires detectron2; run_dir must contain model_best.pth and a detectron2-format config (e.g. from model zoo)."""
    raise NotImplementedError(
        "SOTA Mask2Former inference: install detectron2, train/export Mask2Former, then run the framework's evaluator. "
        "To report pipeline metrics (Dice/IoU/AP50/mAP50-95), convert framework predictions to pred_masks/ and run "
        "evaluation using the same GT masks; write results to out_dir/metrics.csv (metric,value format)."
    )


def main():
    parser = argparse.ArgumentParser(description="Unified baseline inference")
    parser.add_argument("--baseline_type", required=True, choices=["early_fusion", "direct_fusion_e2e", "sota_mask2former"])
    parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_best.pt or ckpt_last.pt")
    parser.add_argument("--test_cam_yaml", default="", help="Test camera split YAML")
    parser.add_argument("--test_man_yaml", default="", help="Test manual split YAML (required for manual-aware)")
    parser.add_argument("--test_mask_root", default="", help="Test mask root (optional)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--student_model", default="", help="Override base YOLO weights path (e.g. local best.pt) when checkpoint was trained elsewhere")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--save_polygons", action="store_true")
    parser.add_argument("--save_logits", action="store_false")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config_resolved", {}))
    if getattr(args, "student_model", None) and args.student_model.strip():
        config["student_model"] = args.student_model.strip()

    if args.baseline_type == "early_fusion":
        if not args.test_cam_yaml or not args.test_man_yaml:
            raise SystemExit("early_fusion requires --test_cam_yaml and --test_man_yaml")
        metrics = run_inference_early_fusion(
            args.ckpt_path,
            config,
            args.test_cam_yaml,
            args.test_man_yaml,
            args.test_mask_root or None,
            args.out_dir,
            device,
            save_masks=args.save_masks,
            save_visualizations=args.save_visualizations,
        )
    elif args.baseline_type == "direct_fusion_e2e":
        if not args.test_cam_yaml or not args.test_man_yaml:
            raise SystemExit("direct_fusion_e2e requires --test_cam_yaml and --test_man_yaml")
        metrics = run_inference_direct_fusion(
            args.ckpt_path, config, args.test_cam_yaml, args.test_man_yaml,
            args.test_mask_root or None, args.out_dir, device,
            save_masks=args.save_masks, save_visualizations=args.save_visualizations,
        )
    else:
        metrics = run_inference_sota_mask2former(
            args.ckpt_path, config, args.test_cam_yaml, args.test_man_yaml,
            args.test_mask_root or None, args.out_dir, device,
            save_masks=args.save_masks, save_visualizations=args.save_visualizations,
        )

    print("Metrics:", metrics)
    print(f"Written to {args.out_dir}/metrics.csv and pred_manifest.json")


if __name__ == "__main__":
    main()
