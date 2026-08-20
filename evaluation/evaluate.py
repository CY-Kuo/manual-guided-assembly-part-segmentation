"""Dataset-level evaluation for the final MAPS inference path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from evaluation.run_inference import (
    CLIP_TO_AUX,
    COVER_THR,
    IMG_SIZE,
    PRIOR_CLOSE_IT,
    PRIOR_ERODE_IT,
    PRIOR_FAR_CUT_PX,
    PRIOR_FAR_THR,
    PRIOR_MIN_AREA,
    PRIOR_NEAR_PX,
    PRIOR_NEAR_THR,
    PRIOR_OPEN_IT,
    RESCORE_ALPHA,
    _collect_instances,
    _promote_teacher_components,
    _save_mask,
    aux_logits_and_prob,
    build_aux_and_fusion,
)
from maps.smart_prior import refine_aux_prior, filter_student_instances_with_prior

BOUND_TOL_PX = 2
MASK_SUFFIX = "_rle.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one final MAPS checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--test-cam-yaml", required=True)
    parser.add_argument("--test-man-yaml", required=True)
    parser.add_argument("--test-mask-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--yolo-conf", type=float, default=0.20)
    parser.add_argument("--aux-thr", type=float, default=0.45)
    parser.add_argument("--save-masks", action="store_true")
    return parser


def pixel_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray, bound_tol: int = BOUND_TOL_PX) -> Dict[str, float]:
    pred = pred_bool.astype(bool)
    gt = gt_bool.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    coverage = pred.mean() * 100.0

    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        structure = np.ones((3, 3), dtype=bool)
        pred_edge = pred ^ binary_erosion(pred, structure=structure, border_value=0)
        gt_edge = gt ^ binary_erosion(gt, structure=structure, border_value=0)
        gt_distance = distance_transform_edt(~gt_edge)
        pred_distance = distance_transform_edt(~pred_edge)
        pred_match = pred_edge & (gt_distance <= bound_tol)
        gt_match = gt_edge & (pred_distance <= bound_tol)
        boundary_precision = pred_match.sum() / max(1, pred_edge.sum())
        boundary_recall = gt_match.sum() / max(1, gt_edge.sum())
        boundary_f = (
            0.0
            if boundary_precision + boundary_recall == 0
            else 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        )
    except Exception:
        boundary_f = 0.0

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "BoundaryF": float(boundary_f),
        "Coverage": float(coverage),
    }


def _decode_instances_from_json(json_path: Path) -> List[np.ndarray]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required for evaluation") from exc

    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data if isinstance(data, list) else data.get("rles", [data]) if isinstance(data, dict) else []
    masks: List[np.ndarray] = []
    for item in items:
        rle = item.get("segmentation", item) if isinstance(item, dict) else None
        if not isinstance(rle, dict) or "counts" not in rle or "size" not in rle:
            continue
        counts = rle["counts"]
        packed = {
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
            "size": [int(rle["size"][0]), int(rle["size"][1])],
        }
        mask = mask_utils.decode(packed)
        if mask.ndim == 3:
            mask = mask[..., 0]
        masks.append(mask > 0)
    return masks


def _merge_binary(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required")
    merged = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        merged |= mask.astype(bool)
    return merged


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return 0.0 if union == 0 else float(intersection) / float(union)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_grid = np.linspace(0, 1, 101)
    precision_envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    values = []
    for recall_value in recall_grid:
        indices = np.where(recalls >= recall_value)[0]
        values.append(precision_envelope[indices[0]] if indices.size else 0.0)
    return float(np.mean(values))


def evaluate_map(
    predictions: List[Dict],
    ground_truth_by_image: Dict[int, List[np.ndarray]],
    iou_thresholds=np.arange(0.5, 0.96, 0.05),
) -> Tuple[float, float]:
    predictions = sorted(predictions, key=lambda item: -item["score"])
    total_ground_truth = sum(len(items) for items in ground_truth_by_image.values())
    average_precisions = {}
    for threshold in iou_thresholds:
        true_positive = np.zeros(len(predictions), dtype=np.float32)
        false_positive = np.zeros(len(predictions), dtype=np.float32)
        matched = {
            image_id: np.zeros(len(items), dtype=bool)
            for image_id, items in ground_truth_by_image.items()
        }
        for index, prediction in enumerate(predictions):
            image_id = prediction["img_id"]
            ground_truth = ground_truth_by_image.get(image_id, [])
            if not ground_truth:
                false_positive[index] = 1.0
                continue
            ious = np.asarray(
                [
                    _mask_iou(prediction["mask"], mask) if not matched[image_id][gt_index] else -1.0
                    for gt_index, mask in enumerate(ground_truth)
                ],
                dtype=np.float32,
            )
            best_index = int(ious.argmax()) if ious.size else -1
            best_iou = float(ious[best_index]) if best_index >= 0 else 0.0
            if best_index >= 0 and best_iou >= threshold:
                true_positive[index] = 1.0
                matched[image_id][best_index] = True
            else:
                false_positive[index] = 1.0
        tp_cumulative = np.cumsum(true_positive)
        fp_cumulative = np.cumsum(false_positive)
        recalls = tp_cumulative / max(1, total_ground_truth)
        precisions = tp_cumulative / np.maximum(1, tp_cumulative + fp_cumulative)
        average_precisions[float(threshold)] = _compute_ap(recalls, precisions)
    ap50 = average_precisions.get(0.5, 0.0)
    map50_95 = float(np.mean(list(average_precisions.values()))) if average_precisions else 0.0
    return float(ap50), map50_95


def _read_images_root(yaml_path: Path, split: str) -> Path:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    split_path = config.get(split)
    if split_path is None:
        raise ValueError(f"{yaml_path} has no '{split}' split")
    if isinstance(split_path, (list, tuple)):
        split_path = split_path[0]
    result = Path(split_path)
    if not result.is_absolute():
        result = (root / result).resolve()
    return result


def _components(mask: np.ndarray) -> List[np.ndarray]:
    try:
        from scipy.ndimage import label

        labels, count = label(mask.astype(np.uint8))
        return [(labels == index) for index in range(1, count + 1) if (labels == index).any()]
    except Exception:
        return [mask.astype(bool)] if mask.any() else []


def evaluate_dataset(
    *,
    ckpt_path: str,
    student_model_path: str,
    teacher_model_path: str,
    camera_yaml: str,
    manual_yaml: str,
    mask_root: str,
    out_dir: str,
    device: str = "0",
    yolo_conf: float = 0.20,
    aux_thr: float = 0.45,
    save_masks: bool = False,
) -> Dict:
    camera_root = _read_images_root(Path(camera_yaml).resolve(), "test")
    manual_root = _read_images_root(Path(manual_yaml).resolve(), "test")
    gt_root = Path(mask_root).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = build_aux_and_fusion(student_model_path, teacher_model_path, ckpt_path, device)
    image_paths = sorted(
        path for path in camera_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No test images found under {camera_root}")

    rows = []
    predictions: List[Dict] = []
    ground_truth: Dict[int, List[np.ndarray]] = {}
    metric_rows = []

    for image_id, camera_path in enumerate(image_paths):
        relative = camera_path.relative_to(camera_root)
        manual_path = manual_root / relative
        if not manual_path.exists():
            alternate = manual_root / camera_path.name
            if not alternate.exists():
                continue
            manual_path = alternate

        gt_path = gt_root / relative.parent / f"{camera_path.stem}{MASK_SUFFIX}"
        gt_instances = _decode_instances_from_json(gt_path)
        if not gt_instances:
            continue
        ground_truth[image_id] = [mask.astype(bool) for mask in gt_instances]
        gt_union = _merge_binary(gt_instances)
        out_hw = gt_union.shape

        student_union, student_masks, student_scores = _collect_instances(
            bundle["yolo_student"], camera_path, yolo_conf, out_hw
        )
        logits, probability = aux_logits_and_prob(bundle, camera_path, manual_path, out_hw)
        aux_prior, _ = refine_aux_prior(
            logits,
            student_union,
            logits_are_probs=False,
            near_thr=PRIOR_NEAR_THR,
            far_thr=PRIOR_FAR_THR,
            max_near_px=PRIOR_NEAR_PX,
            far_cut_px=PRIOR_FAR_CUT_PX,
            keep_only_teacher_minus_student=False,
            min_comp_area=PRIOR_MIN_AREA,
            light_erosion_iters=PRIOR_ERODE_IT,
            open_iters=PRIOR_OPEN_IT,
            close_iters=PRIOR_CLOSE_IT,
        )
        filtered, refined_union = filter_student_instances_with_prior(
            student_masks,
            np.asarray(student_scores, dtype=np.float32),
            aux_prior.astype(bool),
            clip_to_prior=CLIP_TO_AUX,
            cover_thr=COVER_THR,
            rescore_alpha=RESCORE_ALPHA,
        )
        refined_union |= _promote_teacher_components(aux_prior, student_union)
        metrics = pixel_metrics(refined_union, gt_union)
        metric_rows.append(metrics)
        rows.append({"image": str(relative), **metrics})

        for prediction in filtered:
            predictions.append(
                {
                    "img_id": image_id,
                    "score": float(prediction.get("score", 1.0)),
                    "mask": prediction["mask"].astype(bool),
                }
            )
        promoted_only = refined_union.copy()
        for prediction in filtered:
            promoted_only &= ~prediction["mask"].astype(bool)
        for component in _components(promoted_only):
            predictions.append({"img_id": image_id, "score": 1.0, "mask": component})

        if save_masks:
            _save_mask(output_root / "pred_masks" / relative.parent / f"{camera_path.stem}.png", refined_union)

    if not metric_rows:
        raise RuntimeError("No test samples with valid ground-truth masks were evaluated")

    summary = {
        key: float(np.mean([metrics[key] for metrics in metric_rows]))
        for key in metric_rows[0]
    }
    ap50, map50_95 = evaluate_map(predictions, ground_truth)

    per_image_csv = output_root / "per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "Dice", "IoU", "Precision", "Recall", "BoundaryF", "Coverage"])
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = output_root / "summary_pixel.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dice", "IoU", "Precision", "Recall", "BoundaryF", "Coverage"])
        writer.writeheader()
        writer.writerow(summary)

    ap_csv = output_root / "summary_ap.csv"
    with ap_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AP50", "mAP50_95"])
        writer.writeheader()
        writer.writerow({"AP50": ap50, "mAP50_95": map50_95})

    return {
        "pixel": summary,
        "AP50": ap50,
        "mAP50_95": map50_95,
        "per_image_csv": str(per_image_csv),
        "summary_pixel_csv": str(summary_csv),
        "summary_ap_csv": str(ap_csv),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_dataset(
        ckpt_path=args.ckpt,
        student_model_path=args.student,
        teacher_model_path=args.teacher,
        camera_yaml=args.test_cam_yaml,
        manual_yaml=args.test_man_yaml,
        mask_root=args.test_mask_root,
        out_dir=args.out_dir,
        device=args.device,
        yolo_conf=args.yolo_conf,
        aux_thr=args.aux_thr,
        save_masks=args.save_masks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
