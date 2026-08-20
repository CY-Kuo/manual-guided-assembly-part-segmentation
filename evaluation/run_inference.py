"""Run the final MAPS teacher-guided segmentation pipeline on one image pair."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch

from maps.aux_heads import AuxMaskHead
from maps.feature_hooks import FeatureHooker
from maps.fusion import StructureFusion
from maps.smart_prior import refine_aux_prior, filter_student_instances_with_prior

IMG_SIZE = 640
PRIOR_NEAR_THR = 0.2
PRIOR_FAR_THR = 0.9
PRIOR_NEAR_PX = 35
PRIOR_FAR_CUT_PX = 60
PRIOR_MIN_AREA = 300
PRIOR_ERODE_IT = 0
PRIOR_OPEN_IT = 1
PRIOR_CLOSE_IT = 0
CLIP_TO_AUX = True
COVER_THR = 0.20
RESCORE_ALPHA = 0.35
ADD_TEACHER_MISSING = True
PROMOTE_MIN_AREA = 1400
PROMOTE_MAX_DIST_PX = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MAPS inference for one camera image and aligned manual image"
    )
    parser.add_argument("--ckpt", required=True, help="MAPS training checkpoint")
    parser.add_argument("--student", required=True, help="Camera-view YOLO11-seg weights")
    parser.add_argument("--teacher", required=True, help="Manual-view YOLO11-seg weights")
    parser.add_argument("--camera", required=True, help="Camera image")
    parser.add_argument("--manual", required=True, help="Aligned manual image")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--yolo-conf", type=float, default=0.20)
    parser.add_argument("--aux-thr", type=float, default=0.45)
    parser.add_argument("--save-intermediate", action="store_true")
    return parser


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required: pip install ultralytics") from exc
    return YOLO


def _torch_device(device: str) -> torch.device:
    if str(device).lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return torch.device(f"cuda:{int(device)}")


def _first_4d_tensor(value):
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_4d_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_4d_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _pad_in(weight: torch.Tensor, in_dim: int) -> torch.Tensor:
    channels = weight.shape[1]
    if channels == in_dim:
        return weight
    if channels > in_dim:
        return weight[:, :in_dim, ...]
    zeros = weight.new_zeros(weight.shape[0], in_dim - channels, *weight.shape[2:])
    return torch.cat([weight, zeros], dim=1)


def _pad_out(weight: torch.Tensor, out_dim: int) -> torch.Tensor:
    channels = weight.shape[0]
    if channels == out_dim:
        return weight
    if channels > out_dim:
        return weight[:out_dim, ...]
    zeros = weight.new_zeros(out_dim - channels, *weight.shape[1:])
    return torch.cat([weight, zeros], dim=0)


def _pad_bias(bias: torch.Tensor, out_dim: int) -> torch.Tensor:
    channels = bias.shape[0]
    if channels == out_dim:
        return bias
    if channels > out_dim:
        return bias[:out_dim]
    return torch.cat([bias, bias.new_zeros(out_dim - channels)], dim=0)


def _extract_substate(state_dict: dict, safe_key: str) -> dict:
    prefix = safe_key + "."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_fusion_with_channel_compat(
    fusion: StructureFusion, fusions_state: dict, safe_key: str
) -> None:
    sub = _extract_substate(fusions_state, safe_key)
    if not sub:
        return

    q_weight = fusion.q_proj.weight
    teacher_weight = fusion.t_token[0].weight
    out_weight = fusion.out_proj.weight
    film_weight = fusion.film[0].weight if hasattr(fusion, "film") else None

    if "t_token.0.weight" in sub:
        fusion.t_token[0].weight.data.copy_(_pad_in(sub["t_token.0.weight"], teacher_weight.shape[1]))
    if "t_token.0.bias" in sub:
        fusion.t_token[0].bias.data.copy_(sub["t_token.0.bias"])
    if "q_proj.weight" in sub:
        fusion.q_proj.weight.data.copy_(_pad_in(sub["q_proj.weight"], q_weight.shape[1]))
    if "q_proj.bias" in sub:
        fusion.q_proj.bias.data.copy_(sub["q_proj.bias"])
    if "out_proj.weight" in sub:
        fusion.out_proj.weight.data.copy_(_pad_out(sub["out_proj.weight"], out_weight.shape[0]))
    if "out_proj.bias" in sub:
        fusion.out_proj.bias.data.copy_(_pad_bias(sub["out_proj.bias"], out_weight.shape[0]))

    if film_weight is not None:
        if "film.0.weight" in sub:
            weight = _pad_in(sub["film.0.weight"], film_weight.shape[1])
            weight = _pad_out(weight, film_weight.shape[0])
            fusion.film[0].weight.data.copy_(weight)
        if "film.0.bias" in sub:
            fusion.film[0].bias.data.copy_(_pad_bias(sub["film.0.bias"], film_weight.shape[0]))


def _load_seg_head_with_channel_compat(seg_head: AuxMaskHead, state: dict) -> None:
    if not state:
        return
    first_weight = state.get("block.0.weight")
    first_bias = state.get("block.0.bias")
    if first_weight is not None:
        seg_head.block[0].weight.data.copy_(_pad_in(first_weight, seg_head.block[0].weight.shape[1]))
    if first_bias is not None:
        seg_head.block[0].bias.data.copy_(first_bias)

    for name, parameter in seg_head.named_parameters():
        if name in {"block.0.weight", "block.0.bias"}:
            continue
        if name in state and state[name].shape == parameter.shape:
            parameter.data.copy_(state[name])
    for name, buffer in seg_head.named_buffers():
        if name in state and state[name].shape == buffer.shape:
            buffer.copy_(state[name])


def _prep_image(path: Path, size: int = IMG_SIZE) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _resize_bool(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    height, width = out_hw
    if mask.shape == (height, width):
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.NEAREST)) > 127


def build_aux_and_fusion(
    student_model_path: str,
    teacher_model_path: str,
    ckpt_path: str,
    device: str,
) -> Dict:
    YOLO = _require_ultralytics()
    torch_device = _torch_device(device)

    yolo_student = YOLO(student_model_path)
    yolo_teacher = YOLO(teacher_model_path)
    yolo_student.model.to(torch_device).eval()
    yolo_teacher.model.to(torch_device).eval()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    student_state = checkpoint.get("student", {})
    seg_state = checkpoint.get("seg_adapter", {}) or {}
    fusions_state = checkpoint.get("fusions", {}) or {}
    fusion_meta = checkpoint.get("fusion_meta", {}) or {}
    name_map = fusion_meta.get("name_map", {}) or {}
    dims = fusion_meta.get("dims", {}) or {}
    tokens = fusion_meta.get("tokens", {}) or {}

    yolo_student.model.load_state_dict(student_state, strict=False)

    candidates = [
        (key, value)
        for key, value in name_map.items()
        if isinstance(value, dict) and str(value.get("anchor", "")).startswith("model.23")
    ]
    if not candidates:
        raise RuntimeError("Checkpoint fusion metadata does not contain an auxiliary model.23 tap")

    def film_out_channels(key: str):
        sub = _extract_substate(fusions_state, key)
        weight = sub.get("film.0.weight")
        return None if weight is None else int(weight.shape[0])

    selected = None
    for key, entry in candidates:
        student_hook = FeatureHooker(yolo_student.model, [entry["student"]])
        teacher_hook = FeatureHooker(yolo_teacher.model, [entry["teacher"]])
        with torch.no_grad():
            yolo_student.model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=torch_device))
            yolo_teacher.model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=torch_device))
        student_sample = _first_4d_tensor(student_hook.buffers.get(entry["student"]))
        teacher_sample = _first_4d_tensor(teacher_hook.buffers.get(entry["teacher"]))
        student_hook.remove()
        teacher_hook.remove()
        if student_sample is None or teacher_sample is None:
            continue
        student_channels = int(student_sample.shape[1])
        teacher_channels = int(teacher_sample.shape[1])
        if film_out_channels(key) == 2 * student_channels:
            selected = (key, entry, student_channels, teacher_channels)
            break

    if selected is None:
        key, entry = candidates[0]
        student_hook = FeatureHooker(yolo_student.model, [entry["student"]])
        teacher_hook = FeatureHooker(yolo_teacher.model, [entry["teacher"]])
        with torch.no_grad():
            yolo_student.model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=torch_device))
            yolo_teacher.model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=torch_device))
        student_sample = _first_4d_tensor(student_hook.buffers.get(entry["student"]))
        teacher_sample = _first_4d_tensor(teacher_hook.buffers.get(entry["teacher"]))
        student_hook.remove()
        teacher_hook.remove()
        if student_sample is None or teacher_sample is None:
            raise RuntimeError("Resolved auxiliary taps did not produce 4D tensors")
        selected = (key, entry, int(student_sample.shape[1]), int(teacher_sample.shape[1]))

    aux_key, aux_entry, student_channels, teacher_channels = selected
    aux_anchor = aux_entry["anchor"]
    fusion = StructureFusion(
        student_channels,
        teacher_channels,
        d=int(dims.get(aux_anchor, 128)),
        num_tokens=int(tokens.get(aux_anchor, 4)),
        use_film=True,
    ).to(torch_device).eval()
    _load_fusion_with_channel_compat(fusion, fusions_state, aux_key)

    seg_head = AuxMaskHead(in_channels=student_channels, num_classes=1).to(torch_device).eval()
    _load_seg_head_with_channel_compat(seg_head, seg_state)

    return {
        "yolo_student": yolo_student,
        "yolo_teacher": yolo_teacher,
        "student_hook": FeatureHooker(yolo_student.model, [aux_entry["student"]]),
        "teacher_hook": FeatureHooker(yolo_teacher.model, [aux_entry["teacher"]]),
        "student_tap": aux_entry["student"],
        "teacher_tap": aux_entry["teacher"],
        "fusion": fusion,
        "seg_head": seg_head,
        "device": torch_device,
    }


def aux_logits_and_prob(
    bundle: Dict, camera_path: Path, manual_path: Path, out_hw: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    student_hook = bundle["student_hook"]
    teacher_hook = bundle["teacher_hook"]
    device = bundle["device"]

    camera = _prep_image(camera_path).to(device)
    manual = _prep_image(manual_path).to(device)
    student_hook.clear()
    teacher_hook.clear()

    with torch.no_grad():
        bundle["yolo_student"].model(camera)
        bundle["yolo_teacher"].model(manual)
        student_feature = _first_4d_tensor(student_hook.buffers.get(bundle["student_tap"]))
        teacher_feature = _first_4d_tensor(teacher_hook.buffers.get(bundle["teacher_tap"]))
        if student_feature is None or teacher_feature is None:
            raise RuntimeError("MAPS feature hooks did not return the expected feature maps")
        fused = bundle["fusion"](student_feature, teacher_feature)
        logits = bundle["seg_head"](fused, out_hw)
        probability = torch.sigmoid(logits)

    return (
        logits[0, 0].cpu().numpy().astype(np.float32),
        probability[0, 0].cpu().numpy().astype(np.float32),
    )


def _collect_instances(yolo_model, image_path: Path, confidence: float, out_hw):
    model_device = next(yolo_model.model.parameters()).device
    device_arg = model_device.index if model_device.type == "cuda" else "cpu"
    result = yolo_model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=confidence,
        iou=0.6,
        max_det=100,
        device=device_arg,
        verbose=False,
        half=(device_arg != "cpu"),
    )[0]

    height, width = out_hw
    masks, scores = [], []
    union = np.zeros((height, width), dtype=bool)
    if getattr(result, "masks", None) is not None:
        predicted = result.masks.data.cpu().numpy()
        confidence_scores = (
            result.boxes.conf.cpu().numpy()
            if getattr(result, "boxes", None) is not None
            else np.ones(len(predicted), dtype=np.float32)
        )
        for index in range(len(predicted)):
            mask = _resize_bool(predicted[index] > 0.5, out_hw)
            masks.append(mask)
            scores.append(float(confidence_scores[index]))
            union |= mask
    return union, masks, scores


def _promote_teacher_components(aux_prior: np.ndarray, student_union: np.ndarray) -> np.ndarray:
    if not ADD_TEACHER_MISSING:
        return np.zeros_like(aux_prior, dtype=bool)
    try:
        from scipy.ndimage import distance_transform_edt, label
    except ImportError:
        return np.zeros_like(aux_prior, dtype=bool)

    near_student = distance_transform_edt(~student_union) <= PROMOTE_MAX_DIST_PX
    labels, count = label(aux_prior.astype(np.uint8))
    promoted = np.zeros_like(aux_prior, dtype=bool)
    for index in range(1, count + 1):
        component = labels == index
        if component.sum() < PROMOTE_MIN_AREA:
            continue
        if (component & near_student).any():
            promoted |= component
    return promoted


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def run_inference(
    *,
    ckpt_path: str,
    student_model_path: str,
    teacher_model_path: str,
    camera_path: str,
    manual_path: str,
    out_dir: str,
    device: str = "0",
    yolo_conf: float = 0.20,
    aux_thr: float = 0.45,
    save_intermediate: bool = False,
) -> dict:
    camera = Path(camera_path).expanduser().resolve()
    manual = Path(manual_path).expanduser().resolve()
    for path in [Path(ckpt_path), Path(student_model_path), Path(teacher_model_path), camera, manual]:
        if not path.exists():
            raise FileNotFoundError(path)

    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    original = Image.open(camera)
    out_hw = (original.height, original.width)
    bundle = build_aux_and_fusion(student_model_path, teacher_model_path, ckpt_path, device)

    student_union, student_masks, student_scores = _collect_instances(
        bundle["yolo_student"], camera, yolo_conf, out_hw
    )
    logits, probability = aux_logits_and_prob(bundle, camera, manual, out_hw)
    aux_binary = probability > float(aux_thr)

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
    _, refined_union = filter_student_instances_with_prior(
        student_masks,
        np.asarray(student_scores, dtype=np.float32),
        aux_prior.astype(bool),
        clip_to_prior=CLIP_TO_AUX,
        cover_thr=COVER_THR,
        rescore_alpha=RESCORE_ALPHA,
    )
    refined_union |= _promote_teacher_components(aux_prior, student_union)

    final_path = output / "maps_mask.png"
    _save_mask(final_path, refined_union)
    outputs = {"refined_mask": str(final_path)}

    if save_intermediate:
        camera_mask_path = output / "camera_checkpoint_mask.png"
        aux_mask_path = output / "auxiliary_mask.png"
        prior_path = output / "refined_prior.png"
        _save_mask(camera_mask_path, student_union)
        _save_mask(aux_mask_path, aux_binary)
        _save_mask(prior_path, aux_prior)
        outputs.update(
            camera_checkpoint_mask=str(camera_mask_path),
            auxiliary_mask=str(aux_mask_path),
            refined_prior=str(prior_path),
        )

    return {
        "outputs": outputs,
        "thresholds": {"yolo_conf": float(yolo_conf), "aux_thr": float(aux_thr)},
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(
        ckpt_path=args.ckpt,
        student_model_path=args.student,
        teacher_model_path=args.teacher,
        camera_path=args.camera,
        manual_path=args.manual,
        out_dir=args.out_dir,
        device=args.device,
        yolo_conf=args.yolo_conf,
        aux_thr=args.aux_thr,
        save_intermediate=args.save_intermediate,
    )
    print(result["outputs"]["refined_mask"])


if __name__ == "__main__":
    main()
