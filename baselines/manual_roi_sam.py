"""Manual-ROI-to-SAM baseline without experiment-specific paths or bookkeeping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment a camera image from manual-derived ROI boxes using SAM")
    parser.add_argument("--image", required=True)
    parser.add_argument("--roi-json", required=True, help="JSON containing one or more xyxy boxes")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_h", choices=("vit_h", "vit_l", "vit_b"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--box-pad", type=int, default=2)
    return parser


def load_roi_boxes(path: str) -> list[list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data if isinstance(data, list) else [data]
    boxes = []
    for item in items:
        if isinstance(item, dict) and item.get("type", "box") == "box" and len(item.get("box", [])) == 4:
            boxes.append([float(value) for value in item["box"]])
    return boxes


def run_manual_roi_sam(*, image_path: str, roi_json: str, sam_checkpoint: str,
                       out_path: str, sam_model_type: str = "vit_h", device: str = "0",
                       box_pad: int = 2) -> str:
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = rgb.shape[:2]
    torch_device = torch.device(
        f"cuda:{device}" if str(device).lower() != "cpu" and torch.cuda.is_available() else "cpu"
    )
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(torch_device)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    union = np.zeros((height, width), dtype=bool)
    for box in load_roi_boxes(roi_json):
        x1, y1, x2, y2 = box
        prompt = np.array([
            max(0, x1 - box_pad), max(0, y1 - box_pad),
            min(width - 1, x2 + box_pad), min(height - 1, y2 + box_pad),
        ], dtype=np.float32)
        masks, _, _ = predictor.predict(box=prompt[None, :], multimask_output=False)
        union |= masks[0].astype(bool)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8) * 255).save(output)
    return str(output)


def main() -> None:
    args = build_parser().parse_args()
    run_manual_roi_sam(
        image_path=args.image,
        roi_json=args.roi_json,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        out_path=args.out,
        device=args.device,
        box_pad=args.box_pad,
    )


if __name__ == "__main__":
    main()
