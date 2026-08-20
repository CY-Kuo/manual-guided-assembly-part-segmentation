"""YOLO-box-to-SAM baseline without experiment-specific paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use YOLO boxes as SAM prompts")
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_h", choices=("vit_h", "vit_l", "vit_b"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--box-pad", type=int, default=2)
    return parser


def run_yolo_box_sam(*, yolo_model_path: str, image_path: str, sam_checkpoint: str,
                     out_path: str, sam_model_type: str = "vit_h", conf: float = 0.20,
                     device: str = "0", box_pad: int = 2) -> str:
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from ultralytics import YOLO

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = rgb.shape[:2]

    yolo = YOLO(yolo_model_path)
    prediction = yolo.predict(
        source=image_path,
        imgsz=640,
        conf=conf,
        device=device,
        verbose=False,
        half=(str(device).lower() != "cpu"),
    )[0]
    boxes = (
        prediction.boxes.xyxy.cpu().numpy()
        if getattr(prediction, "boxes", None) is not None
        else np.zeros((0, 4), dtype=np.float32)
    )

    torch_device = torch.device(
        f"cuda:{device}" if str(device).lower() != "cpu" and torch.cuda.is_available() else "cpu"
    )
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(torch_device)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    union = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
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
    run_yolo_box_sam(
        yolo_model_path=args.yolo_model,
        image_path=args.image,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        out_path=args.out,
        conf=args.conf,
        device=args.device,
        box_pad=args.box_pad,
    )


if __name__ == "__main__":
    main()
