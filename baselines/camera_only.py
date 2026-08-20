"""Camera-only YOLO11-seg baseline used as the primary MAPS comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run camera-only YOLO segmentation")
    parser.add_argument("--model", required=True, help="YOLO11-seg checkpoint")
    parser.add_argument("--image", required=True, help="Camera image")
    parser.add_argument("--out", required=True, help="Output binary-mask PNG")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser


def run_camera_only(*, model_path: str, image_path: str, out_path: str,
                    conf: float = 0.20, img_size: int = 640, device: str = "0") -> str:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.predict(
        source=image_path,
        imgsz=img_size,
        conf=conf,
        iou=0.6,
        max_det=100,
        device=device,
        verbose=False,
        half=(str(device).lower() != "cpu"),
    )[0]

    with Image.open(image_path) as image:
        width, height = image.size
    union = np.zeros((height, width), dtype=bool)

    if getattr(result, "masks", None) is not None:
        for mask in result.masks.data.cpu().numpy():
            resized = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
            union |= np.asarray(resized) > 127

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8) * 255).save(output)
    return str(output)


def main() -> None:
    args = build_parser().parse_args()
    run_camera_only(
        model_path=args.model,
        image_path=args.image,
        out_path=args.out,
        conf=args.conf,
        img_size=args.img_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
