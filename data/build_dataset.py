"""Build the paired camera/manual dataset used by MAPS.

This wrapper keeps dataset paths outside the source code. The upstream
IKEA-Manuals-at-Work / IKEAVideo package must be installed separately; see
THIRD_PARTY_DATA.md.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from data.yolo_dataset_builder import build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the MAPS paired camera/manual YOLO-seg dataset"
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root of an IKEA-Manuals-at-Work checkout containing data/",
    )
    parser.add_argument("--output-root", required=True, help="Output dataset directory")
    parser.add_argument(
        "--num-data",
        type=int,
        default=98,
        help="Maximum number of source sequences passed to KeyframeDataset",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Keep every Nth source frame (default matches the paper-development script)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def build_dataset(
    *,
    source_root: str,
    output_root: str,
    num_data: int = 98,
    frame_stride: int = 3,
    seed: int = 0,
) -> None:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    try:
        from IKEAVideo.dataloader.dataset_keyframe import KeyframeDataset
    except ImportError as exc:
        raise ImportError(
            "The upstream IKEAVideo package is required for dataset preparation. "
            "See THIRD_PARTY_DATA.md for the official source."
        ) from exc

    root = Path(source_root).expanduser().resolve()
    data_root = root / "data"
    annotation = data_root / "data.json"
    video_dir = data_root / "video"
    manual_dir = data_root / "manual_img"
    obj_dir = data_root / "parts"
    pdf_dir = data_root / "pdfs"

    required = [annotation, video_dir, manual_dir, obj_dir, pdf_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected IKEA-Manuals-at-Work data paths were not found:\n- "
            + "\n- ".join(missing)
        )

    random.seed(seed)

    ds = KeyframeDataset(
        str(annotation),
        str(video_dir),
        transform=None,
        load_into_mem=False,
        verbose=False,
        debug=False,
        obj_dir=str(obj_dir),
        num_of_data=num_data,
        manual_img_dir=str(manual_dir),
        pdf_dir=str(pdf_dir),
        demo_print=False,
        demo_viz=False,
    )

    keyframes = [frame for i, frame in enumerate(ds.data) if i % frame_stride == 0]
    build(Path(output_root), keyframes, str(video_dir))


def main() -> None:
    args = build_parser().parse_args()
    build_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        num_data=args.num_data,
        frame_stride=args.frame_stride,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
