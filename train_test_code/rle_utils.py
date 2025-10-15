# rle_utils.py  (no frPyObjects)
import json
import numpy as np
from pycocotools import mask as mask_utils

def _decode_one(rle_like):
    """
    rle_like: dict with {'counts': <str>, 'size': [H, W]}
              or dict with {'segmentation': {'counts': <str>, 'size': [H,W]}}
    Returns (H,W) uint8 in {0,1}
    """
    # unwrap "segmentation" if present
    if "segmentation" in rle_like and isinstance(rle_like["segmentation"], dict):
        rle_like = rle_like["segmentation"]

    if not ("counts" in rle_like and "size" in rle_like):
        return None

    counts = rle_like["counts"]
    size = rle_like["size"]  # [H, W]

    # We only accept compressed RLE (counts is str). No frPyObjects fallback.
    if not isinstance(counts, str):
        raise ValueError(
            "RLE 'counts' must be a string (compressed RLE). "
            "Your file uses list-format counts; please export compressed RLE."
        )

    rle = {"counts": counts.encode("utf-8"), "size": [int(size[0]), int(size[1])]}
    m = mask_utils.decode(rle)  # (H,W,1) or (H,W)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)

def load_rle_json(json_path):
    """
    Accepts:
      - list of compressed-RLE dicts
      - single dict with compressed RLE (or under 'segmentation')
    Returns merged (H,W) uint8 mask or None.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Normalize to a list of items to decode
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # support {"rles": [...]} or a single RLE dict
        items = data.get("rles", [data])
    else:
        return None

    masks = []
    for it in items:
        m = _decode_one(it)
        if m is not None:
            masks.append(m)

    if not masks:
        return None

    merged = (np.stack(masks, 0).max(0) > 0).astype(np.uint8)
    return merged
