"""Clean entry point for the MAPS training configurations used in the study.

The underlying optimization/training behavior remains in the original research engine for
now so that repository cleanup does not silently alter the published training procedure.
This wrapper removes HPRC-specific paths and exposes one explicit ablation at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# The original training engine uses sibling imports (for example
# ``from dataset_pairs import PairedSegDataset``). Add only that directory to
# the module search path so the engine runs exactly as it did on HPRC while the
# public CLI remains path-independent. After HPRC verification we can fold the
# engine into the cleaned package without changing behavior.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_ENGINE_DIR = _REPO_ROOT / "train_test_code"
if str(_LEGACY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ENGINE_DIR))

from train_ablation import (  # noqa: E402
    ABLATION_GRID,
    Cfg,
    _strip_unknown_cfg_keys,
    load_cfg,
    run_one_training,
)

ABLATION_CHOICES = (
    "base",
    "fuse_10_aux",
    "fuse_6_10_aux",
    "fuse_6_aux",
    "fuse_aux_only",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one MAPS ablation on one prepared train/validation fold"
    )
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, required=True)
    parser.add_argument(
        "--fold-root",
        required=True,
        help="Fold directory containing train/ and val/ subdirectories",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional CUDA device override; otherwise use the YAML setting",
    )
    return parser


def _resolve_ablation(name: str):
    for ablation in ABLATION_GRID:
        if ablation.name == name:
            return ablation
    raise ValueError(
        f"Ablation '{name}' is not available in the training engine. "
        f"Expected one of {ABLATION_CHOICES}."
    )


def build_fold_config(config_path: str, fold_root: str, device: int | None = None) -> Cfg:
    raw = load_cfg(config_path)
    raw = _strip_unknown_cfg_keys(raw, Cfg)
    cfg = Cfg(**raw)

    fold = Path(fold_root).expanduser().resolve()
    train_root = fold / "train"
    val_root = fold / "val"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training split not found: {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"Validation split not found: {val_root}")

    camera_dir = Path(cfg.images_camera_dirname).name
    manual_dir = Path(cfg.images_manual_dirname).name

    replacements = {
        "camera_root": str(train_root / "images" / camera_dir),
        "mask_root_camera": str(train_root / "mask_labels" / camera_dir),
        "mask_root_manual": str(train_root / "mask_labels" / manual_dir),
        "val_camera_root": str(val_root / "images" / camera_dir),
        "val_mask_root_camera": str(val_root / "mask_labels" / camera_dir),
    }
    if device is not None:
        replacements["device"] = int(device)
    return dataclasses.replace(cfg, **replacements)


def train_one(
    *,
    config_path: str,
    ablation_name: str,
    fold_root: str,
    out_dir: str,
    device: int | None = None,
) -> None:
    cfg = build_fold_config(config_path, fold_root, device=device)
    ablation = _resolve_ablation(ablation_name)
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_one_training(cfg, ablation, str(output))


def main() -> None:
    args = build_parser().parse_args()
    train_one(
        config_path=args.config,
        ablation_name=args.ablation,
        fold_root=args.fold_root,
        out_dir=args.out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
