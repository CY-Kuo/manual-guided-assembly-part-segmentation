#!/usr/bin/env python3
# scripts/run_eval_baseline_all_tracks.py
# Run baseline inference on all evaluation tracks (test, unseen category, unseen types).
# Uses Harvard HPC paths by default; data_root points to dataset_code/mini_2 (e.g. IKEAVideo_3 or IKEAVideo_2).
# Writes per-track metrics and an aggregated summary CSV.

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_test_sets(data_root: str) -> list[dict]:
    """Build TEST_SETS in same structure as test_ablation_5tracks_multi_harvard (Harvard paths)."""
    base = Path(data_root).resolve()
    return [
        {
            "name": "test",
            "cam_yaml": str(base / "parts_single_cam.yaml"),
            "man_yaml": str(base / "parts_single_man.yaml"),
            "mask_root": str(base / "mask_labels" / "test" / "camera"),
        },
        {
            "name": "test_unseen_category",
            "cam_yaml": str(base / "cam_unseen_cat.yaml"),
            "man_yaml": str(base / "man_unseen_cat.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_category" / "camera"),
        },
        {
            "name": "test_unseen_type_Bench",
            "cam_yaml": str(base / "cam_unseen_bench.yaml"),
            "man_yaml": str(base / "man_unseen_bench.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_type_Bench" / "camera"),
        },
        {
            "name": "test_unseen_type_Chair",
            "cam_yaml": str(base / "cam_unseen_chair.yaml"),
            "man_yaml": str(base / "man_unseen_chair.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_type_Chair" / "camera"),
        },
        {
            "name": "test_unseen_type_Desk",
            "cam_yaml": str(base / "cam_unseen_desk.yaml"),
            "man_yaml": str(base / "man_unseen_desk.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_type_Desk" / "camera"),
        },
        {
            "name": "test_unseen_type_Shelf",
            "cam_yaml": str(base / "cam_unseen_shelf.yaml"),
            "man_yaml": str(base / "man_unseen_shelf.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_type_Shelf" / "camera"),
        },
        {
            "name": "test_unseen_type_Table",
            "cam_yaml": str(base / "cam_unseen_table.yaml"),
            "man_yaml": str(base / "man_unseen_table.yaml"),
            "mask_root": str(base / "mask_labels" / "test_unseen_type_Table" / "camera"),
        },
    ]


def read_metrics_csv(csv_path: Path) -> dict[str, float]:
    """Read metrics.csv (metric,value) into a dict."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "metric" in row and "value" in row:
                try:
                    out[row["metric"]] = float(row["value"])
                except (ValueError, TypeError):
                    out[row["metric"]] = row["value"]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline inference on all tracks (test + unseen) and aggregate metrics."
    )
    parser.add_argument(
        "--baseline_type",
        required=True,
        choices=["early_fusion", "direct_fusion_e2e"],
        help="Baseline model type.",
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_best.pt (or ckpt_last.pt).")
    parser.add_argument(
        "--data_root",
        required=True,
        help="Dataset root (Harvard: .../IKEAVideo_*/dataset_code/mini_2).",
    )
    parser.add_argument(
        "--out_parent",
        default=None,
        help="Parent output dir; per-track outputs go to out_parent/eval_<track_name>. Default: dir of ckpt_path.",
    )
    parser.add_argument("--skip_missing_tracks", action="store_true", help="Skip tracks whose YAML/mask dirs are missing.")
    parser.add_argument("--save_masks", action="store_true", help="Pass --save_masks to inference.")
    parser.add_argument("--save_visualizations", action="store_true", help="Pass --save_visualizations to inference.")
    parser.add_argument("--device", default="0", help="Device for inference.")
    args = parser.parse_args()

    out_parent = Path(args.out_parent or str(Path(args.ckpt_path).resolve().parent))
    out_parent.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "scripts" / "run_inference_baseline.py"
    if not script.exists():
        raise FileNotFoundError(f"Inference script not found: {script}")

    test_sets = build_test_sets(args.data_root)
    summary_rows = []

    for ts in test_sets:
        track_name = ts["name"]
        cam_yaml = ts["cam_yaml"]
        man_yaml = ts["man_yaml"]
        mask_root = ts["mask_root"]
        track_out = out_parent / f"eval_{track_name}"

        if args.skip_missing_tracks:
            if not Path(cam_yaml).exists():
                print(f"[SKIP] {track_name}: cam_yaml not found: {cam_yaml}")
                continue
            if not Path(mask_root).exists():
                print(f"[SKIP] {track_name}: mask_root not found: {mask_root}")
                continue

        cmd = [
            sys.executable,
            str(script),
            "--baseline_type", args.baseline_type,
            "--ckpt_path", args.ckpt_path,
            "--test_cam_yaml", cam_yaml,
            "--test_man_yaml", man_yaml,
            "--test_mask_root", mask_root,
            "--out_dir", str(track_out),
            "--device", args.device,
        ]
        if args.save_masks:
            cmd.append("--save_masks")
        if args.save_visualizations:
            cmd.append("--save_visualizations")

        print(f"[RUN] {track_name} -> {track_out}")
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if ret.returncode != 0:
            print(f"[WARN] {track_name} exited with code {ret.returncode}")
            summary_rows.append({"track": track_name, "status": "failed", "exit_code": str(ret.returncode)})
            continue

        metrics_csv = track_out / "metrics.csv"
        metrics = read_metrics_csv(metrics_csv)
        row = {"track": track_name, "status": "ok"}
        for k, v in metrics.items():
            row[k] = v
        summary_rows.append(row)

    # Write aggregated summary
    if summary_rows:
        all_keys = set()
        for r in summary_rows:
            all_keys.update(r.keys())
        fieldnames = ["track", "status"] + sorted(all_keys - {"track", "status"})
        summary_path = out_parent / "eval_all_tracks_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
