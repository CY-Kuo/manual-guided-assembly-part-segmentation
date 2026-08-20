# MAPS Portfolio Cleanup Design

Date: 2026-08-20

## Goal

Prepare the private MAPS repository for eventual public portfolio use while preserving `main` unchanged until review. The public-facing version should expose only the clean core MAPS implementation, the minimum data-preparation utilities needed to reproduce inputs, a clear training/evaluation path, selected baselines, and publication-facing documentation. Exploratory ablation and paper-development scripts should not appear in the cleaned branch.

## Source Context

Repository: `CY-Kuo/MAPS-Manual-Guided-Assembly-Parts-Segmentation`

Paper: https://link.springer.com/article/10.1007/s00170-026-18468-w

Current repository structure includes `baseline_code/`, `data_build_code/`, and `train_test_code/`. The core reusable pieces are currently mixed with experiment-specific runners, ablation scripts, and HPRC-path-dependent evaluation code.

## Design Principles

1. Keep the repository private throughout cleanup.
2. Perform all cleanup on `portfolio-cleanup`; do not alter `main` until the user explicitly approves the finished branch.
3. Preserve the original development history; do not rewrite history merely to hide exploratory scripts.
4. Favor a small, understandable public API over exhaustive reproduction of every paper-development experiment.
5. Avoid hard-coded HPRC paths, placeholder assignments, or internal experiment names in public-facing entry points.
6. Keep code changes as conservative as possible: relocate and lightly adapt existing implementation rather than reimplementing the method.
7. Do not include third-party datasets or pretrained weights unless redistribution rights are clear.

## Target Repository Structure

```text
.
├── README.md
├── LICENSE
├── THIRD_PARTY_DATA.md
├── requirements.txt
├── maps/
│   ├── __init__.py
│   ├── dataset_pairs.py
│   ├── smart_prior.py
│   ├── feature_hooks.py
│   ├── aux_heads.py
│   ├── distill_losses.py
│   └── rle_utils.py
├── data/
│   ├── build_dataset.py
│   └── yolo_dataset_builder.py
├── training/
│   ├── train.py
│   └── config.yaml
├── evaluation/
│   ├── evaluate.py
│   └── run_inference.py
├── baselines/
│   ├── camera_only.py
│   ├── manual_roi_sam.py
│   ├── sam_box_from_yolo.py
│   └── direct_fusion.py
└── assets/
    ├── method_overview.png
    └── qualitative_results.png
```

`assets/` may initially be empty or contain only user-approved figures. Images should be added manually if that is more reliable than programmatic upload.

## File Mapping

### Core MAPS package

Move or copy these existing modules from `train_test_code/` into `maps/` with import paths updated as needed:

- `dataset_pairs.py`
- `smart_prior.py`
- `feature_hooks.py`
- `aux_heads.py`
- `distill_losses.py`
- `rle_utils.py`

`smart_prior.py` is a reusable part of the inference/evaluation path and includes teacher-prior refinement and student-instance filtering. It should remain as a library module rather than an entry-point script.

### Data preparation

Map:

- `data_build_code/main.py` -> `data/build_dataset.py`
- `data_build_code/yolo_dataset_builder.py` -> `data/yolo_dataset_builder.py`

These should remain focused on building the paired camera/manual dataset and YOLO-compatible annotations from the supported third-party source. Dataset acquisition details remain in `THIRD_PARTY_DATA.md`.

### Training

Use the current `train_test_code/train_ablation.py` as the implementation source, but publish a single `training/train.py` entry point representing the intended MAPS training configuration rather than an ablation runner.

The public training entry point should:

- read all machine-specific paths from YAML or CLI configuration;
- keep the required teacher/student, feature-hook, distillation, and auxiliary-head logic;
- remove public-facing multi-ablation orchestration when it is not required to train the final method;
- avoid internal comments and naming tied to paper-development runs;
- retain configuration needed to reproduce the final selected MAPS setup.

`training/config.yaml` should use relative/example paths and document required fields without embedding local/HPRC paths.

### Evaluation and inference

Use the current test/evaluation scripts as implementation sources, but expose one clean evaluation path and one clean inference path.

The current `test_ablation_5tracks_multi.py` contains useful metric and test-set logic but is not suitable as a public entry point because it has hard-coded model paths, checkpoint roots, fold lists, ablation names, dataset paths, and experiment-specific output names.

The cleaned `evaluation/evaluate.py` should:

- accept checkpoint/model paths and dataset config through CLI/YAML;
- compute the publication-relevant segmentation metrics used by the repository;
- support the final MAPS inference/refinement path;
- avoid automatic multi-fold/multi-ablation sweeps;
- produce a concise summary and optional per-image CSV.

`evaluation/run_inference.py` should provide a simpler qualitative inference path for a camera image plus aligned manual guidance input.

### Baselines

Retain only baselines that are useful for understanding the paper and reproducing meaningful comparisons:

- camera-only YOLO segmentation;
- manual-ROI-to-SAM;
- SAM box from YOLO;
- direct fusion.

Do not expose every baseline or tuning variant simply because it exists. Files such as CRF/CAM-ROI variants may be retained only if they correspond directly to a reported comparison that materially improves reproducibility; otherwise they should be omitted from the cleaned branch.

## Exploratory Files to Exclude from the Cleaned Branch

The following classes of files should not appear in the final public-facing tree:

- k-fold orchestration scripts;
- multi-track ablation runners;
- test-time-only ablation variants;
- duplicate evaluation drivers;
- experimental configuration files for combinations that are not the final MAPS method;
- scripts with unresolved hard-coded HPRC paths or placeholder assignments;
- temporary research utilities that are not needed to train, infer, evaluate, or understand the published method.

Examples from the current tree include:

- `train_test_code/train_ablation_kfold.py`
- `train_test_code/test_ablation_5tracks_multi.py` as a public entry-point file (its useful logic may be adapted)
- `train_test_code/test_ablation_5tracks_test_time_only.py`
- `train_test_code/test_dual_input_fusion_compare.py`
- `train_test_code/config_ablation_kfold.yaml`

Original versions remain recoverable from Git history and `main` until the user approves replacement.

## README Design

The README should be the primary portfolio surface and should be understandable to a robotics/computer-vision recruiter without reading the paper first.

Recommended order:

1. **Title** — `MAPS: Manual-Guided Assembly-Part Segmentation`
2. **One-sentence problem/solution statement**
3. **Publication link and citation**
4. **Method overview figure**
5. **Why it matters** — clutter, occlusion, visually difficult assembly scenes; manual guidance as a structural prior
6. **Key published results** — only values verified against the paper
7. **Method overview** — camera stream, manual guidance stream, teacher/student or dual-input design, final segmentation
8. **Qualitative results**
9. **Repository structure**
10. **Environment/setup**
11. **Dataset preparation**
12. **Training**
13. **Evaluation/inference**
14. **Third-party data/licensing**
15. **Citation**

The README should not overstate deployment assumptions or invent metrics not reported in the publication.

## Naming

Keep the repository name unchanged during cleanup. Renaming, if desired, should happen only after the cleaned branch is reviewed. A shorter final name such as `MAPS-assembly-part-segmentation` may be considered later, but it is outside this cleanup implementation until explicitly approved.

## Validation

Before asking the user to make the repo public:

1. Confirm `main` remains private and unchanged until merge approval.
2. Verify all Python files in the cleaned tree parse successfully.
3. Check that all local imports resolve under the new directory structure.
4. Scan the cleaned tree for hard-coded absolute paths, usernames, cluster paths, secrets, tokens, private IPs, and placeholder assignments.
5. Verify README commands refer to files that actually exist.
6. Verify the selected training/evaluation entry points provide `--help` or clear configuration instructions.
7. Confirm no dataset files, model weights, caches, generated outputs, or internal experiment artifacts are tracked.
8. Compare publication-facing claims and reported metrics against the published paper before finalizing the README.

## Out of Scope

- rewriting Git history;
- publishing the repository;
- uploading third-party data;
- distributing model weights without a separate licensing review;
- recreating every ablation reported during paper development;
- changing the scientific method beyond cleanup required to expose the final implementation.

## Success Criteria

A recruiter or researcher should be able to open the repository and, within roughly two minutes, understand:

- what MAPS solves;
- how manual guidance enters the segmentation system;
- what technical components the author implemented;
- where the published results come from;
- how to prepare data, train the final method, and run evaluation/inference;
- which parts are third-party and subject to separate licensing.

The final branch should look like a maintained research/engineering project rather than a snapshot of exploratory experiment scripts.
