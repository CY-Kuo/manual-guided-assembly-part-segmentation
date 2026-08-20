# MAPS Portfolio Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the private MAPS research repository into a clean, recruiter-facing paper companion that exposes only the final training, inference/evaluation, and minimum data-preparation code while leaving exploratory research runners out of the cleaned branch.

**Architecture:** Keep reusable MAPS components in a small `maps/` package, publish one final training entry point under `training/`, and split the current 2,200-line HPRC inference/evaluation driver into `evaluation/run_inference.py` and `evaluation/evaluate.py`. Data-building utilities remain separate under `data/`. All cleanup stays on `portfolio-cleanup`; `main` remains unchanged until explicit user approval.

**Tech Stack:** Python 3, PyTorch, Ultralytics YOLO11-seg, NumPy, Pillow, PyYAML, pycocotools, SciPy, OpenCV where already required by existing code.

**Spec:** `docs/superpowers/specs/2026-08-20-maps-portfolio-cleanup-design.md`

## Global Constraints

- Keep the repository private throughout cleanup.
- Do not modify `main` until the user explicitly approves the cleaned branch.
- Do not rewrite Git history.
- Do not publish third-party datasets or pretrained weights unless redistribution rights are separately confirmed.
- Do not expose hard-coded HPRC paths, usernames, private cluster locations, credentials, tokens, private IPs, or unresolved placeholder assignments in public-facing code.
- Preserve scientific behavior of the final MAPS method; cleanup may reorganize code and configuration but must not silently change the published method.
- Keep only code required to prepare supported inputs, train the final MAPS configuration, run inference, and evaluate the final model plus a small set of meaningful baselines.
- Exclude fold/ablation orchestration and one-off paper-development runners from the final public-facing tree.

---

## File Structure Locked for Implementation

```text
.
├── README.md
├── LICENSE
├── THIRD_PARTY_DATA.md
├── requirements.txt
├── maps/
│   ├── __init__.py
│   ├── dataset_pairs.py
│   ├── feature_hooks.py
│   ├── aux_heads.py
│   ├── distill_losses.py
│   ├── smart_prior.py
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
├── tests/
│   ├── test_imports.py
│   ├── test_smart_prior.py
│   ├── test_inference_cli.py
│   ├── test_evaluation_metrics.py
│   └── test_training_cli.py
└── assets/
    └── .gitkeep
```

---

### Task 1: Establish the clean MAPS package

**Files:**
- Create: `maps/__init__.py`
- Create from existing source with import cleanup: `maps/dataset_pairs.py`
- Create from existing source with import cleanup: `maps/feature_hooks.py`
- Create from existing source with import cleanup: `maps/aux_heads.py`
- Create from existing source with import cleanup: `maps/distill_losses.py`
- Create from existing source with import cleanup: `maps/smart_prior.py`
- Create from existing source with import cleanup: `maps/rle_utils.py`
- Test: `tests/test_imports.py`
- Test: `tests/test_smart_prior.py`

**Interfaces:**
- Consumes: existing implementations in `train_test_code/`.
- Produces: importable modules under `maps.*` used by training and evaluation tasks.
- Required public functions from `maps.smart_prior`: `refine_aux_prior(...)` and `filter_student_instances_with_prior(...)`.

- [ ] **Step 1: Write import tests that fail before the package exists**

```python
# tests/test_imports.py

def test_core_modules_import():
    import maps.dataset_pairs  # noqa: F401
    import maps.feature_hooks  # noqa: F401
    import maps.aux_heads  # noqa: F401
    import maps.distill_losses  # noqa: F401
    import maps.smart_prior  # noqa: F401
    import maps.rle_utils  # noqa: F401
```

- [ ] **Step 2: Run the import test and confirm failure**

Run: `pytest tests/test_imports.py -v`

Expected: FAIL because the `maps` package does not yet exist.

- [ ] **Step 3: Copy the reusable modules into `maps/` and normalize project-local imports**

Required import style inside moved modules:

```python
from maps.feature_hooks import FeatureHooker
from maps.aux_heads import AuxMaskHead
from maps.distill_losses import DistillCfg, Distiller
from maps.smart_prior import refine_aux_prior, filter_student_instances_with_prior
```

Do not alter algorithmic logic except where required for the new import paths.

- [ ] **Step 4: Add focused tests for `smart_prior` that do not require GPU/model weights**

```python
# tests/test_smart_prior.py
import numpy as np
from maps.smart_prior import refine_aux_prior, filter_student_instances_with_prior


def test_refine_aux_prior_returns_boolean_mask_and_probability():
    logits = np.zeros((8, 8), dtype=np.float32)
    student = np.zeros((8, 8), dtype=bool)
    student[3:5, 3:5] = True
    mask, prob = refine_aux_prior(logits, student, auto_tighten=False, min_comp_area=0, light_erosion_iters=0, close_iters=0)
    assert mask.shape == (8, 8)
    assert mask.dtype == bool
    assert prob.shape == (8, 8)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_filter_student_instances_returns_union():
    prior = np.ones((6, 6), dtype=bool)
    masks = [np.eye(6, dtype=bool)]
    preds, union = filter_student_instances_with_prior(masks, np.array([0.9]), prior)
    assert len(preds) == 1
    assert union.shape == prior.shape
    assert union.any()
```

- [ ] **Step 5: Run the package tests**

Run: `pytest tests/test_imports.py tests/test_smart_prior.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add maps tests/test_imports.py tests/test_smart_prior.py
git commit -m "refactor: extract reusable MAPS modules"
```

---

### Task 2: Clean the data-preparation entry point

**Files:**
- Create from `data_build_code/main.py`: `data/build_dataset.py`
- Create from `data_build_code/yolo_dataset_builder.py`: `data/yolo_dataset_builder.py`
- Modify: `THIRD_PARTY_DATA.md`
- Test: extend `tests/test_imports.py`

**Interfaces:**
- Consumes: third-party IKEA-Manuals-at-Work / IKEAVideo data as documented in `THIRD_PARTY_DATA.md`.
- Produces: paired camera/manual images, YOLO-compatible labels, and RLE mask files in the existing repository data format.

- [ ] **Step 1: Extend the import test**

```python

def test_data_modules_import():
    import data.yolo_dataset_builder  # noqa: F401
```

- [ ] **Step 2: Run the test and confirm failure before files are created**

Run: `pytest tests/test_imports.py::test_data_modules_import -v`

Expected: FAIL.

- [ ] **Step 3: Move the existing builder with only path/import cleanup**

Create `data/yolo_dataset_builder.py` from the current implementation. Preserve functions such as `rle_valid`, `rle_to_mask_u8`, `contours_to_rows`, and `build` without scientific changes.

- [ ] **Step 4: Convert `data_build_code/main.py` into a configurable `data/build_dataset.py`**

The entry point must accept paths from CLI rather than machine-specific constants:

```python
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Build the MAPS paired camera/manual dataset")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser
```

- [ ] **Step 5: Update `THIRD_PARTY_DATA.md` to state that data is not redistributed and must be obtained from the official source**

Do not add unverified redistribution instructions.

- [ ] **Step 6: Run import and syntax checks**

Run: `python -m compileall data maps`

Expected: successful compilation.

- [ ] **Step 7: Commit**

```bash
git add data THIRD_PARTY_DATA.md tests/test_imports.py
git commit -m "refactor: clean MAPS data preparation utilities"
```

---

### Task 3: Extract the final MAPS inference path

**Files:**
- Create: `evaluation/run_inference.py`
- Test: `tests/test_inference_cli.py`
- Source basis: uploaded `test_ablation_5tracks_multi_save_masks.py` and existing `maps/smart_prior.py`.

**Interfaces:**
- Consumes: student YOLO weights, teacher YOLO weights, MAPS checkpoint, one camera image, one aligned manual image, threshold configuration, device.
- Produces: final MAPS/refined binary mask and optional intermediate masks.
- CLI arguments: `--ckpt`, `--student`, `--teacher`, `--camera`, `--manual`, `--out-dir`, `--device`, `--yolo-conf`, `--aux-thr`, `--save-intermediate`.

- [ ] **Step 1: Write CLI parser tests**

```python
# tests/test_inference_cli.py
from evaluation.run_inference import build_parser


def test_inference_cli_requires_model_and_input_paths():
    parser = build_parser()
    args = parser.parse_args([
        "--ckpt", "maps.pt",
        "--student", "student.pt",
        "--teacher", "teacher.pt",
        "--camera", "camera.jpg",
        "--manual", "manual.jpg",
        "--out-dir", "out",
    ])
    assert args.ckpt == "maps.pt"
    assert args.camera == "camera.jpg"
    assert args.manual == "manual.jpg"
```

- [ ] **Step 2: Run the parser test and confirm failure**

Run: `pytest tests/test_inference_cli.py -v`

Expected: FAIL because `evaluation/run_inference.py` does not yet exist.

- [ ] **Step 3: Extract model reconstruction from the HPRC inference script**

Keep the behavior represented by the uploaded script's `FeatureHooker`, `StructureFusion`, `AuxMaskHead`, channel-compatible checkpoint loaders, `build_aux_and_fusion`, `aux_logits_and_prob`, YOLO mask collection, and smart-prior refinement.

Prefer importing reusable classes from `maps.feature_hooks` and `maps.aux_heads` instead of redefining them in `run_inference.py`.

- [ ] **Step 4: Remove HPRC globals and experiment orchestration**

The public inference file must contain no values beginning with `/scratch/`, no `FOLDS`, no `ABLATIONS`, no validation sweep, and no multi-test-set loop.

- [ ] **Step 5: Implement one-image inference function**

Public interface:

```python
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
    """Run MAPS inference and return output paths plus threshold metadata."""
```

- [ ] **Step 6: Run parser/import tests**

Run: `pytest tests/test_inference_cli.py tests/test_imports.py -v`

Expected: PASS without requiring model weights.

- [ ] **Step 7: Run a static scan for private paths**

Run: `grep -R -nE '/scratch/|kuocy|tamu\.edu|CKPT_ROOT|FOLDS|ABLATIONS' evaluation/run_inference.py`

Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add evaluation/run_inference.py tests/test_inference_cli.py
git commit -m "feat: add clean MAPS inference entry point"
```

---

### Task 4: Extract publication-facing evaluation

**Files:**
- Create: `evaluation/evaluate.py`
- Test: `tests/test_evaluation_metrics.py`
- Source basis: metric/calibration/dataset evaluation portions of uploaded `test_ablation_5tracks_multi_save_masks.py`.

**Interfaces:**
- Consumes: the same three model/checkpoint paths plus camera/manual dataset YAMLs, GT mask root, output directory, fixed or calibration-derived thresholds.
- Produces: `per_image.csv`, `summary_pixel.csv`, `summary_ap.csv`, and optional saved masks.
- Public metric helpers: `pixel_metrics(pred_bool, gt_bool, bound_tol=2)` and `evaluate_map(preds, gts_by_img, iou_thresholds=...)`.

- [ ] **Step 1: Write deterministic metric tests**

```python
# tests/test_evaluation_metrics.py
import numpy as np
from evaluation.evaluate import pixel_metrics


def test_pixel_metrics_perfect_overlap():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = pixel_metrics(mask, mask)
    assert metrics["Dice"] == 1.0
    assert metrics["IoU"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0


def test_pixel_metrics_no_overlap():
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    gt = np.array([[0, 0], [0, 1]], dtype=bool)
    metrics = pixel_metrics(pred, gt)
    assert metrics["Dice"] < 1e-5
    assert metrics["IoU"] < 1e-5
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `pytest tests/test_evaluation_metrics.py -v`

Expected: FAIL because `evaluation/evaluate.py` does not yet exist.

- [ ] **Step 3: Extract metric, RLE-loading, and AP logic from the uploaded evaluation driver**

Retain the scientific metric behavior used in the existing script: Dice, IoU, Precision, Recall, tolerant BoundaryF, coverage, AP50, and mAP50-95.

- [ ] **Step 4: Extract dataset-level evaluation without ablation orchestration**

The evaluator must operate on one final MAPS checkpoint at a time and accept all dataset/model paths through CLI arguments.

- [ ] **Step 5: Keep threshold calibration optional**

Support both:

```text
--test-only --yolo-conf-raw 0.20 --yolo-conf-ckpt 0.20 --aux-thr 0.45
```

and validation-based calibration when validation inputs are explicitly supplied. Do not encode fold-specific winners as hidden defaults beyond documented final defaults.

- [ ] **Step 6: Run metric tests and syntax checks**

Run: `pytest tests/test_evaluation_metrics.py -v && python -m compileall evaluation`

Expected: PASS.

- [ ] **Step 7: Scan for HPRC-specific content**

Run: `grep -R -nE '/scratch/|kuocy|tamu\.edu|fold3|fold4|fuse_6|fuse_10' evaluation`

Expected: no matches in public entry-point code.

- [ ] **Step 8: Commit**

```bash
git add evaluation/evaluate.py tests/test_evaluation_metrics.py
git commit -m "feat: add clean MAPS evaluation pipeline"
```

---

### Task 5: Extract the final training entry point

**Files:**
- Create: `training/train.py`
- Create: `training/config.yaml`
- Test: `tests/test_training_cli.py`
- Source basis: `train_test_code/train_ablation.py` plus reusable `maps/` modules.

**Interfaces:**
- Consumes: YAML config defining student/teacher models, camera/manual training roots, mask roots, feature taps, loss weights, optimizer settings, and output directory.
- Produces: final MAPS checkpoint compatible with `evaluation/run_inference.py` and `evaluation/evaluate.py`.
- CLI: `python training/train.py --config training/config.yaml`.

- [ ] **Step 1: Write a parser/config smoke test**

```python
# tests/test_training_cli.py
from training.train import build_parser


def test_training_cli_accepts_config():
    parser = build_parser()
    args = parser.parse_args(["--config", "training/config.yaml"])
    assert args.config == "training/config.yaml"
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/test_training_cli.py -v`

Expected: FAIL because training entry point does not yet exist.

- [ ] **Step 3: Extract final-method training logic from `train_ablation.py`**

Retain teacher/student loading, feature hooks, distillation losses, auxiliary head training, fusion module construction, optimization, checkpoint metadata, and the exact final fusion configuration required by the chosen MAPS checkpoint format.

Remove public orchestration that iterates over multiple ablation combinations or folds.

- [ ] **Step 4: Implement config-driven entry point**

Minimum parser:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Train the final MAPS model")
    parser.add_argument("--config", required=True)
    return parser
```

- [ ] **Step 5: Create `training/config.yaml` with relative/example paths**

Use keys already consumed by the training implementation. Example path values must be relative placeholders such as:

```yaml
student_model: weights/student.pt
teacher_model: weights/teacher.pt
camera_root: data/mini/images/train/camera
manual_root: data/mini/images/train/manual
out_dir: runs/maps
```

No `/scratch/` paths are allowed.

- [ ] **Step 6: Ensure checkpoint compatibility with inference**

The final training checkpoint must preserve keys expected by inference:

```text
student
seg_adapter
fusions
fusion_meta
```

- [ ] **Step 7: Run parser/import/static checks**

Run: `pytest tests/test_training_cli.py tests/test_imports.py -v`

Run: `grep -R -nE '/scratch/|kuocy|tamu\.edu|fold[0-9]|ABLATIONS' training maps`

Expected: no machine-specific matches in public-facing training code/config.

- [ ] **Step 8: Commit**

```bash
git add training tests/test_training_cli.py
git commit -m "feat: add final MAPS training entry point"
```

---

### Task 6: Retain only the useful baselines

**Files:**
- Create from selected existing baseline implementations: `baselines/camera_only.py`
- Create from selected existing baseline implementations: `baselines/manual_roi_sam.py`
- Create from selected existing baseline implementations: `baselines/sam_box_from_yolo.py`
- Create from selected existing baseline implementations: `baselines/direct_fusion.py`
- Remove from cleaned tree: remaining `baseline_code/` files after useful logic has been preserved.

**Interfaces:**
- Consumes: the same data/model conventions documented by each baseline.
- Produces: standalone comparison scripts, not dependencies of MAPS inference.

- [ ] **Step 1: Select source mappings**

Use:

```text
baseline_code/baseline_manual_roi_to_sam.py -> baselines/manual_roi_sam.py
baseline_code/baseline_sam_box_from_yolo.py -> baselines/sam_box_from_yolo.py
baseline_code/direct_fusion_baseline.py -> baselines/direct_fusion.py
```

For `camera_only.py`, extract only the direct camera-only YOLO inference/evaluation path rather than keeping an orchestration script.

- [ ] **Step 2: Remove hard-coded paths from selected baselines**

Each baseline must accept paths through CLI/config.

- [ ] **Step 3: Syntax check**

Run: `python -m compileall baselines`

Expected: success.

- [ ] **Step 4: Remove the old baseline tree from the cleaned branch**

Delete `baseline_code/` after selected implementations are present under `baselines/`.

- [ ] **Step 5: Commit**

```bash
git add baselines
git rm -r baseline_code
git commit -m "refactor: keep portfolio-relevant MAPS baselines"
```

---

### Task 7: Remove exploratory training/evaluation research runners

**Files:**
- Remove from cleaned branch: `train_test_code/`
- Remove from cleaned branch: `data_build_code/` after Task 2 migration

**Interfaces:**
- Consumes: completed Tasks 1-6.
- Produces: final public-facing tree without duplicate experimental scripts.

- [ ] **Step 1: Verify all required source logic has been migrated**

Run:

```bash
python -m compileall maps data training evaluation baselines
pytest tests -v
```

Expected: PASS before deleting old trees.

- [ ] **Step 2: Remove old source trees**

```bash
git rm -r train_test_code data_build_code
```

- [ ] **Step 3: Re-run tests after removal**

Run: `pytest tests -v`

Expected: PASS, proving no cleaned code imports the removed experimental paths.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove exploratory research runners from portfolio branch"
```

---

### Task 8: Rewrite README as a paper companion and portfolio surface

**Files:**
- Modify: `README.md`
- Create: `assets/.gitkeep`

**Interfaces:**
- Consumes: final repository structure and verified publication claims.
- Produces: recruiter/researcher-facing documentation.

- [ ] **Step 1: Replace the current minimal README with the approved structure**

Required sections in order:

```markdown
# MAPS: Manual-Guided Assembly-Part Segmentation

[one-sentence problem/solution statement]

## Publication
## Why MAPS
## Method Overview
## Key Results
## Qualitative Results
## Repository Structure
## Setup
## Data Preparation
## Training
## Inference
## Evaluation
## Third-Party Data
## Citation
## License
```

- [ ] **Step 2: Include only publication claims verified against the paper**

Use the published paper link supplied by the user. Do not introduce new metrics or deployment claims not supported by the publication.

- [ ] **Step 3: Document the actual clean commands**

Examples must reference files that exist:

```bash
python data/build_dataset.py --source-root /path/to/source --output-root data/mini
python training/train.py --config training/config.yaml
python evaluation/run_inference.py --ckpt weights/maps.pt --student weights/student.pt --teacher weights/teacher.pt --camera sample_camera.jpg --manual sample_manual.jpg --out-dir outputs/demo
python evaluation/evaluate.py --help
```

- [ ] **Step 4: Create `assets/.gitkeep` until user-approved figures are added**

Do not publish paper figures unless their reuse is appropriate and the user explicitly chooses them.

- [ ] **Step 5: Commit**

```bash
git add README.md assets/.gitkeep
git commit -m "docs: turn MAPS repo into published paper companion"
```

---

### Task 9: Final privacy, path, and reproducibility audit

**Files:**
- Modify as required based on audit results: `.gitignore`, `requirements.txt`, README, source files.

**Interfaces:**
- Consumes: all cleaned code and docs.
- Produces: review-ready private branch.

- [ ] **Step 1: Scan tracked files for private/machine-specific content**

Run:

```bash
grep -R -nE '/scratch/|/home/|kuocy|tamu\.edu|password|passwd|secret|token|api[_-]?key|private[_-]?key' . \
  --exclude-dir=.git --exclude='*.md'
```

Review every match; legitimate variable names such as `num_tokens` are not secrets, but credentials and absolute cluster paths must be removed.

- [ ] **Step 2: Scan for exploratory naming**

Run:

```bash
grep -R -nE 'ablation|fold[0-9]|test_time_only|batch_full|my_run|mini_2' \
  maps data training evaluation baselines README.md
```

Expected: no unexplained paper-development orchestration in public-facing files. Dataset terminology that is scientifically necessary may remain only when documented.

- [ ] **Step 3: Run the full test/static suite**

Run:

```bash
python -m compileall maps data training evaluation baselines
pytest tests -v
python training/train.py --help
python evaluation/run_inference.py --help
python evaluation/evaluate.py --help
```

Expected: all commands succeed without model weights or dataset access.

- [ ] **Step 4: Confirm repository contents are clean**

Expected top-level tree:

```text
README.md
LICENSE
THIRD_PARTY_DATA.md
requirements.txt
maps/
data/
training/
evaluation/
baselines/
tests/
assets/
docs/
```

No `train_test_code/`, `data_build_code/`, tracked datasets, weights, caches, or experiment output folders.

- [ ] **Step 5: Compare branch against `main` for review**

Run: `git diff --stat main...portfolio-cleanup`

Do not merge.

- [ ] **Step 6: Commit any audit fixes**

```bash
git add -A
git commit -m "chore: finalize MAPS portfolio cleanup"
```

- [ ] **Step 7: Stop for user review**

Present the cleaned branch, repository tree, README, and diff summary. The repository remains private and `main` remains unchanged until the user explicitly approves merge/publication.
