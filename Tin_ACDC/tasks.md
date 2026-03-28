# tasks.md

## Purpose

This document tracks implementation work, experimental progress, and near-term priorities for the ACDC segmentation project.

It is subordinate to `AGENTS.md`.

Use this file as the operational execution layer:
- what is finished
- what is in progress
- what is blocked
- what comes next

Update it frequently. Keep it concrete.

---

## Project objective

Deliver a thesis-grade and paper-ready benchmark on ACDC with:

- 3 segmentation architectures
- 2 initialization modes:
  - scratch
  - fine-tune from external pretrained weights
- clean final evaluation using official `training` for development and official `testing` as holdout

---

## Global status legend

Use these labels consistently:

- `[TODO]` not started
- `[DOING]` currently in progress
- `[BLOCKED]` waiting on decision, bug fix, or missing resource
- `[DONE]` completed and verified
- `[DROP]` intentionally removed from scope

---

## Phase 0 — governance and setup

- `[DONE]` Select project direction: ACDC cardiac MRI segmentation
- `[DONE]` Select protocol direction: official training for development, official testing as holdout
- `[DONE]` Decide that the primary pipeline is 2D slice-wise
- `[DONE]` Decide main architecture set:
  - UNetR34
  - AttUNetR34
  - BoundaryDSUNetR34
- `[DONE]` Decide main initialization modes:
  - scratch
  - fine-tune from external pretrained weights
- `[DONE]` Freeze exact pretrained source: torchvision ResNet34 IMAGENET1K_V1
- `[DONE]` Freeze experiment naming convention: `{model}_{init}_seed{seed}[_fold{N}]`
- `[DONE]` Create project repository structure
- `[DONE]` Create experiment log template (`experiments/registry.csv`)
- `[DONE]` Create central path config (`configs/paths.yaml`) pointing to NFS
- `[DONE]` Create data config (`configs/data.yaml`)
- `[DONE]` Create model config (`configs/model.yaml`)
- `[DONE]` Create training config (`configs/train.yaml`)

### Deliverables for Phase 0

- stable project scope ✓
- stable benchmark definition ✓
- stable directory structure ✓

---

## Phase 1 — dataset acquisition and validation

### 1.1 Download and storage

- `[DONE]` Download official ACDC `training` — 100 patients on NFS
- `[TODO]` Download official ACDC `testing` — not yet on NFS
- `[DONE]` Verify patient counts: 100 in training
- `[DONE]` Record storage location: `/mnt/nfs-data/tin_dataset/ACDC/training/`
- `[DONE]` Archive raw dataset path in `configs/paths.yaml`

### 1.2 File structure validation

- `[DONE]` Verify presence of `Info.cfg`, `*_4d.nii.gz`, `*_frameXX.nii.gz`, `*_frameXX_gt.nii.gz`
- `[DONE]` Confirm which frames correspond to ED and ES using `Info.cfg` — implemented in `parse_acdc.py`
- `[DONE]` Confirm that masks match images in shape — implemented in `verify_masks()`

### 1.3 Dataset parsing

- `[DONE]` Write parser to enumerate all patients — `src/data/parse_acdc.py`
- `[DONE]` Write parser to extract all labeled frame paths — `find_labeled_frames()`
- `[DONE]` Build metadata table — `parse_dataset()` returns full DataFrame

### Deliverables for Phase 1

- reproducible raw-data inventory ✓
- metadata table for all usable labeled volumes ✓
- confidence that dataset parsing is correct ✓ (code written, needs first run)

---

## Phase 2 — preprocessing pipeline

### 2.1 Basic preprocessing

- `[DONE]` Implement NIfTI loader — `src/data/preprocess.py:load_nifti_pair()`
- `[DONE]` Implement image-mask paired loading with shape check
- `[DONE]` Implement per-volume z-score normalization — `normalize_volume()`
- `[DONE]` Implement resize to `256 x 256` — `resize_slice()`
- `[DONE]` Ensure image interpolation is bilinear (order=1)
- `[DONE]` Ensure mask interpolation is nearest-neighbor (order=0)

### 2.2 Slice extraction

- `[DONE]` Convert labeled 3D ED/ES volumes into 2D slices — `preprocess_volume()`
- `[DONE]` Keep mapping for each slice: patient_id, source_split, frame_id, phase, slice_index
- `[DONE]` Filter empty slices configurable via `keep_empty` flag
- `[DONE]` Save processed metadata as slice-level CSV — `run_preprocessing()`
- `[DONE]` Support two modes: metadata-only vs cache .npz arrays

### 2.3 Label verification

- `[DONE]` Inspect unique class values — `verify_class_ids()` in preprocess
- `[DONE]` Freeze class mapping: 0=BG, 1=RV, 2=MYO, 3=LV — in `configs/data.yaml`
- `[DONE]` Visualization code for sanity checks — `src/utils/visualization.py`

### 2.4 Augmentation pipeline

- `[DONE]` Implement baseline augmentations — `src/data/transforms.py`
- `[DONE]` Freeze the main augmentation policy in `configs/data.yaml`
- `[DONE]` Add augmentation on/off switch through config (`augmentation.enabled`)

### Deliverables for Phase 2

- stable dataset loader ✓
- verified 2D training pipeline ✓ (code written, needs first run)
- sample visualization confirming correctness ✓ (code written, needs first run)

---

## Phase 3 — split protocol and evaluation preparation

### 3.1 Development split

- `[DONE]` Create fixed patient-level development split — `fixed_split()` in `build_splits.py`
- `[DONE]` Ensure no patient leakage — `check_no_leakage()` raises `LeakageError`
- `[DONE]` Save split file to disk — `dev_split.json`

### 3.2 Cross-validation protocol

- `[DONE]` Design 5-fold patient-level CV split files — `kfold_splits()`
- `[DONE]` Save all fold definitions — `fold_0.json` … `fold_4.json`
- `[DONE]` Verify fold balance — optional `--stratify` flag by pathology
- `[DONE]` Completeness check — every patient assigned exactly once

### 3.3 Holdout usage

- `[DONE]` Define strict rule for official testing usage — documented in `AGENTS.md`
- `[DONE]` No hyperparameter tuning on testing — enforced by split structure
- `[DONE]` Reserve testing for final report only

### 3.4 Evaluation code

- `[DONE]` Implement Dice — `src/eval/metrics.py:dice_score()`
- `[DONE]` Implement IoU — `iou_score()`
- `[DONE]` Implement HD95 — `hausdorff95()` via medpy
- `[DONE]` Support per-class reporting: RV, MYO, LV — `compute_volume_metrics()`
- `[DONE]` Support mean metric reporting
- `[DONE]` Support patient-volume reconstruction — `src/eval/reconstruct_volume.py`
- `[DONE]` Full evaluation pipeline — `src/eval/evaluate.py`

### 3.5 Split reporting

- `[DONE]` Generate split report CSV + TXT — `generate_split_report()`
- `[DONE]` CLI script — `scripts/build_splits.py`

### Deliverables for Phase 3

- saved split files ✓ (code written, needs first run)
- trusted evaluation utilities ✓
- explicit no-leakage policy in code ✓

---

## Phase 4 — baseline model M1: UNetR34

### 4.1 Implementation

- `[DONE]` Implement ResNet34-style encoder — `src/models/backbones/resnet34_encoder.py`
- `[DONE]` Implement standard U-Net decoder — `src/models/decoders/unet_decoder.py`
- `[DONE]` Implement segmentation head — `src/models/heads/seg_head.py`
- `[DONE]` Create `UNetR34` model wrapper — `src/models/unetr34.py`

### 4.2 Sanity checks

- `[TODO]` Run dummy forward pass
- `[TODO]` Verify output shape
- `[TODO]` Verify loss computation
- `[TODO]` Verify backward pass
- `[TODO]` Overfit on a tiny subset
- `[TODO]` Visualize first predictions

### 4.3 Training modes

- `[TODO]` Train `UNetR34-scratch`
- `[TODO]` Train `UNetR34-finetune`
- `[TODO]` Compare training curves
- `[TODO]` Record convergence speed and best val scores

### Deliverables for Phase 4

- working baseline architecture ✓ (code done)
- first scratch vs fine-tune result pair — pending

---

## Phase 5 — strong baseline model M2: AttUNetR34

### 5.1 Implementation

- `[DONE]` Implement attention gate module — `AttentionGate` in `unet_decoder.py`
- `[DONE]` Integrate attention into skip connections — `use_attention=True`
- `[DONE]` Create `AttUNetR34` wrapper — `src/models/attunetr34.py`

### 5.2 Sanity checks

- `[TODO]` Run dummy forward pass
- `[TODO]` Verify output shape
- `[TODO]` Verify training stability
- `[TODO]` Overfit on a tiny subset

### 5.3 Training modes

- `[TODO]` Train `AttUNetR34-scratch`
- `[TODO]` Train `AttUNetR34-finetune`
- `[TODO]` Compare against M1 under same protocol

### Deliverables for Phase 5

- attention baseline fully benchmarked — pending

---

## Phase 6 — proposed model M3: BoundaryDSUNetR34

### 6.1 Boundary target generation

- `[DONE]` Choose boundary-target rule: morphological erosion edge
- `[DONE]` Implement boundary map derivation — `generate_boundary_target()` in `boundary_losses.py`
- `[TODO]` Visualize boundary targets
- `[DONE]` Freeze the rule for the main benchmark — in `configs/model.yaml`

### 6.2 Model implementation

- `[DONE]` Implement boundary head — `src/models/heads/boundary_head.py`
- `[DONE]` Implement deep supervision heads — `src/models/heads/deep_supervision_head.py`
- `[DONE]` Create `BoundaryDSUNetR34` wrapper — `src/models/boundary_dsunetr34.py`

### 6.3 Loss implementation

- `[DONE]` Implement main segmentation loss — `DiceCELoss` in `dice_ce.py`
- `[DONE]` Implement boundary loss — `BoundaryBCELoss`
- `[DONE]` Implement deep supervision loss — `DeepSupervisionLoss`
- `[DONE]` Add weighted total-loss config — `configs/train.yaml` with lambda_b, lambda_ds

### 6.4 Sanity checks

- `[TODO]` Run dummy forward pass
- `[TODO]` Verify all outputs and losses
- `[TODO]` Overfit on tiny subset
- `[TODO]` Verify no NaN or unstable gradients

### 6.5 Training modes

- `[TODO]` Train `BoundaryDSUNetR34-scratch`
- `[TODO]` Train `BoundaryDSUNetR34-finetune`
- `[TODO]` Compare against M1 and M2

### Deliverables for Phase 6

- proposed model fully implemented ✓ (code done)
- all 6 primary conditions available — pending training

---

## Phase 7–12

Unchanged from original. All `[TODO]`.

---

## Immediate next actions

1. `[DOING]` Install dependencies and run preprocessing pipeline for first time
2. `[TODO]` Run `scripts/build_splits.py` to generate splits
3. `[TODO]` Run sanity checks: forward pass + overfit test for all 3 models
4. `[TODO]` Launch first `UNetR34-scratch` baseline
5. `[TODO]` Download ACDC testing set to NFS

---

## Blockers to watch

- `[RESOLVED]` exact pretrained source: torchvision ResNet34 IMAGENET1K_V1
- `[RESOLVED]` repository structure: created
- `[RESOLVED]` experiment log template: `experiments/registry.csv`
- `[RESOLVED]` boundary-target generation: morphological erosion edge
- `[BLOCKED]` testing set not yet downloaded to NFS

---

## Definition of done for the project

The project is considered experimentally complete only when all of the following are true:

- all 6 primary conditions have been trained and evaluated
- results are reproducible across seeds or folds
- final holdout testing has been run cleanly
- ablation results support the chosen conclusions
- benchmark tables and visualizations are locked
- thesis writing can cite stable numbers
- paper draft can state a clear contribution with evidence
