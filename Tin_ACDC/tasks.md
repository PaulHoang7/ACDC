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
- `[TODO]` Freeze exact pretrained source for all fine-tune experiments
- `[TODO]` Freeze experiment naming convention in code and logs
- `[TODO]` Create project repository structure
- `[TODO]` Create experiment log template

### Deliverables for Phase 0

- stable project scope
- stable benchmark definition
- stable directory structure

---

## Phase 1 — dataset acquisition and validation

### 1.1 Download and storage

- `[TODO]` Download official ACDC `training`
- `[TODO]` Download official ACDC `testing`
- `[TODO]` Verify patient counts:
  - 100 in training
  - 50 in testing
- `[TODO]` Record storage locations on server
- `[TODO]` Archive raw dataset path in a config or README

### 1.2 File structure validation

- `[TODO]` Inspect at least 5 random patient folders from training
- `[TODO]` Inspect at least 3 random patient folders from testing
- `[TODO]` Verify presence of:
  - `Info.cfg`
  - `*_4d.nii.gz`
  - `*_frameXX.nii.gz`
  - `*_frameXX_gt.nii.gz`
- `[TODO]` Confirm which frames correspond to ED and ES using `Info.cfg`
- `[TODO]` Confirm that masks match images in shape and orientation

### 1.3 Dataset parsing

- `[TODO]` Write parser to enumerate all patients
- `[TODO]` Write parser to extract all labeled frame paths
- `[TODO]` Build metadata table with:
  - patient ID
  - split source (`training` or `testing`)
  - frame ID
  - phase if available
  - pathology group if available
  - image path
  - mask path

### Deliverables for Phase 1

- reproducible raw-data inventory
- metadata table for all usable labeled volumes
- confidence that dataset parsing is correct

---

## Phase 2 — preprocessing pipeline

### 2.1 Basic preprocessing

- `[TODO]` Implement NIfTI loader
- `[TODO]` Implement image-mask paired loading
- `[TODO]` Implement per-volume normalization
- `[TODO]` Implement resize to `256 x 256`
- `[TODO]` Ensure image interpolation is bilinear
- `[TODO]` Ensure mask interpolation is nearest-neighbor

### 2.2 Slice extraction

- `[TODO]` Convert labeled 3D ED/ES volumes into 2D slices
- `[TODO]` Keep mapping for each slice:
  - patient ID
  - source volume
  - phase
  - slice index
- `[TODO]` Filter obviously empty or invalid slices only if rule is documented
- `[TODO]` Save processed metadata for slice-level dataset access

### 2.3 Label verification

- `[TODO]` Inspect unique class values from masks
- `[TODO]` Freeze class mapping:
  - 0 background
  - 1 RV
  - 2 MYO
  - 3 LV
- `[TODO]` Visualize random samples to verify image-mask alignment

### 2.4 Augmentation pipeline

- `[TODO]` Implement baseline augmentations
- `[TODO]` Freeze the main augmentation policy for benchmark fairness
- `[TODO]` Add augmentation on/off switch through config

### Deliverables for Phase 2

- stable dataset loader
- verified 2D training pipeline
- sample visualization confirming correctness

---

## Phase 3 — split protocol and evaluation preparation

### 3.1 Development split

- `[TODO]` Create fixed patient-level development split inside official training
- `[TODO]` Ensure no patient leakage across train and val
- `[TODO]` Save split file to disk

### 3.2 Cross-validation protocol

- `[TODO]` Design 5-fold patient-level cross-validation split files
- `[TODO]` Save all fold definitions
- `[TODO]` Verify fold balance as much as possible

### 3.3 Holdout usage

- `[TODO]` Define strict rule for official testing usage
- `[TODO]` Write down: no hyperparameter tuning on testing
- `[TODO]` Reserve testing for final report only

### 3.4 Evaluation code

- `[TODO]` Implement Dice
- `[TODO]` Implement IoU
- `[TODO]` Implement HD95
- `[TODO]` Support per-class reporting:
  - RV
  - MYO
  - LV
- `[TODO]` Support mean metric reporting
- `[TODO]` Support patient-volume reconstruction from slice predictions if needed

### Deliverables for Phase 3

- saved split files
- trusted evaluation utilities
- explicit no-leakage policy in code and docs

---

## Phase 4 — baseline model M1: UNetR34

### 4.1 Implementation

- `[TODO]` Implement ResNet34-style encoder
- `[TODO]` Implement standard U-Net decoder
- `[TODO]` Implement segmentation head
- `[TODO]` Create `UNetR34` model wrapper

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

- working baseline architecture
- first scratch vs fine-tune result pair

---

## Phase 5 — strong baseline model M2: AttUNetR34

### 5.1 Implementation

- `[TODO]` Implement attention gate module
- `[TODO]` Integrate attention into skip connections
- `[TODO]` Create `AttUNetR34` wrapper

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

- attention baseline fully benchmarked
- clean comparison with M1

---

## Phase 6 — proposed model M3: BoundaryDSUNetR34

### 6.1 Boundary target generation

- `[TODO]` Choose one boundary-target generation rule
- `[TODO]` Implement boundary map derivation from masks
- `[TODO]` Visualize boundary targets
- `[TODO]` Freeze the rule for the main benchmark

### 6.2 Model implementation

- `[TODO]` Implement boundary head
- `[TODO]` Implement deep supervision heads
- `[TODO]` Create `BoundaryDSUNetR34` wrapper

### 6.3 Loss implementation

- `[TODO]` Implement main segmentation loss
- `[TODO]` Implement boundary loss
- `[TODO]` Implement deep supervision loss
- `[TODO]` Add weighted total-loss config

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

- proposed model fully implemented
- all 6 primary conditions available

---

## Phase 7 — main benchmark table

This is the first thesis-grade milestone.

### Mandatory experiment set

- `[TODO]` `UNetR34-scratch`
- `[TODO]` `UNetR34-finetune`
- `[TODO]` `AttUNetR34-scratch`
- `[TODO]` `AttUNetR34-finetune`
- `[TODO]` `BoundaryDSUNetR34-scratch`
- `[TODO]` `BoundaryDSUNetR34-finetune`

### Mandatory outputs

- `[TODO]` Validation metric table
- `[TODO]` Training-curve plots
- `[TODO]` Parameter count table
- `[TODO]` Inference-time table
- `[TODO]` Qualitative comparison figures

### Required benchmark columns

- `[TODO]` Dice RV
- `[TODO]` Dice MYO
- `[TODO]` Dice LV
- `[TODO]` Mean Dice
- `[TODO]` IoU
- `[TODO]` HD95

### Deliverables for Phase 7

- first stable benchmark table
- clear ranking of architectures and init strategies

---

## Phase 8 — reproducibility strengthening

### 8.1 Re-run policy

- `[TODO]` Fix random seeds
- `[TODO]` Train at least 3 seeds for the main development protocol
- `[TODO]` Compute mean and standard deviation

### 8.2 Cross-validation

- `[TODO]` Run 5-fold cross-validation for the final chosen protocol
- `[TODO]` Aggregate fold metrics
- `[TODO]` Check ranking stability

### 8.3 Logging discipline

- `[TODO]` Save configs for every run
- `[TODO]` Save checkpoints with consistent names
- `[TODO]` Save prediction examples by patient ID
- `[TODO]` Save environment details:
  - CUDA
  - PyTorch
  - package versions

### Deliverables for Phase 8

- reproducible benchmark evidence
- thesis-safe reporting

---

## Phase 9 — ablation studies

Run ablations only after the main benchmark is stable.

### 9.1 Fine-tune strategy ablation

- `[TODO]` Full end-to-end fine-tune from epoch 0
- `[TODO]` Freeze encoder for early epochs then unfreeze
- `[TODO]` Decoder-only warm start if relevant

### 9.2 Data-fraction ablation

- `[TODO]` Train on 25% of training patients
- `[TODO]` Train on 50% of training patients
- `[TODO]` Train on 100% of training patients
- `[TODO]` Compare scratch vs fine-tune under low-data settings

### 9.3 Loss ablation for M3

- `[TODO]` Segmentation loss only
- `[TODO]` Segmentation + boundary loss
- `[TODO]` Segmentation + boundary + deep supervision

### 9.4 Architecture ablation for M3

- `[TODO]` M3 without boundary head
- `[TODO]` M3 without deep supervision
- `[TODO]` Full M3

### Deliverables for Phase 9

- ablation tables
- stronger scientific conclusions

---

## Phase 10 — final holdout evaluation

This phase happens once the main protocol is frozen.

### Holdout execution

- `[TODO]` Retrain best configurations on full official training set
- `[TODO]` Run exactly one clean evaluation on official testing set
- `[TODO]` Save test-set predictions and metrics
- `[TODO]` Lock final benchmark tables for thesis writing

### Rules

- `[TODO]` Do not use holdout results to redesign architecture
- `[TODO]` Do not use holdout results to tune hyperparameters
- `[TODO]` If a rerun is required, record exactly why

### Deliverables for Phase 10

- final thesis-grade holdout results
- locked numbers for thesis and paper draft

---

## Phase 11 — qualitative analysis and error analysis

### Visualization

- `[TODO]` Prepare overlay figures:
  - image
  - ground truth
  - M1 prediction
  - M2 prediction
  - M3 prediction
- `[TODO]` Include both success and failure cases

### Error analysis

- `[TODO]` Analyze which class is hardest
- `[TODO]` Analyze whether MYO boundary improves in M3
- `[TODO]` Analyze typical failure modes:
  - missing contour
  - over-segmentation
  - boundary irregularity
  - small-structure confusion

### Paper-facing outputs

- `[TODO]` Write down 3 to 5 key findings from experiments
- `[TODO]` Identify the cleanest contribution statement

### Deliverables for Phase 11

- qualitative figure set
- error-analysis section material
- paper storyline draft

---

## Phase 12 — thesis and demo support

### Thesis support

- `[TODO]` Prepare final metric tables
- `[TODO]` Prepare architecture diagrams
- `[TODO]` Prepare training protocol summary
- `[TODO]` Prepare dataset and split summary

### Demo support

Demo is secondary. Keep it lightweight.

- `[TODO]` Build a minimal local or web demo
- `[TODO]` Allow model selection
- `[TODO]` Show input slice and predicted mask
- `[TODO]` Show side-by-side comparison between models
- `[TODO]` Do not spend thesis-critical time on UI polish before experiments are stable

### Deliverables for Phase 12

- thesis-ready figures and tables
- optional demo for presentation day

---

## Immediate next actions

These are the next concrete steps to do now.

1. `[TODO]` Download full ACDC training and testing to the server
2. `[TODO]` Build the metadata parser for patient folders and labeled frames
3. `[TODO]` Implement preprocessing and 2D slice extraction
4. `[TODO]` Implement evaluation metrics and patient-level split files
5. `[TODO]` Implement `UNetR34`
6. `[TODO]` Run sanity checks and tiny-subset overfit test
7. `[TODO]` Launch first `UNetR34-scratch` baseline
8. `[TODO]` Freeze the external pretrained source for all fine-tune experiments

---

## Blockers to watch

Use this section to track current blockers.

- `[BLOCKED]` exact pretrained source not yet frozen
- `[BLOCKED]` repository structure not yet created
- `[BLOCKED]` experiment log template not yet created
- `[BLOCKED]` boundary-target generation method not yet frozen

Update this section as soon as blockers are resolved.

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
