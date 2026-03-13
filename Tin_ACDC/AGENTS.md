# AGENTS.md

## Project identity

**Project name:** Cardiac MRI segmentation on ACDC with architecture benchmarking and scratch-vs-fine-tune comparison.

**Primary goal:** Build a rigorous graduation-thesis pipeline for cardiac MRI segmentation on ACDC, benchmark 3 closely related segmentation models, and compare **training from scratch** versus **fine-tuning from external pretrained weights**.

**Secondary goal:** Produce thesis-grade artifacts and a paper-ready experimental story. A lightweight web demo is allowed, but it is always secondary to the AI experiments.

---

## Source-of-truth decisions

These decisions are fixed unless explicitly changed in this file.

1. **Dataset:** Use the official ACDC dataset.
2. **Protocol direction:** Use the official `training` set for model development and the official `testing` set as the final holdout set.
3. **No test leakage:** The `testing` set must never be used for architecture selection, hyperparameter tuning, augmentation tuning, or loss tuning.
4. **Task type:** Multi-class segmentation of cardiac MRI structures from the labeled ED/ES frames.
5. **Default modeling choice:** Start with a **2D slice-wise pipeline**. Only move to 2.5D or 3D after the 2D benchmark is stable and reproducible.
6. **Benchmark scope:** Compare 3 architectures under 2 initialization strategies:
   - scratch
   - fine-tune from **external pretrained weights**
7. **Scientific priority:** Reproducibility, fairness, and clean comparison matter more than chasing a slightly better number with a messy protocol.
8. **UI priority:** Web/demo work must not begin until the benchmark table is stable.

---

## Terminology

Use the following terms consistently:

- **Train from scratch** = random initialization; no external pretrained checkpoint.
- **Fine-tune** = initialize some or all model weights from an **external pretrained source**, then adapt on ACDC.
- **Resume training** = continue training our own prior checkpoint. This is **not** the same as fine-tuning.
- **Main benchmark** = comparisons that keep preprocessing, augmentation, optimization, and splits identical across models.
- **Ablation** = controlled experiment where exactly one design choice is changed.

Do not misuse the word "fine-tune" to describe continuing our own checkpoint.

---

## Official dataset usage plan

### ACDC structure

The official dataset is organized into:

- `training/` with 100 patients
- `testing/` with 50 patients

Each patient folder typically contains:

- `Info.cfg`
- one `*_4d.nii.gz` cine MRI file
- two labeled 3D frames such as:
  - `patientXXX_frameYY.nii.gz`
  - `patientXXX_frameYY_gt.nii.gz`

### What to use for segmentation

For the segmentation task, the default data source is:

- `patientXXX_frameYY.nii.gz`
- `patientXXX_frameYY_gt.nii.gz`

These are the labeled ED/ES frames and are the main inputs for the project.

### What the 4D file is for

`*_4d.nii.gz` is the full cine MRI volume over time. It is useful for:

- sanity-checking ED/ES selection
- future extension to temporal modeling
- data inspection and visualization

It is **not required** for the primary 2D segmentation benchmark.

### Role of `Info.cfg`

Use `Info.cfg` to record:

- which frames correspond to ED and ES
- pathology group if available
- any patient-level metadata useful for stratified reporting

### Dataset policy

- Development happens only on `training/`.
- Final reporting on the official holdout happens on `testing/`.
- Never tune using `testing/` results.

---

## Project-level research questions

The project should answer the following:

### RQ1
How much does architecture choice affect ACDC cardiac segmentation performance under a controlled protocol?

### RQ2
Does fine-tuning from **external pretrained weights** outperform training from scratch on ACDC?

### RQ3
Does pretraining help mainly with:
- final performance,
- convergence speed,
- training stability,
- or low-data robustness?

### RQ4
Does an explicit boundary-aware design improve segmentation quality, especially for harder contours such as myocardium?

These questions are the backbone of the thesis and any paper draft.

---

## Core experimental design

## High-level benchmark matrix

There are **3 architectures** and **2 initialization modes**.

### Initialization modes

- **Scratch**: random initialization for the whole network.
- **Fine-tune**: initialize the encoder from an external pretrained source; decoder/head remain randomly initialized unless stated otherwise.

### Architecture family rule

To make the scratch-vs-fine-tune comparison fair, all 3 models should share the **same encoder skeleton family** whenever possible.

Preferred encoder family:

- **ResNet34-style encoder**

Reason:

- allows clean external pretraining
- widely available
- moderate size
- strong enough for ACDC without being overkill

---

## The 3 models to implement

## Model M1: UNet-R34 baseline

**Name:** `UNetR34`

**Definition:**
- Standard U-Net style decoder
- ResNet34-style encoder
- 4 output classes
- No attention
- No boundary head
- No deep supervision

**Purpose:**
- canonical baseline for the project
- simplest model that still supports fine-tune through encoder initialization

---

## Model M2: Attention UNet-R34

**Name:** `AttUNetR34`

**Definition:**
- same ResNet34-style encoder family as M1
- U-Net decoder
- attention gates on skip connections
- 4 output classes
- no boundary head
- no deep supervision in the primary version

**Purpose:**
- strong architectural baseline
- test whether attention improves localization and suppresses irrelevant regions

---

## Model M3: Boundary-aware UNet-R34 with deep supervision

**Name:** `BoundaryDSUNetR34`

**Definition:**
- same ResNet34-style encoder family as M1/M2
- U-Net decoder
- additional boundary prediction head
- deep supervision on one or more decoder stages
- 4 output classes for main segmentation

**Purpose:**
- proposed model for the thesis
- test whether explicit boundary learning improves contour quality and HD95
- strongest candidate for paper contribution

### Important note

Do not add more novelty than necessary in the first stable version.
The first stable M3 should remain interpretable and defendable:

- boundary head
- deep supervision
- optionally tuned boundary-aware loss

Do **not** stack too many ideas at once in the first paper-oriented version.

---

## Primary experiment table

The main benchmark has 6 conditions:

1. `UNetR34-scratch`
2. `UNetR34-finetune`
3. `AttUNetR34-scratch`
4. `AttUNetR34-finetune`
5. `BoundaryDSUNetR34-scratch`
6. `BoundaryDSUNetR34-finetune`

This 6-condition benchmark is mandatory.

---

## Fine-tuning policy

### What counts as valid external pretraining

Acceptable primary source:

- public ImageNet-pretrained encoder weights with clear provenance

Acceptable secondary source only if fully documented:

- external medically pretrained weights with public citation, license, and exact dataset origin

### Default fine-tune protocol

- initialize encoder from external pretrained weights
- initialize decoder, segmentation head, boundary head, and deep-supervision heads randomly
- train end-to-end from epoch 0

### Optional fine-tune ablation

If time permits, compare:

- full end-to-end fine-tuning from epoch 0
- freeze encoder for the first 5-10 epochs, then unfreeze

### What not to do

- Do not mix multiple pretrained sources in the primary benchmark table.
- Do not compare a scratch model against a fine-tuned model with a much larger encoder and call it fair.
- Do not use foundation models or massive segmentation backbones unless the project scope is formally revised.

---

## Data split and evaluation protocol

## Main protocol

### Training set usage

Use the official `training/` data only for development.

Preferred protocol:

- **5-fold cross-validation inside the 100-patient training set** for model selection and stability reporting

Final reporting protocol:

1. choose the best configuration using only the official training set
2. retrain that configuration on the full 100-patient training set
3. evaluate **once** on the official 50-patient testing set

### Fast development protocol

Allowed during early implementation only:

- fixed patient-level split inside `training/`
  - e.g. 80 train / 20 val

But the thesis-grade final benchmark should use either:

- 5-fold cross-validation on training plus final test holdout
- or at minimum 3 independent seeds on a fixed patient-level split plus final test holdout

### Patient-level rule

All splits must be done at the **patient level**, never at the slice level.

This is non-negotiable.

---

## Data preprocessing protocol

The preprocessing pipeline must be identical across all models in the main benchmark.

## Step 1: load labeled frames

For each patient:

- identify the two labeled frames
- load the image volume and corresponding ground-truth volume
- log frame names, patient ID, and phase information

## Step 2: orientation and metadata sanity checks

Before training:

- verify image/mask shape match
- verify affine/orientation consistency where possible
- inspect random samples visually

## Step 3: normalization

Default normalization for MRI:

- per-volume z-score normalization

Recommended implementation:

- optionally clip extreme intensities first
- compute mean and std on non-zero voxels if appropriate
- apply the same normalization policy to all models

If normalization is changed, it must be documented as an ablation, not silently changed.

## Step 4: spatial standardization

Preferred baseline:

- resize slices to `256 x 256`

Interpolation rules:

- images: bilinear
- masks: nearest-neighbor

If spacing-aware resampling is later adopted, it must be applied identically across all models and documented as a protocol change.

## Step 5: slice extraction

Default task setup:

- 2D slice-wise training on the labeled 3D ED/ES volumes

Keep a mapping from each slice back to:

- patient ID
- phase (ED or ES)
- original volume index
- slice index

This mapping is required for patient-level reconstruction and evaluation.

## Step 6: class verification

During dataset parsing:

- inspect unique mask values
- confirm class IDs in the data
- log the label mapping in code and thesis

Do not assume class order blindly; verify it once and freeze it.

---

## Recommended augmentations

Use the same augmentation policy for all models in the primary benchmark.

Recommended training augmentations:

- random rotation within a small range
- random scaling / zoom
- random translation or crop jitter
- horizontal or vertical flips only if confirmed anatomically valid for the chosen orientation pipeline
- mild elastic deformation if stable
- mild intensity shift / contrast jitter
- mild Gaussian noise

Rules:

- do not make augmentation so strong that anatomy becomes unrealistic
- do not change augmentation policy between models in the main benchmark
- augmentation changes belong in ablation experiments

---

## Loss functions

## Primary segmentation loss for M1 and M2

Use a combined region loss:

- `DiceLoss + CrossEntropyLoss`

This is the default safe baseline.

## Primary loss for M3

Use:

- main segmentation loss: `DiceLoss + CrossEntropyLoss`
- boundary loss for the boundary head
- weighted sum of main and auxiliary terms

Recommended formulation:

`TotalLoss = SegLoss + lambda_b * BoundaryLoss + lambda_ds * DeepSupervisionLoss`

Where:

- `SegLoss` is the main multi-class loss
- `BoundaryLoss` can be BCE, Dice-on-boundary, or another documented boundary-aware loss
- `DeepSupervisionLoss` is applied to auxiliary outputs if enabled

### Boundary-loss policy

Keep the first version simple and reproducible.
Do not switch among many exotic boundary losses unless it is part of a clearly documented ablation.

Start with one of the following and keep it fixed:

- BCE on a derived boundary map
- Dice on a derived boundary map
- a clearly cited boundary-aware loss

---

## Optimizer and training defaults

These are the project defaults unless a controlled ablation says otherwise.

- optimizer: `AdamW`
- mixed precision: enabled
- gradient clipping: allowed if instability appears
- scheduler: cosine annealing or ReduceLROnPlateau
- early stopping: enabled on validation metric
- checkpoint selection: best validation `Mean Dice`

### Suggested initial hyperparameters

These are starting points, not immutable rules:

- image size: `256 x 256`
- batch size: as large as stable GPU memory allows
- epochs: 100-200 for first-pass runs
- scratch learning rate: around `1e-3`
- fine-tune learning rate:
  - encoder lower than decoder
  - e.g. encoder `1e-4`, decoder/head `1e-3`
- weight decay: around `1e-4`

If learning rates are changed, keep them documented in the experiment registry.

---

## Evaluation protocol

## Required metrics

Report at minimum:

- Dice for RV
- Dice for MYO
- Dice for LV
- mean Dice across foreground classes
- IoU / Jaccard
- HD95

### Strong recommendation

Compute metrics at the **patient-volume level**, not only slice level.

That means:

1. predict all slices of a labeled volume
2. reconstruct the 3D prediction volume
3. compute metrics against the full 3D ground truth

This is much more defensible for thesis and paper writing.

## Optional metrics

If time permits, also report:

- ASSD
- number of parameters
- inference time per volume
- training time to convergence

## Optional clinical metrics

If resources allow, derive and report:

- LV volume estimates
- RV volume estimates
- ejection fraction related estimates

These are not mandatory for the first stable thesis version, but they strengthen the medical relevance.

---

## Statistical reporting

At thesis/paper level, do not report a single lucky run if avoidable.

Preferred reporting:

- mean ± std across folds
- and/or mean ± std across 3 random seeds

If comparing two strong models, optionally include:

- paired statistical tests on patient-level metrics

The goal is to support claims with stable evidence, not one-off numbers.

---

## Mandatory fairness rules for the main benchmark

The following must be held constant across models in the main benchmark:

- data split
- preprocessing
- image size
- augmentation policy
- optimizer family
- epoch budget
- early stopping rule
- evaluation code
- metric definitions

Only architecture and initialization strategy should change.

If another factor changes, it is no longer the main benchmark and must be labeled as an ablation.

---

## Experiment registry requirements

Every experiment must log at least the following:

- experiment ID
- date
- git commit hash
- model name
- initialization mode
- pretrained source if used
- split or fold
- seed
- image size
- loss config
- optimizer config
- augmentation config
- best validation metrics
- final test metrics
- notes on failures / anomalies

If an experiment cannot be reproduced from logs and config, it does not count as a valid result.

---

## Minimum directory structure

Use a project layout close to this:

```text
project/
  AGENTS.md
  README.md
  data/
    raw/
      acdc/
        training/
        testing/
    processed/
      2d_slices/
      metadata/
  configs/
    data/
    model/
    train/
    experiment/
  src/
    data/
      parse_acdc.py
      build_splits.py
      dataset_2d.py
      transforms.py
    models/
      unet_r34.py
      att_unet_r34.py
      boundary_ds_unet_r34.py
    losses/
      dice_ce.py
      boundary_losses.py
    train/
      engine.py
      trainer.py
      callbacks.py
    eval/
      metrics.py
      reconstruct_volume.py
      evaluate.py
    utils/
      io.py
      seed.py
      logging.py
      visualization.py
  experiments/
    registry.csv
    notes.md
  outputs/
    checkpoints/
    logs/
    predictions/
    figures/
    tables/
  app/
```

This does not need to be followed byte-for-byte, but the separation of concerns should remain.

---

## Order of implementation

Follow this order unless there is a compelling reason not to.

## Phase 1: dataset and evaluation foundation

1. parse ACDC reliably
2. verify patient folder structure
3. load labeled frames and masks
4. create patient-level split logic
5. implement 2D slice dataset
6. implement volume reconstruction for evaluation
7. implement Dice and HD95 correctly
8. save qualitative visualizations

Do not build fancy models before this phase is correct.

## Phase 2: baseline model

1. implement `UNetR34`
2. run a tiny overfit test on a few samples
3. run a short sanity training
4. verify masks visually
5. produce first baseline table

## Phase 3: stronger baselines

1. implement `AttUNetR34`
2. benchmark scratch and fine-tune
3. compare against `UNetR34`

## Phase 4: proposed model

1. implement `BoundaryDSUNetR34`
2. verify boundary-target generation
3. benchmark scratch and fine-tune
4. confirm that any gain is real and not from accidental protocol drift

## Phase 5: ablations

Only after the main benchmark is stable:

- boundary loss choice
- deep supervision on/off
- freeze/unfreeze schedule in fine-tuning
- low-data comparison
- augmentation sensitivity

## Phase 6: thesis/paper artifacts

- final benchmark tables
- qualitative figures
- convergence curves
- ablation tables
- error analysis
- final holdout test results

## Phase 7: lightweight web demo

Only after all above are stable.

---

## Required sanity checks

Before trusting any metric, run these checks:

### Loader sanity

- image and mask shapes match
- masks are not empty for all samples
- labels are in expected set
- ED/ES sample pairs are correctly loaded

### Training sanity

- model can overfit a tiny subset
- loss decreases on a tiny subset
- predictions align spatially with masks

### Evaluation sanity

- reconstruct a full patient volume correctly
- verify one patient manually with saved overlays
- test metric code on trivial examples

---

## Error analysis requirements

A thesis-grade project must analyze failures, not only report averages.

Required analyses:

- per-class performance: RV, MYO, LV
- best-case and worst-case patient examples
- qualitative failure modes at apical / basal slices if visible
- contour quality differences, especially for myocardium
- scratch vs fine-tune convergence behavior

Optional stronger analyses:

- by pathology subgroup if metadata is reliable
- by ED vs ES phase
- by slice position category

---

## Low-data study (recommended for publishability)

A strong optional study is to compare scratch vs fine-tune under reduced training data.

Recommended subsets inside official training data:

- 25%
- 50%
- 100%

This can answer whether external pretraining helps more when labeled cardiac MRI data is limited.

If time permits, prioritize this study after the main benchmark.

---

## What counts as a publishable contribution here

The project does **not** need to invent a radically new architecture.
A publishable student-level contribution can come from a clean combination of:

- strong benchmark protocol
- careful scratch-vs-fine-tune study
- well-motivated proposed model
- good ablation design
- patient-level evaluation
- insightful error analysis

The likely paper story is:

1. establish a clean ACDC benchmark with 3 related U-Net-family models
2. compare scratch vs external fine-tuning under the same protocol
3. show whether a boundary-aware design improves contour quality
4. analyze where and why the gains happen

---

## Out-of-scope items unless the project is formally revised

Do not expand into these by default:

- diffusion models
- large vision transformers as the main model family
- SAM / MedSAM / foundation-model-centered pipelines
- multimodal report generation
- temporal cine modeling from the full 4D sequence
- federated learning
- production-grade deployment work

These can distract from the core thesis.

---

## Deliverables that must exist before thesis writing finishes

1. reproducible training code
2. reproducible evaluation code
3. 6-condition main benchmark table
4. at least 1 ablation table
5. qualitative visualization figure set
6. final testing-set results
7. thesis-ready method section diagrams
8. experiment registry with config traceability

Optional but valuable:

9. lightweight web demo
10. conference-style paper draft

---

## Writing rules for future agents

If you are another AI agent working on this project, follow these rules:

1. Do not silently change the protocol.
2. Do not use the official testing set for tuning.
3. Do not compare models trained under different augmentation or resize settings and call it fair.
4. Do not add extra modules to M3 without documenting them as a new variant.
5. Do not delete failed experiments from the registry; mark them as failed and explain why.
6. Do not optimize the web app before the benchmark is done.
7. Do not report slice-level metrics alone as the main result.
8. Do not claim a contribution that is not backed by an ablation or a controlled comparison.

---

## Suggested benchmark tables

## Table A: Main benchmark

| Model | Init | Dice RV | Dice MYO | Dice LV | Mean Dice | HD95 |
|------|------|---------|----------|---------|-----------|------|
| UNetR34 | Scratch |  |  |  |  |  |
| UNetR34 | Fine-tune |  |  |  |  |  |
| AttUNetR34 | Scratch |  |  |  |  |  |
| AttUNetR34 | Fine-tune |  |  |  |  |  |
| BoundaryDSUNetR34 | Scratch |  |  |  |  |  |
| BoundaryDSUNetR34 | Fine-tune |  |  |  |  |  |

## Table B: Efficiency

| Model | Init | Params | Best Epoch | Train Time | Inference Time |
|------|------|--------|------------|------------|----------------|
| UNetR34 | Scratch |  |  |  |  |
| UNetR34 | Fine-tune |  |  |  |  |
| AttUNetR34 | Scratch |  |  |  |  |
| AttUNetR34 | Fine-tune |  |  |  |  |
| BoundaryDSUNetR34 | Scratch |  |  |  |  |
| BoundaryDSUNetR34 | Fine-tune |  |  |  |  |

## Table C: Ablation for M3

| Variant | Boundary Head | Deep Supervision | Boundary Loss | Init | Mean Dice | HD95 |
|--------|----------------|------------------|---------------|------|-----------|------|
| M3-A | No | No | No | Scratch |  |  |
| M3-B | Yes | No | Yes | Scratch |  |  |
| M3-C | Yes | Yes | Yes | Scratch |  |  |
| M3-D | Yes | Yes | Yes | Fine-tune |  |  |

---

## Minimal milestone plan

### Milestone 1
Dataset parser, split logic, 2D loader, visualization, metrics.

### Milestone 2
`UNetR34` scratch and fine-tune working end-to-end.

### Milestone 3
`AttUNetR34` scratch and fine-tune benchmarked.

### Milestone 4
`BoundaryDSUNetR34` scratch and fine-tune benchmarked.

### Milestone 5
Ablations and error analysis complete.

### Milestone 6
Final test results, thesis tables, figures, and paper outline complete.

---

## Key references to keep in mind

### Dataset and benchmark
- Official ACDC challenge pages
- ACDC challenge paper: Bernard et al., IEEE TMI 2018

### Baseline architecture
- U-Net: Ronneberger et al., 2015

### Strong benchmarking culture
- nnU-Net: Isensee et al., Nature Methods 2021

### Attention baseline inspiration
- Attention U-Net: Oktay et al., 2018

### Boundary-aware design inspiration
- boundary-aware or boundary-loss literature for medical segmentation

Keep exact citations in the thesis bibliography, not only here.

---

## Final project philosophy

This project should look like a real AI thesis, not a demo-first software project.

The order of importance is:

1. correct data protocol
2. fair benchmark design
3. reproducible training and evaluation
4. defensible scientific conclusions
5. optional demo polish

If there is any conflict between a prettier demo and a cleaner experiment, choose the cleaner experiment.
