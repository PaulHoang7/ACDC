# architecture.md

## Purpose

This document defines the model architecture plan for the ACDC cardiac MRI segmentation project.

It is subordinate to `AGENTS.md`. If there is any conflict, `AGENTS.md` wins.

This file exists to prevent architecture drift during implementation, training, and benchmarking.

---

## Project modeling goal

Build and benchmark **3 closely related segmentation architectures** on ACDC under **2 initialization strategies**:

- `scratch`
- `fine-tune` from **external pretrained weights**

The objective is not to maximize novelty at all costs. The objective is to produce a **fair, reproducible, thesis-grade benchmark** and a **defensible proposed model**.

---

## Global architecture policy

### Fixed modeling rules

1. All models must solve the **same task**:
   - multi-class segmentation of cardiac MRI structures on ACDC
   - default pipeline is **2D slice-wise**
   - input is a single MRI slice
   - output is a 4-channel segmentation logit map

2. All models should stay in the **same family**:
   - U-Net-style encoder-decoder
   - same encoder backbone family whenever possible
   - differences between models must be intentional and interpretable

3. The primary comparison must be fair:
   - same preprocessing
   - same augmentations
   - same optimizer family
   - same training budget
   - same split policy
   - same evaluation protocol

4. The proposed model must remain **simple enough to defend**:
   - no stacking many unrelated ideas in the first stable version
   - prefer one clean contribution over many weak ones

---

## Output classes

Use a 4-class output convention:

- class `0`: background
- class `1`: right ventricle (RV)
- class `2`: myocardium (MYO)
- class `3`: left ventricle (LV)

This mapping must be verified against parsed mask values and then frozen in code.

---

## Input specification

### Default input

- single 2D MRI slice
- shape after preprocessing: `1 x 256 x 256`

### Default output

- segmentation logits: `4 x 256 x 256`

### Optional auxiliary outputs

Only for the proposed model:

- boundary logits: `1 x 256 x 256`
- deep supervision logits at decoder stages if enabled

---

## Shared encoder family

### Default backbone family

Use a **ResNet34-style encoder** for all 3 models.

Reason:

- supports scratch training cleanly
- supports external pretrained initialization cleanly
- moderate parameter count
- familiar and defensible
- strong enough without being excessive

### Encoder stage structure

Preferred 5-stage hierarchy:

- stem
- stage 1
- stage 2
- stage 3
- stage 4

Typical feature widths:

- stem / early block: `64`
- stage 1: `64`
- stage 2: `128`
- stage 3: `256`
- stage 4: `512`

You may adapt the exact implementation details, but the feature hierarchy should remain consistent across models unless explicitly documented.

---

## Shared decoder policy

All models use a U-Net-like decoder with:

- progressive upsampling
- skip connections from encoder stages
- feature fusion after concatenation or gated fusion
- final segmentation head projecting to 4 classes

Preferred decoder widths:

- `256`
- `128`
- `64`
- `32`

These may be adjusted if memory or stability issues appear, but keep them identical across M1/M2/M3 unless the difference is the point of the experiment.

---

## Initialization modes

## Mode A: Scratch

Definition:

- all weights randomly initialized
- no external pretrained checkpoint
- standard initialization such as He/Kaiming for conv layers is allowed

Label for experiments:

- `scratch`

## Mode B: Fine-tune

Definition:

- initialize the encoder from an **external pretrained checkpoint**
- decoder and task heads are randomly initialized unless otherwise documented
- train on ACDC end-to-end

Label for experiments:

- `finetune`

### Valid pretrained source

Primary default:

- public ImageNet-pretrained ResNet34 weights with clear provenance

### Fine-tune fairness rule

If M1, M2, and M3 all use a ResNet34-style encoder, then all 3 fine-tune variants must use the **same pretrained source**.

Do not mix ImageNet for one model and a medical pretrained encoder for another in the primary benchmark.

---

## Model M1: UNetR34

### Name

`UNetR34`

### Role

Primary baseline.

### Architectural definition

- ResNet34-style encoder
- standard U-Net decoder
- plain skip connections
- no attention gates
- no boundary head
- no deep supervision

### Purpose

- establish the baseline performance floor
- establish the baseline scratch vs fine-tune effect
- validate the end-to-end pipeline

### Forward outputs

- `seg_logits`

### Notes

This model should be kept simple. Do not quietly add optional modules here.

---

## Model M2: AttUNetR34

### Name

`AttUNetR34`

### Role

Strong architectural baseline.

### Architectural definition

- same ResNet34-style encoder family as M1
- U-Net decoder
- attention gates applied to skip connections before fusion
- no boundary head
- no deep supervision in the primary version

### Purpose

- evaluate whether attention improves focus on relevant anatomical structures
- test whether attention provides gains over the plain skip-connection baseline

### Forward outputs

- `seg_logits`

### Notes

Attention should be inserted only at skip fusion points. Avoid adding unrelated extra modules in the first stable version.

---

## Model M3: BoundaryDSUNetR34

### Name

`BoundaryDSUNetR34`

### Role

Proposed thesis model.

### Architectural definition

- same ResNet34-style encoder family as M1 and M2
- U-Net decoder
- segmentation head for 4 classes
- auxiliary boundary prediction head
- deep supervision on one or more decoder levels

### Purpose

- improve contour quality, especially on difficult boundaries
- improve HD95 and boundary-sensitive behavior
- provide the clearest paper-worthy contribution

### Forward outputs

- `seg_logits`
- `boundary_logits`
- optional `aux_seg_logits_stage_k`

### Design intent

This model tests the hypothesis that explicit boundary guidance plus deep supervision improves cardiac structure segmentation beyond plain and attention-only U-Net variants.

### Important restraint

Do not add attention gates to M3 in the first stable benchmark unless this is explicitly planned as a later ablation.
The initial M3 should stay conceptually focused on:

- boundary-aware learning
- deep supervision

---

## Decoder fusion policy

### M1

- concatenate upsampled decoder feature with encoder skip
- conv fusion block

### M2

- apply attention gate to encoder skip
- concatenate gated skip with upsampled decoder feature
- conv fusion block

### M3

- same basic skip fusion as M1 unless otherwise specified
- decoder features also feed:
  - segmentation output
  - auxiliary deep supervision outputs
  - boundary head pathway

---

## Convolution block policy

Use one block style and keep it consistent.

Preferred block for decoder and non-backbone layers:

- `Conv -> Norm -> ReLU`
- repeated twice per fusion block

Recommended normalization:

- BatchNorm or InstanceNorm

Choose one and keep it fixed across the primary benchmark.

---

## Upsampling policy

Preferred default:

- bilinear upsampling
- followed by convolutional fusion

Acceptable alternative:

- transposed convolution

Use one default choice for all 3 models in the main benchmark.

Do not compare models where the only hidden difference is different upsampling operators unless you are running an explicit ablation.

---

## Boundary head policy for M3

### Goal

Teach the model to explicitly predict contour regions so that segmentation learning pays more attention to hard boundaries.

### Input source

Preferred:

- last decoder feature map
- optionally fused with an intermediate decoder feature if needed

### Output

- single-channel boundary probability map

### Boundary target generation

Derived from the segmentation mask during preprocessing or in the data pipeline.

Possible method:

- morphological edge extraction from each class or from the combined foreground mask

The exact target-generation rule must be documented in code and frozen once selected.

### Keep it simple

Use one stable boundary-target generation rule for the main benchmark.
If you want to try more than one method, do it only as a later ablation.

---

## Deep supervision policy for M3

### Goal

Improve gradient flow and encourage semantically meaningful intermediate decoder outputs.

### Default usage

- attach auxiliary segmentation heads to 1 or 2 decoder stages
- upsample auxiliary logits to full resolution for loss computation

### Loss policy

Auxiliary losses should be weighted less than the main segmentation loss.

Do not over-weight deep supervision.

---

## Recommended loss mapping

### M1 and M2

Primary loss:

- `DiceLoss + CrossEntropyLoss`

### M3

Primary total loss:

- `SegLoss + lambda_b * BoundaryLoss + lambda_ds * DeepSupervisionLoss`

Where:

- `SegLoss = DiceLoss + CrossEntropyLoss`
- `BoundaryLoss` is a simple documented loss such as BCE or Dice on boundary maps
- `DeepSupervisionLoss` is applied to auxiliary segmentation heads

### Initial weighting suggestion

A safe starting point:

- `lambda_b = 0.2`
- `lambda_ds = 0.2`

These are starting values, not fixed truths. Tune only on the training-development protocol, never on official test results.

---

## Parameter-count policy

The models should remain in a similar complexity range.

Target expectation:

- M1 is the lightest
- M2 slightly heavier because of attention
- M3 slightly heavier because of boundary head and deep supervision

Do not let one model become dramatically larger and then claim gains are only due to architecture type.

If parameter counts differ meaningfully, report them explicitly.

---

## Experiment naming convention

Use consistent names in code, logs, and tables.

### Model IDs

- `m1_unetr34`
- `m2_attunetr34`
- `m3_boundarydsunetr34`

### Initialization IDs

- `scratch`
- `finetune`

### Full run IDs

Examples:

- `m1_unetr34_scratch_seed42`
- `m1_unetr34_finetune_seed42`
- `m2_attunetr34_scratch_seed42`
- `m3_boundarydsunetr34_finetune_seed42`

---

## Minimal implementation checklist

Before large-scale training, all three architectures must pass:

1. forward pass on a dummy tensor
2. output shape check
3. loss computation without crash
4. backward pass without NaN
5. overfit-on-small-batch sanity test
6. visualization of one sample prediction

If a model fails any of these, do not launch long training.

---

## Implementation modules

Recommended code structure:

```text
src/
  models/
    backbones/
      resnet34_encoder.py
    decoders/
      unet_decoder.py
    heads/
      seg_head.py
      boundary_head.py
      deep_supervision_head.py
    unetr34.py
    attunetr34.py
    boundary_dsunetr34.py
```

### Design principle

Reuse components when possible.

Do not implement each model as a completely separate codebase if the shared pieces can be factored cleanly.

---

## What counts as a protocol change

Any of the following requires explicit logging and documentation:

- changing input resolution
- changing encoder family
- changing normalization type
- changing augmentation policy
- changing pretrained source
- changing upsampling operator
- changing loss weights
- changing boundary target generation method

Silent changes are not allowed.

---

## What not to do

- do not mix 2D and 3D models in the primary benchmark
- do not compare different encoder families in the primary table unless the benchmark is redesigned
- do not stack attention, boundary, transformer blocks, and extra losses all at once in the first stable version
- do not hide architecture changes inside helper functions
- do not use the official testing set to decide architectural design

---

## Recommended development order

1. implement `UNetR34`
2. verify full training pipeline
3. implement `AttUNetR34`
4. verify benchmark fairness
5. implement `BoundaryDSUNetR34`
6. verify boundary target generation and auxiliary losses
7. launch full benchmark
8. run ablations only after the main 6-condition benchmark is stable

---

## Thesis-facing architecture summary

The thesis should describe the architecture story as:

1. a plain U-Net-style baseline with ResNet34 encoder
2. an attention-enhanced variant to test skip-connection refinement
3. a proposed boundary-aware deeply supervised variant to improve contour quality

This framing is clean, defendable, and suitable for both thesis writing and paper drafting.
