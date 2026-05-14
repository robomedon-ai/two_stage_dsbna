# DSBANet with Two-Stage Cascade for Prostate Segmentation

Two-stage cascade extension of DSBANet for multi-class prostate MR
segmentation (PZ / CG / Tumor) on the **Prostate158** dataset. Built to
address the major-revision concern that the published 3D tumor DSC was
near zero on a small (n=19) test set.

The cascade decouples *where the prostate is* from *what is inside it*:

1. **Stage 1 — `ProstateROINet`**: binary 3D segmentation of the whole
   prostate gland. Its sole job is to localise the gland and produce a
   bounding box.
2. **Stage 2 — `TumorSegNet`**: 4-class segmentation
   (background / PZ / CG / Tumor) restricted to the Stage 1 bounding box,
   with class-weighted loss and case-level tumour oversampling.

The same two-stage scaffold works with two interchangeable Stage 2
backbones (`dsba_net_3d`, `unet3d`) and an optional multimodal input
(T2 + ADC + DWI on the registered Prostate158 grid).


## Why a cascade

A single 3D network trying to segment four classes on the entire
volume has to:
- localise the prostate inside a much larger box,
- separate PZ from CG,
- separate tumour from the rest,
- all under heavy class imbalance (tumour voxels are < 0.1 % of the
  whole volume).

With 19 test cases this is hard to learn reliably and the published
DSBANet 3D tumour DSC was 0.003. Cropping to the Stage 1 prostate ROI
removes ≈ 90 % of the input voxels, fixes the FOV, and lets Stage 2
spend its capacity on the three foreground classes only.

The expected (and observed) benefit is concentrated on the rare class:
PZ and CG were already learnable on the full volume; tumour was not.


## Architecture

```
                       full T2 (and optionally ADC, DWI) volume
                                       |
                                       v
                      +-------------------------------+
                      |   Stage 1 : ProstateROINet    |
                      |   (binary 3D U-Net /           |
                      |    DSBA-Net 3D, 1 channel out) |
                      +---------------+---------------+
                                      |
                       largest connected component
                                      |
                              tight bbox + margin
                                      |
                                      v
                      +-------------------------------+
                      |   Stage 2 : TumorSegNet       |
                      |   4-class on cropped ROI       |
                      |   (DSBA-Net 3D or U-Net 3D)    |
                      +---------------+---------------+
                                      |
                              paste back to full
                                      |
                                      v
                               final segmentation
```

Key implementation details (all in `src/cascade.py` and
`src/models/cascade_unet3d.py`):

- **ROI bbox** (`compute_roi_bbox`) — largest connected component of
  the binary mask, asymmetric margin `(2, 8, 8)` voxels (z, y, x),
  centre-expand to a minimum size of `(24, 96, 96)`, volume-bound clamp.
- **Stage 2 input volume** — resampled to `(48, 128, 128)`.
- **Training-time jitter** — Stage 2 sees bbox-jittered crops (±5
  voxels per axis) so it does not depend on a single ROI definition.
- **Paste-back** — Stage 2 logits are upsampled to the bbox size and
  written back into a full-volume tensor; background (class 0)
  everywhere outside the bbox.
- **Loss** — class-weighted CE + Dice with a high tumour weight
  (default `cascade_stage2_tumor_weight = 10.0`) and case-level
  tumour-positive oversampling (`cascade_oversample_factor = 3.0`).


## Multimodal input

When `--multimodal` is passed:

- The dataset loader stacks T2 + ADC + DWI as channels (already
  pre-registered on the Prostate158 grid), giving an input tensor of
  shape `(M=3, D, H, W)`.
- Each modality is Z-score normalised independently.
- Both Stage 1 and Stage 2 are trained with `in_channels = 3`.
- Output paths are routed under `output/prostate158/multimodal/...`
  so that T2-only and multimodal runs do not overwrite each other.

The ADC/DWI signal is most informative for the tumour sub-class — see
the results below.


## Running the cascade

All commands run from this directory.

### 1. Train Stage 1

```bash
python -u main.py --mode cascade_stage1 \
  --cascade_arch dsba_net_3d \
  --dataset prostate158 \
  --epochs 100
```

Add `--multimodal` to train the 3-channel variant.

### 2. Predict prostate bboxes for val + test

```bash
python -u main.py --mode cascade_predict_bboxes \
  --cascade_arch dsba_net_3d \
  --dataset prostate158
```

Streams one case at a time (memory-bounded) and writes
`cascade_predicted_bboxes_{val,test}.json`. Add `--multimodal` if
Stage 1 was multimodal.

### 3. Train Stage 2

```bash
python -u main.py --mode cascade_stage2 \
  --cascade_arch dsba_net_3d \
  --dataset prostate158 \
  --epochs 120
```

### 4. End-to-end evaluation

Run Stage 1 → bbox → Stage 2 → paste-back and report per-case DSC:

```bash
# real pipeline (predicted bbox)
python -u main.py --mode cascade_evaluate \
  --cascade_arch dsba_net_3d \
  --bbox_source predicted \
  --dataset prostate158

# Stage 2 upper bound (oracle bbox)
python -u main.py --mode cascade_evaluate \
  --cascade_arch dsba_net_3d \
  --bbox_source gt \
  --dataset prostate158

# no-cascade sanity check (Stage 2 on the full volume)
python -u main.py --mode cascade_evaluate \
  --cascade_arch dsba_net_3d \
  --bbox_source full \
  --dataset prostate158
```

Per-case DSC is dumped to
`output/prostate158/cascade/stage2/per_case_dsc_cascade_*.json`.

### 5. Save NIfTI predictions

```bash
python -u main.py --mode cascade_save_nifti \
  --cascade_arch dsba_net_3d \
  --dataset prostate158
```


## Configuration knobs

Everything lives in [`src/config.py`](src/config.py) under the
`# Two-stage cascade` section:

| Key | Default | What it does |
|---|---|---|
| `cascade_mode` | `"off"` | `"stage1"` / `"stage2"` switch routing in the trainer. |
| `cascade_arch` | `"dsba_net_3d"` | `"dsba_net_3d"` or `"unet3d"`. |
| `cascade_stage2_volume_size` | `(48, 128, 128)` | Stage 2 input size after ROI resampling. |
| `cascade_bbox_margin_voxels` | `(2, 8, 8)` | Symmetric padding around the prostate bbox. |
| `cascade_min_bbox_size` | `(24, 96, 96)` | Centre-expand the bbox to at least this. |
| `cascade_bbox_jitter_voxels` | `5` | Training-time bbox jitter. |
| `cascade_stage1_loss` | `"combined"` | Binary Stage 1 loss. |
| `cascade_stage2_tumor_weight` | `10.0` | Tumor class weight in Stage 2 CE / Dice. |
| `cascade_oversample_factor` | `3.0` | Case-level oversampling of tumour-positive volumes. |
| `multimodal` | `False` | Stack T2 + ADC + DWI as input channels. |
| `modalities` | `("t2", "adc", "dwi")` | Modality order for multimodal input. |


## Results (Prostate158, test set, n=19, per-case DSC)

Cascade comparison with DSBA-Net 3D as Stage 2 backbone:

| Configuration | PZ | CG | Tumor | Tumor non-zero |
|---|---|---|---|---|
| Single-model 3D (paper baseline) | 0.81 | 0.65 | 0.003 | 1 / 19 |
| T2 cascade (DSBA-Net 3D Stage 2)        | 0.830 | 0.656 | 0.298 | 15 / 19 |
| T2 cascade (U-Net 3D Stage 2)           | 0.828 | 0.658 | 0.289 | 15 / 19 |
| **Multimodal cascade (DSBA-Net 3D)**    | **0.821** | **0.640** | **0.433** | **18 / 19** |
| Multimodal cascade — oracle bbox (UB)   | 0.833 | 0.660 | 0.445 | 18 / 19 |
| Multimodal Stage 2 on full volume (no cascade) | 0.557 | 0.477 | 0.198 | 18 / 19 |

Observations:

- **Cascade is the larger lever** for tumour: +0.235 DSC over the
  no-cascade full-volume baseline at the same modalities
  (0.198 → 0.433).
- **Multimodal adds further +0.135 tumour DSC** over the T2 cascade
  (0.298 → 0.433), and brings the number of non-zero-tumour cases
  from 15 / 19 to 18 / 19.
- **Stage 1 is essentially solved**: replacing the predicted bbox
  with the GT bbox costs only ≈ 0.012 tumour DSC.
- PZ and CG are saturated under the cascade — improvements there
  would have to come from a larger labelled set, not from
  architecture or modality.

See [`output/prostate158/figures/cascade3d_boxplots_v2.png`](../output/prostate158/figures/cascade3d_boxplots_v2.png)
for a per-case DSC boxplot of these conditions.


## File map

```
dsbanet_with_cascade/
├── main.py                          # entry point, dispatches all cascade modes
├── src/
│   ├── cascade.py                   # pure-NumPy bbox + paste-back utilities
│   ├── config.py                    # Config dataclass (cascade keys here)
│   ├── dataset_prostate158.py       # Stage 1/2 datasets, multimodal-aware
│   ├── losses.py                    # Dice / CE / Focal / Tversky / Focal-Tversky
│   ├── train.py                     # auto-resume training loop
│   ├── evaluate.py                  # per-case and per-slice metrics
│   └── models/
│       ├── cascade_unet3d.py        # CascadeUNet3D (Stage 1 + Stage 2 wrapper)
│       ├── dsba_net3d.py            # DSBA-Net 3D
│       ├── unet3d.py                # U-Net 3D
│       └── ...                      # other 3D baselines
└── build_ablation_report.py         # effect-size tables for the paper
```


## Reproducing the headline numbers

Multimodal DSBA-Net 3D cascade, end-to-end on the test set:

```bash
python -u main.py --mode cascade_stage1   --cascade_arch dsba_net_3d --multimodal --epochs 100 --dataset prostate158
python -u main.py --mode cascade_predict_bboxes --cascade_arch dsba_net_3d --multimodal --dataset prostate158
python -u main.py --mode cascade_stage2   --cascade_arch dsba_net_3d --multimodal --epochs 120 --dataset prostate158
python -u main.py --mode cascade_evaluate --cascade_arch dsba_net_3d --multimodal --bbox_source predicted --dataset prostate158
```

## Multimodal U-Net 3D cascade
```bash
python -u main.py --mode cascade_stage1   --cascade_arch unet3d --multimodal --epochs 100 --dataset prostate158
python -u main.py --mode cascade_predict_bboxes --cascade_arch unet3d --multimodal --dataset prostate158
python -u main.py --mode cascade_stage2   --cascade_arch unet3d --multimodal --epochs 120 --dataset prostate158
python -u main.py --mode cascade_evaluate --cascade_arch unet3d --multimodal --bbox_source predicted --dataset prostate158
```

Outputs land under
`output/prostate158/multimodal/cascade/{stage1,stage2}/...`.
