"""
3D Cascade U-Net for prostate zonal + tumour segmentation.

Two-stage volumetric architecture that decomposes a hard multi-class task
(small-volume tumour against large-volume PZ + CG anatomy) into two easier
sub-tasks:

  Stage 1  ProstateROINet  — binary U-Net3D (out_channels = 2) that locates
                              the prostate as one foreground class.
  Stage 2  TumorSegNet     — multi-class U-Net3D (out_channels = 4) that
                              operates on a tightly-cropped ROI extracted
                              from Stage 1's prediction and outputs the
                              four-class map (background, PZ, CG, tumour).

The cascade is *not* end-to-end differentiable — the ROI extraction step
between the stages (largest-connected-component → axis-aligned bounding box)
is non-differentiable by construction. Each stage is trained independently
with its own loss; this class only wraps the two trained stages for
inference. Training scripts live in ``main.py`` under the
``cascade_stage1`` and ``cascade_stage2`` modes.

Theoretical motivation:
  * Within the prostate ROI, the tumour-to-volume ratio rises by roughly
    one to two orders of magnitude, which restores a usable gradient
    signal for the rare class.
  * The cropped ROI can be cast at a higher effective resolution (the
    default ``stage2_volume_size = 48×128×128``) than the original full
    volume (``32×256×256``) without exceeding memory, recovering spatial
    detail lost to single-stage downsampling.
  * Stage 1 is a comparatively easy task (well-defined organ with strong
    T2 contrast against surrounding tissue) so a lightweight network with
    fewer parameters is sufficient; Stage 2 carries the full capacity.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import UNet3D
from ..cascade import compute_roi_bbox, paste_back


class CascadeUNet3D(nn.Module):
    """
    Cascaded 3D U-Net that wraps a binary ProstateROINet (Stage 1) and a
    four-class TumorSegNet (Stage 2). Provides a unified inference API.

    Args:
        in_channels: number of input image channels (1 for T2 MRI).
        out_channels: number of foreground classes for Stage 2
            (default 4 = background + PZ + CG + tumour).
        stage1_base_filters: base filter count for the binary Stage 1 U-Net.
            Defaults to 16 (≈8× lighter than the 32-filter Stage 2).
        stage2_base_filters: base filter count for the multi-class Stage 2
            U-Net.
        stage2_volume_size: fixed input shape for Stage 2 after cropping +
            resampling the ROI; must be divisible by ``2**4 = 16`` per axis
            because of four max-pool stages inside ``UNet3D``.
        bbox_margin_voxels: per-axis padding added to the predicted bbox
            from Stage 1 before cropping. Default ``(2, 8, 8)`` reflects
            anisotropic spacing typical of Prostate158 (≈3 mm slice,
            ≈0.5 mm in-plane).
        bbox_min_size: minimum bbox extent per axis after padding; the box
            is centre-expanded if smaller than this and clamped to volume
            bounds. Guarantees a stable Stage 2 input shape.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 stage1_base_filters: int = 16,
                 stage2_base_filters: int = 32,
                 stage2_volume_size: Tuple[int, int, int] = (48, 128, 128),
                 bbox_margin_voxels: Tuple[int, int, int] = (2, 8, 8),
                 bbox_min_size: Tuple[int, int, int] = (24, 96, 96)):
        super().__init__()
        if out_channels < 2:
            raise ValueError(
                f"out_channels must be >= 2 (got {out_channels}); the "
                f"cascade is designed for multi-class downstream tasks.")
        for axis, ext in zip("DHW", stage2_volume_size):
            if ext % 16 != 0:
                raise ValueError(
                    f"stage2_volume_size[{axis}] = {ext} must be divisible "
                    f"by 16 (four max-pool stages inside UNet3D).")

        # Stage 1: binary prostate localisation.
        self.stage1 = UNet3D(in_channels=in_channels, out_channels=2,
                              base_filters=stage1_base_filters)
        # Stage 2: in-ROI multi-class segmentation.
        self.stage2 = UNet3D(in_channels=in_channels, out_channels=out_channels,
                              base_filters=stage2_base_filters)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage2_volume_size = tuple(stage2_volume_size)
        self.bbox_margin = tuple(bbox_margin_voxels)
        self.bbox_min_size = tuple(bbox_min_size)

    # ----------------------------------------------------------------- #
    # Stage-level forwards (used during stage-by-stage training)         #
    # ----------------------------------------------------------------- #

    def stage1_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Stage 1 only. Used for binary prostate-localisation training.

        Args:
            x: (B, C, D, H, W) input volume(s).

        Returns:
            (B, 2, D, H, W) logits over {background, prostate}.
        """
        return self.stage1(x)

    def stage2_forward(self, roi: torch.Tensor) -> torch.Tensor:
        """
        Forward Stage 2 only on already-cropped, fixed-shape ROI tensors.

        Args:
            roi: (B, C, D, H, W) ROI volume(s). Spatial extents must match
                ``stage2_volume_size``.

        Returns:
            (B, out_channels, D, H, W) logits over the four classes.
        """
        expected = self.stage2_volume_size
        if tuple(roi.shape[2:]) != expected:
            raise ValueError(
                f"stage2_forward: expected spatial shape {expected}, got "
                f"{tuple(roi.shape[2:])}. Resample the ROI before calling.")
        return self.stage2(roi)

    # ----------------------------------------------------------------- #
    # End-to-end inference                                               #
    # ----------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                return_intermediate: bool = False
                ) -> torch.Tensor:
        """
        Run the full cascade pipeline on a batch of full-resolution volumes.

        For each sample independently:
          1. Stage 1 produces a binary prostate prediction.
          2. ``compute_roi_bbox`` extracts the bounding box of the largest
             connected component, padded by ``bbox_margin`` and centre-
             expanded to ``bbox_min_size``.
          3. The original volume is cropped to that box and trilinearly
             resampled to ``stage2_volume_size``.
          4. Stage 2 produces 4-class logits at ROI resolution; the argmax
             is taken, nearest-neighbour resampled back to the bbox-native
             shape, and pasted into a zero-initialised full-volume canvas.

        Args:
            x: (B, C, D, H, W) input volumes.
            return_intermediate: if True, also return the binary Stage 1
                masks and the list of per-sample bboxes (useful for
                debugging or evaluation tooling).

        Returns:
            preds: (B, D, H, W) integer class labels in ``[0, out_channels)``.
            (optional) stage1_masks: (B, D, H, W) uint8 binary masks.
            (optional) bboxes: list of (z0, z1, y0, y1, x0, x1) tuples,
                one per sample.
        """
        if x.dim() != 5:
            raise ValueError(
                f"predict expects (B, C, D, H, W); got shape {tuple(x.shape)}")
        device = x.device
        B, _, D, H, W = x.shape

        # --- Stage 1 ---
        s1_logits = self.stage1(x)                          # (B, 2, D, H, W)
        s1_pred = s1_logits.argmax(dim=1).to(torch.uint8)   # (B, D, H, W)

        preds = torch.zeros((B, D, H, W), dtype=torch.long, device=device)
        bboxes = []
        for b in range(B):
            mask_b = s1_pred[b].detach().cpu().numpy()
            bbox = compute_roi_bbox(mask_b,
                                     margin_voxels=self.bbox_margin,
                                     min_size=self.bbox_min_size,
                                     use_largest_cc=True)
            bboxes.append(bbox)
            if mask_b.sum() == 0:
                # Stage 1 failed — emit all-background and warn quietly.
                continue
            z0, z1, y0, y1, x0, x1 = bbox

            # Crop + resample to fixed Stage 2 input shape.
            roi = x[b:b + 1, :, z0:z1, y0:y1, x0:x1]
            roi_resampled = F.interpolate(
                roi, size=self.stage2_volume_size,
                mode="trilinear", align_corners=False)

            # --- Stage 2 ---
            s2_logits = self.stage2(roi_resampled)          # (1, C, *S2)
            s2_pred = s2_logits.argmax(dim=1).to(torch.uint8)  # (1, *S2)

            # Resample 4-class prediction back to bbox-native shape using
            # nearest-neighbour to preserve label integrity, then paste back.
            bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
            # Add fake channel dim so we can use F.interpolate on a long tensor.
            s2_pred_native = F.interpolate(
                s2_pred.unsqueeze(0).float(),
                size=bbox_shape, mode="nearest"
            ).squeeze(0).squeeze(0).to(torch.long)
            # paste_back operates on numpy; round-trip through cpu for safety.
            pasted = paste_back(
                s2_pred_native.detach().cpu().numpy(), bbox,
                full_shape=(D, H, W), fill_value=0, dtype=np.int64)
            preds[b] = torch.from_numpy(pasted).to(device).long()

        if return_intermediate:
            return preds, s1_pred, bboxes
        return preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward = end-to-end inference. For stage-specific training,
        use :meth:`stage1_forward` and :meth:`stage2_forward` directly.
        """
        return self.predict(x)

    # ----------------------------------------------------------------- #
    # Parameter accounting                                               #
    # ----------------------------------------------------------------- #

    def count_parameters(self) -> int:
        """Total trainable parameter count across both stages."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_stage(self) -> Tuple[int, int]:
        """Return (stage1_params, stage2_params)."""
        s1 = sum(p.numel() for p in self.stage1.parameters() if p.requires_grad)
        s2 = sum(p.numel() for p in self.stage2.parameters() if p.requires_grad)
        return s1, s2

    # ----------------------------------------------------------------- #
    # Checkpoint utilities                                               #
    # ----------------------------------------------------------------- #

    def load_stage_checkpoint(self, stage: int, checkpoint_path: str,
                              map_location: Optional[str] = None) -> None:
        """
        Load a previously-trained checkpoint for one stage. Accepts either
        a raw ``state_dict`` or the wrapped ``{"model_state_dict": ...}``
        format produced by the project's training loop.
        """
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        ck = torch.load(checkpoint_path, map_location=map_location,
                         weights_only=False)
        state = ck["model_state_dict"] if "model_state_dict" in ck else ck
        # The checkpoint was saved from a *standalone* UNet3D, so its keys do
        # not have the "stage1." / "stage2." prefix. Load into the matching
        # sub-module directly.
        target = self.stage1 if stage == 1 else self.stage2
        target.load_state_dict(state, strict=True)
