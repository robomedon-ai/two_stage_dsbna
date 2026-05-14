"""
Utility functions for the two-stage cascade pipeline:

  Stage 1 (ProstateROINet)  →  binary prostate mask  →  bounding box
                              compute_roi_bbox()
  Stage 2 (TumorSegNet)     →  cropped 4-class prediction  →  pasted back
                              paste_back()

The functions are pure NumPy with no model dependencies so that they can be
unit-tested in isolation and used both at training time (for cropping
ground-truth volumes) and at inference time (for cropping based on stage-1
predictions).
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import label as cc_label

# ---------------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------------


def _largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Vraća binarnu masku koja sadrži samo najveću 3D povezanu komponentu
    iz `binary_mask`. Koristi 6-konektivnost (faces only), što je standardno
    za volumetrijske maske organa i konzervativnije od 18- ili 26-konekt.

    Args:
        binary_mask: 3D ndarray (D, H, W) tipa bool ili uint8.

    Returns:
        ndarray istog oblika i tipa uint8 koji sadrži samo najveću komponentu.
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.sum() == 0:
        return binary_mask
    structure = np.array(
        [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.uint8)
    labeled, n = cc_label(binary_mask, structure=structure)
    if n <= 1:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # exclude background
    largest_label = sizes.argmax()
    return (labeled == largest_label).astype(np.uint8)


def compute_roi_bbox(
    binary_mask: np.ndarray,
    margin_voxels: Tuple[int, int, int] = (2, 8, 8),
    min_size: Tuple[int, int, int] = (24, 96, 96),
    use_largest_cc: bool = True,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute an axis-aligned 3D bounding box around the foreground voxels of
    `binary_mask`, optionally restricted to the largest connected component,
    padded by `margin_voxels` along each axis, and centre-expanded to at
    least `min_size` per axis. The box is clamped to the volume bounds.

    Args:
        binary_mask: 3D ndarray (D, H, W). Treated as boolean (any nonzero
            value is foreground).
        margin_voxels: (margin_d, margin_h, margin_w) padding added on each
            side along the slice (D), in-plane height (H) and in-plane
            width (W) axes. The default (2, 8, 8) reflects the typical
            anisotropic spacing of Prostate158 (≈3 mm in z, ≈0.5 mm in plane);
            margins are equivalent to ≈6 mm of padding in every direction.
        min_size: minimum bounding-box size per axis. If the box is smaller
            after padding, it is symmetrically expanded around its centre to
            reach `min_size`, then clamped to volume bounds.
        use_largest_cc: if True, restrict the bounding box to the largest
            connected component of `binary_mask`. Recommended at inference
            time to suppress false-positive blobs from stage 1.

    Returns:
        Tuple (z0, z1, y0, y1, x0, x1) with half-open conventions
        (mask[z0:z1, y0:y1, x0:x1] indexes the ROI). If the mask is empty
        the function returns (0, 0, 0, 0, 0, 0); callers should treat this
        as a stage-1 failure and fall back to a sentinel output.
    """
    if binary_mask.ndim != 3:
        raise ValueError(f"binary_mask must be 3D, got shape {binary_mask.shape}")
    D, H, W = binary_mask.shape

    if use_largest_cc:
        mask = _largest_connected_component(binary_mask)
    else:
        mask = (binary_mask > 0).astype(np.uint8)

    if mask.sum() == 0:
        return (0, 0, 0, 0, 0, 0)

    # Axis-aligned bbox of nonzero voxels
    zs, ys, xs = np.where(mask > 0)
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Padding
    md, mh, mw = margin_voxels
    z0 = max(0, z0 - md); z1 = min(D, z1 + md)
    y0 = max(0, y0 - mh); y1 = min(H, y1 + mh)
    x0 = max(0, x0 - mw); x1 = min(W, x1 + mw)

    # Centre-expand to minimum size, then clamp
    z0, z1 = _expand_to_min(z0, z1, min_size[0], D)
    y0, y1 = _expand_to_min(y0, y1, min_size[1], H)
    x0, x1 = _expand_to_min(x0, x1, min_size[2], W)

    return (z0, z1, y0, y1, x0, x1)


def _expand_to_min(lo: int, hi: int, min_len: int, bound: int) -> Tuple[int, int]:
    """Centre-expand [lo, hi) to length >= min_len, clamping to [0, bound]."""
    cur = hi - lo
    if cur >= min_len:
        return lo, hi
    need = min_len - cur
    half = need // 2
    new_lo = max(0, lo - half)
    new_hi = min(bound, hi + (need - half))
    # If clamped on one side, push the other to make up the length
    deficit = min_len - (new_hi - new_lo)
    if deficit > 0:
        if new_lo == 0:
            new_hi = min(bound, new_hi + deficit)
        else:
            new_lo = max(0, new_lo - deficit)
    return new_lo, new_hi


def jitter_bbox(
    bbox: Tuple[int, int, int, int, int, int],
    volume_shape: Tuple[int, int, int],
    max_jitter_voxels: int,
    min_size: Tuple[int, int, int],
    rng: np.random.Generator,
) -> Tuple[int, int, int, int, int, int]:
    """
    Apply integer translation jitter to each face of a bounding box. Used
    during stage-2 training to teach the network robustness to imperfect
    stage-1 localisation. The jittered box is clamped to volume bounds and
    re-expanded if it falls below `min_size` on any axis.
    """
    z0, z1, y0, y1, x0, x1 = bbox
    D, H, W = volume_shape
    j = max_jitter_voxels
    if j > 0:
        z0 = int(np.clip(z0 + rng.integers(-j, j + 1), 0, D))
        z1 = int(np.clip(z1 + rng.integers(-j, j + 1), z0 + 1, D))
        y0 = int(np.clip(y0 + rng.integers(-j, j + 1), 0, H))
        y1 = int(np.clip(y1 + rng.integers(-j, j + 1), y0 + 1, H))
        x0 = int(np.clip(x0 + rng.integers(-j, j + 1), 0, W))
        x1 = int(np.clip(x1 + rng.integers(-j, j + 1), x0 + 1, W))
        z0, z1 = _expand_to_min(z0, z1, min_size[0], D)
        y0, y1 = _expand_to_min(y0, y1, min_size[1], H)
        x0, x1 = _expand_to_min(x0, x1, min_size[2], W)
    return (z0, z1, y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# Paste-back: write a small ROI volume back into a full-volume canvas
# ---------------------------------------------------------------------------


def paste_back(
    roi_volume: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int],
    full_shape: Tuple[int, int, int],
    fill_value: int = 0,
    dtype: np.dtype = None,
) -> np.ndarray:
    """
    Paste a small 3D volume `roi_volume` (which has bounding-box-shape
    `(z1-z0, y1-y0, x1-x0)`) back into a zero-initialised volume of shape
    `full_shape`. Voxels outside the bounding box are filled with
    `fill_value` (default 0 = background class).

    Args:
        roi_volume: 3D ndarray with shape matching the bbox extents.
        bbox: (z0, z1, y0, y1, x0, x1) — half-open.
        full_shape: target volume shape (D, H, W).
        fill_value: background value to fill outside-ROI voxels.
        dtype: optional override of output dtype (defaults to roi_volume.dtype).
    """
    z0, z1, y0, y1, x0, x1 = bbox
    expected = (z1 - z0, y1 - y0, x1 - x0)
    if roi_volume.shape != expected:
        raise ValueError(
            f"roi_volume.shape={roi_volume.shape} does not match bbox extents "
            f"{expected}")
    out_dtype = dtype if dtype is not None else roi_volume.dtype
    full = np.full(full_shape, fill_value, dtype=out_dtype)
    full[z0:z1, y0:y1, x0:x1] = roi_volume
    return full


# ---------------------------------------------------------------------------
# Inline self-test (run `python -m src.cascade`)
# ---------------------------------------------------------------------------


def _run_self_test():
    rng = np.random.default_rng(0)

    # 1) empty mask → zero bbox
    empty = np.zeros((10, 20, 20), dtype=np.uint8)
    assert compute_roi_bbox(empty) == (0, 0, 0, 0, 0, 0)

    # 2) single voxel → bbox enlarged by margin and min_size, clamped to volume
    single = np.zeros((20, 40, 40), dtype=np.uint8)
    single[10, 20, 20] = 1
    bb = compute_roi_bbox(single, margin_voxels=(2, 8, 8),
                          min_size=(8, 16, 16))
    z0, z1, y0, y1, x0, x1 = bb
    assert z1 - z0 >= 8 and y1 - y0 >= 16 and x1 - x0 >= 16
    assert 0 <= z0 < z1 <= 20 and 0 <= y0 < y1 <= 40 and 0 <= x0 < x1 <= 40

    # 3) blob plus a small false-positive blob → largest CC dominates
    fp = np.zeros((30, 60, 60), dtype=np.uint8)
    fp[10:20, 20:40, 20:40] = 1       # big blob (10×20×20 = 4000 voxels)
    fp[2:4, 2:4, 2:4] = 1             # tiny FP blob (8 voxels)
    bb1 = compute_roi_bbox(fp, use_largest_cc=True,
                            margin_voxels=(0, 0, 0), min_size=(1, 1, 1))
    z0, z1, y0, y1, x0, x1 = bb1
    # The bbox should be of the big blob only — slices [10,20), [20,40), [20,40)
    assert (z0, z1, y0, y1, x0, x1) == (10, 20, 20, 40, 20, 40), bb1

    # 4) without largest-CC filtering, the small blob enlarges the bbox
    bb2 = compute_roi_bbox(fp, use_largest_cc=False,
                            margin_voxels=(0, 0, 0), min_size=(1, 1, 1))
    z0, z1, y0, y1, x0, x1 = bb2
    assert z0 == 2 and y0 == 2 and x0 == 2  # tiny blob expanded the box

    # 5) jitter respects bounds and min_size
    bb3 = jitter_bbox(bb1, volume_shape=fp.shape, max_jitter_voxels=3,
                       min_size=(1, 1, 1), rng=rng)
    z0, z1, y0, y1, x0, x1 = bb3
    assert 0 <= z0 < z1 <= 30
    assert 0 <= y0 < y1 <= 60
    assert 0 <= x0 < x1 <= 60

    # 6) paste_back round-trip
    roi = np.ones((10, 20, 20), dtype=np.float32) * 7.0
    pasted = paste_back(roi, bb1, full_shape=fp.shape, fill_value=0)
    assert pasted.shape == fp.shape
    assert pasted[10:20, 20:40, 20:40].mean() == 7.0
    assert pasted[0, 0, 0] == 0.0

    # 7) paste_back rejects shape mismatch
    bad = np.ones((10, 20, 19), dtype=np.float32)
    try:
        paste_back(bad, bb1, fp.shape)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shape mismatch")

    # 8) bbox is clamped when min_size > volume
    small_vol = np.zeros((4, 4, 4), dtype=np.uint8)
    small_vol[2, 2, 2] = 1
    bb4 = compute_roi_bbox(small_vol, margin_voxels=(0, 0, 0),
                            min_size=(10, 10, 10))
    z0, z1, y0, y1, x0, x1 = bb4
    assert (z0, z1, y0, y1, x0, x1) == (0, 4, 0, 4, 0, 4)

    print("[cascade.py] self-test passed.")


if __name__ == "__main__":
    _run_self_test()
