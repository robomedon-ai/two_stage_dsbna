"""
Convert .npy multi-class predictions to NIfTI volumes.

Handles:
  - mask_*.npy: multi-class labels (0=bg, 1=PZ, 2=CG, 3=tumor) -> nearest-neighbor resize
  - prob_*.npy: 4-channel softmax probabilities -> per-channel linear resize
  - image_*.npy: MRI input slices (ground_truth folder only)

2D/2.5D models: 476 slices, stacked back into per-case volumes.
3D models: 19 volumes (depth padded to 32), cropped back.

Output: output/prostate158/test_predictions_nifti/<architecture>/<case>_<prefix>.nii.gz
"""

import numpy as np
import nibabel as nib
import os
from scipy.ndimage import zoom

BASE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE, "output", "prostate158", "test_predictions")
REF_DIR = os.path.join(BASE, "prostate158", "prostate158_test", "prostate158_test", "test")
OUT_DIR = os.path.join(BASE, "output", "prostate158", "test_predictions_nifti")

# Build case info
cases = sorted(os.listdir(REF_DIR))
case_info = []
cumulative = 0
for case_id in cases:
    ref_path = os.path.join(REF_DIR, case_id, "t2.nii.gz")
    ref_img = nib.load(ref_path)
    n_slices = ref_img.shape[2]
    case_info.append((case_id, cumulative, cumulative + n_slices, ref_path, ref_img.shape))
    cumulative += n_slices

print(f"Found {len(cases)} cases, {cumulative} total slices")


def resize_2d(arr, target_h, target_w, is_mask=False):
    if arr.shape[-2] == target_h and arr.shape[-1] == target_w:
        return arr
    if is_mask:
        return zoom(arr.astype(np.float64),
                     (target_h / arr.shape[0], target_w / arr.shape[1]),
                     order=0).astype(arr.dtype)
    # For multi-channel prob maps: (C, H, W)
    if arr.ndim == 3:
        factors = (1, target_h / arr.shape[1], target_w / arr.shape[2])
    else:
        factors = (target_h / arr.shape[0], target_w / arr.shape[1])
    return zoom(arr, factors, order=1)


def resize_3d_volume(vol, target_shape, is_mask=False):
    """Resize volume. For masks use order=0, for probs use order=1."""
    order = 0 if is_mask else 1
    if vol.ndim == 3:  # (D, H, W)
        factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
    elif vol.ndim == 4:  # (C, D, H, W)
        factors = (1,) + tuple(t / s for t, s in zip(target_shape, vol.shape[1:]))
    else:
        return vol
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return vol
    if is_mask:
        return zoom(vol.astype(np.float64), factors, order=0).astype(vol.dtype)
    return zoom(vol, factors, order=1)


def convert_folder(folder_name):
    folder_path = os.path.join(PRED_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return

    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if not npy_files:
        return

    # Detect if 3D (per-case volumes) or 2D (per-slice)
    sample = np.load(os.path.join(folder_path, npy_files[0]), allow_pickle=True)
    # 3D model masks: (D, H, W); 3D model probs: (C, D, H, W)
    # 2D model masks: (H, W); 2D model probs: (C, H, W)
    # Detect by file count: 3D has 19 per prefix, 2D has 476
    prefixes = sorted(set(f.split("_")[0] for f in npy_files))
    first_prefix_count = len([f for f in npy_files if f.startswith(prefixes[0] + "_")])
    is_3d = first_prefix_count == len(case_info)  # 19 files = per-case

    out_folder = os.path.join(OUT_DIR, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    for prefix in prefixes:
        prefix_files = sorted([f for f in npy_files if f.startswith(prefix + "_")])
        if not prefix_files:
            continue

        is_mask = prefix == "mask"

        print(f"  {prefix}: {len(prefix_files)} files ({'3D' if is_3d else '2D'}, "
              f"{'labels' if is_mask else 'continuous'})")

        if is_3d:
            for i, (case_id, _, _, ref_path, orig_shape) in enumerate(case_info):
                if i >= len(prefix_files):
                    break
                vol = np.load(os.path.join(folder_path, prefix_files[i]),
                              allow_pickle=True)
                orig_d = orig_shape[2]

                if vol.ndim == 3:  # (D, H, W) mask or image
                    vol = vol[:orig_d, :, :]
                    target_h, target_w = orig_shape[0], orig_shape[1]
                    if vol.shape[1] != target_h or vol.shape[2] != target_w:
                        vol = resize_3d_volume(vol, (orig_d, target_h, target_w),
                                               is_mask=is_mask)
                    vol = np.transpose(vol, (1, 2, 0))  # (H, W, D)
                elif vol.ndim == 4:  # (C, D, H, W) prob
                    vol = vol[:, :orig_d, :, :]
                    target_h, target_w = orig_shape[0], orig_shape[1]
                    if vol.shape[2] != target_h or vol.shape[3] != target_w:
                        vol = resize_3d_volume(vol, (orig_d, target_h, target_w),
                                               is_mask=False)
                    # Save each channel separately or as 4D NIfTI
                    vol = np.transpose(vol, (2, 3, 1, 0))  # (H, W, D, C)

                ref_img = nib.load(ref_path)
                nii = nib.Nifti1Image(vol.astype(np.float32), ref_img.affine,
                                       ref_img.header)
                nib.save(nii, os.path.join(out_folder, f"{case_id}_{prefix}.nii.gz"))
        else:
            # Load all 2D slices
            all_slices = []
            for f in prefix_files:
                arr = np.load(os.path.join(folder_path, f), allow_pickle=True)
                all_slices.append(arr)

            for case_id, start, end, ref_path, orig_shape in case_info:
                case_slices = all_slices[start:end]
                target_h, target_w = orig_shape[0], orig_shape[1]

                resized = [resize_2d(s, target_h, target_w, is_mask=is_mask)
                           for s in case_slices]

                if resized[0].ndim == 2:  # (H, W) -> stack to (H, W, D)
                    volume = np.stack(resized, axis=2)
                else:  # (C, H, W) -> stack to (H, W, D, C)
                    volume = np.stack(resized, axis=0)  # (D, C, H, W)
                    volume = np.transpose(volume, (2, 3, 0, 1))  # (H, W, D, C)

                ref_img = nib.load(ref_path)
                nii = nib.Nifti1Image(volume.astype(np.float32), ref_img.affine,
                                       ref_img.header)
                nib.save(nii, os.path.join(out_folder, f"{case_id}_{prefix}.nii.gz"))

        print(f"    -> Saved {len(case_info)} volumes")


# Clear old output
import shutil
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# Process all folders
for folder in sorted(os.listdir(PRED_DIR)):
    if not os.path.isdir(os.path.join(PRED_DIR, folder)):
        continue
    print(f"\nConverting: {folder}")
    convert_folder(folder)

print(f"\nDone! NIfTI files saved to:\n{OUT_DIR}")
