"""
Učitavanje i predobrada Prostate158 skupa podataka za segmentaciju prostate.

Prostate158 sadrži T2-weighted MRI volumene u NIfTI formatu s anotacijama
za perifernu zonu (label=1) i centralnu žlijezdu (label=2) u anatomy datoteci,
te tumor (label=3) u zasebnoj tumor datoteci.

Multi-class segmentacija: 0=pozadina, 1=PZ, 2=CG, 3=tumor.

Podržava tri načina rada: 2D, 2.5D i 3D.
"""

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate, map_coordinates, gaussian_filter

from .config import Config


# ---------------------------------------------------------------------------
# Pomoćne funkcije
# ---------------------------------------------------------------------------

def _case_id_from_path(t2_path: str) -> str:
    """Izvuci case identifier iz t2.nii.gz putanje (= naziv parent direktorija)."""
    return os.path.basename(os.path.dirname(t2_path))


def load_case_nifti(t2_path: str, seg_path: str,
                    tumor_path: Optional[str] = None
                    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Učitava NIfTI T2 sliku, anatomy masku i opcionalno tumor masku.

    Vraća:
        image: numpy niz oblika (D, H, W)
        mask: numpy niz oblika (D, H, W) s vrijednostima {0, 1, 2, 3}
              0=pozadina, 1=PZ, 2=CG, 3=tumor
        spacing: (sx, sy, sz) voxel spacing u mm
    """
    nii_img = nib.load(t2_path)
    nii_seg = nib.load(seg_path)

    image = nii_img.get_fdata().astype(np.float32)
    anatomy = nii_seg.get_fdata().astype(np.float32)
    spacing = nii_img.header.get_zooms()  # (sx, sy, sz)

    # NIfTI format: (H, W, D) -> transponiraj u (D, H, W)
    image = np.transpose(image, (2, 0, 1))
    anatomy = np.transpose(anatomy, (2, 0, 1))

    # Anatomy maska: label 1 = PZ, label 2 = CG
    mask = np.round(anatomy).astype(np.int64)
    mask = np.clip(mask, 0, 2)

    # Tumor maska: label 3 (overrides anatomy where present)
    if tumor_path and os.path.exists(tumor_path):
        nii_tumor = nib.load(tumor_path)
        tumor = nii_tumor.get_fdata().astype(np.float32)
        tumor = np.transpose(tumor, (2, 0, 1))
        tumor_binary = (tumor > 0.5)
        mask[tumor_binary] = 3

    return image, mask, (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def load_case_multimodal(t2_path: str, seg_path: str,
                          tumor_path: Optional[str] = None,
                          modalities: Tuple[str, ...] = ("t2", "adc", "dwi")
                          ) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Load a multi-modality Prostate158 case as a stacked 4D volume.

    The modalities (default: T2 + ADC + DWI) are read from the same case
    directory and stacked along a new leading axis. Prostate158 ships
    the three modalities already co-registered to the T2 grid (verified:
    shape, spacing and affine all match per case), so no resampling is
    required at load time. Each modality is normalised independently
    later in ``preprocess_volume`` via Z-score; that step is the natural
    place to handle the very different intensity ranges of the three
    sequences (T2 ≈ 0–1200, ADC ≈ 0–3200, DWI ≈ 10–200).

    Args:
        t2_path: path to ``t2.nii.gz`` of the case. ADC / DWI paths are
            derived by sibling lookup in the same directory.
        seg_path: anatomy mask path (``t2_anatomy_reader1.nii.gz``).
        tumor_path: optional tumour mask path.
        modalities: ordered tuple of modality names to load. Each name
            must correspond to ``<name>.nii.gz`` in the case directory.

    Returns:
        image: numpy array of shape ``(M, D, H, W)`` where M = len(modalities).
        mask:  numpy array of shape ``(D, H, W)`` with values {0, 1, 2, 3}.
        spacing: ``(sx, sy, sz)`` voxel spacing in mm, taken from T2.
    """
    case_dir = os.path.dirname(t2_path)
    nii_t2 = nib.load(t2_path)
    spacing = nii_t2.header.get_zooms()

    # Load every modality in the requested order
    channels = []
    for m in modalities:
        m_path = os.path.join(case_dir, f"{m}.nii.gz")
        if not os.path.exists(m_path):
            raise FileNotFoundError(
                f"Modality '{m}' missing for case {case_dir}: {m_path}")
        arr = nib.load(m_path).get_fdata().astype(np.float32)
        # NIfTI is (H, W, D); transpose to (D, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        channels.append(arr)

    # Verify all modalities share the same (D, H, W) shape
    shapes = {c.shape for c in channels}
    if len(shapes) != 1:
        raise ValueError(
            f"Multimodal load failed: shapes differ across modalities "
            f"{dict(zip(modalities, [c.shape for c in channels]))}. "
            f"Prostate158 should already be co-registered.")
    image = np.stack(channels, axis=0)   # (M, D, H, W)

    # Mask construction is identical to the single-modal loader
    nii_seg = nib.load(seg_path)
    anatomy = np.transpose(nii_seg.get_fdata().astype(np.float32), (2, 0, 1))
    mask = np.round(anatomy).astype(np.int64)
    mask = np.clip(mask, 0, 2)
    if tumor_path and os.path.exists(tumor_path):
        tumor = nib.load(tumor_path).get_fdata().astype(np.float32)
        tumor = np.transpose(tumor, (2, 0, 1))
        mask[tumor > 0.5] = 3

    return image, mask, (float(spacing[0]), float(spacing[1]), float(spacing[2]))


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def resample_to_uniform_spacing(image: np.ndarray, mask: np.ndarray,
                                 original_spacing: tuple,
                                 target_spacing: tuple = (0.5, 0.5, 3.0)
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample volumena na uniformni voxel spacing.

    Ovo osigurava da svi volumeni imaju iste fizičke proporcije,
    što je ključno jer Prostate158 ima varijabilni spacing.
    """
    sx, sy, sz = original_spacing
    tx, ty, tz = target_spacing

    zoom_factors = (sz / tz, sx / tx, sy / ty)  # (D, H, W)
    image_resampled = zoom(image, zoom_factors, order=3)
    mask_resampled = zoom(mask.astype(np.float64), zoom_factors, order=0)
    mask_resampled = np.round(mask_resampled).astype(np.int64)

    return image_resampled, mask_resampled


def apply_clahe_2d(image_slice: np.ndarray, clip_limit: float = 2.0,
                   grid_size: int = 8) -> np.ndarray:
    """
    Primjenjuje CLAHE (Contrast Limited Adaptive Histogram Equalization)
    na 2D rez za poboljšanje lokalnog kontrasta na rubovima prostate.
    """
    try:
        import cv2
        # Normaliziraj u [0, 255] za OpenCV
        smin, smax = image_slice.min(), image_slice.max()
        if smax - smin < 1e-8:
            return image_slice
        normalized = ((image_slice - smin) / (smax - smin) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(grid_size, grid_size))
        enhanced = clahe.apply(normalized)
        # Vrati u originalni raspon
        result = enhanced.astype(np.float32) / 255.0 * (smax - smin) + smin
        return result
    except ImportError:
        return image_slice


def apply_clahe_volume(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Primjenjuje CLAHE na svaki rez volumena."""
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        result[i] = apply_clahe_2d(image[i], clip_limit)
    return result


def crop_to_roi(image: np.ndarray, mask: np.ndarray,
                margin: int = 20, target_size: Tuple[int, int] = (256, 256)
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Izrezuje ROI oko prostate koristeći centar mase maske + marginu.

    Smanjuje pozadinu i fokusira model na relevantno područje.
    Za rezove bez prostate, koristi centar slike.
    """
    D, H, W = image.shape

    # Nađi bounding box prostate preko svih rezova
    nonzero = np.where(mask > 0)
    if len(nonzero[0]) > 0:
        # Centar mase u H, W dimenzijama
        h_center = int(np.mean(nonzero[1]))
        w_center = int(np.mean(nonzero[2]))

        # Bounding box s marginom
        h_min = max(0, np.min(nonzero[1]) - margin)
        h_max = min(H, np.max(nonzero[1]) + margin)
        w_min = max(0, np.min(nonzero[2]) - margin)
        w_max = min(W, np.max(nonzero[2]) + margin)

        # Osiguraj minimalni ROI
        roi_h = max(h_max - h_min, target_size[0] // 2)
        roi_w = max(w_max - w_min, target_size[1] // 2)

        # Centriraj ROI
        h_start = max(0, h_center - roi_h // 2)
        h_end = min(H, h_start + roi_h)
        if h_end == H:
            h_start = max(0, H - roi_h)

        w_start = max(0, w_center - roi_w // 2)
        w_end = min(W, w_start + roi_w)
        if w_end == W:
            w_start = max(0, W - roi_w)

        image = image[:, h_start:h_end, w_start:w_end]
        mask = mask[:, h_start:h_end, w_start:w_end]

    return image, mask


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Z-score normalizacija s clipom na [0.5, 99.5] percentil."""
    p_low, p_high = np.percentile(image, [0.5, 99.5])
    image = np.clip(image, p_low, p_high)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    else:
        image = image - mean
    return image


def preprocess_volume(image: np.ndarray, mask: np.ndarray,
                      spacing: tuple, config: Config
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kompletni preprocessing pipeline:
      1. Resampling na uniformni spacing
      2. CLAHE za poboljšanje kontrasta
      3. ROI crop oko prostate
      4. Z-score normalizacija (per-modality kada je ``image`` 4D)

    Podržava i 3D ulaz ``(D, H, W)`` (single-modal) i 4D ulaz
    ``(M, D, H, W)`` (multimodal — M kanala). Advanced preprocessing
    (resampling, CLAHE, ROI crop) trenutno nije implementirano za 4D
    ulaz i bit će preskočeno uz upozorenje ako je multimodal=True.
    """
    is_multimodal = image.ndim == 4
    use_advanced = getattr(config, "use_advanced_preprocessing", False)

    if use_advanced and not is_multimodal:
        target_spacing = getattr(config, "target_spacing", (0.5, 0.5, 3.0))
        image, mask = resample_to_uniform_spacing(image, mask, spacing,
                                                   target_spacing)
        clahe_clip = getattr(config, "clahe_clip_limit", 2.0)
        image = apply_clahe_volume(image, clip_limit=clahe_clip)
        roi_margin = getattr(config, "roi_margin", 20)
        image, mask = crop_to_roi(image, mask, margin=roi_margin)
    elif use_advanced and is_multimodal:
        print("preprocess_volume: advanced preprocessing skipped for "
              "multimodal input (not implemented for 4D arrays).")

    # Z-score normalisation — per modality for multimodal, otherwise per volume.
    if is_multimodal:
        for m in range(image.shape[0]):
            image[m] = normalize_intensity(image[m])
    else:
        image = normalize_intensity(image)

    return image, mask


def resize_slice(image_slice: np.ndarray, target_size: Tuple[int, int],
                 is_mask: bool = False) -> np.ndarray:
    """Resize a 2D slice ``(H, W)`` or a multi-channel 2D slice ``(M, H, W)``."""
    th, tw = target_size
    order = 0 if is_mask else 3
    if image_slice.ndim == 2:
        h, w = image_slice.shape
        factors = (th / h, tw / w)
    elif image_slice.ndim == 3:
        # Multimodal slice (M, H, W): keep channels axis, resize spatial dims
        _, h, w = image_slice.shape
        factors = (1.0, th / h, tw / w)
    else:
        raise ValueError(
            f"resize_slice: unsupported ndim={image_slice.ndim}")
    resized = zoom(image_slice.astype(np.float64) if is_mask else image_slice,
                    factors, order=order)
    if is_mask:
        resized = np.round(resized).astype(np.int64)
    return resized


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int],
                  is_mask: bool = False) -> np.ndarray:
    """Resize a volume ``(D, H, W)`` or multimodal volume ``(M, D, H, W)``."""
    td, th, tw = target_size
    order = 0 if is_mask else 3
    if volume.ndim == 3:
        d, h, w = volume.shape
        factors = (td / d, th / h, tw / w)
    elif volume.ndim == 4:
        # Multimodal (M, D, H, W): keep channels axis, resize spatial dims
        _, d, h, w = volume.shape
        factors = (1.0, td / d, th / h, tw / w)
    else:
        raise ValueError(
            f"resize_volume: unsupported ndim={volume.ndim}")
    resized = zoom(volume.astype(np.float64) if is_mask else volume,
                    factors, order=order)
    if is_mask:
        resized = np.round(resized).astype(np.int64)
    return resized


def elastic_deformation_2d(image: np.ndarray, mask: np.ndarray,
                           alpha: float = 100.0, sigma: float = 10.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    shape = image.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.clip(y + dy, 0, shape[0] - 1),
               np.clip(x + dx, 0, shape[1] - 1)]
    image_def = map_coordinates(image, indices, order=3, mode='reflect')
    mask_def = map_coordinates(mask.astype(np.float64), indices, order=0, mode='reflect')
    mask_def = np.round(mask_def).astype(np.int64)
    return image_def, mask_def


def get_prostate158_paths(data_root: str) -> Tuple[List, List, List]:
    """
    Vraća liste (t2_path, seg_path, tumor_path) za train, val i test split.

    Struktura:
      data_root/prostate158_train/train.csv  -> train split
      data_root/prostate158_train/valid.csv  -> val split
      data_root/prostate158_test/prostate158_test/test/  -> test split
    """
    train_dir = os.path.join(data_root, "prostate158_train")
    test_dir = os.path.join(data_root, "prostate158_test", "prostate158_test")

    def parse_csv(csv_path, base_dir):
        triples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t2_rel = row["t2"]
                seg_rel = row.get("t2_anatomy_reader1", "")
                if not seg_rel:
                    continue
                t2_abs = os.path.join(base_dir, t2_rel)
                seg_abs = os.path.join(base_dir, seg_rel)
                if not (os.path.exists(t2_abs) and os.path.exists(seg_abs)):
                    continue
                # Tumor path (may be empty or missing)
                tumor_rel = row.get("t2_tumor_reader1", "")
                tumor_abs = None
                if tumor_rel and tumor_rel.strip():
                    tumor_abs = os.path.join(base_dir, tumor_rel)
                    if not os.path.exists(tumor_abs):
                        tumor_abs = None
                triples.append((t2_abs, seg_abs, tumor_abs))
        return triples

    train_triples = parse_csv(os.path.join(train_dir, "train.csv"), train_dir)
    val_triples = parse_csv(os.path.join(train_dir, "valid.csv"), train_dir)

    # Test: skeneri direktorij
    test_triples = []
    test_case_dir = os.path.join(test_dir, "test")
    if os.path.isdir(test_case_dir):
        for case_name in sorted(os.listdir(test_case_dir)):
            case_path = os.path.join(test_case_dir, case_name)
            t2 = os.path.join(case_path, "t2.nii.gz")
            seg = os.path.join(case_path, "t2_anatomy_reader1.nii.gz")
            tumor = os.path.join(case_path, "t2_tumor_reader1.nii.gz")
            if os.path.exists(t2) and os.path.exists(seg):
                tumor_path = tumor if os.path.exists(tumor) else None
                test_triples.append((t2, seg, tumor_path))

    return train_triples, val_triples, test_triples


# ---------------------------------------------------------------------------
# Dataset klase za Prostate158
# ---------------------------------------------------------------------------

class Prostate158Dataset2D(Dataset):
    """
    2D Dataset za Prostate158: svaki rez kao zasebni uzorak.

    Ako je min_mask_ratio > 0, rezovi s manje od tog postotka prostate
    piksela se odbacuju iz treniranja (ali se zadržavaju za evaluaciju).
    """

    def __init__(self, case_triples: List[Tuple[str, str, Optional[str]]],
                 config: Config, augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.image_size_2d
        self.multimodal = bool(getattr(config, "multimodal", False))
        self.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
        # Multimodal slices: each entry holds (case_idx, (M, H, W) image, (H, W) mask).
        # Single-modal slices: (case_idx, (H, W) image, (H, W) mask).
        self.slices: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self.case_ids: List[str] = []
        self.has_tumor: List[bool] = []

        min_ratio = getattr(config, "min_mask_ratio", 0.0)
        filter_empty = augment and min_ratio > 0

        for case_idx, (t2_path, seg_path, tumor_path) in enumerate(case_triples):
            self.case_ids.append(_case_id_from_path(t2_path))
            if self.multimodal:
                image, mask, spacing = load_case_multimodal(
                    t2_path, seg_path, tumor_path, modalities=self.modalities)
            else:
                image, mask, spacing = load_case_nifti(
                    t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            # image shape is (M, D, H, W) when multimodal else (D, H, W)
            D = image.shape[1] if self.multimodal else image.shape[0]
            for i in range(D):
                if self.multimodal:
                    img_s = resize_slice(image[:, i], self.target_size)  # (M, H, W)
                else:
                    img_s = resize_slice(image[i], self.target_size)     # (H, W)
                msk_s = resize_slice(mask[i], self.target_size, is_mask=True)
                if filter_empty and (msk_s > 0).mean() < min_ratio:
                    continue
                self.slices.append((case_idx, img_s, msk_s))
                self.has_tumor.append(bool((msk_s == 3).any()))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_idx, img, msk = self.slices[idx]
        if self.augment:
            img, msk = self._augment(img, msk)
        # For multimodal img is already (M, H, W); for single-modal it's (H, W)
        # and needs an explicit channel axis to become (1, H, W).
        img_t = torch.from_numpy(img.copy()).float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        return {
            "image": img_t,
            "mask": torch.from_numpy(msk.copy()).long(),
            "case_idx": case_idx,
        }

    def _augment(self, img, msk):
        # img is (H, W) when single-modal and (M, H, W) when multimodal.
        # Apply spatial flips along the last two axes so the same code works
        # for both layouts. Mask is always (H, W).
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-1).copy()
            msk = np.flip(msk, axis=-1).copy()
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-2).copy()
            msk = np.flip(msk, axis=-2).copy()
        if getattr(self.config, "use_enhanced_aug", False):
            mm = img.ndim == 3   # (M, H, W) multimodal flag
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                rot_axes = (1, 2) if mm else (0, 1)
                img = rotate(img, angle, axes=rot_axes, reshape=False, order=3)
                msk = rotate(msk.astype(np.float64), angle,
                              reshape=False, order=0)
                msk = np.round(msk).astype(np.int64)
            if np.random.random() < 0.3 and not mm:
                # Elastic deformation operates on (H, W); skip for multimodal.
                img, msk = elastic_deformation_2d(
                    img, msk, self.config.elastic_alpha, self.config.elastic_sigma)
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)
        return img, msk


class Prostate158Dataset25D(Dataset):
    """2.5D Dataset za Prostate158: N susjednih rezova."""

    def __init__(self, case_triples: List[Tuple[str, str, Optional[str]]],
                 config: Config, augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.image_size_2d
        self.n_adj = config.num_adjacent_slices
        self.half = self.n_adj // 2

        self.entries = []
        self.volumes = []
        self.masks = []
        self.case_ids: List[str] = []
        self.has_tumor: List[bool] = []

        for t2_path, seg_path, tumor_path in case_triples:
            image, mask, spacing = load_case_nifti(t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            D = image.shape[0]
            resized_img = np.stack([
                resize_slice(image[i], self.target_size) for i in range(D)])
            resized_msk = np.stack([
                resize_slice(mask[i], self.target_size, is_mask=True) for i in range(D)])
            vol_idx = len(self.volumes)
            self.volumes.append(resized_img)
            self.masks.append(resized_msk)
            self.case_ids.append(_case_id_from_path(t2_path))
            for s in range(D):
                self.entries.append((vol_idx, s))
                self.has_tumor.append(bool((resized_msk[s] == 3).any()))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.entries[idx]
        volume = self.volumes[vol_idx]
        mask = self.masks[vol_idx]
        D = volume.shape[0]
        channels = []
        for offset in range(-self.half, self.half + 1):
            s = max(0, min(slice_idx + offset, D - 1))
            channels.append(volume[s])
        img_multi = np.stack(channels, axis=0)
        msk_slice = mask[slice_idx]
        if self.augment:
            img_multi, msk_slice = self._augment(img_multi, msk_slice)
        return {
            "image": torch.from_numpy(img_multi.copy()).float(),
            "mask": torch.from_numpy(msk_slice.copy()).long(),
            "case_idx": vol_idx,
        }

    def _augment(self, img, msk):
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=1).copy()
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=0).copy()
        if getattr(self.config, "use_enhanced_aug", False):
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                for c in range(img.shape[0]):
                    img[c] = rotate(img[c], angle, reshape=False, order=3)
                msk = rotate(msk.astype(np.float64), angle, reshape=False, order=0)
                msk = np.round(msk).astype(np.int64)
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)
        return img, msk


class Prostate158Dataset3D(Dataset):
    """3D Dataset za Prostate158: čitav volumen."""

    def __init__(self, case_triples: List[Tuple[str, str, Optional[str]]],
                 config: Config, augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.volume_size_3d
        self.volumes = []
        self.case_ids: List[str] = []

        for t2_path, seg_path, tumor_path in case_triples:
            image, mask, spacing = load_case_nifti(t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            image = resize_volume(image, self.target_size)
            mask = resize_volume(mask, self.target_size, is_mask=True)
            self.volumes.append((image, mask))
            self.case_ids.append(_case_id_from_path(t2_path))

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        img, msk = self.volumes[idx]
        if self.augment:
            img, msk = self._augment(img, msk)
        return {
            "image": torch.from_numpy(img.copy()).unsqueeze(0).float(),
            "mask": torch.from_numpy(msk.copy()).long(),
            "case_idx": idx,
        }

    def _augment(self, img, msk):
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=2).copy()
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()
        if getattr(self.config, "use_enhanced_aug", False):
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                img = rotate(img, angle, axes=(1, 2), reshape=False, order=3)
                msk = rotate(msk.astype(np.float64), angle, axes=(1, 2),
                             reshape=False, order=0)
                msk = np.round(msk).astype(np.int64)
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)
        return img, msk


# ---------------------------------------------------------------------------
# Cascade datasets (Stage 1 + Stage 2)
# ---------------------------------------------------------------------------


class Prostate158CascadeStage1Dataset(Dataset):
    """
    Dataset za Stage 1 cascade (ProstateROINet): vraća puni 3D T2 volumen i
    binarnu prostate masku (PZ ∪ CG ∪ tumour → 1, ostalo → 0). Volumen se
    resamplea na ``config.volume_size_3d`` točno kao za jednofazni 3D model,
    kako bi cijela 3D infrastruktura (sampler, AMP, batching) ostala
    nepromijenjena.
    """

    def __init__(self, case_triples: List[Tuple[str, str, Optional[str]]],
                 config: Config, augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.volume_size_3d
        self.multimodal = bool(getattr(config, "multimodal", False))
        self.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
        # Single-modal: (D, H, W); multimodal: (M, D, H, W)
        self.volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.case_ids: List[str] = []

        for t2_path, seg_path, tumor_path in case_triples:
            if self.multimodal:
                image, mask, spacing = load_case_multimodal(
                    t2_path, seg_path, tumor_path, modalities=self.modalities)
            else:
                image, mask, spacing = load_case_nifti(
                    t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            image = resize_volume(image, self.target_size)
            mask = resize_volume(mask, self.target_size, is_mask=True)
            binary_mask = (mask > 0).astype(np.int64)
            self.volumes.append((image, binary_mask))
            self.case_ids.append(_case_id_from_path(t2_path))

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        img, msk = self.volumes[idx]
        if self.augment:
            img, msk = self._augment(img, msk)
        # img is (D, H, W) single-modal or (M, D, H, W) multimodal
        img_t = torch.from_numpy(img.copy()).float()
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)         # (1, D, H, W)
        return {
            "image": img_t,
            "mask": torch.from_numpy(msk.copy()).long(),
            "case_idx": idx,
        }

    def _augment(self, img, msk):
        # Multimodal flag from ndim: 3D = single-modal, 4D = (M, D, H, W)
        mm = img.ndim == 4
        # Flip along the last two spatial axes
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-1).copy()
            msk = np.flip(msk, axis=-1).copy()
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-2).copy()
            msk = np.flip(msk, axis=-2).copy()
        if getattr(self.config, "use_enhanced_aug", False):
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                rot_axes = (2, 3) if mm else (1, 2)
                img = rotate(img, angle, axes=rot_axes, reshape=False, order=3)
                msk = rotate(msk.astype(np.float64), angle, axes=(1, 2),
                             reshape=False, order=0)
                msk = np.round(msk).astype(np.int64)
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)
        return img, msk


class Prostate158CascadeStage2Dataset(Dataset):
    """
    Dataset za Stage 2 cascade (TumorSegNet): vraća cropani T2 volumen i
    odgovarajuću 4-class masku unutar prostate-ROI bounding-boxa.

    Pri treningu se koristi ground-truth bounding box (čist signal), s
    opcionalnim integer jitterom ±``config.cascade_bbox_jitter_voxels`` po
    osi kako bi mreža bila robustna na netočne Stage-1 predikcije pri
    inferenciji. Pri evaluaciji jitter je isključen; vanjski skripta
    ``predict_stage1_bboxes.py`` zamijeni GT bboxove predikcijama Stage-1.

    Cropani volumen se resamplea na ``config.cascade_stage2_volume_size``
    (default 48 × 128 × 128) kako bi sva tri foreground razreda dobila
    veću efektivnu prostornu rezoluciju nego pri jednofaznom 3D treningu.
    """

    def __init__(self, case_triples: List[Tuple[str, str, Optional[str]]],
                 config: Config, augment: bool = False,
                 predicted_bboxes: Optional[Dict[str, Tuple[int, int, int,
                                                            int, int, int]]] = None):
        from .cascade import compute_roi_bbox, jitter_bbox
        self.config = config
        self.augment = augment
        self.full_size = config.volume_size_3d
        self.target_size = getattr(
            config, "cascade_stage2_volume_size", (48, 128, 128))
        self.margin = tuple(getattr(
            config, "cascade_bbox_margin_voxels", (2, 8, 8)))
        self.min_size = tuple(getattr(
            config, "cascade_min_bbox_size", (24, 96, 96)))
        self.jitter = int(getattr(config, "cascade_bbox_jitter_voxels", 5))
        self.predicted_bboxes = predicted_bboxes
        self.multimodal = bool(getattr(config, "multimodal", False))
        self.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
        self._compute_roi_bbox = compute_roi_bbox
        self._jitter_bbox = jitter_bbox

        # Single-modal: image is (D, H, W); multimodal: (M, D, H, W).
        self.volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.case_ids: List[str] = []
        self.gt_bboxes: List[Tuple[int, int, int, int, int, int]] = []
        self.has_tumor: List[bool] = []

        for t2_path, seg_path, tumor_path in case_triples:
            if self.multimodal:
                image, mask, spacing = load_case_multimodal(
                    t2_path, seg_path, tumor_path, modalities=self.modalities)
            else:
                image, mask, spacing = load_case_nifti(
                    t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            image = resize_volume(image, self.full_size)
            mask = resize_volume(mask, self.full_size, is_mask=True)
            self.volumes.append((image, mask))
            self.case_ids.append(_case_id_from_path(t2_path))
            # bbox computed from GT mask (always 3D regardless of multimodal)
            bbox = self._compute_roi_bbox(
                (mask > 0).astype(np.uint8),
                margin_voxels=self.margin, min_size=self.min_size,
                use_largest_cc=False)
            self.gt_bboxes.append(bbox)
            self.has_tumor.append(bool((mask == 3).any()))

    def __len__(self):
        return len(self.volumes)

    def _select_bbox(self, idx: int):
        """Pri evaluaciji koristi predikciju Stage-1 ako je dostupna; inače GT."""
        if self.predicted_bboxes is not None:
            cid = self.case_ids[idx]
            if cid in self.predicted_bboxes:
                return tuple(self.predicted_bboxes[cid])
        return self.gt_bboxes[idx]

    def __getitem__(self, idx):
        img_full, msk_full = self.volumes[idx]
        bbox = self._select_bbox(idx)

        # jitter_bbox needs the (D, H, W) spatial shape (not channel-aware)
        spatial_shape = img_full.shape[-3:] if self.multimodal else img_full.shape
        if self.augment and self.predicted_bboxes is None and self.jitter > 0:
            rng = np.random.default_rng()
            bbox = self._jitter_bbox(bbox, volume_shape=spatial_shape,
                                      max_jitter_voxels=self.jitter,
                                      min_size=self.min_size, rng=rng)

        z0, z1, y0, y1, x0, x1 = bbox
        if self.multimodal:
            img_roi = img_full[:, z0:z1, y0:y1, x0:x1]   # (M, dz, dy, dx)
        else:
            img_roi = img_full[z0:z1, y0:y1, x0:x1]
        msk_roi = msk_full[z0:z1, y0:y1, x0:x1]

        img_roi = resize_volume(img_roi, self.target_size)
        msk_roi = resize_volume(msk_roi, self.target_size, is_mask=True)

        if self.augment:
            img_roi, msk_roi = self._augment(img_roi, msk_roi)

        img_t = torch.from_numpy(img_roi.copy()).float()
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        return {
            "image": img_t,
            "mask": torch.from_numpy(msk_roi.copy()).long(),
            "case_idx": idx,
            "bbox": torch.tensor(bbox, dtype=torch.long),
        }

    def _augment(self, img, msk):
        # img is (D, H, W) single-modal or (M, D, H, W) multimodal
        mm = img.ndim == 4
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-1).copy()
            msk = np.flip(msk, axis=-1).copy()
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=-2).copy()
            msk = np.flip(msk, axis=-2).copy()
        if getattr(self.config, "use_enhanced_aug", False):
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                rot_axes = (2, 3) if mm else (1, 2)
                img = rotate(img, angle, axes=rot_axes, reshape=False, order=3)
                msk = rotate(msk.astype(np.float64), angle, axes=(1, 2),
                             reshape=False, order=0)
                msk = np.round(msk).astype(np.int64)
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)
        return img, msk


# ---------------------------------------------------------------------------
# Factory funkcija
# ---------------------------------------------------------------------------

def create_prostate158_datasets(config: Config) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """Stvara train, validation i test dataset za Prostate158."""
    data_root = config.prostate158_dir

    train_triples, val_triples, test_triples = get_prostate158_paths(data_root)

    print(f"Prostate158 — train: {len(train_triples)}, val: {len(val_triples)}, "
          f"test: {len(test_triples)}")

    # Odaberi Dataset klasu
    if config.model_name == "unet3d" or "_3d" in config.model_name:
        DatasetClass = Prostate158Dataset3D
    elif "_25d" in config.model_name or config.model_name == "unet25d":
        DatasetClass = Prostate158Dataset25D
    else:
        DatasetClass = Prostate158Dataset2D

    use_aug = config.use_augmentation
    train_dataset = DatasetClass(train_triples, config, augment=use_aug)
    val_dataset = DatasetClass(val_triples, config, augment=False)

    test_dataset = None
    if test_triples:
        test_dataset = DatasetClass(test_triples, config, augment=False)

    print(f"Broj uzoraka — train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}"
          + (f", test: {len(test_dataset)}" if test_dataset else ""))

    return train_dataset, val_dataset, test_dataset


def create_prostate158_cascade_stage1_datasets(
        config: Config) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """Cascade Stage 1 (binary prostate localisation)."""
    train_triples, val_triples, test_triples = get_prostate158_paths(
        config.prostate158_dir)
    print(f"Prostate158 cascade Stage 1 — train: {len(train_triples)}, "
          f"val: {len(val_triples)}, test: {len(test_triples)}")
    use_aug = config.use_augmentation
    train_dataset = Prostate158CascadeStage1Dataset(
        train_triples, config, augment=use_aug)
    val_dataset = Prostate158CascadeStage1Dataset(
        val_triples, config, augment=False)
    test_dataset = Prostate158CascadeStage1Dataset(
        test_triples, config, augment=False) if test_triples else None
    return train_dataset, val_dataset, test_dataset


def create_prostate158_cascade_stage2_datasets(
        config: Config,
        predicted_bboxes_val: Optional[Dict[str, Tuple[int, int, int,
                                                         int, int, int]]] = None,
        predicted_bboxes_test: Optional[Dict[str, Tuple[int, int, int,
                                                         int, int, int]]] = None
        ) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Cascade Stage 2 (in-ROI 4-class segmentation).

    Trening dataset uvijek koristi GT bbox + jitter (čist signal).
    Validacijski i test dataset koriste ``predicted_bboxes_*`` ako su predane;
    inače padaju na GT bbox (oracle ceiling, korisno za phase E ablaciju).
    """
    train_triples, val_triples, test_triples = get_prostate158_paths(
        config.prostate158_dir)
    print(f"Prostate158 cascade Stage 2 — train: {len(train_triples)}, "
          f"val: {len(val_triples)}, test: {len(test_triples)}")
    use_aug = config.use_augmentation
    train_dataset = Prostate158CascadeStage2Dataset(
        train_triples, config, augment=use_aug, predicted_bboxes=None)
    val_dataset = Prostate158CascadeStage2Dataset(
        val_triples, config, augment=False,
        predicted_bboxes=predicted_bboxes_val)
    test_dataset = (Prostate158CascadeStage2Dataset(
        test_triples, config, augment=False,
        predicted_bboxes=predicted_bboxes_test)
        if test_triples else None)
    return train_dataset, val_dataset, test_dataset
