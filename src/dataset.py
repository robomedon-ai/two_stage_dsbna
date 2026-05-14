"""
Učitavanje i predobrada PROMISE12 skupa podataka za segmentaciju prostate.

Podržava tri načina rada:
  - 2D: svaki rez volumena kao zasebni uzorak
  - 2.5D: N susjednih rezova kao višekanalni ulaz
  - 3D: čitav volumen kao ulaz
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate, map_coordinates, gaussian_filter

from .config import Config


# ---------------------------------------------------------------------------
# Pomoćne funkcije za učitavanje i predobradu
# ---------------------------------------------------------------------------

def load_case(image_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Učitava MHD sliku i pripadajuću segmentacijsku masku.

    Vraća:
        image: numpy niz oblika (D, H, W)
        mask: numpy niz oblika (D, H, W) s vrijednostima {0, 1}
        metadata: rječnik s informacijama o spacingu, originu itd.
    """
    seg_path = image_path.replace(".mhd", "_segmentation.mhd")

    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(seg_path)

    image = sitk.GetArrayFromImage(image_sitk).astype(np.float32)  # (D, H, W)
    mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

    # Binarna maska
    mask = (mask > 0.5).astype(np.float32)

    metadata = {
        "spacing": image_sitk.GetSpacing(),
        "origin": image_sitk.GetOrigin(),
        "direction": image_sitk.GetDirection(),
        "original_size": image_sitk.GetSize(),
    }

    return image, mask, metadata


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalizacija intenziteta pomoću Z-score normalizacije.
    Obrezuje outliere na percentile [0.5, 99.5] prije normalizacije.
    """
    p_low, p_high = np.percentile(image, [0.5, 99.5])
    image = np.clip(image, p_low, p_high)

    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    else:
        image = image - mean

    return image


def resize_slice(image_slice: np.ndarray, target_size: Tuple[int, int],
                 is_mask: bool = False) -> np.ndarray:
    """Mijenja veličinu 2D reza na ciljanu dimenziju."""
    h, w = image_slice.shape
    th, tw = target_size
    zoom_h = th / h
    zoom_w = tw / w

    order = 0 if is_mask else 3  # nearest za masku, cubic za sliku
    resized = zoom(image_slice, (zoom_h, zoom_w), order=order)

    if is_mask:
        resized = (resized > 0.5).astype(np.float32)

    return resized


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int],
                  is_mask: bool = False) -> np.ndarray:
    """Mijenja veličinu 3D volumena na ciljane dimenzije (D, H, W)."""
    d, h, w = volume.shape
    td, th, tw = target_size
    zoom_factors = (td / d, th / h, tw / w)

    order = 0 if is_mask else 3
    resized = zoom(volume, zoom_factors, order=order)

    if is_mask:
        resized = (resized > 0.5).astype(np.float32)

    return resized


def elastic_deformation_2d(image: np.ndarray, mask: np.ndarray,
                           alpha: float = 100.0, sigma: float = 10.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Primjenjuje elastičnu deformaciju na 2D sliku i masku."""
    shape = image.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.clip(y + dy, 0, shape[0] - 1),
               np.clip(x + dx, 0, shape[1] - 1)]

    image_def = map_coordinates(image, indices, order=3, mode='reflect')
    mask_def = map_coordinates(mask, indices, order=0, mode='reflect')
    mask_def = (mask_def > 0.5).astype(np.float32)
    return image_def, mask_def


def get_case_paths(data_dir: str) -> List[str]:
    """Vraća sortirani popis putanja do .mhd datoteka slika (bez segmentacija)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "Case*.mhd")))
    paths = [p for p in paths if "segmentation" not in p]
    return paths


# ---------------------------------------------------------------------------
# Dataset klase
# ---------------------------------------------------------------------------

class ProstateDataset2D(Dataset):
    """
    2D Dataset: svaki rez volumena je zasebni uzorak.
    Ulaz: (1, H, W), Izlaz: (1, H, W)
    """

    def __init__(self, case_paths: List[str], config: Config,
                 augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.image_size_2d

        # Učitaj sve rezove
        self.slices: List[Tuple[np.ndarray, np.ndarray]] = []
        for path in case_paths:
            image, mask, _ = load_case(path)
            image = normalize_intensity(image)
            for i in range(image.shape[0]):
                img_slice = resize_slice(image[i], self.target_size)
                msk_slice = resize_slice(mask[i], self.target_size, is_mask=True)
                self.slices.append((img_slice, msk_slice))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, msk = self.slices[idx]

        if self.augment:
            img, msk = self._augment(img, msk)

        # Dodaj dimenziju kanala: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()

        return {"image": img_tensor, "mask": msk_tensor}

    def _augment(self, img: np.ndarray, msk: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontalni flip
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()

        # Vertikalni flip
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=0).copy()
            msk = np.flip(msk, axis=0).copy()

        # Poboljšana augmentacija
        if getattr(self.config, "use_enhanced_aug", False):
            # Rotacija
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                img = rotate(img, angle, reshape=False, order=3)
                msk = rotate(msk, angle, reshape=False, order=0)
                msk = (msk > 0.5).astype(np.float32)

            # Elastična deformacija
            if np.random.random() < 0.3:
                img, msk = elastic_deformation_2d(
                    img, msk, self.config.elastic_alpha, self.config.elastic_sigma)

            # Augmentacija intenziteta
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)

        return img, msk


class ProstateDataset25D(Dataset):
    """
    2.5D Dataset: N susjednih rezova kao višekanalni ulaz.
    Ulaz: (N, H, W), Izlaz: (1, H, W) za središnji rez.
    """

    def __init__(self, case_paths: List[str], config: Config,
                 augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.image_size_2d
        self.n_adj = config.num_adjacent_slices
        self.half = self.n_adj // 2

        # Učitaj sve volumene i popamti informacije o rezovima
        self.entries: List[Tuple[int, int]] = []  # (volume_idx, slice_idx)
        self.volumes: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []

        for path in case_paths:
            image, mask, _ = load_case(path)
            image = normalize_intensity(image)

            # Preoblikuj svaki rez na ciljanu veličinu
            D = image.shape[0]
            resized_img = np.stack([
                resize_slice(image[i], self.target_size) for i in range(D)
            ])
            resized_msk = np.stack([
                resize_slice(mask[i], self.target_size, is_mask=True) for i in range(D)
            ])

            vol_idx = len(self.volumes)
            self.volumes.append(resized_img)
            self.masks.append(resized_msk)

            for s in range(D):
                self.entries.append((vol_idx, s))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx, slice_idx = self.entries[idx]
        volume = self.volumes[vol_idx]
        mask = self.masks[vol_idx]
        D = volume.shape[0]

        # Prikupi N susjednih rezova s refleksijskim paddingom
        channels = []
        for offset in range(-self.half, self.half + 1):
            s = slice_idx + offset
            s = max(0, min(s, D - 1))  # refleksija na rubovima
            channels.append(volume[s])

        img_multi = np.stack(channels, axis=0)  # (N, H, W)
        msk_slice = mask[slice_idx]  # (H, W)

        if self.augment:
            img_multi, msk_slice = self._augment(img_multi, msk_slice)

        img_tensor = torch.from_numpy(img_multi).float()
        msk_tensor = torch.from_numpy(msk_slice).unsqueeze(0).float()

        return {"image": img_tensor, "mask": msk_tensor}

    def _augment(self, img: np.ndarray, msk: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontalni flip (os W)
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=1).copy()

        # Vertikalni flip (os H)
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=0).copy()

        # Poboljšana augmentacija
        if getattr(self.config, "use_enhanced_aug", False):
            # Rotacija (svaki kanal zasebno)
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                for c in range(img.shape[0]):
                    img[c] = rotate(img[c], angle, reshape=False, order=3)
                msk = rotate(msk, angle, reshape=False, order=0)
                msk = (msk > 0.5).astype(np.float32)

            # Augmentacija intenziteta
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)

        return img, msk


class ProstateDataset3D(Dataset):
    """
    3D Dataset: čitav volumen kao ulaz.
    Ulaz: (1, D, H, W), Izlaz: (1, D, H, W)
    """

    def __init__(self, case_paths: List[str], config: Config,
                 augment: bool = False):
        self.config = config
        self.augment = augment
        self.target_size = config.volume_size_3d

        self.volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        for path in case_paths:
            image, mask, _ = load_case(path)
            image = normalize_intensity(image)

            image = resize_volume(image, self.target_size)
            mask = resize_volume(mask, self.target_size, is_mask=True)

            self.volumes.append((image, mask))

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, msk = self.volumes[idx]

        if self.augment:
            img, msk = self._augment(img, msk)

        # Dodaj dimenziju kanala: (D, H, W) -> (1, D, H, W)
        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).float()
        msk_tensor = torch.from_numpy(msk.copy()).unsqueeze(0).float()

        return {"image": img_tensor, "mask": msk_tensor}

    def _augment(self, img: np.ndarray, msk: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontalni flip (os W)
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=2).copy()

        # Vertikalni flip (os H)
        if np.random.random() < self.config.flip_prob:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()

        # Poboljšana augmentacija
        if getattr(self.config, "use_enhanced_aug", False):
            # Rotacija u aksijalnoj ravnini
            if np.random.random() < 0.5:
                angle = np.random.uniform(-self.config.rotation_degrees,
                                          self.config.rotation_degrees)
                img = rotate(img, angle, axes=(1, 2), reshape=False, order=3)
                msk = rotate(msk, angle, axes=(1, 2), reshape=False, order=0)
                msk = (msk > 0.5).astype(np.float32)

            # Augmentacija intenziteta
            if np.random.random() < 0.5:
                brightness = np.random.uniform(*self.config.brightness_range)
                contrast = np.random.uniform(*self.config.contrast_range)
                img = img * contrast + (brightness - 1.0)

        return img, msk


def create_datasets(config: Config) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Stvara train, validation i opcionalni test dataset.

    Vraća:
        (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(config.random_seed)

    # Dohvati putanje
    train_paths = get_case_paths(config.train_data_dir)
    test_paths = get_case_paths(config.test_data_dir)

    # Podjela na train/validation
    n_total = len(train_paths)
    n_val = int(n_total * config.val_split)
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_case_paths = [train_paths[i] for i in train_indices]
    val_case_paths = [train_paths[i] for i in val_indices]

    print(f"Broj slučajeva za treniranje: {len(train_case_paths)}")
    print(f"Broj validacijskih slučajeva: {len(val_case_paths)}")
    print(f"Broj testnih slučajeva: {len(test_paths)}")

    # Odaberi odgovarajuću Dataset klasu
    if config.model_name == "unet3d" or "_3d" in config.model_name:
        DatasetClass = ProstateDataset3D
    elif "_25d" in config.model_name or config.model_name == "unet25d":
        DatasetClass = ProstateDataset25D
    else:
        DatasetClass = ProstateDataset2D

    use_aug = config.use_augmentation

    train_dataset = DatasetClass(train_case_paths, config, augment=use_aug)
    val_dataset = DatasetClass(val_case_paths, config, augment=False)

    # Test dataset - provjeri ima li segmentacija
    test_dataset = None
    test_with_seg = [p for p in test_paths
                     if os.path.exists(p.replace(".mhd", "_segmentation.mhd"))]
    if test_with_seg:
        test_dataset = DatasetClass(test_with_seg, config, augment=False)

    print(f"Broj uzoraka - train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}"
          + (f", test: {len(test_dataset)}" if test_dataset else ""))

    return train_dataset, val_dataset, test_dataset
