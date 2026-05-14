"""
Konfiguracija za segmentaciju prostate na MR slikama.
Sadrži sve hiperparametre i postavke za treniranje i evaluaciju.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- Putanje ----
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = ""
    train_data_dir: str = ""
    test_data_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""

    # ---- Model ----
    # "unet2d", "unet25d", "unet3d"
    model_name: str = "unet2d"
    in_channels: int = 1
    out_channels: int = 4  # 0=background, 1=PZ, 2=CG, 3=tumor
    base_filters: int = 32

    # Za 2.5D U-Net: broj susjednih rezova (kontekst)
    num_adjacent_slices: int = 3  # 1 rez iznad + trenutni + 1 rez ispod

    # ---- Predobrada ----
    # Ciljana veličina reza za 2D i 2.5D
    image_size_2d: Tuple[int, int] = (256, 256)
    # Ciljana veličina volumena za 3D
    volume_size_3d: Tuple[int, int, int] = (32, 256, 256)

    # ---- Treniranje ----
    batch_size: int = 8
    batch_size_3d: int = 2
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20

    # ---- Funkcija gubitka ----
    # "dice", "bce", "combined", "focal", "tversky", "focal_tversky",
    # "combined_focal_tversky"
    loss_function: str = "combined"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    # Class imbalance handling (za rare classes poput tumor):
    #   tumor_weight  — multiplikator za tumor klasu (zadnja klasa) u
    #                   class-weighted CE / Tversky / Focal Tversky.
    #   class_weights — eksplicitni override (lista dužine num_classes).
    tumor_weight: float = 1.0
    class_weights: Tuple[float, float, float, float] = None  # type: ignore
    # Focal / Tversky hiperparametri
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_tversky_gamma: float = 4.0 / 3.0
    ft_weight: float = 0.5
    focal_in_combo_weight: float = 0.5
    # Tumour-positive slice oversampling u 2D / 2.5D dataloaderu
    oversample_tumor: bool = False
    tumor_oversample_factor: float = 5.0

    # ---- Podaci ----
    dataset_name: str = "promise12"  # "promise12" ili "prostate158"
    prostate158_dir: str = ""
    val_split: float = 0.2
    num_workers: int = 4
    random_seed: int = 42
    min_mask_ratio: float = 0.0  # Minimalni udio prostate piksela (0 = bez filtriranja)

    # Napredni preprocessing
    use_advanced_preprocessing: bool = False
    target_spacing: Tuple[float, float, float] = (0.5, 0.5, 3.0)
    clahe_clip_limit: float = 2.0
    roi_margin: int = 20

    # ---- Augmentacija ----
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotation_degrees: int = 15
    scale_range: Tuple[float, float] = (0.9, 1.1)

    # ---- MSDA-Net specifično ----
    deep_supervision: bool = False
    boundary_weight: float = 0.5
    aux_weights: Tuple[float, float, float] = (0.4, 0.3, 0.2)

    # Poboljšano treniranje
    use_cosine_annealing: bool = False
    cosine_T_0: int = 20
    cosine_T_mult: int = 2
    cosine_eta_min: float = 1e-6

    # DSBANet pretrained encoder
    use_pretrained: bool = False
    freeze_encoder_epochs: int = 10  # Zamrzni encoder prvih N epoha
    use_amp: bool = True  # Mixed Precision Training
    use_gan: bool = False  # Adversarial training s PatchGAN
    adv_weight: float = 0.01  # Težina adversarial gubitka

    # Evaluacija
    use_tta: bool = False
    use_postprocess: bool = False

    # Ablacijska studija (DSBANet komponente)
    ablation_use_se: bool = True
    ablation_use_aspp: bool = True
    ablation_use_dag: bool = True
    ablation_use_msaf: bool = True
    ablation_use_ffm: bool = True
    ablation_use_ds: bool = True
    ablation_use_brm: bool = True

    # Poboljšana augmentacija
    use_enhanced_aug: bool = False
    use_elastic_deform: bool = False
    elastic_alpha: float = 100.0
    elastic_sigma: float = 10.0
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)

    # ---- Two-stage cascade (Stage 1 = binary localisation, Stage 2 = in-ROI seg) ----
    cascade_mode: str = "off"                  # "off", "stage1", "stage2"
    cascade_stage2_volume_size: Tuple[int, int, int] = (48, 128, 128)
    cascade_bbox_margin_voxels: Tuple[int, int, int] = (2, 8, 8)
    cascade_min_bbox_size: Tuple[int, int, int] = (24, 96, 96)
    cascade_bbox_jitter_voxels: int = 5
    cascade_stage1_loss: str = "combined"      # "dice", "bce", "combined" — binary task
    cascade_stage2_tumor_weight: float = 10.0  # mirrors the 2D class-imbalance fix
    cascade_oversample_factor: float = 3.0     # case-level oversampling for Stage 2
    cascade_arch: str = "dsba_net_3d"          # "dsba_net_3d" or "unet3d"
    cascade_stage1_base_filters: int = 16      # lightweight Stage 1 when arch=unet3d
    cascade_stage2_base_filters: int = 32      # Stage 2 base filters when arch=unet3d

    # ---- Multimodal input (T2 + ADC + DWI on the Prostate158-registered grid) ----
    multimodal: bool = False
    modalities: Tuple[str, ...] = ("t2", "adc", "dwi")
    cascade_predicted_bboxes_val: dict = None  # type: ignore
    cascade_predicted_bboxes_test: dict = None  # type: ignore

    def __post_init__(self):
        self._update_paths()

    def _update_paths(self):
        """Recalculate all derived paths from current config values."""
        self.dataset_dir = os.path.join(self.project_root, "dataset")
        self.train_data_dir = os.path.join(self.dataset_dir, "training_data")
        self.test_data_dir = os.path.join(self.dataset_dir, "test_data")
        self.prostate158_dir = os.path.join(self.project_root, "prostate158")
        self.output_dir = os.path.join(self.project_root, "output", self.dataset_name)
        if getattr(self, "multimodal", False):
            self.output_dir = os.path.join(self.output_dir, "multimodal")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def effective_batch_size(self) -> int:
        if self.model_name == "unet3d" or "_3d" in self.model_name:
            return self.batch_size_3d
        return self.batch_size
