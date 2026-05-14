"""
Generate all figures and tables for the paper.

Figures:
  1. Bar chart: Test DSC comparison (all models, grouped by dimension)
  2. Per-class DSC heatmap
  3. Box plots: Per-slice DSC distributions
  4. Training curves: Val DSC over epochs (2D, 2.5D, 3D panels)
  5. Training curves: Val Loss over epochs
  6. Segmentation examples: GT vs predictions for all 2D and 2.5D models
  7. Radar chart: multi-metric comparison of top models
  8. Dimension comparison: Mean DSC, PZ, Tumor per architecture
  9. Per-class box plots for top models

Output: output/prostate158/paper_figures/
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE, "output", "prostate158", "test_predictions")
HIST_DIR = os.path.join(BASE, "output", "prostate158")
FIG_DIR = os.path.join(BASE, "output", "prostate158", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Configuration ----
MODELS_2D = [
    ("unet2d", "U-Net"),
    ("attention_unet", "Att. U-Net"),
    ("unet_plus_plus", "U-Net++"),
    ("resunet", "ResUNet"),
    ("transunet", "TransUNet"),
    ("swin_unet", "Swin-UNet"),
    ("msda_net", "MSDA-Net"),
    ("dsba_net", "DSBANet"),
]
MODELS_25D = [
    ("unet25d", "U-Net"),
    ("attention_unet_25d", "Att. U-Net"),
    ("unet_plus_plus_25d", "U-Net++"),
    ("resunet_25d", "ResUNet"),
    ("transunet_25d", "TransUNet"),
    ("swin_unet_25d", "Swin-UNet"),
    ("msda_net_25d", "MSDA-Net"),
    ("dsba_net_25d", "DSBANet"),
]
MODELS_3D = [
    ("unet3d", "U-Net"),
    ("attention_unet_3d", "Att. U-Net"),
    ("unet_plus_plus_3d", "U-Net++"),
    ("resunet_3d", "ResUNet"),
    ("transunet_3d", "TransUNet"),
    ("swin_unet_3d", "Swin-UNet"),
    ("msda_net_3d", "MSDA-Net"),
    ("dsba_net_3d", "DSBANet"),
]

ARCH_NAMES = ["U-Net", "Att. U-Net", "U-Net++", "ResUNet",
              "TransUNet", "Swin-UNet", "MSDA-Net", "DSBANet"]

COLOR_2D = "#2196F3"
COLOR_25D = "#4CAF50"
COLOR_3D = "#FF9800"
ARCH_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_metadata(model_key):
    path = os.path.join(PRED_DIR, model_key, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_history(model_key):
    for name in [model_key, model_key.replace("_25d", "25d")]:
        path = os.path.join(HIST_DIR, f"history_{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


# =========================================================================
# Tumour-positive-only evaluation helpers
# =========================================================================
def _get_tumor_positive_slices():
    """Return list of global slice indices that have tumour in GT."""
    gt_dir = os.path.join(PRED_DIR, "ground_truth")
    tumor_slices = []
    for i in range(476):
        mask_path = os.path.join(gt_dir, f"mask_{i:04d}.npy")
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path, allow_pickle=True)
        if 3 in np.unique(mask).astype(int):
            tumor_slices.append(i)
    return tumor_slices

# Cache it once
_TUMOR_SLICES = None
def get_tumor_slices():
    global _TUMOR_SLICES
    if _TUMOR_SLICES is None:
        _TUMOR_SLICES = _get_tumor_positive_slices()
    return _TUMOR_SLICES


def corrected_metrics(model_key):
    """Return (pz, cg, tum, mean_dsc) with tumour evaluated on positive slices only."""
    meta = load_metadata(model_key)
    if not meta or not meta.get("per_slice"):
        return 0, 0, 0, 0
    per = meta["per_slice"]
    pz = np.mean([s.get("DSC_PZ", 0) for s in per])
    cg = np.mean([s.get("DSC_CG", 0) for s in per])

    tumor_slices = get_tumor_slices()
    n_items = len(per)

    if n_items == 19:
        # 3D model: all 19 volumes have tumour
        tum = np.mean([s.get("DSC_Tumor", 0) for s in per])
    else:
        # 2D/2.5D: evaluate only on tumour-positive slices
        tum_vals = [per[s]["DSC_Tumor"] for s in tumor_slices if s < n_items]
        tum = np.mean(tum_vals) if tum_vals else 0.0

    mean_dsc = (pz + cg + tum) / 3.0
    return pz, cg, tum, mean_dsc


# =========================================================================
# Figure 1: Bar chart — Test DSC by model and dimension
# =========================================================================
def fig1_dsc_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ARCH_NAMES))
    width = 0.25

    dsc_2d, dsc_25d, dsc_3d = [], [], []
    for models, store in [(MODELS_2D, dsc_2d), (MODELS_25D, dsc_25d), (MODELS_3D, dsc_3d)]:
        for key, _ in models:
            _, _, _, mean_dsc = corrected_metrics(key)
            store.append(mean_dsc)

    bars1 = ax.bar(x - width, dsc_2d, width, label="2D", color=COLOR_2D, edgecolor="white")
    bars2 = ax.bar(x, dsc_25d, width, label="2.5D", color=COLOR_25D, edgecolor="white")
    bars3 = ax.bar(x + width, dsc_3d, width, label="3D", color=COLOR_3D, edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_ylabel("Dice Similarity Coefficient (DSC)")
    ax.set_title("Test Set Performance: Multi-class Prostate Segmentation")
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_NAMES, rotation=15, ha="right")
    ax.legend(title="Input Dimension")
    ax.set_ylim(0, 0.95)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_dsc_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_dsc_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 1: DSC comparison bar chart")


# =========================================================================
# Figure 2: Per-class DSC heatmap
# =========================================================================
def fig2_perclass_heatmap():
    all_models = MODELS_2D + MODELS_25D + MODELS_3D
    labels = []
    data = []

    for key, name in MODELS_2D:
        labels.append(f"{name} (2D)")
    for key, name in MODELS_25D:
        labels.append(f"{name} (2.5D)")
    for key, name in MODELS_3D:
        labels.append(f"{name} (3D)")

    for key, name in all_models:
        pz, cg, tum, mean_dsc = corrected_metrics(key)
        data.append([pz, cg, tum, mean_dsc])

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["PZ", "CG", "Tumor", "Mean DSC"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(4):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.axhline(y=7.5, color="black", linewidth=2)
    ax.axhline(y=15.5, color="black", linewidth=2)

    ax.set_title("Per-class Dice Scores on Test Set")
    plt.colorbar(im, ax=ax, label="DSC", shrink=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_perclass_heatmap.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig2_perclass_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 2: Per-class DSC heatmap")


# =========================================================================
# Figure 3: Box plots — Per-case DSC distributions
# =========================================================================
def _build_case_boundaries():
    """Return list of (start, end) slice indices for each of the 19 test cases."""
    import nibabel as nib
    ref_dir = os.path.join(BASE, "prostate158", "prostate158_test",
                           "prostate158_test", "test")
    cases = sorted(os.listdir(ref_dir))
    boundaries = []
    cum = 0
    for c in cases:
        ref = nib.load(os.path.join(ref_dir, c, "t2.nii.gz"))
        n = ref.shape[2]
        boundaries.append((cum, cum + n))
        cum += n
    return boundaries


def _per_case_dsc(meta, case_boundaries):
    """Aggregate per-slice corrected DSC into per-case DSC.

    Uses tumour-positive-only evaluation: for each slice, recompute
    mean DSC as (PZ + CG + Tum_corrected) / 3, where Tum is only
    counted on tumour-positive slices.
    """
    if not meta or not meta.get("per_slice"):
        return []
    per = meta["per_slice"]
    n_items = len(per)
    tumor_slices = set(get_tumor_slices())

    if n_items == len(case_boundaries):
        # Already per-case (3D models): recompute mean with corrected tumour
        case_dscs = []
        for s in per:
            pz = s.get("DSC_PZ", 0)
            cg = s.get("DSC_CG", 0)
            tum = s.get("DSC_Tumor", 0)
            case_dscs.append((pz + cg + tum) / 3.0)
        return case_dscs

    # Per-slice (2D/2.5D): recompute per-slice DSC, aggregate by case
    corrected_dscs = []
    for i, s in enumerate(per):
        pz = s.get("DSC_PZ", 0)
        cg = s.get("DSC_CG", 0)
        if i in tumor_slices:
            tum = s.get("DSC_Tumor", 0)
        else:
            tum = None  # exclude from mean for this slice
        if tum is not None:
            corrected_dscs.append((pz + cg + tum) / 3.0)
        else:
            corrected_dscs.append((pz + cg) / 2.0)

    case_dscs = []
    for start, end in case_boundaries:
        if end <= n_items:
            case_dscs.append(float(np.mean(corrected_dscs[start:end])))
    return case_dscs


def fig3_boxplots():
    case_boundaries = _build_case_boundaries()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, models, title in [
        (axes[0], MODELS_2D, "2D Models"),
        (axes[1], MODELS_25D, "2.5D Models"),
        (axes[2], MODELS_3D, "3D Models"),
    ]:
        box_data = []
        box_labels = []
        for key, name in models:
            meta = load_metadata(key)
            dscs = _per_case_dsc(meta, case_boundaries)
            if dscs:
                box_data.append(dscs)
                box_labels.append(name)

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        showfliers=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))

        for patch, color in zip(bp["boxes"], ARCH_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_ylabel("DSC" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    fig.suptitle("Distribution of Per-case Dice Scores on Test Set (19 cases)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_boxplots.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_boxplots.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 3: Box plots per-case (2D, 2.5D, 3D)")


# =========================================================================
# Figure 4: Training curves — Val DSC
# =========================================================================
def fig4_training_curves_dsc():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, models, title in [
        (axes[0], MODELS_2D, "2D Models"),
        (axes[1], MODELS_25D, "2.5D Models"),
        (axes[2], MODELS_3D, "3D Models"),
    ]:
        for i, (key, name) in enumerate(models):
            hist = load_history(key)
            if hist and "val_dsc" in hist:
                epochs = range(1, len(hist["val_dsc"]) + 1)
                ax.plot(epochs, hist["val_dsc"], label=name,
                        color=ARCH_COLORS[i], linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if ax == axes[0]:
            ax.set_ylabel("Validation DSC")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.95)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Validation DSC During Training", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_training_dsc.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig4_training_dsc.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 4: Training curves (DSC)")


# =========================================================================
# Figure 5: Training curves — Val Loss
# =========================================================================
def fig5_training_curves_loss():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, models, title in [
        (axes[0], MODELS_2D, "2D Models"),
        (axes[1], MODELS_25D, "2.5D Models"),
        (axes[2], MODELS_3D, "3D Models"),
    ]:
        for i, (key, name) in enumerate(models):
            hist = load_history(key)
            if hist and "val_loss" in hist:
                epochs = range(1, len(hist["val_loss"]) + 1)
                ax.plot(epochs, hist["val_loss"], label=name,
                        color=ARCH_COLORS[i], linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if ax == axes[0]:
            ax.set_ylabel("Validation Loss")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Validation Loss During Training", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_training_loss.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig5_training_loss.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 5: Training curves (Loss)")


# =========================================================================
# Figure 6: Segmentation examples
# =========================================================================
def fig6_segmentation_examples():
    gt_dir = os.path.join(PRED_DIR, "ground_truth")
    if not os.path.isdir(gt_dir):
        print("Fig 6: SKIP — no ground truth")
        return

    # Find slices with all 3 foreground classes
    good_slices = []
    for i in range(476):
        mask_path = os.path.join(gt_dir, f"mask_{i:04d}.npy")
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path, allow_pickle=True)
        uniq = set(np.unique(mask).astype(int))
        if {1, 2, 3}.issubset(uniq):
            good_slices.append(i)

    if len(good_slices) < 3:
        for i in range(476):
            mask_path = os.path.join(gt_dir, f"mask_{i:04d}.npy")
            if not os.path.exists(mask_path):
                continue
            mask = np.load(mask_path, allow_pickle=True)
            uniq = set(np.unique(mask).astype(int))
            if {1, 2}.issubset(uniq) and i not in good_slices:
                good_slices.append(i)

    indices = [good_slices[0],
               good_slices[len(good_slices)//2],
               good_slices[-1]]

    def overlay_mask(img, mask, alpha=0.45):
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rgb = np.stack([img_norm] * 3, axis=-1)
        class_colors = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0.4, 1]),
        }
        for cls_id, color in class_colors.items():
            region = mask == cls_id
            if region.any():
                rgb[region] = rgb[region] * (1 - alpha) + color * alpha
        return np.clip(rgb, 0, 1)

    legend_patches = [
        mpatches.Patch(color=[0, 1, 0], label="PZ"),
        mpatches.Patch(color=[1, 0, 0], label="CG"),
        mpatches.Patch(color=[0, 0.4, 1], label="Tumor"),
    ]

    # 2D models figure
    for suffix, models, title in [
        ("2d", MODELS_2D, "2D Models"),
        ("25d", MODELS_25D, "2.5D Models"),
    ]:
        n_cols = 2 + len(models)
        n_rows = len(indices)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.5 * n_cols, 2.5 * n_rows))

        for row, slice_idx in enumerate(indices):
            img = np.load(os.path.join(gt_dir, f"image_{slice_idx:04d}.npy"),
                          allow_pickle=True)
            gt_mask = np.load(os.path.join(gt_dir, f"mask_{slice_idx:04d}.npy"),
                              allow_pickle=True)

            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("MRI", fontsize=10, fontweight="bold")

            axes[row, 1].imshow(overlay_mask(img, gt_mask))
            axes[row, 1].axis("off")
            if row == 0:
                axes[row, 1].set_title("Ground Truth", fontsize=10, fontweight="bold")

            for col, (key, name) in enumerate(models):
                pred_path = os.path.join(PRED_DIR, key, f"mask_{slice_idx:04d}.npy")
                if os.path.exists(pred_path):
                    pred = np.load(pred_path, allow_pickle=True)
                    axes[row, col + 2].imshow(overlay_mask(img, pred))
                else:
                    axes[row, col + 2].imshow(img, cmap="gray")
                axes[row, col + 2].axis("off")
                if row == 0:
                    axes[row, col + 2].set_title(name, fontsize=10, fontweight="bold")

        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                   fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Segmentation Results: {title}", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"fig6_segmentation_{suffix}.png"),
                    bbox_inches="tight")
        plt.savefig(os.path.join(FIG_DIR, f"fig6_segmentation_{suffix}.pdf"),
                    bbox_inches="tight")
        plt.close()

    # 3D models figure: need to extract slices from 3D volumes
    # Map 2D slice indices to (case_volume_idx, local_slice_idx)
    import nibabel as nib
    ref_dir = os.path.join(BASE, "prostate158", "prostate158_test",
                           "prostate158_test", "test")
    ref_cases = sorted(os.listdir(ref_dir))
    cumulative = 0
    case_map = []  # (case_id, global_start, global_end, n_slices)
    for c in ref_cases:
        ref = nib.load(os.path.join(ref_dir, c, "t2.nii.gz"))
        n = ref.shape[2]
        case_map.append((c, cumulative, cumulative + n, n))
        cumulative += n

    def global_to_3d(global_idx):
        """Convert global 2D slice index to (volume_idx, local_slice)."""
        for vol_idx, (_, start, end, _) in enumerate(case_map):
            if start <= global_idx < end:
                return vol_idx, global_idx - start
        return None, None

    n_cols = 2 + len(MODELS_3D)
    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows))

    for row, slice_idx in enumerate(indices):
        # Load same MRI image and GT as 2D figures
        img = np.load(os.path.join(gt_dir, f"image_{slice_idx:04d}.npy"),
                      allow_pickle=True)
        gt_mask = np.load(os.path.join(gt_dir, f"mask_{slice_idx:04d}.npy"),
                          allow_pickle=True)

        vol_idx, local_slice = global_to_3d(slice_idx)

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("MRI", fontsize=10, fontweight="bold")

        axes[row, 1].imshow(overlay_mask(img, gt_mask))
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Ground Truth", fontsize=10, fontweight="bold")

        for col, (key, name) in enumerate(MODELS_3D):
            pred_path = os.path.join(PRED_DIR, key, f"mask_{vol_idx:04d}.npy")
            if os.path.exists(pred_path):
                vol = np.load(pred_path, allow_pickle=True)
                # vol shape: (D_padded, H, W) where D_padded=32
                if local_slice < vol.shape[0]:
                    pred_slice = vol[local_slice]
                else:
                    pred_slice = np.zeros_like(img, dtype=np.int64)
                # Resize from 256x256 to match GT image size if needed
                if pred_slice.shape != img.shape:
                    from scipy.ndimage import zoom
                    pred_slice = zoom(pred_slice.astype(np.float64),
                                     (img.shape[0] / pred_slice.shape[0],
                                      img.shape[1] / pred_slice.shape[1]),
                                     order=0).astype(np.int64)
                axes[row, col + 2].imshow(overlay_mask(img, pred_slice))
            else:
                axes[row, col + 2].imshow(img, cmap="gray")
            axes[row, col + 2].axis("off")
            if row == 0:
                axes[row, col + 2].set_title(name, fontsize=10, fontweight="bold")

    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Segmentation Results: 3D Models", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_segmentation_3d.png"),
                bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig6_segmentation_3d.pdf"),
                bbox_inches="tight")
    plt.close()

    print("Fig 6: Segmentation examples (2D + 2.5D + 3D)")


# =========================================================================
# Figure 7: Radar chart
# =========================================================================
def fig7_radar_chart():
    top_models = [
        ("dsba_net", "DSBANet 2D"),
        ("dsba_net_25d", "DSBANet 2.5D"),
        ("msda_net", "MSDA-Net 2D"),
        ("msda_net_25d", "MSDA-Net 2.5D"),
        ("resunet_25d", "ResUNet 2.5D"),
        ("attention_unet_25d", "Att. U-Net 2.5D"),
    ]

    metrics = ["DSC (Mean)", "DSC (PZ)", "DSC (CG)", "DSC (Tumor)"]
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(top_models)))

    for i, (key, label) in enumerate(top_models):
        pz, cg, tum, mean_dsc = corrected_metrics(key)
        values = [mean_dsc, pz, cg, tum]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Multi-metric Comparison of Top Models", y=1.1, fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_radar_chart.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig7_radar_chart.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 7: Radar chart")


# =========================================================================
# Figure 8: Dimension comparison
# =========================================================================
def fig8_dimension_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric_key, metric_name in [
        (axes[0], "DSC", "Mean DSC"),
        (axes[1], "DSC_PZ", "DSC (PZ)"),
        (axes[2], "DSC_Tumor", "DSC (Tumor)"),
    ]:
        vals_2d, vals_25d, vals_3d = [], [], []

        for models, store in [(MODELS_2D, vals_2d), (MODELS_25D, vals_25d),
                               (MODELS_3D, vals_3d)]:
            for key, name in models:
                pz, cg, tum, mean_dsc = corrected_metrics(key)
                if metric_key == "DSC":
                    store.append(mean_dsc)
                elif metric_key == "DSC_PZ":
                    store.append(pz)
                elif metric_key == "DSC_Tumor":
                    store.append(tum)
                else:
                    store.append(0)

        x = np.arange(len(ARCH_NAMES))
        width = 0.25
        ax.bar(x - width, vals_2d, width, label="2D", color=COLOR_2D)
        ax.bar(x, vals_25d, width, label="2.5D", color=COLOR_25D)
        ax.bar(x + width, vals_3d, width, label="3D", color=COLOR_3D)

        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(ARCH_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Impact of Input Dimensionality on Segmentation Performance",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_dimension_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig8_dimension_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 8: Dimension comparison")


# =========================================================================
# Figure 9: Per-class box plots for top models
# =========================================================================
def fig9_perclass_boxplots():
    fig, ax = plt.subplots(figsize=(12, 6))

    models_to_show = [
        ("dsba_net", "DSBANet 2D"),
        ("dsba_net_25d", "DSBANet 2.5D"),
        ("msda_net", "MSDA-Net 2D"),
        ("msda_net_25d", "MSDA-Net 2.5D"),
        ("resunet_25d", "ResUNet 2.5D"),
        ("attention_unet", "Att. U-Net 2D"),
    ]

    class_info = [("PZ", "#4CAF50"), ("CG", "#F44336"), ("Tumor", "#2196F3")]
    positions = []
    box_data = []
    colors_list = []
    group_centers = []

    tumor_slices = set(get_tumor_slices())

    pos = 0
    for key, name in models_to_show:
        meta = load_metadata(key)
        if not meta:
            continue
        per = meta["per_slice"]
        n_items = len(per)
        group_start = pos
        for cls_name, color in class_info:
            if cls_name == "Tumor" and n_items > 19:
                # 2D/2.5D: only tumour-positive slices
                data = [per[s].get("DSC_Tumor", 0)
                        for s in tumor_slices if s < n_items]
            else:
                data = [s.get(f"DSC_{cls_name}", 0) for s in per]
            box_data.append(data)
            positions.append(pos)
            colors_list.append(color)
            pos += 1
        group_centers.append((group_start + pos - 1) / 2.0)
        pos += 1  # gap

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                    showfliers=False, widths=0.7,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([n for _, n in models_to_show], rotation=15, ha="right")

    legend_patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
                      for l, c in class_info]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_ylabel("Dice Similarity Coefficient")
    ax.set_title("Per-class DSC Distribution for Top Models")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_perclass_boxplots.png"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig9_perclass_boxplots.pdf"), bbox_inches="tight")
    plt.close()
    print("Fig 9: Per-class box plots")


# =========================================================================
# LaTeX results table
# =========================================================================
def generate_latex_table():
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-class prostate segmentation results on the Prostate158 test set. "
                 r"DSC values are reported as mean $\pm$ standard deviation.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & \textbf{Dim.} & \textbf{DSC (Mean)} & "
                 r"\textbf{DSC (PZ)} & \textbf{DSC (CG)} & \textbf{DSC (Tumor)} \\")
    lines.append(r"\hline")

    for dim_label, models in [("2D", MODELS_2D), ("2.5D", MODELS_25D), ("3D", MODELS_3D)]:
        for key, name in models:
            pz, cg, tum, mean_dsc = corrected_metrics(key)
            lines.append(f"{name} & {dim_label} & {mean_dsc:.4f} & "
                         f"{pz:.4f} & {cg:.4f} & {tum:.4f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(os.path.join(FIG_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(lines))

    # Print plain text table
    print("\n" + "=" * 100)
    print(f"{'Model':<15} {'Dim':<5} {'Mean DSC':>8} {'PZ':>8} {'CG':>8} {'Tumor':>8}")
    print("-" * 100)
    for dim_label, models in [("2D", MODELS_2D), ("2.5D", MODELS_25D), ("3D", MODELS_3D)]:
        for key, name in models:
            pz, cg, tum, mean_dsc = corrected_metrics(key)
            print(f"{name:<15} {dim_label:<5} {mean_dsc:>8.4f}"
                  f" {pz:>8.4f} {cg:>8.4f} {tum:>8.4f}")
        print("-" * 100)
    print("LaTeX table saved")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print(f"Generating paper figures in: {FIG_DIR}\n")

    fig1_dsc_comparison()
    fig2_perclass_heatmap()
    fig3_boxplots()
    fig4_training_curves_dsc()
    fig5_training_curves_loss()
    fig6_segmentation_examples()
    fig7_radar_chart()
    fig8_dimension_comparison()
    fig9_perclass_boxplots()
    generate_latex_table()

    print(f"\nAll figures saved to: {FIG_DIR}")
    print(f"Files: {sorted(os.listdir(FIG_DIR))}")
