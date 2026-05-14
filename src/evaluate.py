"""
Evaluacija i vizualizacija rezultata multi-class segmentacije prostate.

Klase: 0=pozadina, 1=PZ (zelena), 2=CG (crvena), 3=tumor (plava).

Uključuje:
  - Evaluaciju modela na testnom skupu
  - Vizualizaciju predikcija (slike, maske, overlay)
  - Grafikone treniranja (loss, DSC, IoU)
  - Usporedbu modela
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .metrics import MetricTracker, compute_all_metrics


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def keep_largest_component_multiclass(mask: np.ndarray, num_classes: int = 4
                                       ) -> np.ndarray:
    """Zadrži samo najveću povezanu komponentu za svaku foreground klasu."""
    result = np.zeros_like(mask)
    for c in range(1, num_classes):
        class_mask = (mask == c).astype(np.float32)
        if class_mask.sum() == 0:
            continue
        labeled, num_features = ndimage.label(class_mask)
        if num_features <= 1:
            result[class_mask > 0] = c
            continue
        component_sizes = ndimage.sum(class_mask, labeled,
                                       range(1, num_features + 1))
        largest = np.argmax(component_sizes) + 1
        result[labeled == largest] = c
    return result


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_tta(model: nn.Module, images: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    Predikcija s Test-Time Augmentation.

    Primjenjuje 4 transformacije (original, h-flip, v-flip, h+v-flip),
    prosječuje softmax predikcije i vraća logite.
    """
    model.eval()
    images = images.to(device)

    # Original
    logits = model(images)
    if isinstance(logits, dict):
        logits = logits["main"]
    probs_sum = F.softmax(logits, dim=1)

    # Horizontalni flip
    flipped_h = torch.flip(images, dims=[-1])
    out_h = model(flipped_h)
    if isinstance(out_h, dict):
        out_h = out_h["main"]
    probs_sum = probs_sum + F.softmax(torch.flip(out_h, dims=[-1]), dim=1)

    # Vertikalni flip
    flipped_v = torch.flip(images, dims=[-2])
    out_v = model(flipped_v)
    if isinstance(out_v, dict):
        out_v = out_v["main"]
    probs_sum = probs_sum + F.softmax(torch.flip(out_v, dims=[-2]), dim=1)

    # Oba flipa
    flipped_hv = torch.flip(images, dims=[-1, -2])
    out_hv = model(flipped_hv)
    if isinstance(out_hv, dict):
        out_hv = out_hv["main"]
    probs_sum = probs_sum + F.softmax(torch.flip(out_hv, dims=[-1, -2]), dim=1)

    # Prosječna predikcija — vrati kao logite (log-probs)
    avg_probs = probs_sum / 4.0
    # Convert back to logits: log(p) (safe)
    avg_probs = torch.clamp(avg_probs, 1e-7, 1.0)
    logits_out = torch.log(avg_probs)
    return logits_out


# ---------------------------------------------------------------------------
# Evaluacija na testnom skupu
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   device: torch.device,
                   use_tta: bool = False,
                   use_postprocess: bool = False,
                   skip_empty: bool = False,
                   num_classes: int = 4) -> Dict[str, float]:
    """
    Evaluira model i vraća prosječne metrike.

    Args:
        use_tta: Koristi Test-Time Augmentation
        use_postprocess: Zadrži samo najveću povezanu komponentu per class
        skip_empty: Preskoči rezove s praznom GT maskom pri izračunu metrika
    """
    model.eval()
    metric_tracker = MetricTracker()

    for batch in tqdm(dataloader, desc="Evaluacija"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if use_tta:
            outputs = predict_with_tta(model, images, device)
        else:
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["main"]

        if use_postprocess:
            preds = outputs.argmax(dim=1).cpu().numpy()
            for i in range(preds.shape[0]):
                preds[i] = keep_largest_component_multiclass(
                    preds[i], num_classes=num_classes)
            # Convert back to one-hot-ish logits for metric computation
            preds_tensor = torch.from_numpy(preds).long().to(device)
            nc = outputs.shape[1]
            outputs = F.one_hot(preds_tensor, nc).permute(0, -1,
                *range(1, preds_tensor.dim())).float() * 10.0

        metrics = compute_all_metrics(outputs, masks, skip_empty=skip_empty,
                                       num_classes=num_classes)
        count = metrics.pop("_count", images.size(0))
        if count > 0:
            metric_tracker.update(metrics, count=count)

    return metric_tracker.compute()


@torch.no_grad()
def evaluate_per_case(model: nn.Module, dataloader: DataLoader,
                      device: torch.device,
                      case_ids: Optional[List[str]] = None,
                      num_classes: int = 4,
                      use_tta: bool = False,
                      use_postprocess: bool = False,
                      smooth: float = 1e-6) -> Dict[str, Dict[str, float]]:
    """
    Računa DSC po slučaju i klasi (PZ, CG, Tumor), agregirajući intersection /
    union sumu kroz sve uzorke pripadajućeg slučaja (case_idx).

    Vraća:
        {case_id: {"PZ": dsc, "CG": dsc, "Tumor": dsc}}
        gdje je case_id `case_ids[case_idx]` ako je predan, inače string indeksa.
    """
    model.eval()

    # Akumulatori po (case_idx, class)
    inter_acc: Dict[int, Dict[int, float]] = {}
    union_acc: Dict[int, Dict[int, float]] = {}

    def _acc(case_idx: int, c: int, inter: float, union: float):
        if case_idx not in inter_acc:
            inter_acc[case_idx] = {cc: 0.0 for cc in range(1, num_classes)}
            union_acc[case_idx] = {cc: 0.0 for cc in range(1, num_classes)}
        inter_acc[case_idx][c] += inter
        union_acc[case_idx][c] += union

    for batch in tqdm(dataloader, desc="Per-case eval"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        case_idxs = batch.get("case_idx")
        if case_idxs is None:
            # Bez case_idx ne možemo grupirati — preskoči
            return {}
        if isinstance(case_idxs, torch.Tensor):
            case_idxs = case_idxs.cpu().tolist()

        if use_tta:
            outputs = predict_with_tta(model, images, device)
        else:
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["main"]

        if use_postprocess:
            preds = outputs.argmax(dim=1).cpu().numpy()
            for i in range(preds.shape[0]):
                preds[i] = keep_largest_component_multiclass(preds[i])
            preds = torch.from_numpy(preds).long().to(device)
        else:
            preds = outputs.argmax(dim=1)

        for b in range(preds.size(0)):
            ci = int(case_idxs[b])
            p = preds[b]
            t = masks[b]
            for c in range(1, num_classes):
                pc = (p == c).float()
                tc = (t == c).float()
                inter = (pc * tc).sum().item()
                union = pc.sum().item() + tc.sum().item()
                _acc(ci, c, inter, union)

    # Compute per-case DSC
    from .metrics import CLASS_NAMES
    result: Dict[str, Dict[str, float]] = {}
    for case_idx in sorted(inter_acc.keys()):
        key = case_ids[case_idx] if case_ids and case_idx < len(case_ids) else str(case_idx)
        per_class = {}
        for c in range(1, num_classes):
            i = inter_acc[case_idx][c]
            u = union_acc[case_idx][c]
            dsc = (2.0 * i + smooth) / (u + smooth)
            per_class[CLASS_NAMES[c]] = float(dsc)
        result[key] = per_class
    return result


def wilcoxon_full_vs_variants(per_case_results: Dict[str, Dict[str, Dict[str, float]]],
                              full_key: str = "DSBANet (full)"
                              ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Wilcoxon signed-rank test: Full vs each ablated variant, per class,
    with Holm-Bonferroni correction over all variants × classes.

    Args:
        per_case_results: {variant_name: {case_id: {class_name: dsc}}}
        full_key: ime varijante koja predstavlja Full model

    Vraća:
        {variant_name: {class_name: {"stat": ..., "p_raw": ...,
                                     "p_holm": ..., "median_diff": ...,
                                     "n": ...}}}
    """
    from scipy.stats import wilcoxon

    if full_key not in per_case_results:
        return {}

    full_data = per_case_results[full_key]
    variant_names = [v for v in per_case_results.keys() if v != full_key]
    classes = ["PZ", "CG", "Tumor"]

    raw: List[Tuple[str, str, float, float, float, int]] = []  # (variant, class, stat, p, median_diff, n)

    for v in variant_names:
        v_data = per_case_results[v]
        # Set zajedničkih case_id
        common = sorted(set(full_data.keys()) & set(v_data.keys()))
        for c in classes:
            f_vals = [full_data[k].get(c) for k in common]
            v_vals = [v_data[k].get(c) for k in common]
            paired = [(a, b) for a, b in zip(f_vals, v_vals)
                      if a is not None and b is not None and not (np.isnan(a) or np.isnan(b))]
            if len(paired) < 2:
                raw.append((v, c, float("nan"), float("nan"), float("nan"), len(paired)))
                continue
            a = np.array([p[0] for p in paired])
            b = np.array([p[1] for p in paired])
            diffs = a - b
            if np.allclose(diffs, 0):
                raw.append((v, c, float("nan"), 1.0, 0.0, len(paired)))
                continue
            try:
                stat, p = wilcoxon(a, b, alternative="two-sided",
                                   zero_method="wilcox")
                stat_v, p_v = float(stat), float(p)
            except Exception:
                stat_v, p_v = float("nan"), float("nan")
            raw.append((v, c, stat_v, p_v, float(np.median(diffs)), len(paired)))

    # Holm-Bonferroni correction over all (variant, class) pairs
    valid = [(i, r[3]) for i, r in enumerate(raw) if not np.isnan(r[3])]
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    p_holm = [float("nan")] * len(raw)
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(valid):
        adj = min(1.0, max(running_max, p * (m - rank)))
        p_holm[orig_idx] = adj
        running_max = adj

    # Pack
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, (v, c, stat, p, mdiff, n) in enumerate(raw):
        out.setdefault(v, {})[c] = {
            "stat": stat,
            "p_raw": p,
            "p_holm": p_holm[i],
            "median_diff_full_minus_variant": mdiff,
            "n": n,
        }
    return out


@torch.no_grad()
def evaluate_ensemble(models: list, dataloader: DataLoader,
                      device: torch.device,
                      use_tta: bool = False,
                      use_postprocess: bool = False) -> Dict[str, float]:
    """Evaluira ensemble modela prosječenjem softmax predikcija."""
    for m in models:
        m.eval()
    metric_tracker = MetricTracker()

    for batch in tqdm(dataloader, desc="Evaluacija (ensemble)"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        probs_sum = None
        for model in models:
            if use_tta:
                logits = predict_with_tta(model, images, device)
            else:
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["main"]
            probs = F.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

        avg_probs = probs_sum / len(models)

        if use_postprocess:
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            for i in range(preds.shape[0]):
                preds[i] = keep_largest_component_multiclass(preds[i])
            preds_tensor = torch.from_numpy(preds).long().to(device)
            num_classes = avg_probs.shape[1]
            outputs = F.one_hot(preds_tensor, num_classes).permute(0, -1,
                *range(1, preds_tensor.dim())).float() * 10.0
        else:
            avg_probs = torch.clamp(avg_probs, 1e-7, 1.0)
            outputs = torch.log(avg_probs)

        metrics = compute_all_metrics(outputs, masks)
        metric_tracker.update(metrics, count=images.size(0))

    return metric_tracker.compute()


@torch.no_grad()
def predict_batch(model: nn.Module, images: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    """Generira class-label predikcije za batch slika."""
    model.eval()
    images = images.to(device)
    outputs = model(images)
    if isinstance(outputs, dict):
        outputs = outputs["main"]
    preds = outputs.argmax(dim=1).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# Vizualizacija
# ---------------------------------------------------------------------------

# Boje za multi-class overlay
# 0=background (transparent), 1=PZ (green), 2=CG (red), 3=tumor (blue)
CLASS_COLORS = {
    1: (0.0, 1.0, 0.0, 0.4),   # PZ - zelena
    2: (1.0, 0.0, 0.0, 0.4),   # CG - crvena
    3: (0.0, 0.0, 1.0, 0.4),   # Tumor - plava
}


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Pretvara integer masku u RGBA sliku za overlay."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        region = mask == cls_id
        if region.any():
            rgba[region] = color
    return rgba


def plot_training_history(history: Dict[str, list], save_path: str):
    """Crta grafikone treniranja: loss, DSC i IoU."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validacija")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Gubitak")
    axes[0].set_title("Funkcija gubitka")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # DSC
    axes[1].plot(epochs, history["train_dsc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dsc"], "r-", label="Validacija")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("DSC")
    axes[1].set_title("Dice Similarity Coefficient")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(epochs, history["train_iou"], "b-", label="Train")
    axes[2].plot(epochs, history["val_iou"], "r-", label="Validacija")
    axes[2].set_xlabel("Epoha")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Intersection over Union")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafikon treniranja spremljen: {save_path}")


def plot_segmentation_examples(model: nn.Module, dataset, device: torch.device,
                               save_path: str, num_examples: int = 6,
                               is_3d: bool = False):
    """
    Vizualizira primjere segmentacije: sliku, ground truth i predikciju.
    Multi-class: zelena=PZ, crvena=CG, plava=tumor.
    """
    model.eval()
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))

    indices = np.linspace(0, len(dataset) - 1, num_examples, dtype=int)

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"]  # (H, W) or (D, H, W) integer labels

        with torch.no_grad():
            output = model(image)
            if isinstance(output, dict):
                output = output["main"]
            pred = output.argmax(dim=1).cpu()  # (1, H, W) or (1, D, H, W)

        if is_3d:
            mid = image.shape[2] // 2
            img_show = image[0, 0, mid].cpu().numpy()
            mask_show = mask[mid].numpy()
            pred_show = pred[0, mid].numpy()
        else:
            if image.shape[1] > 1:
                mid_ch = image.shape[1] // 2
                img_show = image[0, mid_ch].cpu().numpy()
            else:
                img_show = image[0, 0].cpu().numpy()
            mask_show = mask.numpy()
            pred_show = pred[0].numpy()

        # Originalna slika
        axes[row, 0].imshow(img_show, cmap="gray")
        axes[row, 0].set_title("MR slika")
        axes[row, 0].axis("off")

        # Ground truth (color overlay)
        axes[row, 1].imshow(img_show, cmap="gray")
        axes[row, 1].imshow(mask_to_rgba(mask_show))
        axes[row, 1].set_title("Ground truth")
        axes[row, 1].axis("off")

        # Predikcija (color overlay)
        axes[row, 2].imshow(img_show, cmap="gray")
        axes[row, 2].imshow(mask_to_rgba(pred_show))
        axes[row, 2].set_title("Predikcija")
        axes[row, 2].axis("off")

        # Overlay s konturama
        axes[row, 3].imshow(img_show, cmap="gray")
        # GT contours
        for c, color in [(1, "lime"), (2, "red"), (3, "blue")]:
            gt_c = (mask_show == c).astype(float)
            if gt_c.max() > 0:
                axes[row, 3].contour(gt_c, levels=[0.5], colors=[color],
                                     linewidths=2, linestyles="solid")
            pr_c = (pred_show == c).astype(float)
            if pr_c.max() > 0:
                axes[row, 3].contour(pr_c, levels=[0.5], colors=[color],
                                     linewidths=2, linestyles="dashed")
        axes[row, 3].set_title("Overlay (solid=GT, dashed=pred)")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Primjeri segmentacije spremljeni: {save_path}")


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str):
    """
    Usporedba performansi modela pomoću bar chart-a.
    """
    models = list(results.keys())
    metrics = ["DSC", "IoU", "Precision", "Recall"]

    x = np.arange(len(metrics))
    width = 0.25
    offsets = np.linspace(-width, width, len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, model_name in enumerate(models):
        values = [results[model_name].get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], values, width * 0.9,
                      label=model_name, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Vrijednost metrike")
    ax.set_title("Usporedba modela segmentacije prostate")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Usporedba modela spremljena: {save_path}")


def create_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """Stvara tablicu rezultata u tekstualnom formatu."""
    metrics = ["DSC", "IoU", "Precision", "Recall"]
    header = f"{'Model':<15}" + "".join(f"{m:<12}" for m in metrics)
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for model_name, model_metrics in results.items():
        values = "".join(
            f"{model_metrics.get(m, 0):<12.4f}" for m in metrics
        )
        lines.append(f"{model_name:<15}{values}")
    lines.append(separator)

    return "\n".join(lines)
