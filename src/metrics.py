"""
Metrike za evaluaciju multi-class segmentacije prostate.

Klase: 0=pozadina, 1=PZ, 2=CG, 3=tumor.

Uključuje:
  - Dice Similarity Coefficient (DSC) — per-class i prosjek
  - Intersection over Union (IoU) / Jaccard indeks
  - Precision i Recall
"""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ["Background", "PZ", "CG", "Tumor"]


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6, num_classes: int = 4
                     ) -> torch.Tensor:
    """
    Izračunava prosječni Dice Similarity Coefficient (DSC) za foreground klase.

    Args:
        pred: logits (B, C, ...) ili argmax predikcija (B, ...)
        target: ground truth maska (B, ...) s integer labelama

    Vraća:
        Prosječni DSC po foreground klasama.
    """
    # Ako je pred logits (B, C, ...), pretvori u class labels
    if pred.dim() > target.dim():
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred

    dice_sum = 0.0
    count = 0
    for c in range(1, num_classes):  # skip background
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target == c).float().view(target.size(0), -1)

        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_sum += dice.mean()
        count += 1

    return dice_sum / max(count, 1)


def dice_per_class(pred: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1e-6, num_classes: int = 4
                   ) -> Dict[str, float]:
    """Izračunava DSC za svaku klasu zasebno."""
    if pred.dim() > target.dim():
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred

    result = {}
    for c in range(1, num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target == c).float().view(target.size(0), -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        result[f"DSC_{CLASS_NAMES[c]}"] = dice.mean().item()

    return result


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6, num_classes: int = 4
              ) -> torch.Tensor:
    """Prosječni IoU za foreground klase."""
    if pred.dim() > target.dim():
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred

    iou_sum = 0.0
    count = 0
    for c in range(1, num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target == c).float().view(target.size(0), -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_sum += iou.mean()
        count += 1

    return iou_sum / max(count, 1)


def precision_score(pred: torch.Tensor, target: torch.Tensor,
                    smooth: float = 1e-6, num_classes: int = 4
                    ) -> torch.Tensor:
    """Prosječna Precision za foreground klase."""
    if pred.dim() > target.dim():
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred

    prec_sum = 0.0
    count = 0
    for c in range(1, num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target == c).float().view(target.size(0), -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        prec = (tp + smooth) / (tp + fp + smooth)
        prec_sum += prec.mean()
        count += 1

    return prec_sum / max(count, 1)


def recall_score(pred: torch.Tensor, target: torch.Tensor,
                 smooth: float = 1e-6, num_classes: int = 4
                 ) -> torch.Tensor:
    """Prosječni Recall za foreground klase."""
    if pred.dim() > target.dim():
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred

    rec_sum = 0.0
    count = 0
    for c in range(1, num_classes):
        p = (pred_labels == c).float().view(pred_labels.size(0), -1)
        t = (target == c).float().view(target.size(0), -1)
        tp = (p * t).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        rec = (tp + smooth) / (tp + fn + smooth)
        rec_sum += rec.mean()
        count += 1

    return rec_sum / max(count, 1)


def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor,
                        num_classes: int = 4,
                        skip_empty: bool = False) -> Dict[str, float]:
    """
    Izračunava sve metrike za dani batch.

    Args:
        logits: izlaz modela (B, C, ...) raw logits
        targets: ground truth maska (B, ...) s integer labelama
        num_classes: broj klasa
        skip_empty: ako True, preskoči uzorke gdje je GT maska prazna

    Vraća:
        Rječnik s metrikama: DSC, IoU, Precision, Recall + per-class DSC
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)

        if skip_empty:
            # Filtriraj uzorke s nepraznom GT maskom
            mask_sums = (targets > 0).float().view(targets.size(0), -1).sum(dim=1)
            non_empty = mask_sums > 0
            if non_empty.sum() == 0:
                pred_sums = (preds > 0).float().view(preds.size(0), -1).sum(dim=1)
                all_correct = (pred_sums == 0).all().item()
                v = 1.0 if all_correct else 0.0
                return {"DSC": v, "IoU": v, "Precision": v, "Recall": v,
                        "_count": 0}
            preds = preds[non_empty]
            targets = targets[non_empty]

        dsc = dice_coefficient(preds, targets, num_classes=num_classes).item()
        iou = iou_score(preds, targets, num_classes=num_classes).item()
        prec = precision_score(preds, targets, num_classes=num_classes).item()
        rec = recall_score(preds, targets, num_classes=num_classes).item()

    result = {
        "DSC": dsc,
        "IoU": iou,
        "Precision": prec,
        "Recall": rec,
    }

    # Add per-class DSC
    per_class = dice_per_class(preds, targets, num_classes=num_classes)
    result.update(per_class)

    if skip_empty:
        result["_count"] = int(preds.size(0))
    return result


class MetricTracker:
    """Praćenje metrika tijekom treniranja."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._metrics = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = 0.0
                self._counts[key] = 0
            self._metrics[key] += value * count
            self._counts[key] += count

    def compute(self) -> Dict[str, float]:
        return {
            key: self._metrics[key] / self._counts[key]
            for key in self._metrics
        }

    def __str__(self) -> str:
        computed = self.compute()
        parts = [f"{k}: {v:.4f}" for k, v in computed.items()]
        return " | ".join(parts)
