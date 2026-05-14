"""
Funkcije gubitka za multi-class segmentaciju prostate.

Klase: 0=pozadina, 1=PZ, 2=CG, 3=tumor.

Uključuje:
  - Multi-class Dice Loss: per-class Dice, prosjek
  - Cross-Entropy Loss
  - Combined Loss: težinska kombinacija Dice i CE gubitka
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss.

    Računa Dice gubitak za svaku klasu zasebno, zatim prosjek.
    Koristi softmax za pretvaranje logita u vjerojatnosti.
    """

    def __init__(self, smooth: float = 1.0, num_classes: int = 4):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, ...) raw logits or dict with "main" key
            targets: (B, ...) integer class labels
        """
        if isinstance(logits, dict):
            logits = logits["main"]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, ...) -> (B, C, ...)
        targets_onehot = F.one_hot(targets, self.num_classes)
        # Move class dim: (B, H, W, C) -> (B, C, H, W) or (B, D, H, W, C) -> (B, C, D, H, W)
        dims = list(range(targets_onehot.dim()))
        dims = [dims[0], dims[-1]] + dims[1:-1]
        targets_onehot = targets_onehot.permute(*dims).float()

        dice_sum = 0.0
        # Compute Dice for each class (skip background class 0)
        for c in range(1, self.num_classes):
            p = probs[:, c].contiguous().view(probs.size(0), -1)
            t = targets_onehot[:, c].contiguous().view(targets_onehot.size(0), -1)
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_sum += dice.mean()

        # Average over foreground classes
        return 1.0 - dice_sum / (self.num_classes - 1)


class CrossEntropyLoss(nn.Module):
    """
    Omotač za PyTorch CrossEntropyLoss s opcionalnim per-class težinama.

    `class_weights` je tuple/list dužine num_classes (npr. (1, 1, 1, 5) da se
    tumor klasa pojača 5×) — prosljeđuje se kao `weight=` u nn.CrossEntropyLoss.
    """

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            w = torch.tensor(list(class_weights), dtype=torch.float32)
            self.loss = nn.CrossEntropyLoss(weight=w)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits["main"]
        # Premjesti weight na isti uređaj kao logits (lazy)
        if (self.loss.weight is not None
                and self.loss.weight.device != logits.device):
            self.loss.weight = self.loss.weight.to(logits.device)
        return self.loss(logits, targets)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss: FL = -α_t * (1 - p_t)^γ * log(p_t).

    γ down-weights easy examples (large background regions), α uvodi per-class
    težine (npr. veće za tumor). Standardno γ=2.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, num_classes: int = 4):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is not None:
            self.register_buffer(
                "alpha",
                torch.tensor(list(alpha), dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits["main"]
        log_probs = F.log_softmax(logits, dim=1)
        # Gather log p_t and p_t at target index
        # targets: (B, ...) ints in [0, num_classes)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Multi-class Tversky Loss.

    Tversky = TP / (TP + α·FP + β·FN). Pri α<β model je kažnjen više za FN,
    što pomaže manjim klasama (tumor). Tipično α=0.3, β=0.7.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, num_classes: int = 4,
                 class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self.class_weights = list(class_weights) if class_weights else None

    def _tversky_per_class(self, logits, targets):
        """Vraća Tversky indeks po klasi i batchu, oblika (B, C-1) za FG klase."""
        if isinstance(logits, dict):
            logits = logits["main"]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        dims = list(range(targets_oh.dim()))
        dims = [dims[0], dims[-1]] + dims[1:-1]
        targets_oh = targets_oh.permute(*dims).float()

        per_class = []
        for c in range(1, self.num_classes):
            p = probs[:, c].contiguous().view(probs.size(0), -1)
            t = targets_oh[:, c].contiguous().view(targets_oh.size(0), -1)
            tp = (p * t).sum(dim=1)
            fp = (p * (1 - t)).sum(dim=1)
            fn = ((1 - p) * t).sum(dim=1)
            tv = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth)
            per_class.append(tv)
        return torch.stack(per_class, dim=1)  # (B, C-1)

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        tv = self._tversky_per_class(logits, targets).mean(dim=0)  # (C-1,)
        loss = 1.0 - tv  # per-class loss
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights[1:],
                             dtype=loss.dtype, device=loss.device)
            return (loss * w).sum() / w.sum()
        return loss.mean()


class FocalTverskyLoss(TverskyLoss):
    """
    Focal Tversky: ((1 - Tversky)^γ).mean() — pojačava primjere s niskim
    Tversky indeksom. Tipično γ=4/3, α=0.3, β=0.7 za neuravnoteženu
    multi-class segmentaciju.
    """

    def __init__(self, gamma: float = 4.0 / 3.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        tv = self._tversky_per_class(logits, targets).mean(dim=0)
        loss = (1.0 - tv) ** self.gamma
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights[1:],
                             dtype=loss.dtype, device=loss.device)
            return (loss * w).sum() / w.sum()
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Kombinirana funkcija gubitka: Multi-class Dice Loss + Cross-Entropy Loss.

    L = α * DiceLoss + β * CELoss

    Ako je `ce_class_weights` predan, CE se računa s per-class težinama.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5,
                 smooth: float = 1.0, num_classes: int = 4,
                 ce_class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = MultiClassDiceLoss(smooth=smooth, num_classes=num_classes)
        self.ce_loss = CrossEntropyLoss(class_weights=ce_class_weights)

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits["main"]
        return (self.dice_weight * self.dice_loss(logits, targets)
                + self.ce_weight * self.ce_loss(logits, targets))


class CombinedFocalTverskyLoss(nn.Module):
    """
    Kombinacija Focal Tversky + Focal Loss. Standardni recipe za rare-class
    medical segmentation: Focal Tversky kažnjava male klase preko FN, dok
    Focal Loss dodatno fokusira gradient na teške piksele.
    """

    def __init__(self, ft_weight: float = 0.5, focal_weight: float = 0.5,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 ft_gamma: float = 4.0 / 3.0,
                 focal_gamma: float = 2.0,
                 num_classes: int = 4,
                 class_weights=None,
                 focal_alpha=None):
        super().__init__()
        self.ft_weight = ft_weight
        self.focal_weight = focal_weight
        self.ft = FocalTverskyLoss(
            gamma=ft_gamma, alpha=tversky_alpha, beta=tversky_beta,
            num_classes=num_classes, class_weights=class_weights)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha,
                                num_classes=num_classes)

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits["main"]
        return (self.ft_weight * self.ft(logits, targets)
                + self.focal_weight * self.focal(logits, targets))


class BoundaryLoss(nn.Module):
    """
    Boundary Loss za poboljšanje segmentacije rubova.

    Izvlači rubove iz ground truth maske pomoću erozije
    i računa CE gubitak između predviđenih rubova i pravih rubova.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Izvlači rubove iz maske (integer labels)."""
        # Create binary foreground mask
        fg = (mask > 0).float().unsqueeze(1)  # (B, 1, H, W) or (B, 1, D, H, W)

        if fg.dim() == 4:  # 2D
            kernel = torch.ones(1, 1, 3, 3, device=fg.device)
            eroded = F.conv2d(fg, kernel, padding=1)
            eroded = (eroded >= 9).float()
        elif fg.dim() == 5:  # 3D
            kernel = torch.ones(1, 1, 3, 3, 3, device=fg.device)
            eroded = F.conv3d(fg, kernel, padding=1)
            eroded = (eroded >= 27).float()
        else:
            return fg.squeeze(1)

        boundary = fg.squeeze(1) - eroded.squeeze(1)
        boundary = torch.clamp(boundary, 0.0, 1.0)
        return boundary

    def forward(self, boundary_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # For boundary, we use binary: foreground vs background
        boundary_gt = self._extract_boundary(targets)
        # boundary_logits: (B, C, ...) -> take max of foreground channels
        if boundary_logits.shape[1] > 1:
            boundary_pred = boundary_logits[:, 1:].max(dim=1)[0]
        else:
            boundary_pred = boundary_logits[:, 0]
        return F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt)


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss za MSDA-Net / DSBANet.

    Kombinira glavni gubitak, pomoćne gubitke s opadajućim težinama,
    i boundary gubitak za precizniju segmentaciju rubova.

    L = L_main + Σ(w_i * L_aux_i) + w_boundary * L_boundary
    """

    def __init__(self, base_loss: nn.Module,
                 aux_weights: tuple = (0.4, 0.3, 0.2),
                 boundary_weight: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights
        self.boundary_weight = boundary_weight
        self.boundary_loss = BoundaryLoss(num_classes=num_classes)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        # Ako model vraća samo tensor (eval mode), koristi osnovni gubitak
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, targets)

        # Dict output (training mode)
        loss = self.base_loss(outputs["main"], targets)

        # Pomoćni gubici
        aux_keys = ["aux4", "aux3", "aux2"]
        for key, weight in zip(aux_keys, self.aux_weights):
            if key in outputs:
                loss = loss + weight * self.base_loss(outputs[key], targets)

        # Boundary gubitak
        if "boundary" in outputs:
            loss = loss + self.boundary_weight * self.boundary_loss(
                outputs["boundary"], targets
            )

        return loss


def _resolve_class_weights(config, num_classes: int):
    """
    Konstruira tuple per-class težina iz config-a.

    Ako je `class_weights` zadan eksplicitno (config.class_weights), koristi to.
    Inače, ako je `tumor_weight` > 1 (tumor klasa = posljednja), kreira
    [1, 1, ..., tumor_weight]. Ako ništa, vraća None.
    """
    cw = getattr(config, "class_weights", None)
    if cw is not None:
        return tuple(cw)
    tumor_w = getattr(config, "tumor_weight", 1.0)
    if tumor_w and tumor_w != 1.0:
        weights = [1.0] * num_classes
        weights[-1] = float(tumor_w)  # zadnja klasa = tumor
        return tuple(weights)
    return None


def get_loss_function(config) -> nn.Module:
    """Stvara funkciju gubitka na temelju konfiguracije."""
    num_classes = getattr(config, "out_channels", 4)
    class_weights = _resolve_class_weights(config, num_classes)

    name = config.loss_function
    if name == "dice":
        base_loss = MultiClassDiceLoss(num_classes=num_classes)
    elif name == "bce":
        # "bce" now means CrossEntropy for multi-class
        base_loss = CrossEntropyLoss(class_weights=class_weights)
    elif name == "combined":
        base_loss = CombinedLoss(
            dice_weight=config.dice_weight,
            ce_weight=config.bce_weight,
            num_classes=num_classes,
            ce_class_weights=class_weights,
        )
    elif name == "focal":
        base_loss = FocalLoss(
            gamma=getattr(config, "focal_gamma", 2.0),
            alpha=class_weights,
            num_classes=num_classes,
        )
    elif name == "tversky":
        base_loss = TverskyLoss(
            alpha=getattr(config, "tversky_alpha", 0.3),
            beta=getattr(config, "tversky_beta", 0.7),
            num_classes=num_classes,
            class_weights=class_weights,
        )
    elif name == "focal_tversky":
        base_loss = FocalTverskyLoss(
            gamma=getattr(config, "focal_tversky_gamma", 4.0 / 3.0),
            alpha=getattr(config, "tversky_alpha", 0.3),
            beta=getattr(config, "tversky_beta", 0.7),
            num_classes=num_classes,
            class_weights=class_weights,
        )
    elif name == "combined_focal_tversky":
        base_loss = CombinedFocalTverskyLoss(
            ft_weight=getattr(config, "ft_weight", 0.5),
            focal_weight=getattr(config, "focal_in_combo_weight", 0.5),
            tversky_alpha=getattr(config, "tversky_alpha", 0.3),
            tversky_beta=getattr(config, "tversky_beta", 0.7),
            ft_gamma=getattr(config, "focal_tversky_gamma", 4.0 / 3.0),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
            num_classes=num_classes,
            class_weights=class_weights,
            focal_alpha=class_weights,
        )
    else:
        raise ValueError(f"Nepoznata funkcija gubitka: {name}")

    # Ako je deep supervision uključen, omota osnovni gubitak
    if getattr(config, "deep_supervision", False):
        return DeepSupervisionLoss(
            base_loss=base_loss,
            aux_weights=getattr(config, "aux_weights", (0.4, 0.3, 0.2)),
            boundary_weight=getattr(config, "boundary_weight", 0.5),
            num_classes=num_classes,
        )

    return base_loss
