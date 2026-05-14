"""
Pipeline za treniranje modela segmentacije prostate.

Uključuje:
  - Petlju treniranja s praćenjem metrika
  - Validaciju nakon svake epohe
  - Learning rate scheduler (ReduceLROnPlateau)
  - Early stopping
  - Spremanje najboljeg modela
"""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
try:
    from .dataset import create_datasets
except ImportError:
    create_datasets = None
try:
    from .dataset_prostate158 import create_prostate158_datasets
except ImportError:
    create_prostate158_datasets = None
from .losses import get_loss_function
from .metrics import MetricTracker, compute_all_metrics
from .models import (
    UNet2D, UNet25D, UNet3D,
    AttentionUNet, UNetPlusPlus, ResUNet, TransUNet, SwinUNet,
    AttentionUNet3D, UNetPlusPlus3D, ResUNet3D, TransUNet3D, SwinUNet3D,
    MSDANet2D, MSDANet3D,
    DSBANet2D, DSBANet3D,
)


def create_model(config: Config) -> nn.Module:
    """Stvara model na temelju konfiguracije."""
    model_classes = {
        "unet2d": UNet2D,
        "unet25d": UNet25D,
        "unet3d": UNet3D,
        "attention_unet": AttentionUNet,
        "unet_plus_plus": UNetPlusPlus,
        "resunet": ResUNet,
        "transunet": TransUNet,
        "swin_unet": SwinUNet,
        "attention_unet_25d": AttentionUNet,
        "unet_plus_plus_25d": UNetPlusPlus,
        "resunet_25d": ResUNet,
        "transunet_25d": TransUNet,
        "swin_unet_25d": SwinUNet,
        "attention_unet_3d": AttentionUNet3D,
        "unet_plus_plus_3d": UNetPlusPlus3D,
        "resunet_3d": ResUNet3D,
        "transunet_3d": TransUNet3D,
        "swin_unet_3d": SwinUNet3D,
        "msda_net": MSDANet2D,
        "msda_net_25d": MSDANet2D,
        "msda_net_3d": MSDANet3D,
    }

    # 2.5D varijante koriste više susjednih rezova kao ulazne kanale
    is_25d = "_25d" in config.model_name or config.model_name == "unet25d"
    in_ch = config.num_adjacent_slices if is_25d else config.in_channels
    # Multimodal: stack T2 + ADC + DWI as separate channels.
    multimodal = bool(getattr(config, "multimodal", False))
    if multimodal:
        n_mod = len(getattr(config, "modalities", ("t2", "adc", "dwi")))
        in_ch = in_ch * n_mod if is_25d else n_mod

    # DSBANet - konfigurabilne komponente za ablacijsku studiju
    if config.model_name.startswith("dsba_net"):
        is_3d = "_3d" in config.model_name
        ModelClass = DSBANet3D if is_3d else DSBANet2D

        kwargs = dict(
            in_channels=in_ch,
            out_channels=config.out_channels,
            base_filters=config.base_filters,
            use_se=getattr(config, "ablation_use_se", True),
            use_aspp=getattr(config, "ablation_use_aspp", True),
            use_dual_attention=getattr(config, "ablation_use_dag", True),
            use_deep_supervision=getattr(config, "ablation_use_ds", True),
            use_boundary=getattr(config, "ablation_use_brm", True),
        )
        # Pretrained, MSAF i FFM za 2D i 3D
        kwargs["use_pretrained"] = getattr(config, "use_pretrained", False)
        kwargs["use_msaf"] = getattr(config, "ablation_use_msaf", True)
        kwargs["use_ffm"] = getattr(config, "ablation_use_ffm", True)
        model = ModelClass(**kwargs)
        print(f"Model: {config.model_name}")
        print(f"Broj parametara: {model.count_parameters():,}")
        return model

    if config.model_name not in model_classes:
        raise ValueError(f"Nepoznati model: {config.model_name}")

    model = model_classes[config.model_name](
        in_channels=in_ch,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
    )

    print(f"Model: {config.model_name}")
    print(f"Broj parametara: {model.count_parameters():,}")
    return model


def _masks_to_onehot(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Pretvara integer masku (B, ...) u one-hot (B, C, ...) za diskriminator."""
    import torch.nn.functional as F
    onehot = F.one_hot(masks, num_classes)  # (B, ..., C)
    # Premjesti class dim na poziciju 1
    dims = list(range(onehot.dim()))
    dims = [dims[0], dims[-1]] + dims[1:-1]
    return onehot.permute(*dims).float()


def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    discriminator: Optional[nn.Module] = None,
                    optimizer_d: Optional[torch.optim.Optimizer] = None,
                    adv_weight: float = 0.01,
                    num_classes: int = 4,
                    ) -> Tuple[float, Dict[str, float]]:
    """Trenira model jednu epohu s opcionim AMP i adversarial trainingom."""
    model.train()
    if discriminator is not None:
        discriminator.train()
    metric_tracker = MetricTracker()
    total_loss = 0.0
    use_amp = scaler is not None
    use_gan = discriminator is not None and optimizer_d is not None
    gan_criterion = nn.MSELoss() if use_gan else None

    for batch in tqdm(dataloader, desc="  Treniranje", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # ----- 1) Treniraj diskriminator -----
        if use_gan:
            optimizer_d.zero_grad()
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs_det = model(images)
                else:
                    outputs_det = model(images)
                main_det = outputs_det["main"] if isinstance(outputs_det, dict) else outputs_det
                fake_mask = torch.softmax(main_det, dim=1).detach()

            masks_onehot = _masks_to_onehot(masks, num_classes)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred_real = discriminator(images, masks_onehot)
                    pred_fake = discriminator(images, fake_mask)
                    real_label = torch.ones_like(pred_real)
                    fake_label = torch.zeros_like(pred_fake)
                    loss_d = (gan_criterion(pred_real, real_label)
                              + gan_criterion(pred_fake, fake_label)) * 0.5
                scaler.scale(loss_d).backward()
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(optimizer_d)
                scaler.update()
            else:
                pred_real = discriminator(images, masks_onehot)
                pred_fake = discriminator(images, fake_mask)
                real_label = torch.ones_like(pred_real)
                fake_label = torch.zeros_like(pred_fake)
                loss_d = (gan_criterion(pred_real, real_label)
                          + gan_criterion(pred_fake, fake_label)) * 0.5
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()

        # ----- 2) Treniraj generator (segmentacijski model) -----
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
                # Adversarial gubitak za generator
                if use_gan:
                    main_out = outputs["main"] if isinstance(outputs, dict) else outputs
                    fake_mask_g = torch.softmax(main_out, dim=1)
                    pred_fake_g = discriminator(images, fake_mask_g)
                    loss_g_adv = gan_criterion(pred_fake_g,
                                               torch.ones_like(pred_fake_g))
                    loss = loss + adv_weight * loss_g_adv
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            if use_gan:
                main_out = outputs["main"] if isinstance(outputs, dict) else outputs
                fake_mask_g = torch.sigmoid(main_out)
                pred_fake_g = discriminator(images, fake_mask_g)
                loss_g_adv = gan_criterion(pred_fake_g,
                                           torch.ones_like(pred_fake_g))
                loss = loss + adv_weight * loss_g_adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        main_outputs = outputs["main"] if isinstance(outputs, dict) else outputs
        metrics = compute_all_metrics(main_outputs.float(), masks,
                                       num_classes=num_classes)
        metric_tracker.update(metrics, count=images.size(0))

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = metric_tracker.compute()
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device,
             num_classes: int = 4
             ) -> Tuple[float, Dict[str, float]]:
    """Evaluira model na validacijskom skupu."""
    model.eval()
    metric_tracker = MetricTracker()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="  Validacija", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item() * images.size(0)
        metrics = compute_all_metrics(outputs, masks, num_classes=num_classes)
        metric_tracker.update(metrics, count=images.size(0))

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = metric_tracker.compute()
    return avg_loss, avg_metrics


class EarlyStopping:
    """Early stopping za sprečavanje pretreniranja."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train(config: Config) -> Dict:
    """
    Glavna funkcija za treniranje.

    Vraća rječnik s poviješću treniranja i putanjom do najboljeg modela.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi se uređaj: {device}")

    # Priprema podataka
    print("\nUčitavanje podataka...")
    cascade_mode = getattr(config, "cascade_mode", "off")
    if getattr(config, "dataset_name", "promise12") == "prostate158":
        if cascade_mode == "stage1":
            from .dataset_prostate158 import create_prostate158_cascade_stage1_datasets
            train_dataset, val_dataset, test_dataset = \
                create_prostate158_cascade_stage1_datasets(config)
        elif cascade_mode == "stage2":
            from .dataset_prostate158 import create_prostate158_cascade_stage2_datasets
            pred_val = getattr(config, "cascade_predicted_bboxes_val", None)
            pred_test = getattr(config, "cascade_predicted_bboxes_test", None)
            train_dataset, val_dataset, test_dataset = \
                create_prostate158_cascade_stage2_datasets(
                    config, predicted_bboxes_val=pred_val,
                    predicted_bboxes_test=pred_test)
        else:
            train_dataset, val_dataset, test_dataset = \
                create_prostate158_datasets(config)
    else:
        train_dataset, val_dataset, test_dataset = create_datasets(config)

    # Tumour-positive slice oversampling (samo 2D / 2.5D, gdje dataset ima
    # `has_tumor` listu po sample-u). 3D dataset ima 1 sample = 1 volumen pa
    # nema smisla per-volume oversampling.
    train_sampler = None
    if (getattr(config, "oversample_tumor", False)
            and hasattr(train_dataset, "has_tumor")
            and len(train_dataset.has_tumor) == len(train_dataset)):
        from torch.utils.data import WeightedRandomSampler
        factor = float(getattr(config, "tumor_oversample_factor", 5.0))
        weights = [factor if t else 1.0 for t in train_dataset.has_tumor]
        train_sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True)
        n_pos = sum(train_dataset.has_tumor)
        n_total = len(train_dataset.has_tumor)
        print(f"WeightedRandomSampler: {n_pos}/{n_total} tumor-positive "
              f"slices boostani {factor}× (efektivno ~{factor * n_pos / (factor * n_pos + (n_total - n_pos)):.0%} "
              f"batch udjela)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.effective_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Model, gubitak, optimizator
    model = create_model(config).to(device)
    criterion = get_loss_function(config)

    # Freeze encoder za prvih N epoha ako je pretrained
    freeze_epochs = getattr(config, "freeze_encoder_epochs", 0)
    use_pretrained = getattr(config, "use_pretrained", False)
    if use_pretrained and freeze_epochs > 0 and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Encoder zamrznut za prvih {freeze_epochs} epoha "
              f"(trainable: {trainable:,})")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if getattr(config, "use_cosine_annealing", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.cosine_T_0,
            T_mult=config.cosine_T_mult,
            eta_min=config.cosine_eta_min,
        )
        use_cosine = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )
        use_cosine = False

    early_patience = 50 if use_cosine else config.early_stopping_patience
    early_stopping = EarlyStopping(patience=early_patience)

    # Mixed Precision (AMP)
    use_amp = device.type == "cuda" and getattr(config, "use_amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed Precision (AMP) uključen")

    # Adversarial training (GAN)
    use_gan = getattr(config, "use_gan", False)
    discriminator = None
    optimizer_d = None
    if use_gan:
        from .discriminator import PatchDiscriminator2D, PatchDiscriminator3D
        is_3d = "_3d" in config.model_name
        in_ch_d = config.in_channels + config.out_channels
        if is_3d:
            discriminator = PatchDiscriminator3D(in_channels=in_ch_d,
                                                 base_filters=32).to(device)
        else:
            # Za 2.5D ulaz je num_adjacent_slices + 1
            is_25d = "_25d" in config.model_name or config.model_name == "unet25d"
            if is_25d:
                in_ch_d = config.num_adjacent_slices + config.out_channels
            discriminator = PatchDiscriminator2D(in_channels=in_ch_d,
                                                 base_filters=64).to(device)
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=config.learning_rate * 0.1,
            betas=(0.5, 0.999))
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Adversarial training uključen (Diskriminator: {d_params:,} param)")

    # Povijest treniranja
    history = {
        "train_loss": [], "val_loss": [],
        "train_dsc": [], "val_dsc": [],
        "train_iou": [], "val_iou": [],
        "lr": [],
    }

    best_val_dsc = 0.0
    best_model_path = os.path.join(config.checkpoint_dir,
                                   f"best_{config.model_name}.pth")
    last_state_path = os.path.join(config.checkpoint_dir,
                                   f"last_{config.model_name}.pth")

    # ---- Auto-resume: učitaj last.pth ako postoji ----
    start_epoch = 1
    if os.path.exists(last_state_path) and getattr(config, "auto_resume", True):
        print(f"\nPronađen last_state checkpoint: {last_state_path}")
        ck = torch.load(last_state_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            if "scheduler_state_dict" in ck and ck["scheduler_state_dict"] is not None:
                try:
                    scheduler.load_state_dict(ck["scheduler_state_dict"])
                except Exception as e:
                    print(f"  Upozorenje: ne mogu vratiti scheduler state ({e}); "
                          f"nastavljam s novim schedulerom.")
            if scaler is not None and ck.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ck["scaler_state_dict"])
            if discriminator is not None and ck.get("discriminator_state_dict") is not None:
                discriminator.load_state_dict(ck["discriminator_state_dict"])
            if optimizer_d is not None and ck.get("optimizer_d_state_dict") is not None:
                optimizer_d.load_state_dict(ck["optimizer_d_state_dict"])
            history = ck.get("history", history)
            best_val_dsc = ck.get("best_val_dsc", 0.0)
            es = ck.get("early_stop_state")
            if es:
                early_stopping.counter = es.get("counter", 0)
                early_stopping.best_score = es.get("best_score")
                early_stopping.should_stop = es.get("should_stop", False)
            start_epoch = ck.get("epoch", 0) + 1
            print(f"  Nastavak od epohe {start_epoch} "
                  f"(best Val DSC dosad: {best_val_dsc:.4f})")
        except Exception as e:
            print(f"  Resume pao ({e}); kreni iz početka.")
            start_epoch = 1

    print(f"\nPočetak treniranja (epohe {start_epoch}..{config.num_epochs})...")
    print("=" * 70)

    if start_epoch > config.num_epochs:
        print("Sve epohe su već istrenirane — preskačem trening.")

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        # Unfreeze encoder nakon freeze_epochs
        if (use_pretrained and freeze_epochs > 0
                and epoch == freeze_epochs + 1
                and hasattr(model, "encoder")):
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Reinicijaliziraj optimizer sa svim parametrima i manjim LR
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate * 0.1,
                weight_decay=config.weight_decay,
            )
            if use_cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=config.cosine_T_0,
                    T_mult=config.cosine_T_mult,
                    eta_min=config.cosine_eta_min)
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max",
                    patience=config.scheduler_patience,
                    factor=config.scheduler_factor)
            print(f"\n>>> Encoder odmrznut na epohi {epoch}, "
                  f"LR smanjen na {config.learning_rate * 0.1:.1e}")

        # Treniranje
        adv_weight = getattr(config, "adv_weight", 0.01)
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            discriminator=discriminator, optimizer_d=optimizer_d,
            adv_weight=adv_weight,
            num_classes=config.out_channels,
        )

        # Validacija
        val_loss, val_metrics = validate(model, val_loader, criterion, device,
                                          num_classes=config.out_channels)

        # Scheduler
        val_dsc = val_metrics["DSC"]
        if use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(val_dsc)

        # Spremi najbolji model
        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dsc": val_dsc,
                "config": config,
            }, best_model_path)

        # Ažuriraj povijest
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dsc"].append(train_metrics["DSC"])
        history["val_dsc"].append(val_dsc)
        history["train_iou"].append(train_metrics["IoU"])
        history["val_iou"].append(val_metrics["IoU"])
        history["lr"].append(current_lr)

        # ---- Spremi puni state nakon svake epohe (za resume) ----
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
                if hasattr(scheduler, "state_dict") else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "discriminator_state_dict": discriminator.state_dict()
                if discriminator is not None else None,
            "optimizer_d_state_dict": optimizer_d.state_dict()
                if optimizer_d is not None else None,
            "history": history,
            "best_val_dsc": best_val_dsc,
            "early_stop_state": {
                "counter": early_stopping.counter,
                "best_score": early_stopping.best_score,
                "should_stop": early_stopping.should_stop,
            },
            "config": config,
        }
        # Atomski zapis: tmp → rename, kako Ctrl-C ne bi pokvario zadnji state.
        tmp_path = last_state_path + ".tmp"
        torch.save(last_state, tmp_path)
        os.replace(tmp_path, last_state_path)

        epoch_time = time.time() - epoch_start

        print(f"Epoha [{epoch:3d}/{config.num_epochs}] "
              f"({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train DSC: {train_metrics['DSC']:.4f} | "
              f"Val DSC: {val_dsc:.4f} | "
              f"Val IoU: {val_metrics['IoU']:.4f} | "
              f"LR: {current_lr:.6f}")

        # Early stopping
        if early_stopping(val_dsc):
            print(f"\nEarly stopping na epohi {epoch}. "
                  f"Najbolji Val DSC: {best_val_dsc:.4f}")
            break

    print("=" * 70)
    print(f"Treniranje završeno. Najbolji Val DSC: {best_val_dsc:.4f}")
    print(f"Najbolji model spremljen: {best_model_path}")

    return {
        "history": history,
        "best_model_path": best_model_path,
        "best_val_dsc": best_val_dsc,
        "test_dataset": test_dataset,
    }
