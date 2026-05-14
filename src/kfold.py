"""
K-Fold Cross-Validation za segmentaciju prostate.

Trenira K modela na K različitih train/val splitova, zatim:
  1. Sprema svaki fold-model
  2. Evaluira ensemble svih K modela na testnom skupu
  3. Sprema prosječne i per-fold rezultate
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from .config import Config
from .dataset_prostate158 import (
    get_prostate158_paths, Prostate158Dataset2D,
    Prostate158Dataset25D, Prostate158Dataset3D,
)
from .evaluate import evaluate_model, evaluate_ensemble
from .losses import get_loss_function
from .metrics import MetricTracker, compute_all_metrics
from .train import create_model, train_one_epoch, validate, EarlyStopping


def _get_dataset_class(config: Config):
    """Vraća odgovarajuću Dataset klasu za model."""
    if config.dataset_name == "prostate158":
        if "_3d" in config.model_name or config.model_name == "unet3d":
            return Prostate158Dataset3D
        elif "_25d" in config.model_name or config.model_name == "unet25d":
            return Prostate158Dataset25D
        else:
            return Prostate158Dataset2D
    else:
        from .dataset import ProstateDataset2D, ProstateDataset25D, ProstateDataset3D
        if "_3d" in config.model_name or config.model_name == "unet3d":
            return ProstateDataset3D
        elif "_25d" in config.model_name or config.model_name == "unet25d":
            return ProstateDataset25D
        else:
            return ProstateDataset2D


def _get_all_pairs(config: Config):
    """Vraća sve (image, mask) parove ili putanje ovisno o datasetu."""
    if config.dataset_name == "prostate158":
        train_pairs, val_pairs, test_pairs = get_prostate158_paths(
            config.prostate158_dir)
        all_pairs = train_pairs + val_pairs
        return all_pairs, test_pairs
    else:
        from .dataset import get_case_paths
        train_paths = get_case_paths(config.train_data_dir)
        test_paths = get_case_paths(config.test_data_dir)
        test_with_seg = [p for p in test_paths
                         if os.path.exists(p.replace(".mhd", "_segmentation.mhd"))]
        return train_paths, test_with_seg


def train_kfold(config: Config, n_folds: int = 5) -> Dict:
    """
    K-Fold Cross-Validation treniranje.

    Trenira n_folds modela, svaki na drugom train/val splitu,
    te evaluira ensemble na testnom skupu.

    Vraća dict s per-fold rezultatima i ensemble rezultatima.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi se uređaj: {device}")
    print(f"K-Fold Cross-Validation: {n_folds} foldova\n")

    all_pairs, test_pairs = _get_all_pairs(config)
    print(f"Ukupno train+val: {len(all_pairs)}, test: {len(test_pairs)}")

    DatasetClass = _get_dataset_class(config)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)

    fold_results = []
    fold_models = []
    best_fold_dsc = 0.0
    best_fold_idx = -1

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_pairs)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"  Train: {len(train_indices)} slučajeva, Val: {len(val_indices)} slučajeva")
        print(f"{'='*70}")

        fold_model_path = os.path.join(
            config.checkpoint_dir,
            f"best_{config.model_name}_fold{fold_idx + 1}.pth")

        # Provjeri postoji li već checkpoint za ovaj fold (resume podrška)
        if os.path.exists(fold_model_path):
            print(f"  Checkpoint pronađen: {fold_model_path}")
            checkpoint = torch.load(fold_model_path, map_location="cpu",
                                    weights_only=False)
            best_val_dsc = checkpoint.get("val_dsc", 0.0)
            print(f"  Preskačem treniranje — učitavam model "
                  f"(epoha {checkpoint.get('epoch', '?')}, "
                  f"Val DSC: {best_val_dsc:.4f})")

            model = create_model(config)
            # Filtriraj ključeve čiji shape ne odgovara novom modelu
            saved_state = checkpoint["model_state_dict"]
            model_state = model.state_dict()
            compatible_state = {
                k: v for k, v in saved_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model.load_state_dict(compatible_state, strict=False)
            fold_models.append(model)

            fold_results.append({"fold": fold_idx + 1, "best_val_dsc": best_val_dsc})
            if best_val_dsc > best_fold_dsc:
                best_fold_dsc = best_val_dsc
                best_fold_idx = fold_idx
            continue

        # Kreiraj fold-specifične parove
        train_fold = [all_pairs[i] for i in train_indices]
        val_fold = [all_pairs[i] for i in val_indices]

        # Kreiraj datasete
        train_dataset = DatasetClass(train_fold, config, augment=config.use_augmentation)
        val_dataset = DatasetClass(val_fold, config, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=config.effective_batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=config.effective_batch_size,
            shuffle=False, num_workers=config.num_workers, pin_memory=True)

        print(f"  Train uzoraka: {len(train_dataset)}, Val uzoraka: {len(val_dataset)}")

        # Model, loss, optimizer
        model = create_model(config).to(device)
        criterion = get_loss_function(config)

        # Freeze encoder
        freeze_epochs = getattr(config, "freeze_encoder_epochs", 0)
        use_pretrained = getattr(config, "use_pretrained", False)
        if use_pretrained and freeze_epochs > 0 and hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=config.weight_decay)

        if getattr(config, "use_cosine_annealing", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.cosine_T_0,
                T_mult=config.cosine_T_mult, eta_min=config.cosine_eta_min)
            use_cosine = True
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=config.scheduler_patience,
                factor=config.scheduler_factor)
            use_cosine = False

        early_stopping = EarlyStopping(
            patience=50 if use_cosine else config.early_stopping_patience)

        # AMP
        use_amp = device.type == "cuda" and getattr(config, "use_amp", True)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # GAN
        discriminator = None
        optimizer_d = None
        use_gan = getattr(config, "use_gan", False)
        if use_gan:
            from .discriminator import PatchDiscriminator2D, PatchDiscriminator3D
            is_3d = "_3d" in config.model_name
            in_ch_d = config.in_channels + 1
            if is_3d:
                discriminator = PatchDiscriminator3D(
                    in_channels=in_ch_d, base_filters=32).to(device)
            else:
                discriminator = PatchDiscriminator2D(
                    in_channels=in_ch_d, base_filters=64).to(device)
            optimizer_d = torch.optim.Adam(
                discriminator.parameters(), lr=config.learning_rate * 0.1,
                betas=(0.5, 0.999))

        # Treniranje folda
        best_val_dsc = 0.0

        for epoch in range(1, config.num_epochs + 1):
            epoch_start = time.time()

            # Unfreeze encoder
            if (use_pretrained and freeze_epochs > 0
                    and epoch == freeze_epochs + 1
                    and hasattr(model, "encoder")):
                for param in model.encoder.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config.learning_rate * 0.1,
                    weight_decay=config.weight_decay)
                if use_cosine:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=config.cosine_T_0,
                        T_mult=config.cosine_T_mult,
                        eta_min=config.cosine_eta_min)

            adv_weight = getattr(config, "adv_weight", 0.01)
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                discriminator=discriminator, optimizer_d=optimizer_d,
                adv_weight=adv_weight)

            val_loss, val_metrics = validate(
                model, val_loader, criterion, device)

            val_dsc = val_metrics["DSC"]
            if use_cosine:
                scheduler.step(epoch)
            else:
                scheduler.step(val_dsc)

            if val_dsc > best_val_dsc:
                best_val_dsc = val_dsc
                torch.save({
                    "epoch": epoch,
                    "fold": fold_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "val_dsc": val_dsc,
                }, fold_model_path)

            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0 or epoch == 1:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Fold {fold_idx+1} | Ep [{epoch:3d}/{config.num_epochs}] "
                      f"({epoch_time:.1f}s) | "
                      f"Train DSC: {train_metrics['DSC']:.4f} | "
                      f"Val DSC: {val_dsc:.4f} | "
                      f"Best: {best_val_dsc:.4f} | LR: {lr:.6f}")

            if early_stopping(val_dsc):
                print(f"  Early stopping na epohi {epoch}")
                break

        print(f"\n  Fold {fold_idx+1} Best Val DSC: {best_val_dsc:.4f}")
        fold_results.append({"fold": fold_idx + 1, "best_val_dsc": best_val_dsc})

        if best_val_dsc > best_fold_dsc:
            best_fold_dsc = best_val_dsc
            best_fold_idx = fold_idx

        # Učitaj best model za ovaj fold i spremi na CPU
        checkpoint = torch.load(fold_model_path, map_location="cpu",
                                weights_only=False)
        model.cpu()
        model.load_state_dict(checkpoint["model_state_dict"])
        fold_models.append(model)

        # Oslobodi memoriju za sljedeći fold
        del optimizer, scheduler, criterion, scaler
        if discriminator:
            del discriminator, optimizer_d
        torch.cuda.empty_cache()

    # ====== Rezultati ======
    val_dscs = [r["best_val_dsc"] for r in fold_results]
    print(f"\n{'='*70}")
    print(f"K-FOLD REZULTATI")
    print(f"{'='*70}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val DSC = {r['best_val_dsc']:.4f}")
    print(f"  Prosjek: {np.mean(val_dscs):.4f} +/- {np.std(val_dscs):.4f}")
    print(f"  Najbolji fold: {best_fold_idx + 1} (DSC = {best_fold_dsc:.4f})")

    # ====== Ensemble evaluacija na testnom skupu ======
    if test_pairs:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE EVALUACIJA NA TESTNOM SKUPU ({len(test_pairs)} slučajeva)")
        print(f"{'='*70}")

        test_dataset = DatasetClass(test_pairs, config, augment=False)
        test_loader = DataLoader(
            test_dataset, batch_size=config.effective_batch_size,
            shuffle=False, num_workers=config.num_workers)

        use_tta = getattr(config, "use_tta", False)
        use_pp = getattr(config, "use_postprocess", False)

        # Pojedinačni fold na testu
        for i, model in enumerate(fold_models):
            model.to(device)
            test_metrics = evaluate_model(model, test_loader, device,
                                          use_tta=use_tta, use_postprocess=use_pp)
            print(f"  Fold {i+1} test: DSC={test_metrics['DSC']:.4f} "
                  f"IoU={test_metrics['IoU']:.4f}")
            fold_results[i]["test_metrics"] = test_metrics

        # Ensemble svih foldova
        for m in fold_models:
            m.to(device)
        ensemble_metrics = evaluate_ensemble(
            fold_models, test_loader, device,
            use_tta=use_tta, use_postprocess=use_pp)
        print(f"\n  ENSEMBLE test: DSC={ensemble_metrics['DSC']:.4f} "
              f"IoU={ensemble_metrics['IoU']:.4f} "
              f"Prec={ensemble_metrics['Precision']:.4f} "
              f"Rec={ensemble_metrics['Recall']:.4f}")
    else:
        ensemble_metrics = {}

    # Spremi rezultate
    results = {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean_val_dsc": float(np.mean(val_dscs)),
        "std_val_dsc": float(np.std(val_dscs)),
        "ensemble_test": ensemble_metrics,
    }
    results_path = os.path.join(config.output_dir,
                                f"kfold_{config.model_name}_{n_folds}fold.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultati spremljeni: {results_path}")

    return results
