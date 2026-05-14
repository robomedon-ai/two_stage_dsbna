"""
Evaluate all trained models on the Prostate158 test set and print results table.

Usage:
    python evaluate_all.py
    python evaluate_all.py --tta          # with test-time augmentation
    python evaluate_all.py --tta --pp     # with TTA + post-processing
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset_prostate158 import create_prostate158_datasets
from src.evaluate import evaluate_model, evaluate_ensemble
from src.train import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", action="store_true", help="Test-Time Augmentation")
    parser.add_argument("--pp", action="store_true", help="Post-processing (largest component)")
    parser.add_argument("--dataset", default="prostate158", choices=["promise12", "prostate158"])
    args = parser.parse_args()

    config = Config()
    config.dataset_name = args.dataset
    config._update_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset_name}")
    print(f"TTA: {args.tta}, Post-processing: {args.pp}\n")

    # Models to evaluate — (display_name, model_name, dim_label)
    baseline_models = [
        # 2D
        ("U-Net", "unet2d", "2D"),
        ("Attention U-Net", "attention_unet", "2D"),
        ("U-Net++", "unet_plus_plus", "2D"),
        ("ResUNet", "resunet", "2D"),
        ("TransUNet", "transunet", "2D"),
        ("Swin-UNet", "swin_unet", "2D"),
        # 2.5D
        ("U-Net", "unet25d", "2.5D"),
        ("Attention U-Net", "attention_unet_25d", "2.5D"),
        ("U-Net++", "unet_plus_plus_25d", "2.5D"),
        ("ResUNet", "resunet_25d", "2.5D"),
        ("TransUNet", "transunet_25d", "2.5D"),
        ("Swin-UNet", "swin_unet_25d", "2.5D"),
        # 3D
        ("U-Net", "unet3d", "3D"),
        ("Attention U-Net", "attention_unet_3d", "3D"),
        ("U-Net++", "unet_plus_plus_3d", "3D"),
        ("ResUNet", "resunet_3d", "3D"),
        ("TransUNet", "transunet_3d", "3D"),
        ("Swin-UNet", "swin_unet_3d", "3D"),
    ]

    all_results = {}

    # Evaluate each baseline
    for display_name, model_name, dim_label in baseline_models:
        checkpoint_path = os.path.join(
            config.checkpoint_dir, f"best_{model_name}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"SKIP {display_name} ({dim_label}) — no checkpoint")
            continue

        config.model_name = model_name
        is_3d = "_3d" in model_name or model_name == "unet3d"

        # Create test dataset for this model type
        _, _, test_dataset = create_prostate158_datasets(config)
        if test_dataset is None or len(test_dataset) == 0:
            print(f"SKIP {display_name} ({dim_label}) — no test data")
            continue

        test_loader = DataLoader(
            test_dataset, batch_size=config.effective_batch_size,
            shuffle=False, num_workers=config.num_workers)

        # Load model
        model = create_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Evaluate
        metrics = evaluate_model(
            model, test_loader, device,
            use_tta=args.tta, use_postprocess=args.pp)

        key = f"{display_name} ({dim_label})"
        all_results[key] = metrics
        print(f"{key:30s} | DSC={metrics['DSC']:.4f} | "
              f"IoU={metrics['IoU']:.4f} | "
              f"Prec={metrics['Precision']:.4f} | "
              f"Rec={metrics['Recall']:.4f}")

        # Free GPU
        del model
        torch.cuda.empty_cache()

    # Evaluate DSBANet (single best fold or ensemble)
    dsba_folds = []
    for fold_idx in range(1, 6):
        fold_path = os.path.join(
            config.checkpoint_dir, f"best_dsba_net_fold{fold_idx}.pth")
        if os.path.exists(fold_path):
            dsba_folds.append(fold_path)

    if dsba_folds:
        config.model_name = "dsba_net"
        config.use_pretrained = True
        config.deep_supervision = True
        _, _, test_dataset = create_prostate158_datasets(config)
        test_loader = DataLoader(
            test_dataset, batch_size=config.effective_batch_size,
            shuffle=False, num_workers=config.num_workers)

        # Evaluate best single fold
        best_fold_dsc = 0
        best_fold_metrics = None
        best_fold_idx = 0
        fold_models = []

        for i, fold_path in enumerate(dsba_folds):
            model = create_model(config).to(device)
            checkpoint = torch.load(fold_path, map_location=device,
                                    weights_only=False)
            # Handle potential shape mismatches from architecture changes
            saved_state = checkpoint["model_state_dict"]
            model_state = model.state_dict()
            compatible_state = {
                k: v for k, v in saved_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model.load_state_dict(compatible_state, strict=False)
            model.eval()

            metrics = evaluate_model(
                model, test_loader, device,
                use_tta=args.tta, use_postprocess=args.pp)

            print(f"{'DSBANet fold ' + str(i+1):30s} | DSC={metrics['DSC']:.4f} | "
                  f"IoU={metrics['IoU']:.4f} | "
                  f"Prec={metrics['Precision']:.4f} | "
                  f"Rec={metrics['Recall']:.4f}")

            if metrics["DSC"] > best_fold_dsc:
                best_fold_dsc = metrics["DSC"]
                best_fold_metrics = metrics
                best_fold_idx = i + 1

            fold_models.append(model)

        all_results["DSBANet best fold (2D)"] = best_fold_metrics

        # Ensemble evaluation
        ensemble_metrics = evaluate_ensemble(
            fold_models, test_loader, device,
            use_tta=args.tta, use_postprocess=args.pp)

        all_results["DSBANet ensemble (2D)"] = ensemble_metrics
        print(f"{'DSBANet ensemble':30s} | DSC={ensemble_metrics['DSC']:.4f} | "
              f"IoU={ensemble_metrics['IoU']:.4f} | "
              f"Prec={ensemble_metrics['Precision']:.4f} | "
              f"Rec={ensemble_metrics['Recall']:.4f}")

        # Free GPU
        for m in fold_models:
            del m
        torch.cuda.empty_cache()

    # Save results
    suffix = ""
    if args.tta:
        suffix += "_tta"
    if args.pp:
        suffix += "_pp"
    results_path = os.path.join(
        config.output_dir, f"test_results_{config.dataset_name}{suffix}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print_latex_table(all_results)


def print_latex_table(results):
    """Print results as a LaTeX table ready for the paper."""
    print(r"\begin{table}[H]")
    print(r"\caption{Comparison of segmentation methods on the Prostate158 test set "
          r"(19 cases). Best results are shown in \textbf{bold}.\label{tab:comparison}}")
    print(r"\centering")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Dim} & \textbf{DSC} & "
          r"\textbf{IoU} & \textbf{Precision} & \textbf{Recall} \\")
    print(r"\midrule")

    # Find best values
    best = {"DSC": 0, "IoU": 0, "Precision": 0, "Recall": 0}
    for metrics in results.values():
        for k in best:
            if metrics.get(k, 0) > best[k]:
                best[k] = metrics[k]

    prev_dim = None
    for name, metrics in results.items():
        # Extract dim from name
        if "(2D)" in name:
            dim = "2D"
        elif "(2.5D)" in name:
            dim = "2.5D"
        elif "(3D)" in name:
            dim = "3D"
        else:
            dim = "2D"

        # Add midrule between dim groups
        clean_name = name.split(" (")[0]
        if prev_dim and dim != prev_dim:
            print(r"\midrule")
        prev_dim = dim

        # Format values, bold if best
        vals = []
        for k in ["DSC", "IoU", "Precision", "Recall"]:
            v = metrics.get(k, 0)
            s = f"{v:.4f}"
            if abs(v - best[k]) < 1e-5:
                s = r"\textbf{" + s + "}"
            vals.append(s)

        # Escape underscores, add cite markers for known architectures
        tex_name = clean_name.replace("_", r"\_")

        print(f"{tex_name:<35s} & {dim:<5s} & "
              f"{vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
