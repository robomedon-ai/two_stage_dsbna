"""
Run inference on the Prostate158 test set for all trained models
and save predicted segmentation masks as .npy files.

Output structure:
  output/prostate158/test_predictions/
    unet2d/
      mask_0000.npy   (256x256 multi-class mask, int64, values 0-3)
      prob_0000.npy   (4x256x256 softmax probabilities, float32)
      ...
    attention_unet/
      ...

Each mask_*.npy file contains a multi-class segmentation mask:
  0=background, 1=PZ, 2=CG, 3=tumor.

A metadata JSON is saved per model with per-slice DSC scores.

Usage:
    python save_test_predictions.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.train import create_model
from src.dataset_prostate158 import create_prostate158_datasets
from src.metrics import dice_coefficient, dice_per_class

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(PROJECT_ROOT, "output", "prostate158", "test_predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def save_model_predictions(config, device, model, model_dir, test_dataset):
    """Run inference and save all predictions + per-slice metrics."""
    os.makedirs(model_dir, exist_ok=True)
    model.eval()
    num_classes = config.out_channels

    per_slice = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)  # (1, H, W) long

            output = model(image)
            if isinstance(output, dict):
                output = output["main"]

            prob = F.softmax(output, dim=1)  # (1, C, H, W)
            pred = prob.argmax(dim=1)  # (1, H, W)

            # Per-slice DSC (mean over foreground classes)
            dsc = dice_coefficient(pred, mask, num_classes=num_classes).item()
            per_class = dice_per_class(pred, mask, num_classes=num_classes)

            # Save multi-class mask
            pred_np = pred.cpu().numpy()[0].astype(np.int64)
            np.save(os.path.join(model_dir, f"mask_{i:04d}.npy"), pred_np)

            # Save probability maps (C, H, W)
            prob_np = prob.cpu().numpy()[0].astype(np.float32)
            np.save(os.path.join(model_dir, f"prob_{i:04d}.npy"), prob_np)

            per_slice.append({"slice": i, "DSC": dsc, **per_class})

    # Save metadata
    dscs = [s["DSC"] for s in per_slice]
    metadata = {
        "model": os.path.basename(model_dir),
        "num_classes": num_classes,
        "num_slices": len(per_slice),
        "mean_DSC": float(np.mean(dscs)),
        "std_DSC": float(np.std(dscs)),
        "per_slice": per_slice,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {PRED_DIR}\n")

    config = Config()
    config.dataset_name = "prostate158"
    config._update_paths()

    num_classes = config.out_channels

    # All models: (model_name, is_pretrained, deep_supervision)
    baseline_models = [
        ("unet2d", False, False),
        ("attention_unet", False, False),
        ("unet_plus_plus", False, False),
        ("resunet", False, False),
        ("transunet", False, False),
        ("swin_unet", False, False),
        # 2.5D
        ("unet25d", False, False),
        ("attention_unet_25d", False, False),
        ("unet_plus_plus_25d", False, False),
        ("resunet_25d", False, False),
        ("transunet_25d", False, False),
        ("swin_unet_25d", False, False),
        # 3D
        ("unet3d", False, False),
        ("attention_unet_3d", False, False),
        ("unet_plus_plus_3d", False, False),
        ("resunet_3d", False, False),
        ("transunet_3d", False, False),
        ("swin_unet_3d", False, False),
        # MSDA-Net
        ("msda_net", False, True),
        ("msda_net_25d", False, True),
        ("msda_net_3d", False, True),
        # DSBANet
        ("dsba_net", False, True),
        ("dsba_net_25d", False, True),
        ("dsba_net_3d", False, True),
    ]

    for model_name, is_pretrained, deep_sup in baseline_models:
        model_dir = os.path.join(PRED_DIR, model_name)
        ckpt_path = os.path.join(config.checkpoint_dir, f"best_{model_name}.pth")

        if not os.path.exists(ckpt_path):
            print(f"SKIP {model_name} — no checkpoint")
            continue

        if os.path.exists(os.path.join(model_dir, "metadata.json")):
            print(f"SKIP {model_name} — already done")
            continue

        print(f"Processing {model_name}...")
        config.model_name = model_name
        config.use_pretrained = is_pretrained
        config.deep_supervision = deep_sup

        _, _, test_dataset = create_prostate158_datasets(config)

        model = create_model(config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = checkpoint["model_state_dict"]
        cur = model.state_dict()
        compat = {k: v for k, v in saved.items() if k in cur and v.shape == cur[k].shape}
        model.load_state_dict(compat, strict=False)

        meta = save_model_predictions(config, device, model, model_dir, test_dataset)
        print(f"  {model_name}: DSC = {meta['mean_DSC']:.4f} ± {meta['std_DSC']:.4f} "
              f"({meta['num_slices']} slices)")

        del model
        torch.cuda.empty_cache()

    # DSBANet folds
    config.model_name = "dsba_net"
    config.use_pretrained = True
    config.deep_supervision = True

    _, _, test_dataset = create_prostate158_datasets(config)

    fold_probs = []  # for ensemble

    for fold_idx in range(1, 6):
        fold_name = f"dsba_net_fold{fold_idx}"
        model_dir = os.path.join(PRED_DIR, fold_name)
        ckpt_path = os.path.join(config.checkpoint_dir, f"best_dsba_net_fold{fold_idx}.pth")

        if not os.path.exists(ckpt_path):
            print(f"SKIP {fold_name} — no checkpoint")
            continue

        if os.path.exists(os.path.join(model_dir, "metadata.json")):
            print(f"SKIP {fold_name} — already done")
            # Load probs for ensemble
            probs = []
            for i in range(len(test_dataset)):
                p = np.load(os.path.join(model_dir, f"prob_{i:04d}.npy"))
                probs.append(p)
            fold_probs.append(probs)
            continue

        print(f"Processing {fold_name}...")

        model = create_model(config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = checkpoint["model_state_dict"]
        cur = model.state_dict()
        compat = {k: v for k, v in saved.items() if k in cur and v.shape == cur[k].shape}
        model.load_state_dict(compat, strict=False)

        meta = save_model_predictions(config, device, model, model_dir, test_dataset)
        print(f"  {fold_name}: DSC = {meta['mean_DSC']:.4f} ± {meta['std_DSC']:.4f} "
              f"({meta['num_slices']} slices)")

        # Collect probs for ensemble
        probs = []
        for i in range(len(test_dataset)):
            p = np.load(os.path.join(model_dir, f"prob_{i:04d}.npy"))
            probs.append(p)
        fold_probs.append(probs)

        del model
        torch.cuda.empty_cache()

    # DSBANet ensemble
    if fold_probs:
        ens_dir = os.path.join(PRED_DIR, "dsba_net_ensemble")

        if os.path.exists(os.path.join(ens_dir, "metadata.json")):
            print("SKIP dsba_net_ensemble — already done")
        else:
            print("Processing dsba_net_ensemble...")
            os.makedirs(ens_dir, exist_ok=True)

            per_slice = []
            for i in range(len(test_dataset)):
                # Average probabilities across folds (C, H, W)
                avg_prob = np.mean([fold_probs[f][i] for f in range(len(fold_probs))],
                                   axis=0)
                pred = avg_prob.argmax(axis=0).astype(np.int64)

                np.save(os.path.join(ens_dir, f"mask_{i:04d}.npy"), pred)
                np.save(os.path.join(ens_dir, f"prob_{i:04d}.npy"),
                        avg_prob.astype(np.float32))

                # DSC
                sample = test_dataset[i]
                mask = sample["mask"]  # (H, W) long
                pred_t = torch.from_numpy(pred).unsqueeze(0).long()
                mask_t = mask.unsqueeze(0)
                dsc = dice_coefficient(pred_t, mask_t, num_classes=num_classes).item()
                per_class = dice_per_class(pred_t, mask_t, num_classes=num_classes)
                per_slice.append({"slice": i, "DSC": dsc, **per_class})

            dscs = [s["DSC"] for s in per_slice]
            metadata = {
                "model": "dsba_net_ensemble",
                "num_folds": len(fold_probs),
                "num_classes": num_classes,
                "num_slices": len(per_slice),
                "mean_DSC": float(np.mean(dscs)),
                "std_DSC": float(np.std(dscs)),
                "per_slice": per_slice,
            }
            with open(os.path.join(ens_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  dsba_net_ensemble: DSC = {metadata['mean_DSC']:.4f} "
                  f"± {metadata['std_DSC']:.4f} ({metadata['num_slices']} slices)")

    # Also save ground truth masks for reference
    gt_dir = os.path.join(PRED_DIR, "ground_truth")
    if not os.path.exists(os.path.join(gt_dir, "metadata.json")):
        print("Saving ground truth masks...")
        os.makedirs(gt_dir, exist_ok=True)
        config.model_name = "unet2d"
        config.use_pretrained = False
        config.deep_supervision = False
        _, _, test_dataset_2d = create_prostate158_datasets(config)
        for i in range(len(test_dataset_2d)):
            sample = test_dataset_2d[i]
            mask = sample["mask"].numpy().astype(np.int64)  # (H, W)
            np.save(os.path.join(gt_dir, f"mask_{i:04d}.npy"), mask)
            # Also save the MRI input
            image = sample["image"]
            if image.shape[0] > 1:
                img = image[image.shape[0] // 2].numpy()
            else:
                img = image[0].numpy()
            np.save(os.path.join(gt_dir, f"image_{i:04d}.npy"), img.astype(np.float32))

        with open(os.path.join(gt_dir, "metadata.json"), "w") as f:
            json.dump({"num_slices": len(test_dataset_2d), "num_classes": num_classes,
                       "type": "ground_truth"}, f, indent=2)
        print(f"  Saved {len(test_dataset_2d)} GT masks + images")

    print("\nDone.")
    print(f"\nSaved predictions for models:")
    for d in sorted(os.listdir(PRED_DIR)):
        meta_path = os.path.join(PRED_DIR, d, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            dsc = m.get("mean_DSC", "—")
            if isinstance(dsc, float):
                print(f"  {d:30s} DSC = {dsc:.4f}")
            else:
                print(f"  {d:30s} ({m.get('type', '')})")


if __name__ == "__main__":
    main()
