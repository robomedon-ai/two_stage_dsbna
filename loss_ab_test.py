"""
A/B test za rješavanje tumour-learning problema.

Trenira Full DSBANet 2D na prostate158 s različitim loss/sampler konfiguracijama
(po default 30 epoha) i ispisuje test-set per-case tumour DSC za svaku.

Primjer:
    python loss_ab_test.py --epochs 30
    python loss_ab_test.py --epochs 30 --only "combined_w5"   # samo jedan
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset_prostate158 import create_prostate158_datasets
from src.evaluate import evaluate_model, evaluate_per_case
from src.train import create_model, train


# --------------------------------------------------------------------------
# Kandidati za testiranje
# --------------------------------------------------------------------------
TRIALS = {
    "combined_w5": dict(
        loss_function="combined",
        tumor_weight=5.0,
        oversample_tumor=False,
    ),
    "combined_w10_oversample": dict(
        loss_function="combined",
        tumor_weight=10.0,
        oversample_tumor=True,
        tumor_oversample_factor=5.0,
    ),
    "focal_tversky_w5": dict(
        loss_function="focal_tversky",
        tumor_weight=5.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_tversky_gamma=4.0 / 3.0,
        oversample_tumor=False,
    ),
    "combined_ft_w5_oversample": dict(
        loss_function="combined_focal_tversky",
        tumor_weight=5.0,
        ft_weight=0.6,
        focal_in_combo_weight=0.4,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_tversky_gamma=4.0 / 3.0,
        focal_gamma=2.0,
        oversample_tumor=True,
        tumor_oversample_factor=5.0,
    ),
}


def run_trial(name: str, settings: dict, epochs: int,
              output_root: str) -> dict:
    cfg = Config()
    cfg.dataset_name = "prostate158"
    cfg.model_name = "dsba_net"
    cfg.num_epochs = epochs
    cfg.deep_supervision = True
    cfg.auto_resume = True
    cfg.use_amp = True
    # Apply trial-specific overrides
    for k, v in settings.items():
        setattr(cfg, k, v)

    trial_dir = os.path.join(output_root, name)
    cfg.output_dir = trial_dir
    cfg.checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"TRIAL: {name}")
    for k, v in settings.items():
        print(f"  {k} = {v}")
    print("=" * 70)

    t0 = time.time()
    result = train(cfg)
    train_time = time.time() - t0
    print(f"  Trening trajao {train_time / 60:.1f} min")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(result["best_model_path"], map_location=device,
                    weights_only=False)
    model = create_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])

    _, _, test_dataset = create_prostate158_datasets(cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.effective_batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    case_ids = getattr(test_dataset, "case_ids", None)
    per_case = evaluate_per_case(model, test_loader, device,
                                  case_ids=case_ids)
    with open(os.path.join(trial_dir, "per_case_dsc.json"), "w") as f:
        json.dump(per_case, f, indent=2)

    pz = np.array([v["PZ"] for v in per_case.values()])
    cg = np.array([v["CG"] for v in per_case.values()])
    tum = np.array([v["Tumor"] for v in per_case.values()])
    summary = {
        "name": name,
        "settings": settings,
        "epochs_trained": len(result["history"]["val_loss"]),
        "best_val_dsc": result["best_val_dsc"],
        "train_minutes": train_time / 60,
        "test_per_case": {
            "PZ_mean": float(pz.mean()),  "PZ_std":  float(pz.std()),
            "CG_mean": float(cg.mean()),  "CG_std":  float(cg.std()),
            "Tumor_mean": float(tum.mean()), "Tumor_std": float(tum.std()),
            "Tumor_nonzero": int((tum > 0.01).sum()),
            "Tumor_n_cases": int(len(tum)),
        },
    }
    with open(os.path.join(trial_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Test PZ:    {summary['test_per_case']['PZ_mean']:.4f}")
    print(f"  Test CG:    {summary['test_per_case']['CG_mean']:.4f}")
    print(f"  Test Tumor: {summary['test_per_case']['Tumor_mean']:.4f} "
          f"(non-zero on {summary['test_per_case']['Tumor_nonzero']}/"
          f"{summary['test_per_case']['Tumor_n_cases']} cases)")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--only", type=str, default=None,
                        help="Pokreni samo jedan trial (po imenu)")
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_root = args.output_root or os.path.join(
        project_root, "output", "prostate158", "loss_trial")
    os.makedirs(output_root, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    trials = TRIALS if args.only is None else {args.only: TRIALS[args.only]}
    summaries = []
    for name, settings in trials.items():
        s = run_trial(name, settings, args.epochs, output_root)
        summaries.append(s)

    # Comparison
    print("\n" + "=" * 90)
    print("USPOREDBA")
    print("=" * 90)
    print(f"{'Trial':<32} {'PZ':>8} {'CG':>8} {'Tumor':>8} {'Tumor>0':>8} {'Best Val DSC':>14}")
    for s in sorted(summaries, key=lambda x: -x["test_per_case"]["Tumor_mean"]):
        nz = s["test_per_case"]
        print(f"{s['name']:<32} "
              f"{nz['PZ_mean']:>8.4f} {nz['CG_mean']:>8.4f} "
              f"{nz['Tumor_mean']:>8.4f} "
              f"{nz['Tumor_nonzero']:>4}/{nz['Tumor_n_cases']:<3} "
              f"{s['best_val_dsc']:>14.4f}")

    with open(os.path.join(output_root, "ab_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSažetak: {os.path.join(output_root, 'ab_summary.json')}")


if __name__ == "__main__":
    main()
