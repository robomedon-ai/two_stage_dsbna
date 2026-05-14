"""
Automatska segmentacija prostate na MR slikama primjenom dubokih neuronskih mreža.

Glavni ulazni skript za pokretanje treniranja i evaluacije modela:
  - 2D U-Net
  - 2.5D U-Net
  - 3D U-Net

Korištenje:
    # Treniranje svih modela:
    python main.py --mode all

    # Treniranje jednog modela:
    python main.py --mode single --model unet2d

    # Evaluacija spremljenog modela:
    python main.py --mode evaluate --model unet2d --checkpoint output/checkpoints/best_unet2d.pth

    # Samo vizualizacija podataka:
    python main.py --mode visualize_data
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # Backend bez GUI-a
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Dodaj src na path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import create_datasets, get_case_paths, load_case, normalize_intensity
from src.dataset_prostate158 import create_prostate158_datasets
from src.evaluate import (
    create_results_table,
    evaluate_model,
    evaluate_per_case,
    plot_model_comparison,
    plot_segmentation_examples,
    plot_training_history,
)
from src.losses import get_loss_function
from src.metrics import compute_all_metrics
from src.train import create_model, train


def visualize_dataset(config: Config):
    """Vizualizira primjere iz dataseta za preliminarnu analizu."""
    print("=" * 70)
    print("VIZUALIZACIJA DATASETA")
    print("=" * 70)

    case_paths = get_case_paths(config.train_data_dir)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, case_idx in enumerate([0, 10, 30]):
        if case_idx >= len(case_paths):
            continue
        image, mask, metadata = load_case(case_paths[case_idx])
        image = normalize_intensity(image)

        mid_slice = image.shape[0] // 2

        # MR slika
        axes[row, 0].imshow(image[mid_slice], cmap="gray")
        axes[row, 0].set_title(f"Case{case_idx:02d} - MR slika (rez {mid_slice})")
        axes[row, 0].axis("off")

        # Segmentacijska maska
        axes[row, 1].imshow(mask[mid_slice], cmap="gray")
        axes[row, 1].set_title(f"Segmentacijska maska")
        axes[row, 1].axis("off")

        # Overlay
        axes[row, 2].imshow(image[mid_slice], cmap="gray")
        axes[row, 2].imshow(mask[mid_slice], cmap="Reds", alpha=0.4)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

        # Histogram intenziteta
        axes[row, 3].hist(image[mid_slice].flatten(), bins=100, color="steelblue",
                          alpha=0.7)
        axes[row, 3].set_title("Histogram intenziteta")
        axes[row, 3].set_xlabel("Intenzitet")
        axes[row, 3].set_ylabel("Frekvencija")

        # Ispiši metapodatke
        spacing = metadata["spacing"]
        print(f"  Case{case_idx:02d}: oblik={image.shape}, "
              f"spacing=({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "dataset_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVizualizacija dataseta spremljena: {save_path}")


def train_single_model(config: Config):
    """Trenira jedan model."""
    print("=" * 70)
    print(f"TRENIRANJE MODELA: {config.model_name.upper()}")
    print("=" * 70)

    result = train(config)

    # Spremi grafikone treniranja
    history_path = os.path.join(config.output_dir,
                                f"training_history_{config.model_name}.png")
    plot_training_history(result["history"], history_path)

    # Spremi povijest kao JSON
    history_json_path = os.path.join(config.output_dir,
                                     f"history_{config.model_name}.json")
    with open(history_json_path, "w") as f:
        json.dump(result["history"], f, indent=2)

    # Evaluacija na validacijskom skupu s vizualizacijom
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(result["best_model_path"], map_location=device,
                            weights_only=False)
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Vizualizacija predikcija
    cascade_mode = getattr(config, "cascade_mode", "off")
    if config.dataset_name == "prostate158":
        if cascade_mode == "stage1":
            from src.dataset_prostate158 import create_prostate158_cascade_stage1_datasets
            _, val_dataset, test_dataset = create_prostate158_cascade_stage1_datasets(config)
        elif cascade_mode == "stage2":
            from src.dataset_prostate158 import create_prostate158_cascade_stage2_datasets
            _, val_dataset, test_dataset = create_prostate158_cascade_stage2_datasets(
                config,
                predicted_bboxes_val=getattr(config, "cascade_predicted_bboxes_val", None),
                predicted_bboxes_test=getattr(config, "cascade_predicted_bboxes_test", None),
            )
        else:
            _, val_dataset, test_dataset = create_prostate158_datasets(config)
    else:
        _, val_dataset, test_dataset = create_datasets(config)

    is_3d = (config.model_name == "unet3d" or "_3d" in config.model_name)
    seg_path = os.path.join(config.output_dir,
                            f"segmentation_examples_{config.model_name}.png")
    dataset_for_vis = test_dataset if test_dataset else val_dataset
    plot_segmentation_examples(model, dataset_for_vis, device, seg_path,
                               num_examples=6, is_3d=is_3d)

    # Evaluacija na testnom skupu
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.effective_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        use_tta = getattr(config, "use_tta", False)
        use_pp = getattr(config, "use_postprocess", False)
        test_metrics = evaluate_model(model, test_loader, device,
                                      use_tta=use_tta,
                                      use_postprocess=use_pp,
                                      num_classes=config.out_channels)
        print(f"\nRezultati na testnom skupu ({config.model_name}):")
        if use_tta:
            print("  (TTA uključen)")
        if use_pp:
            print("  (Post-processing uključen)")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Per-case DSC za statističke testove (samo prostate158, gdje
        # dataset prosljeđuje case_idx u batch dict-u).
        if config.dataset_name == "prostate158":
            case_ids = getattr(test_dataset, "case_ids", None)
            per_case = evaluate_per_case(model, test_loader, device,
                                         case_ids=case_ids,
                                         use_tta=use_tta,
                                         use_postprocess=use_pp,
                                         num_classes=config.out_channels)
            if per_case:
                per_case_path = os.path.join(config.output_dir,
                                             "per_case_dsc.json")
                with open(per_case_path, "w") as f:
                    json.dump(per_case, f, indent=2)
                print(f"  Per-case DSC spremljen: {per_case_path}")
        return test_metrics

    return {"DSC": result["best_val_dsc"]}


def train_all_models(config: Config):
    """Trenira sva tri modela i uspoređuje rezultate."""
    print("=" * 70)
    print("TRENIRANJE SVIH MODELA")
    print("=" * 70)

    model_names = [
        "unet2d", "unet25d", "unet3d",
        "attention_unet", "unet_plus_plus", "resunet",
        "transunet", "swin_unet",
        "attention_unet_25d", "unet_plus_plus_25d", "resunet_25d",
        "transunet_25d", "swin_unet_25d",
        "attention_unet_3d", "unet_plus_plus_3d", "resunet_3d",
        "transunet_3d", "swin_unet_3d",
        "msda_net", "msda_net_25d", "msda_net_3d",
        "dsba_net", "dsba_net_25d", "dsba_net_3d",
    ]
    all_results = {}

    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*70}\n")

        config.model_name = model_name
        metrics = train_single_model(config)
        all_results[model_name] = metrics

    # Usporedba modela
    print("\n" + "=" * 70)
    print("USPOREDBA MODELA")
    print("=" * 70)

    table = create_results_table(all_results)
    print(table)

    # Spremi tablicu
    table_path = os.path.join(config.output_dir, "results_comparison.txt")
    with open(table_path, "w") as f:
        f.write(table)

    # Spremi grafikon usporedbe
    comparison_path = os.path.join(config.output_dir, "model_comparison.png")
    plot_model_comparison(all_results, comparison_path)

    # Spremi rezultate kao JSON
    results_json_path = os.path.join(config.output_dir, "all_results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSvi rezultati spremljeni u: {config.output_dir}")


def run_ablation_study(config: Config, max_variants: int = None,
                        retrain_done: bool = False):
    """
    Ablacijska studija za DSBANet.

    Trenira 8 varijanti modela:
      1. DSBANet (puni — MSAF + sve ostale komponente)
      2. DSBANet w/o SE        (bez SE-Residual blokova)
      3. DSBANet w/o ASPP      (bez ASPP u bottlenecku)
      4. DSBANet w/o MSAF (DAG)  (MSAF zamijenjen jednostavnim Dual Attention Gate)
      5. DSBANet w/o skip-att.   (bez ikakve attention na skip konekcijama)
      6. DSBANet w/o FFM       (bez Feature Fusion Module)
      7. DSBANet w/o DS        (bez Deep Supervision pomoćnih izlaza)
      8. DSBANet w/o BRM       (bez Boundary Refinement Module)

    Svaka varijanta se trenira s istim hiperparametrima i usporedbom rezultata.
    Per-case DSC po klasama (PZ, CG, tumor) se sprema za statističko testiranje.
    Izlazi idu u: <output_dir>/ablation/<dim>/<variant>/.
    """
    print("=" * 70)
    print("ABLACIJSKA STUDIJA - DSBANet")
    print("=" * 70)

    # Odredi bazni model (2D, 2.5D ili 3D) na temelju --model argumenta
    base_model = config.model_name
    if base_model not in ("dsba_net", "dsba_net_25d", "dsba_net_3d"):
        base_model = "dsba_net"  # default: 2D

    is_3d_ablation = base_model.endswith("_3d")
    is_25d_ablation = base_model.endswith("_25d")
    dim_label = "3D" if is_3d_ablation else ("2.5D" if is_25d_ablation else "2D")
    dim_dir = "3d" if is_3d_ablation else ("25d" if is_25d_ablation else "2d")
    print(f"Bazni model: DSBANet {dim_label}")

    # Dedicirani direktorij: <output_dir>/ablation/<dim>/
    ablation_root = os.path.join(config.output_dir, "ablation", dim_dir)
    os.makedirs(ablation_root, exist_ok=True)
    original_output_dir = config.output_dir

    # Definicija ablacijskih varijanti
    # (naziv, use_se, use_aspp, use_msaf, use_ffm, use_dag, use_ds, use_brm)
    # Napomena: kad use_msaf=True, DAG se ne gradi (use_dag flag se ignorira u modelu).
    # Kad use_msaf=False, use_dag=True znači DAG zamjenjuje MSAF; use_dag=False znači
    # potpuno bez attentiona na skip konekcijama (samo plain skip + concat).
    ablation_variants = [
        ("DSBANet (full)",         True,  True,  True,  True,  False, True,  True),
        ("DSBANet w/o SE",         False, True,  True,  True,  False, True,  True),
        ("DSBANet w/o ASPP",       True,  False, True,  True,  False, True,  True),
        ("DSBANet w/o MSAF (DAG)", True,  True,  False, True,  True,  True,  True),
        ("DSBANet w/o skip-att.",  True,  True,  False, True,  False, True,  True),
        ("DSBANet w/o FFM",        True,  True,  True,  False, False, True,  True),
        ("DSBANet w/o DS",         True,  True,  True,  True,  False, False, True),
        ("DSBANet w/o BRM",        True,  True,  True,  True,  False, True,  False),
    ]
    if max_variants is not None:
        ablation_variants = ablation_variants[:max_variants]
        print(f"Ograničeno na prvih {len(ablation_variants)} varijanti.")

    all_results = {}
    all_per_case = {}

    for variant_name, use_se, use_aspp, use_msaf, use_ffm, use_dag, use_ds, use_brm in ablation_variants:
        print(f"\n{'='*70}")
        print(f"Varijanta: {variant_name}")
        print(f"  SE={use_se}, ASPP={use_aspp}, MSAF={use_msaf}, FFM={use_ffm}, "
              f"DAG={use_dag}, DS={use_ds}, BRM={use_brm}")
        print(f"{'='*70}\n")

        # Postavi ablacijske flagove
        config.ablation_use_se = use_se
        config.ablation_use_aspp = use_aspp
        config.ablation_use_msaf = use_msaf
        config.ablation_use_ffm = use_ffm
        config.ablation_use_dag = use_dag
        config.ablation_use_ds = use_ds
        config.ablation_use_brm = use_brm

        # Deep supervision u loss-u mora pratiti model
        config.deep_supervision = use_ds or use_brm

        # Suffix za naziv modela (i dataset routing prema _3d/_25d substringu)
        suffix = variant_name.replace(" ", "_").replace("(", "").replace(")", "")
        suffix = suffix.replace("/", "").replace(".", "").lower()
        config.model_name = f"{base_model}__ablation__{suffix}"

        # Per-variant output direktorij (override-aj path-ove direktno;
        # NE pozivaj _update_paths jer bi resetirao output_dir na default).
        variant_dir = os.path.join(ablation_root, suffix)
        variant_ckpt_dir = os.path.join(variant_dir, "checkpoints")
        os.makedirs(variant_ckpt_dir, exist_ok=True)
        config.output_dir = variant_dir
        config.checkpoint_dir = variant_ckpt_dir

        done_flag = os.path.join(variant_dir, "done.flag")
        per_case_path = os.path.join(variant_dir, "per_case_dsc.json")
        results_cache = os.path.join(variant_dir, "test_metrics.json")

        if os.path.exists(done_flag) and not retrain_done:
            print(f"  → done.flag postoji, preskačem trening.")
            if os.path.exists(results_cache):
                with open(results_cache) as f:
                    all_results[variant_name] = json.load(f)
            else:
                all_results[variant_name] = {}
            if os.path.exists(per_case_path):
                with open(per_case_path) as f:
                    all_per_case[variant_name] = json.load(f)
            continue

        metrics = train_single_model(config)
        all_results[variant_name] = metrics

        # Cache test_metrics za buduće preskakanje
        with open(results_cache, "w") as f:
            json.dump(metrics, f, indent=2)

        # Spremi per-case DSC (ako je evaluator vratio strukturu)
        if os.path.exists(per_case_path):
            with open(per_case_path) as f:
                all_per_case[variant_name] = json.load(f)

        # Označi varijantu kao završenu
        with open(done_flag, "w") as f:
            f.write("done\n")

    # Vrati output_dir za izvještaj
    config.output_dir = original_output_dir

    # Ispis rezultata
    print("\n" + "=" * 70)
    print("REZULTATI ABLACIJSKE STUDIJE")
    print("=" * 70)

    header = f"{'Varijanta':<28} {'DSC':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in all_results.items():
        dsc = metrics.get("DSC", 0)
        iou = metrics.get("IoU", 0)
        prec = metrics.get("Precision", 0)
        rec = metrics.get("Recall", 0)
        print(f"{name:<28} {dsc:>8.4f} {iou:>8.4f} {prec:>10.4f} {rec:>8.4f}")

    # Spremi sažete rezultate i per-case strukturu na razini studije
    summary_path = os.path.join(ablation_root, "ablation_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    if all_per_case:
        per_case_summary = os.path.join(ablation_root, "ablation_per_case_dsc.json")
        with open(per_case_summary, "w") as f:
            json.dump(all_per_case, f, indent=2)

    # Statistički testovi: Wilcoxon Full vs each variant, per class, Holm-Bonferroni
    if all_per_case and "DSBANet (full)" in all_per_case:
        from src.evaluate import wilcoxon_full_vs_variants
        stats = wilcoxon_full_vs_variants(all_per_case,
                                          full_key="DSBANet (full)")
        stats_path = os.path.join(ablation_root, "ablation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistički testovi spremljeni u: {stats_path}")

    # Grafikon ablacijske studije
    _plot_ablation_results(all_results, ablation_root,
                           per_case=all_per_case)

    print(f"\nRezultati ablacijske studije spremljeni u: {ablation_root}")


def _plot_ablation_results(results: dict, output_dir: str,
                            per_case: dict = None):
    """
    Generira grafikone za ablacijsku studiju.

    Ako je `per_case` predan (mapa varijanta -> {case_id -> {PZ, CG, Tumor}}),
    crta odvojeni bar-plot po klasama s 95% bootstrap CI; inače fallback na
    skupne DSC/IoU vrijednosti.
    """
    names = list(results.keys())
    short_names = ["Full" if "full" in n else n.replace("DSBANet w/o ", "w/o ")
                   for n in names]

    # ---- Plot 1: skupni DSC/IoU bar plot ----
    dsc_values = [results[n].get("DSC", 0) for n in names]
    iou_values = [results[n].get("IoU", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, dsc_values, width, label="DSC", color="steelblue")
    bars2 = ax.bar(x + width / 2, iou_values, width, label="IoU", color="coral")
    ax.set_ylabel("Score")
    ax.set_title("Ablacijska studija - DSBANet (skupni DSC/IoU)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8)
    plt.tight_layout()
    overall_path = os.path.join(output_dir, "ablation_study_overall.png")
    plt.savefig(overall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafikon ablacijske studije (skupni): {overall_path}")

    # ---- Plot 2: per-class DSC s 95% bootstrap CI ----
    if per_case:
        classes = ["PZ", "CG", "Tumor"]
        class_means = {c: [] for c in classes}
        class_lo = {c: [] for c in classes}
        class_hi = {c: [] for c in classes}

        rng = np.random.default_rng(42)
        for n in names:
            cases = per_case.get(n, {})
            for c in classes:
                vals = np.array([cases[case_id].get(c, np.nan)
                                 for case_id in cases], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    class_means[c].append(0.0)
                    class_lo[c].append(0.0)
                    class_hi[c].append(0.0)
                    continue
                # Bootstrap 95% CI
                B = 2000
                boot = rng.choice(vals, size=(B, len(vals)), replace=True).mean(axis=1)
                class_means[c].append(float(vals.mean()))
                class_lo[c].append(float(np.percentile(boot, 2.5)))
                class_hi[c].append(float(np.percentile(boot, 97.5)))

        x = np.arange(len(names))
        width = 0.27
        offsets = [-width, 0, width]
        colors = {"PZ": "tab:blue", "CG": "tab:orange", "Tumor": "tab:red"}

        fig, ax = plt.subplots(figsize=(13, 6))
        for c, off in zip(classes, offsets):
            means = np.array(class_means[c])
            lo = np.array(class_lo[c])
            hi = np.array(class_hi[c])
            err = np.vstack([means - lo, hi - means])
            ax.bar(x + off, means, width, yerr=err, capsize=3,
                   label=c, color=colors[c])

        ax.set_ylabel("DSC")
        ax.set_title("Ablacijska studija - per-class DSC (95% bootstrap CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        per_class_path = os.path.join(output_dir, "ablation_study_per_class.png")
        plt.savefig(per_class_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grafikon ablacijske studije (per-class): {per_class_path}")


def evaluate_cascade_full(config: Config,
                          stage1_ckpt: str = None,
                          stage2_ckpt: str = None,
                          bbox_source: str = "predicted"):
    """
    End-to-end cascade evaluation on the Prostate158 test set.

    For each test case:
      (1) load the full-resolution T2 volume + GT mask,
      (2) run Stage 1 to obtain a binary prostate prediction → bbox,
          (or use the GT bbox / no bbox at all per ``bbox_source``),
      (3) crop and resample to (48, 128, 128), run Stage 2,
      (4) resample the 4-class prediction back to bbox-native shape,
      (5) paste back into a zero-initialised full-volume canvas,
      (6) compute per-class volume-level DSC against GT.

    Args:
        bbox_source: "predicted" (use Stage 1's predicted bbox, the realistic
            cascade), "gt" (use GT bbox; oracle ceiling), or "full" (no crop;
            single-stage baseline emulation — Stage 2 sees the full volume).
    """
    from src.dataset_prostate158 import (
        get_prostate158_paths,
        load_case_nifti,
        preprocess_volume,
        resize_volume,
    )
    from src.cascade import compute_roi_bbox, paste_back

    print("=" * 70)
    print(f"CASCADE end-to-end evaluation (bbox_source={bbox_source})")
    print("=" * 70)

    arch = getattr(config, "cascade_arch", "dsba_net_3d")
    project_root = os.path.dirname(os.path.abspath(__file__))
    cascade_root = _cascade_root(config, arch)
    if stage1_ckpt is None:
        stage1_ckpt = os.path.join(cascade_root, "stage1", "checkpoints",
                                    f"best_{arch}.pth")
    if stage2_ckpt is None:
        stage2_ckpt = os.path.join(cascade_root, "stage2", "checkpoints",
                                    f"best_{arch}.pth")
    if bbox_source == "predicted" and not os.path.exists(stage1_ckpt):
        raise FileNotFoundError(stage1_ckpt)
    if not os.path.exists(stage2_ckpt):
        raise FileNotFoundError(stage2_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.dataset_name = "prostate158"

    # ---- Build Stage 1 (binary) if needed ----
    s1_model = None
    if bbox_source == "predicted":
        s1_cfg = Config()
        s1_cfg.dataset_name = "prostate158"
        s1_cfg.model_name = arch
        s1_cfg.out_channels = 2
        s1_cfg.deep_supervision = False
        s1_cfg.ablation_use_ds = False
        s1_cfg.ablation_use_brm = False
        s1_cfg.multimodal = bool(getattr(config, "multimodal", False))
        s1_cfg.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
        if arch == "unet3d":
            s1_cfg.base_filters = int(
                getattr(config, "cascade_stage1_base_filters", 16))
        s1_model = create_model(s1_cfg).to(device)
        s1_model.load_state_dict(
            torch.load(stage1_ckpt, map_location=device,
                       weights_only=False)["model_state_dict"])
        s1_model.eval()
        print(f"  Stage 1 ({arch}) loaded: {stage1_ckpt}")

    # ---- Build Stage 2 (4-class) ----
    s2_cfg = Config()
    s2_cfg.dataset_name = "prostate158"
    s2_cfg.model_name = arch
    s2_cfg.out_channels = 4
    s2_cfg.multimodal = bool(getattr(config, "multimodal", False))
    s2_cfg.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
    if arch == "dsba_net_3d":
        s2_cfg.deep_supervision = True
        s2_cfg.ablation_use_se = True
        s2_cfg.ablation_use_aspp = True
        s2_cfg.ablation_use_msaf = True
        s2_cfg.ablation_use_ffm = True
        s2_cfg.ablation_use_ds = True
        s2_cfg.ablation_use_brm = True
    elif arch == "unet3d":
        s2_cfg.deep_supervision = False
        s2_cfg.ablation_use_ds = False
        s2_cfg.ablation_use_brm = False
        s2_cfg.base_filters = int(
            getattr(config, "cascade_stage2_base_filters", 32))
    s2_model = create_model(s2_cfg).to(device)
    s2_model.load_state_dict(
        torch.load(stage2_ckpt, map_location=device,
                   weights_only=False)["model_state_dict"])
    s2_model.eval()
    print(f"  Stage 2 ({arch}) loaded: {stage2_ckpt}")

    full_size = config.volume_size_3d  # (32, 256, 256)
    target = getattr(config, "cascade_stage2_volume_size", (48, 128, 128))
    margin = tuple(getattr(config, "cascade_bbox_margin_voxels", (2, 8, 8)))
    min_size = tuple(getattr(config, "cascade_min_bbox_size", (24, 96, 96)))

    multimodal = bool(getattr(config, "multimodal", False))
    modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))

    _, _, test_triples = get_prostate158_paths(config.prostate158_dir)
    per_case = {}
    smooth = 1e-6

    for t2_path, seg_path, tumor_path in test_triples:
        case_id = os.path.basename(os.path.dirname(t2_path))

        # Load & preprocess to common (32, 256, 256) frame
        if multimodal:
            from src.dataset_prostate158 import load_case_multimodal
            image, mask, spacing = load_case_multimodal(
                t2_path, seg_path, tumor_path, modalities=modalities)
        else:
            image, mask, spacing = load_case_nifti(
                t2_path, seg_path, tumor_path)
        image, mask = preprocess_volume(image, mask, spacing, config)
        image = resize_volume(image, full_size)              # (M, D, H, W) or (D, H, W)
        gt = resize_volume(mask, full_size, is_mask=True).astype(np.int64)

        # ---- Determine bbox per bbox_source ----
        if bbox_source == "predicted":
            with torch.no_grad():
                if multimodal:
                    x = torch.from_numpy(image).unsqueeze(0).float().to(device)  # (1, M, D, H, W)
                else:
                    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
                out1 = s1_model(x)
                if isinstance(out1, dict):
                    out1 = out1["main"]
                pred_binary = out1.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            bbox = compute_roi_bbox(pred_binary, margin_voxels=margin,
                                     min_size=min_size, use_largest_cc=True)
        elif bbox_source == "gt":
            bbox = compute_roi_bbox((gt > 0).astype(np.uint8),
                                     margin_voxels=margin,
                                     min_size=min_size, use_largest_cc=False)
        elif bbox_source == "full":
            bbox = (0, full_size[0], 0, full_size[1], 0, full_size[2])
        else:
            raise ValueError(f"Unknown bbox_source: {bbox_source}")

        z0, z1, y0, y1, x0, x1 = bbox

        # ---- Stage 2 forward on the cropped+resampled ROI ----
        if multimodal:
            img_roi = image[:, z0:z1, y0:y1, x0:x1]          # (M, dz, dy, dx)
        else:
            img_roi = image[z0:z1, y0:y1, x0:x1]
        img_roi_resampled = resize_volume(img_roi, target)
        with torch.no_grad():
            if multimodal:
                x = torch.from_numpy(img_roi_resampled).unsqueeze(0).float().to(device)
            else:
                x = torch.from_numpy(img_roi_resampled).unsqueeze(0).unsqueeze(0).float().to(device)
            out2 = s2_model(x)
            if isinstance(out2, dict):
                out2 = out2["main"]
            pred_4c_resampled = out2.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Resample 4-class prediction back to bbox-native shape, then paste back
        bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
        pred_4c_native = resize_volume(pred_4c_resampled, bbox_shape,
                                        is_mask=True).astype(np.int64)
        full_pred = paste_back(pred_4c_native, bbox, full_size,
                                fill_value=0, dtype=np.int64)

        # ---- Per-class volume-level DSC ----
        class_names = ["PZ", "CG", "Tumor"]
        dscs = {}
        for c, name in zip([1, 2, 3], class_names):
            p = (full_pred == c).astype(np.float32)
            g = (gt == c).astype(np.float32)
            inter = float((p * g).sum())
            union = float(p.sum() + g.sum())
            dscs[name] = (2.0 * inter + smooth) / (union + smooth)
        per_case[case_id] = dscs

        print(f"  {case_id}: PZ={dscs['PZ']:.4f}  CG={dscs['CG']:.4f}  "
              f"Tumor={dscs['Tumor']:.4f}  "
              f"bbox=(z={z0}-{z1}, y={y0}-{y1}, x={x0}-{x1})")

    # ---- Aggregate ----
    pz = np.array([v["PZ"] for v in per_case.values()])
    cg = np.array([v["CG"] for v in per_case.values()])
    tum = np.array([v["Tumor"] for v in per_case.values()])
    nz = int((tum > 0.01).sum())
    print("\n" + "=" * 70)
    print(f"CASCADE ({bbox_source}) — test set summary ({len(per_case)} cases):")
    print(f"  PZ:    {pz.mean():.4f} ± {pz.std():.4f}")
    print(f"  CG:    {cg.mean():.4f} ± {cg.std():.4f}")
    print(f"  Tumor: {tum.mean():.4f} ± {tum.std():.4f}  "
          f"(non-zero on {nz}/{len(per_case)} cases)")
    print("=" * 70)

    # ---- Save ----
    out_dir = os.path.join(cascade_root, "stage2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"per_case_dsc_cascade_{bbox_source}.json")
    with open(out_path, "w") as f:
        json.dump(per_case, f, indent=2)
    print(f"  Per-case DSC saved: {out_path}")
    print(f"  Architecture:       {arch}")
    return per_case


def _load_predicted_bboxes(cascade_root: str):
    """Load bboxes_{train,val,test}.json; return (val_dict, test_dict) or None."""
    val_path = os.path.join(cascade_root, "bboxes_val.json")
    test_path = os.path.join(cascade_root, "bboxes_test.json")
    val_b, test_b = None, None
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_b = {k: tuple(v) for k, v in json.load(f).items()}
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_b = {k: tuple(v) for k, v in json.load(f).items()}
    return val_b, test_b


def run_cascade_stage2(config: Config):
    """
    Treniraj Stage 2 (TumorSegNet) na cropanim 48×128×128 ROI volumenima s
    imbalance-corrected loss receptom (class-weighted CE w_tumour=10 +
    Dice [+ deep supervision + boundary refinement za DSBANet]) i case-level
    oversamplingom (factor 3) za tumour-positive slučajeve.

    Arhitektura određena s ``config.cascade_arch``:
      - "dsba_net_3d": puna DSBANet 3D mašinerija (SE, ASPP, MSAF, FFM, DS, BRM)
      - "unet3d": vanilla 3D U-Net (lakša baselina, bez DS/BRM)
    """
    arch = getattr(config, "cascade_arch", "dsba_net_3d")
    print("=" * 70)
    print(f"CASCADE STAGE 2 ({arch}): TumorSegNet (in-ROI multi-class)")
    print("=" * 70)

    config.dataset_name = "prostate158"
    config.cascade_mode = "stage2"
    config.model_name = arch
    config.out_channels = 4
    config.loss_function = "combined"
    # Imbalance correction inherited from the 2D ablation winner
    config.tumor_weight = getattr(config, "cascade_stage2_tumor_weight", 10.0)
    config.oversample_tumor = True
    config.tumor_oversample_factor = float(
        getattr(config, "cascade_oversample_factor", 3.0))
    config.auto_resume = True

    if arch == "dsba_net_3d":
        # Full DSBANet machinery: SE, ASPP, MSAF, FFM, deep supervision, BRM
        config.ablation_use_se = True
        config.ablation_use_aspp = True
        config.ablation_use_msaf = True
        config.ablation_use_ffm = True
        config.ablation_use_dag = False
        config.ablation_use_ds = True
        config.ablation_use_brm = True
        config.deep_supervision = True
    elif arch == "unet3d":
        # Vanilla 3D U-Net: no deep supervision, no boundary loss
        config.deep_supervision = False
        config.ablation_use_ds = False
        config.ablation_use_brm = False
        config.base_filters = int(
            getattr(config, "cascade_stage2_base_filters", 32))

    # Load predicted bboxes from Stage 1 (val + test eval; training uses GT+jitter).
    # Stage 2 always reads bboxes from its OWN arch's Stage 1 outputs.
    stage1_root = _cascade_root(config, arch, "stage1")
    val_b, test_b = _load_predicted_bboxes(stage1_root)
    if val_b is None or test_b is None:
        raise FileNotFoundError(
            f"Missing bboxes JSONs under {stage1_root}. "
            f"Run `--mode cascade_predict_bboxes --cascade_arch {arch}` first.")
    print(f"Loaded predicted bboxes ({arch}): val={len(val_b)} cases, "
          f"test={len(test_b)} cases.")
    config.cascade_predicted_bboxes_val = val_b
    config.cascade_predicted_bboxes_test = test_b

    cascade_root = _cascade_root(config, arch, "stage2")
    cascade_ckpt = os.path.join(cascade_root, "checkpoints")
    os.makedirs(cascade_ckpt, exist_ok=True)
    config.output_dir = cascade_root
    config.checkpoint_dir = cascade_ckpt

    metrics = train_single_model(config)
    print("\n" + "=" * 70)
    print(f"STAGE 2 ({arch}) — final test metrics (with predicted Stage-1 bboxes):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 70)
    return metrics


def predict_cascade_bboxes(config: Config, checkpoint_path: str = None):
    """
    Učitaj Stage 1 checkpoint, izvedi inferenciju na train + val + test splitu,
    izračunaj ROI bounding box za svaki slučaj i spremi rezultat u JSON.
    Stage 2 trening i evaluacija konzumiraju ove bboxove u inferenciji.

    Per-case stage-1 DSC se također spremi u JSON, kao sanity check
    (radi gate-a iz §6 plana: ako je stage-1 DSC < 0.85, ne nastavljamo
    sa Stage 2 prije zamjene arhitekture).
    """
    from src.dataset_prostate158 import (
        get_prostate158_paths, load_case_nifti, load_case_multimodal,
        preprocess_volume, resize_volume, _case_id_from_path,
    )
    from src.cascade import compute_roi_bbox

    print("=" * 70)
    print("CASCADE: predict prostate ROI bounding boxes for all splits")
    print("=" * 70)

    arch = getattr(config, "cascade_arch", "dsba_net_3d")
    config.dataset_name = "prostate158"
    config.cascade_mode = "stage1"
    config.model_name = arch
    config.out_channels = 2
    config.deep_supervision = False
    config.ablation_use_ds = False
    config.ablation_use_brm = False
    if arch == "unet3d":
        config.base_filters = int(
            getattr(config, "cascade_stage1_base_filters", 16))

    # Resolve checkpoint
    cascade_root = _cascade_root(config, arch, "stage1")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            cascade_root, "checkpoints", f"best_{arch}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {checkpoint_path}. "
            f"Run `--mode cascade_stage1 --cascade_arch {arch}` first.")
    print(f"Loading Stage 1 checkpoint ({arch}): {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output_dir = cascade_root
    config.checkpoint_dir = os.path.join(cascade_root, "checkpoints")

    model = create_model(config).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Iterate triples directly instead of preloading the full Dataset to keep
    # memory bounded — multimodal loading triples per-case memory usage and
    # the Dataset class would otherwise keep all 158 cases × 3 modalities
    # in RAM simultaneously (~14 GB for prostate158).
    train_triples, val_triples, test_triples = get_prostate158_paths(
        config.prostate158_dir)
    splits = {"train": train_triples, "val": val_triples,
              "test": test_triples}

    multimodal = bool(getattr(config, "multimodal", False))
    modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
    full_size = config.volume_size_3d

    margin = tuple(getattr(config, "cascade_bbox_margin_voxels", (2, 8, 8)))
    min_size = tuple(getattr(config, "cascade_min_bbox_size", (24, 96, 96)))

    all_bboxes = {}
    all_stage1_dsc = {}
    smooth = 1e-6

    for split_name, triples in splits.items():
        if not triples:
            continue
        print(f"\nSplit: {split_name} ({len(triples)} cases)")
        split_bboxes = {}
        for t2_path, seg_path, tumor_path in triples:
            case_id = _case_id_from_path(t2_path)

            # Load + preprocess + resize for one case only
            if multimodal:
                image, mask, spacing = load_case_multimodal(
                    t2_path, seg_path, tumor_path, modalities=modalities)
            else:
                image, mask, spacing = load_case_nifti(
                    t2_path, seg_path, tumor_path)
            image, mask = preprocess_volume(image, mask, spacing, config)
            image = resize_volume(image, full_size)
            mask = resize_volume(mask, full_size, is_mask=True)
            gt = (mask > 0).astype(np.uint8)

            # Forward through Stage 1
            if multimodal:
                x = torch.from_numpy(image).unsqueeze(0).float().to(device)
            else:
                x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, dict):
                    out = out["main"]
                pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Bbox from predicted mask
            bbox = compute_roi_bbox(pred, margin_voxels=margin,
                                     min_size=min_size, use_largest_cc=True)
            all_bboxes[case_id] = list(bbox)
            split_bboxes[case_id] = list(bbox)

            # Per-case stage-1 DSC vs GT
            inter = float((pred * gt).sum())
            union = float(pred.sum() + gt.sum())
            dsc = (2.0 * inter + smooth) / (union + smooth)
            all_stage1_dsc[case_id] = float(dsc)
            print(f"  {case_id}: DSC={dsc:.4f}  bbox=(z={bbox[0]}-{bbox[1]}, "
                  f"y={bbox[2]}-{bbox[3]}, x={bbox[4]}-{bbox[5]})")

            # Free per-case memory before next iteration
            del image, mask, gt, x, out, pred

        # Save per-split JSON
        out_path = os.path.join(cascade_root, f"bboxes_{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_bboxes, f, indent=2)
        print(f"  Saved: {out_path}")

    # Save aggregate DSC for the gate check
    dsc_path = os.path.join(cascade_root, "stage1_per_case_dsc.json")
    with open(dsc_path, "w") as f:
        json.dump(all_stage1_dsc, f, indent=2)
    mean_dsc = float(np.mean(list(all_stage1_dsc.values())))
    print(f"\nStage 1 DSC across all splits: mean={mean_dsc:.4f} "
          f"({len(all_stage1_dsc)} cases)")
    if mean_dsc < 0.85:
        print("  ⚠  Stage 1 DSC below 0.85 — review before proceeding to Stage 2.")
    else:
        print("  ✓ Stage 1 DSC ≥ 0.85 — safe to proceed to Stage 2.")
    print(f"  All bboxes: {cascade_root}/bboxes_{{train,val,test}}.json")
    print(f"  All DSCs:   {dsc_path}")


def save_cascade_test_predictions_nifti(config: Config,
                                          arch: str = None,
                                          stage1_ckpt: str = None,
                                          stage2_ckpt: str = None):
    """
    Generate NIfTI prediction files for all 19 test cases using the trained
    cascade. The prediction is computed at the canonical (32, 256, 256)
    voxel grid (matching training preprocessing) and then resampled back to
    the original T2 volume grid using nearest-neighbour interpolation, so
    that the saved ``prediction.nii.gz`` overlays correctly with
    ``t2.nii.gz`` in any standard medical-image viewer (3D Slicer,
    ITK-SNAP, MRIcron, etc.).

    Output structure:
        output/prostate158/<cascade_root>/test_predictions_nifti/<case>/
            ├── prediction.nii.gz       (4-class label map, uint8)
            └── prediction_overlay.nii.gz  (optional same content; can be loaded as label map)
    """
    import nibabel as nib
    from scipy.ndimage import zoom as _zoom
    from src.dataset_prostate158 import (
        get_prostate158_paths,
        load_case_nifti,
        preprocess_volume,
        resize_volume,
    )
    from src.cascade import compute_roi_bbox, paste_back

    if arch is None:
        arch = getattr(config, "cascade_arch", "dsba_net_3d")

    print("=" * 70)
    print(f"CASCADE: saving NIfTI test-set predictions ({arch})")
    print("=" * 70)

    cascade_root = _cascade_root(config, arch)
    if stage1_ckpt is None:
        stage1_ckpt = os.path.join(cascade_root, "stage1", "checkpoints",
                                    f"best_{arch}.pth")
    if stage2_ckpt is None:
        stage2_ckpt = os.path.join(cascade_root, "stage2", "checkpoints",
                                    f"best_{arch}.pth")
    for p in [stage1_ckpt, stage2_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Cascade checkpoint not found: {p}. "
                f"Train both stages first with --cascade_arch {arch}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.dataset_name = "prostate158"

    # ---- Build Stage 1 ----
    s1_cfg = Config()
    s1_cfg.dataset_name = "prostate158"
    s1_cfg.model_name = arch
    s1_cfg.out_channels = 2
    s1_cfg.deep_supervision = False
    s1_cfg.ablation_use_ds = False
    s1_cfg.ablation_use_brm = False
    s1_cfg.multimodal = bool(getattr(config, "multimodal", False))
    s1_cfg.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
    if arch == "unet3d":
        s1_cfg.base_filters = int(
            getattr(config, "cascade_stage1_base_filters", 16))
    s1_model = create_model(s1_cfg).to(device)
    s1_model.load_state_dict(
        torch.load(stage1_ckpt, map_location=device,
                   weights_only=False)["model_state_dict"])
    s1_model.eval()

    # ---- Build Stage 2 ----
    s2_cfg = Config()
    s2_cfg.dataset_name = "prostate158"
    s2_cfg.model_name = arch
    s2_cfg.out_channels = 4
    s2_cfg.multimodal = bool(getattr(config, "multimodal", False))
    s2_cfg.modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))
    if arch == "dsba_net_3d":
        s2_cfg.deep_supervision = True
        s2_cfg.ablation_use_se = True
        s2_cfg.ablation_use_aspp = True
        s2_cfg.ablation_use_msaf = True
        s2_cfg.ablation_use_ffm = True
        s2_cfg.ablation_use_ds = True
        s2_cfg.ablation_use_brm = True
    elif arch == "unet3d":
        s2_cfg.deep_supervision = False
        s2_cfg.ablation_use_ds = False
        s2_cfg.ablation_use_brm = False
        s2_cfg.base_filters = int(
            getattr(config, "cascade_stage2_base_filters", 32))
    s2_model = create_model(s2_cfg).to(device)
    s2_model.load_state_dict(
        torch.load(stage2_ckpt, map_location=device,
                   weights_only=False)["model_state_dict"])
    s2_model.eval()

    full_size = config.volume_size_3d
    target = getattr(config, "cascade_stage2_volume_size", (48, 128, 128))
    margin = tuple(getattr(config, "cascade_bbox_margin_voxels", (2, 8, 8)))
    min_size = tuple(getattr(config, "cascade_min_bbox_size", (24, 96, 96)))
    multimodal = bool(getattr(config, "multimodal", False))
    modalities = tuple(getattr(config, "modalities", ("t2", "adc", "dwi")))

    out_dir = os.path.join(cascade_root, "test_predictions_nifti")
    os.makedirs(out_dir, exist_ok=True)

    _, _, test_triples = get_prostate158_paths(config.prostate158_dir)
    print(f"Test cases: {len(test_triples)}")

    for t2_path, seg_path, tumor_path in test_triples:
        case_id = os.path.basename(os.path.dirname(t2_path))

        # 1) Load the original T2 NIfTI to get affine + native shape
        ref_nii = nib.load(t2_path)
        ref_affine = ref_nii.affine
        # nibabel arrays are (X, Y, Z); we treat them as (W, H, D) inside the
        # pipeline (slice axis last). Final NIfTI output keeps this convention.
        native_shape = ref_nii.shape  # (Wx, Wy, Wz)

        # 2) Preprocess + resize to canonical (32, 256, 256) grid (matches training)
        if multimodal:
            from src.dataset_prostate158 import load_case_multimodal
            image, _mask, spacing = load_case_multimodal(
                t2_path, seg_path, tumor_path, modalities=modalities)
        else:
            image, _mask, spacing = load_case_nifti(
                t2_path, seg_path, tumor_path)
        image, _ = preprocess_volume(image, _mask, spacing, config)
        image = resize_volume(image, full_size)        # (M, D, H, W) or (D, H, W)

        # 3) Run cascade pipeline at canonical grid
        with torch.no_grad():
            if multimodal:
                x = torch.from_numpy(image).unsqueeze(0).float().to(device)
            else:
                x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            out1 = s1_model(x)
            if isinstance(out1, dict):
                out1 = out1["main"]
            pred_binary = out1.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            bbox = compute_roi_bbox(pred_binary, margin_voxels=margin,
                                     min_size=min_size, use_largest_cc=True)
            z0, z1, y0, y1, x0, x1 = bbox

            if multimodal:
                img_roi = image[:, z0:z1, y0:y1, x0:x1]
            else:
                img_roi = image[z0:z1, y0:y1, x0:x1]
            img_roi_resampled = resize_volume(img_roi, target)
            if multimodal:
                xr = torch.from_numpy(img_roi_resampled).unsqueeze(0).float().to(device)
            else:
                xr = torch.from_numpy(img_roi_resampled).unsqueeze(0).unsqueeze(0).float().to(device)
            out2 = s2_model(xr)
            if isinstance(out2, dict):
                out2 = out2["main"]
            pred_4c_resampled = out2.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Resample 4-class prediction back to bbox-native shape, paste to canonical full volume
        bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
        pred_4c_native = resize_volume(pred_4c_resampled, bbox_shape,
                                        is_mask=True).astype(np.int64)
        full_pred_canonical = paste_back(pred_4c_native, bbox, full_size,
                                          fill_value=0, dtype=np.int64)

        # 4) Resample from canonical (32, 256, 256) → native T2 shape (D, H, W).
        # nibabel ref_nii.shape is (Wx, Wy, Wz). Our canonical-frame array is
        # (D, H, W) = (Wz, Wy, Wx) after the (axes-transposing) preprocess.
        # We restore the original axis order by transposing back to (W, H, D).
        d_t = native_shape[2]   # slice count
        h_t = native_shape[1]   # in-plane height
        w_t = native_shape[0]   # in-plane width
        factors = (d_t / full_size[0], h_t / full_size[1], w_t / full_size[2])
        if any(abs(f - 1.0) > 1e-6 for f in factors):
            pred_native = _zoom(full_pred_canonical.astype(np.float64),
                                 factors, order=0).astype(np.uint8)
        else:
            pred_native = full_pred_canonical.astype(np.uint8)
        # Transpose (D, H, W) → (W, H, D) to match nibabel/T2 axis order
        pred_native = pred_native.transpose(2, 1, 0)
        assert pred_native.shape == native_shape, (
            f"shape mismatch: pred {pred_native.shape} vs T2 {native_shape}")

        # 5) Save NIfTI with the T2's affine (overlay-correct in any viewer)
        case_out = os.path.join(out_dir, case_id)
        os.makedirs(case_out, exist_ok=True)
        nii = nib.Nifti1Image(pred_native, ref_affine, ref_nii.header)
        nii.set_data_dtype(np.uint8)
        out_path = os.path.join(case_out, "prediction.nii.gz")
        nib.save(nii, out_path)
        # Per-class voxel counts (for a quick sanity snapshot)
        cnt = [int((pred_native == c).sum()) for c in range(4)]
        print(f"  {case_id}: bg={cnt[0]}  PZ={cnt[1]}  CG={cnt[2]}  "
              f"Tumor={cnt[3]}  shape={pred_native.shape}  → {out_path}")

    print(f"\nAll predictions saved under: {out_dir}/<case>/prediction.nii.gz")
    return out_dir


def _cascade_subdir(arch: str) -> str:
    """Return the 'cascade' or 'cascade_unet3d' subdirectory name."""
    return "cascade" if arch == "dsba_net_3d" else f"cascade_{arch}"


def _cascade_root(config: Config, arch: str, stage: str = "") -> str:
    """
    Return the cascade output root for the given architecture, automatically
    routing under a ``multimodal/`` subdirectory when ``config.multimodal``
    is True. ``stage`` (one of "stage1", "stage2") is appended when non-empty.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    multimodal_subdir = "multimodal" if getattr(config, "multimodal", False) else ""
    parts = [project_root, "output", config.dataset_name, multimodal_subdir,
             _cascade_subdir(arch)]
    if stage:
        parts.append(stage)
    # Empty strings are silently dropped by os.path.join.
    return os.path.join(*parts)


def run_cascade_stage1(config: Config):
    """
    Treniraj Stage 1 (ProstateROINet): binarna prostate lokalizacija na punom
    3D volumenu. Arhitektura određena s ``config.cascade_arch``:
      - "dsba_net_3d": DSBANet 3D s ``out_channels=2`` (deep supervision i
        boundary loss su isključeni), izlazi u ``output/.../cascade/stage1/``.
      - "unet3d": vanilla 3D U-Net s ``out_channels=2`` i
        ``base_filters = cascade_stage1_base_filters`` (default 16),
        izlazi u ``output/.../cascade_unet3d/stage1/``.
    """
    arch = getattr(config, "cascade_arch", "dsba_net_3d")
    print("=" * 70)
    print(f"CASCADE STAGE 1 ({arch}): ProstateROINet (binary localisation)")
    print("=" * 70)

    config.dataset_name = "prostate158"
    config.cascade_mode = "stage1"
    config.model_name = arch
    config.out_channels = 2
    config.loss_function = getattr(config, "cascade_stage1_loss", "combined")
    config.deep_supervision = False
    config.ablation_use_ds = False
    config.ablation_use_brm = False
    config.tumor_weight = 1.0
    config.oversample_tumor = False
    config.auto_resume = True
    if arch == "unet3d":
        # Vanilla 3D U-Net is much lighter; use the dedicated Stage 1 base
        # filter count (default 16, ≈8× lighter than DSBANet 3D).
        config.base_filters = int(
            getattr(config, "cascade_stage1_base_filters", 16))

    cascade_root = _cascade_root(config, arch, "stage1")
    cascade_ckpt = os.path.join(cascade_root, "checkpoints")
    os.makedirs(cascade_ckpt, exist_ok=True)
    config.output_dir = cascade_root
    config.checkpoint_dir = cascade_ckpt

    metrics = train_single_model(config)
    print("\n" + "=" * 70)
    print(f"STAGE 1 ({arch}) — final test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 70)
    return metrics


def evaluate_saved_model(config: Config, checkpoint_path: str):
    """Evaluira prethodno spremljeni model."""
    print("=" * 70)
    print(f"EVALUACIJA MODELA: {config.model_name.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Učitaj model
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model učitan iz: {checkpoint_path}")
    print(f"Epoha treniranja: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val DSC pri spremanju: {checkpoint.get('val_dsc', 'N/A')}")

    # Stvori dataset
    if config.dataset_name == "prostate158":
        _, val_dataset, test_dataset = create_prostate158_datasets(config)
    else:
        _, val_dataset, test_dataset = create_datasets(config)
    eval_dataset = test_dataset if test_dataset else val_dataset

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.effective_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Evaluacija
    metrics = evaluate_model(model, eval_loader, device)
    print("\nRezultati evaluacije:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Vizualizacija
    is_3d = (config.model_name == "unet3d" or "_3d" in config.model_name)
    seg_path = os.path.join(config.output_dir,
                            f"eval_examples_{config.model_name}.png")
    plot_segmentation_examples(model, eval_dataset, device, seg_path,
                               num_examples=6, is_3d=is_3d)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentacija prostate na MR slikama - PROMISE12"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "single", "evaluate", "visualize_data", "ablation", "kfold",
                 "cascade_stage1", "cascade_predict_bboxes", "cascade_stage2",
                 "cascade_evaluate", "cascade_save_nifti"],
        help="Način rada: all (svi modeli), single (jedan model), "
             "evaluate (evaluacija), visualize_data (vizualizacija dataseta), "
             "ablation (ablacijska studija DSBANet), "
             "kfold (K-fold cross-validation s ensemble), "
             "cascade_stage1 (Stage 1 binary prostate localisation), "
             "cascade_predict_bboxes (run Stage 1 inference → save bboxes JSON), "
             "cascade_stage2 (Stage 2 in-ROI multi-class segmentation), "
             "cascade_evaluate (end-to-end Stage 1 + Stage 2 pipeline; "
             "use --bbox_source {predicted,gt,full}), "
             "cascade_save_nifti (write per-case NIfTI predictions resampled "
             "back to original T2 grid)"
    )
    parser.add_argument(
        "--model", type=str, default="unet2d",
        choices=["unet2d", "unet25d", "unet3d",
                 "attention_unet", "unet_plus_plus", "resunet",
                 "transunet", "swin_unet",
                 "attention_unet_25d", "unet_plus_plus_25d", "resunet_25d",
                 "transunet_25d", "swin_unet_25d",
                 "attention_unet_3d", "unet_plus_plus_3d", "resunet_3d",
                 "transunet_3d", "swin_unet_3d",
                 "msda_net", "msda_net_25d", "msda_net_3d",
                 "dsba_net", "dsba_net_25d", "dsba_net_3d"],
        help="Naziv modela za treniranje/evaluaciju"
    )
    parser.add_argument("--dataset", type=str, default="promise12",
                        choices=["promise12", "prostate158"],
                        help="Dataset za treniranje/evaluaciju")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Putanja do checkpointa za evaluaciju")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Broj epoha treniranja")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Veličina batcha")
    parser.add_argument("--lr", type=float, default=None,
                        help="Stopa učenja")
    parser.add_argument("--base_filters", type=int, default=None,
                        help="Broj baznih filtera u modelu")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Isključi augmentaciju podataka")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["dice", "bce", "combined", "focal",
                                 "tversky", "focal_tversky",
                                 "combined_focal_tversky"],
                        help="Funkcija gubitka")
    parser.add_argument("--tumor_weight", type=float, default=None,
                        help="Multiplikator za tumor klasu u CE/Tversky/Focal "
                             "(npr. 5.0 = tumor gradient 5×)")
    parser.add_argument("--oversample_tumor", action="store_true",
                        help="WeightedRandomSampler s tumor-positive slice "
                             "boostom (samo 2D / 2.5D)")
    parser.add_argument("--tumor_oversample_factor", type=float, default=None,
                        help="Faktor pojačanja za tumor-positive slice")
    parser.add_argument("--deep_supervision", action="store_true",
                        help="Uključi deep supervision (za MSDA-Net)")
    parser.add_argument("--cosine_annealing", action="store_true",
                        help="Koristi CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--enhanced_aug", action="store_true",
                        help="Uključi poboljšanu augmentaciju (rotacija, elastic, intenzitet)")
    parser.add_argument("--boundary_weight", type=float, default=None,
                        help="Težina boundary gubitka")
    parser.add_argument("--tta", action="store_true",
                        help="Koristi Test-Time Augmentation pri evaluaciji")
    parser.add_argument("--postprocess", action="store_true",
                        help="Zadrži samo najveću povezanu komponentu")
    parser.add_argument("--pretrained", action="store_true",
                        help="Koristi pretrained ResNet50 enkoder (za DSBANet)")
    parser.add_argument("--gan", action="store_true",
                        help="Uključi adversarial training s PatchGAN diskriminatorom")
    parser.add_argument("--adv_weight", type=float, default=None,
                        help="Težina adversarial gubitka (default: 0.01)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Broj foldova za K-fold cross-validation")
    parser.add_argument("--advanced_preprocess", action="store_true",
                        help="Uključi napredni preprocessing (resampling + CLAHE + ROI crop)")
    parser.add_argument("--volume_size", type=int, nargs=3, default=None,
                        metavar=("D", "H", "W"),
                        help="Veličina 3D volumena (D H W), npr. --volume_size 16 128 128")
    parser.add_argument("--max_variants", type=int, default=None,
                        help="Ograniči ablacijsku studiju na prvih N varijanti "
                             "(za smoke testove)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ne učitavaj last.pth checkpoint; kreni od epohe 1.")
    parser.add_argument("--retrain_done", action="store_true",
                        help="Ablacija: ponovo treniraj i varijante koje "
                             "imaju done.flag (default: preskoči gotove).")
    parser.add_argument("--bbox_source", type=str, default="predicted",
                        choices=["predicted", "gt", "full"],
                        help="cascade_evaluate: 'predicted' = Stage 1 output "
                             "(realistic cascade); 'gt' = ground-truth bbox "
                             "(oracle ceiling); 'full' = no crop (single-stage "
                             "baseline emulation).")
    parser.add_argument("--cascade_arch", type=str, default="dsba_net_3d",
                        choices=["dsba_net_3d", "unet3d"],
                        help="Architecture used for both cascade stages. "
                             "'dsba_net_3d' (default) uses the full DSBANet 3D "
                             "machinery; 'unet3d' uses a vanilla 3D U-Net "
                             "(lightweight baseline cascade). Outputs go to "
                             "separate subdirectories.")
    parser.add_argument("--multimodal", action="store_true",
                        help="Use multimodal input (T2 + ADC + DWI stacked as "
                             "channels). All three modalities are read from "
                             "the same case directory; Prostate158 ships them "
                             "pre-registered. Outputs route to "
                             "output/<dataset>/multimodal/.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Primijeni argumente iz naredbenog retka
    config.dataset_name = args.dataset
    config.model_name = args.model
    config.cascade_arch = args.cascade_arch
    if args.multimodal:
        config.multimodal = True
        # Re-resolve output paths now that multimodal is set, so that all
        # downstream artifacts route under output/<dataset>/multimodal/.
        config._update_paths()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        config.batch_size_3d = max(1, args.batch_size // 4)
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.base_filters is not None:
        config.base_filters = args.base_filters
    if args.no_augmentation:
        config.use_augmentation = False
    if args.loss is not None:
        config.loss_function = args.loss
    if args.tumor_weight is not None:
        config.tumor_weight = args.tumor_weight
    if args.oversample_tumor:
        config.oversample_tumor = True
    if args.tumor_oversample_factor is not None:
        config.tumor_oversample_factor = args.tumor_oversample_factor
    if args.deep_supervision:
        config.deep_supervision = True
    if args.cosine_annealing:
        config.use_cosine_annealing = True
    if args.enhanced_aug:
        config.use_enhanced_aug = True
    if args.boundary_weight is not None:
        config.boundary_weight = args.boundary_weight
    if args.pretrained:
        config.use_pretrained = True
    if args.gan:
        config.use_gan = True
    if args.adv_weight is not None:
        config.adv_weight = args.adv_weight
    if args.advanced_preprocess:
        config.use_advanced_preprocessing = True
    if args.volume_size is not None:
        config.volume_size_3d = tuple(args.volume_size)
    if args.tta:
        config.use_tta = True
    if args.postprocess:
        config.use_postprocess = True
    config.auto_resume = not args.no_resume

    # Ažuriraj putanje nakon što su svi argumenti primijenjeni
    config._update_paths()

    # Postavi seed za reproducibilnost
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    # Ispiši konfiguraciju
    print("=" * 70)
    print("AUTOMATSKA SEGMENTACIJA PROSTATE NA MR SLIKAMA")
    print(f"{config.dataset_name.upper()} Dataset | PyTorch")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Epohe: {config.num_epochs}")
    print(f"Batch size: {config.effective_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Base filters: {config.base_filters}")
    print(f"Funkcija gubitka: {config.loss_function}")
    print(f"Augmentacija: {config.use_augmentation}")
    print(f"Uređaj: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    if args.mode == "visualize_data":
        visualize_dataset(config)

    elif args.mode == "single":
        visualize_dataset(config)
        train_single_model(config)

    elif args.mode == "all":
        visualize_dataset(config)
        train_all_models(config)

    elif args.mode == "kfold":
        from src.kfold import train_kfold
        train_kfold(config, n_folds=args.n_folds)

    elif args.mode == "ablation":
        # visualize_dataset koristi promise12 path; preskoči za prostate158 ablation
        if config.dataset_name != "prostate158":
            visualize_dataset(config)
        run_ablation_study(config, max_variants=args.max_variants,
                            retrain_done=args.retrain_done)

    elif args.mode == "cascade_stage1":
        run_cascade_stage1(config)

    elif args.mode == "cascade_predict_bboxes":
        predict_cascade_bboxes(config, checkpoint_path=args.checkpoint)

    elif args.mode == "cascade_stage2":
        run_cascade_stage2(config)

    elif args.mode == "cascade_evaluate":
        evaluate_cascade_full(config, bbox_source=args.bbox_source)

    elif args.mode == "cascade_save_nifti":
        save_cascade_test_predictions_nifti(config, arch=args.cascade_arch)

    elif args.mode == "evaluate":
        if args.checkpoint is None:
            print("Greška: potrebno je navesti --checkpoint za evaluaciju.")
            sys.exit(1)
        evaluate_saved_model(config, args.checkpoint)


if __name__ == "__main__":
    main()
