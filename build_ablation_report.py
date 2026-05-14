"""
Generira paper-ready izvještaj za ablacijsku studiju DSBANet — effect-size
framing.

Za svaku ablaciju vs. Full računa per-class:
  - mean DSC ± std (i 95% bootstrap CI)
  - mean difference (Δ = Full - variant) s 95% bootstrap CI
  - rank-biserial correlation r (paired) — non-parametric effect size
  - Cohen's d_z (paired) — parametric effect size kao referenca

Generira:
  - ablation_table.md  (Markdown, ljudski-čitljivo)
  - ablation_table.tex (LaTeX, spremno za umetanje)
  - ablation_effect_sizes.json (numerički, za reproducibilnost)

Korištenje:
    python build_ablation_report.py --dim 2d
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.train import create_model


# Mapa naziva varijante na ablation flagove (mora biti ista kao u main.py)
ABLATION_FLAGS = {
    "DSBANet (full)":         (True,  True,  True,  True,  False, True,  True),
    "DSBANet w/o SE":         (False, True,  True,  True,  False, True,  True),
    "DSBANet w/o ASPP":       (True,  False, True,  True,  False, True,  True),
    "DSBANet w/o MSAF (DAG)": (True,  True,  False, True,  True,  True,  True),
    "DSBANet w/o skip-att.":  (True,  True,  False, True,  False, True,  True),
    "DSBANet w/o FFM":        (True,  True,  True,  False, False, True,  True),
    "DSBANet w/o DS":         (True,  True,  True,  True,  False, False, True),
    "DSBANet w/o BRM":        (True,  True,  True,  True,  False, True,  False),
}

CLASSES = ("PZ", "CG", "Tumor")
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 5000


# ---------------------------------------------------------------------------
# Učitavanje
# ---------------------------------------------------------------------------

def load_per_case(ablation_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Učita per-case DSC za svaku varijantu iz pojedinačnih JSON-ova."""
    per_case_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in ABLATION_FLAGS:
        suffix = (variant.replace("DSBANet ", "")
                          .replace(" ", "_")
                          .replace("(", "").replace(")", "")
                          .replace("/", "")
                          .replace(".", "")
                          .lower())
        if "full" in suffix:
            suffix = "dsbanet_full"
        else:
            suffix = "dsbanet_" + suffix
        path = os.path.join(ablation_dir, suffix, "per_case_dsc.json")
        if os.path.exists(path):
            with open(path) as f:
                per_case_all[variant] = json.load(f)
    return per_case_all


# ---------------------------------------------------------------------------
# Effect-size statistike
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, alpha: float = 0.05,
                 n: int = N_BOOTSTRAP) -> Tuple[float, float]:
    """Percentilni bootstrap 95% CI za mean."""
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = RNG.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_diff_ci(full: np.ndarray, variant: np.ndarray,
                   alpha: float = 0.05, n: int = N_BOOTSTRAP
                   ) -> Tuple[float, float, float]:
    """Bootstrap CI za mean(Full - variant) na uparenim slučajevima."""
    diffs = full - variant
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    boot = RNG.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi


def rank_biserial(full: np.ndarray, variant: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation r.
    r = (W+ - W-) / (W+ + W-) gdje su W± sume ranking-a apsolutnih razlika
    nad pozitivnim/negativnim parovima. r ∈ [-1, 1]; predznak = smjer
    (Full > variant ⇒ pozitivno).
    """
    diffs = full - variant
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    # Rangiraj |diffs|
    abs_d = np.abs(nonzero)
    # average ranks for ties
    order = np.argsort(abs_d)
    ranks = np.empty(len(abs_d), dtype=float)
    i = 0
    while i < len(abs_d):
        j = i
        while j + 1 < len(abs_d) and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg = (i + j + 2) / 2  # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    Wp = ranks[nonzero > 0].sum()
    Wn = ranks[nonzero < 0].sum()
    total = Wp + Wn
    return float((Wp - Wn) / total) if total > 0 else 0.0


def cohens_dz(full: np.ndarray, variant: np.ndarray) -> float:
    """Cohen's d_z za uparene uzorke = mean(diff) / std(diff)."""
    diffs = full - variant
    if len(diffs) < 2:
        return float("nan")
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def magnitude(r: float) -> str:
    """Cohen-style descriptor za |rank-biserial r| (small/medium/large)."""
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Per-variant statistike
# ---------------------------------------------------------------------------

def variant_summary(per_case: Dict[str, Dict[str, Dict[str, float]]],
                    full_key: str = "DSBANet (full)"
                    ) -> Dict[str, Dict[str, dict]]:
    """
    Vrati po varijanti (osim full) i klasi:
      mean_full, ci_full, mean_var, ci_var, delta, ci_delta,
      r_rb, d_z, magnitude
    Plus za full: mean + CI.
    """
    out: Dict[str, Dict[str, dict]] = {}
    full_data = per_case[full_key]
    common = sorted(full_data.keys())  # sve casee

    # Full row
    out[full_key] = {}
    for c in CLASSES:
        full_vals = np.array([full_data[k].get(c, np.nan) for k in common])
        full_vals = full_vals[~np.isnan(full_vals)]
        lo, hi = bootstrap_ci(full_vals)
        out[full_key][c] = {
            "mean": float(full_vals.mean()),
            "std": float(full_vals.std(ddof=1)) if len(full_vals) > 1 else 0.0,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": int(len(full_vals)),
        }

    # Variant rows
    for variant, var_data in per_case.items():
        if variant == full_key:
            continue
        out[variant] = {}
        keys = sorted(set(full_data) & set(var_data))
        for c in CLASSES:
            f = np.array([full_data[k].get(c, np.nan) for k in keys])
            v = np.array([var_data[k].get(c, np.nan) for k in keys])
            mask = ~(np.isnan(f) | np.isnan(v))
            f, v = f[mask], v[mask]
            v_lo, v_hi = bootstrap_ci(v)
            d_mean, d_lo, d_hi = paired_diff_ci(f, v)
            r = rank_biserial(f, v)
            dz = cohens_dz(f, v)
            out[variant][c] = {
                "mean": float(v.mean()) if len(v) else float("nan"),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "ci_lo": v_lo,
                "ci_hi": v_hi,
                "delta": d_mean,
                "delta_ci_lo": d_lo,
                "delta_ci_hi": d_hi,
                "rank_biserial": r,
                "cohens_dz": dz,
                "magnitude": magnitude(r),
                "n": int(len(v)),
            }
    return out


# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------

def count_params(dim: str) -> Dict[str, int]:
    """Stvara model za svaku varijantu i broji parametre."""
    cfg = Config()
    cfg.model_name = {"2d": "dsba_net", "25d": "dsba_net_25d",
                      "3d": "dsba_net_3d"}[dim]
    counts = {}
    for variant, (se, aspp, msaf, ffm, dag, ds, brm) in ABLATION_FLAGS.items():
        cfg.ablation_use_se = se
        cfg.ablation_use_aspp = aspp
        cfg.ablation_use_msaf = msaf
        cfg.ablation_use_ffm = ffm
        cfg.ablation_use_dag = dag
        cfg.ablation_use_ds = ds
        cfg.ablation_use_brm = brm
        cfg.deep_supervision = ds or brm
        m = create_model(cfg)
        counts[variant] = m.count_parameters()
        del m
    return counts


# ---------------------------------------------------------------------------
# Renderiranje
# ---------------------------------------------------------------------------

def short_name(name: str) -> str:
    if "full" in name:
        return "Full"
    return name.replace("DSBANet w/o ", "w/o ")


def render_markdown(stats: Dict[str, Dict[str, dict]],
                    params: Dict[str, int],
                    full_key: str = "DSBANet (full)") -> str:
    lines = []
    lines.append("# DSBANet 2D ablation — effect-size summary")
    lines.append("")
    lines.append("Per-class DSC mean ± std on 19 prostate158 test cases. "
                 "Δ = Full − variant (positive = Full is better). "
                 "95% CIs from 5000-sample percentile bootstrap. "
                 "*r* = matched-pairs rank-biserial correlation. "
                 "*d_z* = Cohen's *d* for paired samples. "
                 "Magnitude: |r| < 0.1 negligible, 0.1–0.3 small, "
                 "0.3–0.5 medium, ≥ 0.5 large.")
    lines.append("")

    # Table 1: Per-class DSC
    lines.append("## Table 1. Per-class DSC")
    lines.append("")
    lines.append("| Variant | Params (M) | PZ | CG | Tumor |")
    lines.append("|---|---|---|---|---|")
    for v in ABLATION_FLAGS:
        if v not in stats:
            continue
        cells = [short_name(v),
                 f"{params.get(v, 0) / 1e6:.2f}"]
        for c in CLASSES:
            s = stats[v][c]
            cells.append(f"{s['mean']:.4f} ± {s['std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table 2: effect sizes per class
    for c in CLASSES:
        lines.append(f"## Table 2{['a', 'b', 'c'][CLASSES.index(c)]}. "
                     f"Effect of removing each component on {c} DSC")
        lines.append("")
        lines.append("| Variant | Mean DSC | Δ vs Full [95% CI] | rank-biserial *r* | Cohen's *d_z* | Magnitude |")
        lines.append("|---|---|---|---|---|---|")
        # Full first
        sf = stats[full_key][c]
        lines.append(f"| **Full** | {sf['mean']:.4f} | — | — | — | (reference) |")
        for v in ABLATION_FLAGS:
            if v == full_key or v not in stats:
                continue
            s = stats[v][c]
            delta = s["delta"]
            d_ci = f"[{s['delta_ci_lo']:+.4f}, {s['delta_ci_hi']:+.4f}]"
            cells = [
                short_name(v),
                f"{s['mean']:.4f}",
                f"{delta:+.4f} {d_ci}",
                f"{s['rank_biserial']:+.3f}",
                f"{s['cohens_dz']:+.3f}",
                s["magnitude"],
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_latex(stats: Dict[str, Dict[str, dict]],
                 params: Dict[str, int],
                 full_key: str = "DSBANet (full)",
                 dim_label: str = "2D") -> str:
    L = []
    # Master table: all classes side-by-side with mean ± std and rank-biserial r
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(rf"\caption{{Ablacijska studija DSBANet ({dim_label}) na "
             r"Prostate158: per-klasna DSC (mean $\pm$ std preko 19 test "
             r"slučajeva) s razlikom $\Delta=$ Full $-$ varijanta i "
             r"matched-pairs rank-biserial efekt-veličinom $r$. "
             r"Pozitivni $\Delta$ znači da uklanjanje komponente smanjuje DSC. "
             r"95\% CI iz 5000 bootstrap uzoraka. "
             r"$|r|<0.1$ zanemariv; $0.1$--$0.3$ mali; $0.3$--$0.5$ srednji; "
             r"$\geq 0.5$ velik efekt.}")
    L.append(rf"\label{{tab:ablation-{dim_label.lower()}}}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{lr|cc|cc|cc}")
    L.append(r"\toprule")
    L.append(r"& & \multicolumn{2}{c|}{PZ} & "
             r"\multicolumn{2}{c|}{CG} & "
             r"\multicolumn{2}{c}{Tumor} \\")
    L.append(r"Variant & Params (M) & DSC & $r$ & DSC & $r$ & DSC & $r$ \\")
    L.append(r"\midrule")
    for v in ABLATION_FLAGS:
        if v not in stats:
            continue
        cells = [short_name(v).replace("w/o", r"w/o\,"),
                 f"{params.get(v, 0) / 1e6:.2f}"]
        for c in CLASSES:
            s = stats[v][c]
            cells.append(f"${s['mean']:.3f}\\!\\pm\\!{s['std']:.3f}$")
            if v == full_key:
                cells.append("---")
            else:
                cells.append(f"${s['rank_biserial']:+.2f}$")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render_figure(stats: Dict[str, Dict[str, dict]], save_path: str,
                  full_key: str = "DSBANet (full)"):
    """Per-class DSC bar chart s 95% bootstrap CI; engleski naslovi za paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = [v for v in ABLATION_FLAGS if v in stats]
    short = [short_name(v) for v in variants]
    x = np.arange(len(variants))
    width = 0.27
    offsets = {"PZ": -width, "CG": 0.0, "Tumor": width}
    colors = {"PZ": "tab:blue", "CG": "tab:orange", "Tumor": "tab:red"}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for c in CLASSES:
        means = np.array([stats[v][c]["mean"] for v in variants])
        lo = np.array([stats[v][c]["ci_lo"] for v in variants])
        hi = np.array([stats[v][c]["ci_hi"] for v in variants])
        err = np.vstack([means - lo, hi - means])
        ax.bar(x + offsets[c], means, width, yerr=err, capsize=4,
               label=c, color=colors[c], edgecolor="white")
    ax.set_ylabel("Dice Similarity Coefficient")
    ax.set_title("DSBANet 2D ablation on Prostate158 — "
                  "per-class DSC (mean and 95% bootstrap CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.legend(title="Class", loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(save_path.replace(".png", f".{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=["2d", "25d", "3d"], default="2d")
    parser.add_argument("--dataset", default="prostate158")
    parser.add_argument("--no_params", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    ablation_dir = os.path.join(project_root, "output", args.dataset,
                                 "ablation", args.dim)
    print(f"Učitavam: {ablation_dir}")

    per_case = load_per_case(ablation_dir)
    if "DSBANet (full)" not in per_case:
        sys.exit("Nema DSBANet (full) per_case_dsc.json — najprije završi "
                 "ablation Full varijantu.")

    stats = variant_summary(per_case)
    params = {} if args.no_params else count_params(args.dim)

    # Spremi numeričke effect-size statistike
    out_json = os.path.join(ablation_dir, "ablation_effect_sizes.json")
    serializable = {
        v: {c: {k: float(val) if isinstance(val, (int, float, np.floating))
                  else val for k, val in cls.items()}
            for c, cls in classes.items()}
        for v, classes in stats.items()
    }
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Effect sizes JSON: {out_json}")

    md = render_markdown(stats, params)
    md_path = os.path.join(ablation_dir, "ablation_table.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown:          {md_path}")

    dim_label = {"2d": "2D", "25d": "2.5D", "3d": "3D"}[args.dim]
    tex = render_latex(stats, params, dim_label=dim_label)
    tex_path = os.path.join(ablation_dir, "ablation_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX:             {tex_path}")

    fig_path = os.path.join(ablation_dir, "fig_ablation_perclass.png")
    render_figure(stats, fig_path)
    print(f"Figure (PNG+PDF):  {fig_path[:-4]}.{{png,pdf}}")
    print()
    print(md)


if __name__ == "__main__":
    main()
