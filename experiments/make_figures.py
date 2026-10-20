"""Generate all figures from saved experiment results."""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vascular_evolution.body import build_body
from vascular_evolution.constants import LITERATURE
from vascular_evolution.growth import GENE_NAMES
from vascular_evolution.io import load_json, load_tree
from vascular_evolution.metrics import strahler_orders
from vascular_evolution.physics import DiffusionModel
from vascular_evolution.viz import draw_tree_3d, draw_tree_projection

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

EVOLVED_C = "#c0392b"
REF_C = "#2471a3"
LIT_C = "#7f8c8d"


def fig1_side_by_side(body, ref, best):
    fig = plt.figure(figsize=(12, 8.5))
    specs = [
        (ref, "Anatomical human arterial tree\n(majors from atlas + canonical infill)"),
        (best, "Evolved optimal tree\n(baseline fitness weights)"),
    ]
    views = [(8, -88, "front"), (8, -5, "side")]
    for col, (tree, title) in enumerate(specs):
        for row, (elev, azim, label) in enumerate(views):
            ax = fig.add_axes([0.005 + (col * 2 + row) * 0.235, -0.14,
                               0.235, 1.20], projection="3d")
            lc = draw_tree_3d(ax, tree, body, elev=elev, azim=azim,
                              body_alpha=0.03, zoom=1.45)
            fig.text(0.005 + (col * 2 + row) * 0.235 + 0.118, 0.10, label,
                     ha="center", color="#555555", fontsize=10)
        fig.text(0.005 + col * 0.47 + 0.235, 0.86, title,
                 ha="center", fontsize=11.5)
    cax = fig.add_axes([0.955, 0.32, 0.013, 0.40])
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label("log10 radius (mm)")
    fig.suptitle("Human arterial tree vs. evolved optimum", y=0.93,
                 fontsize=14)
    fig.savefig(FIGS / "fig1_side_by_side_3d.png", dpi=170)
    plt.close(fig)


def fig2_concentration(body, model, ref, best):
    def conc_grid(tree):
        C = model.solve(tree)["concentration"]
        g = np.full(body.shape, np.nan)
        g[body.mask] = C
        return g

    gr, gb = conc_grid(ref), conc_grid(best)
    yslice = body.shape[1] // 2
    fig, axes = plt.subplots(1, 4, figsize=(11, 7),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})
    for ax, g, tree, title in [
            (axes[0], gr, ref, "Anatomical reference"),
            (axes[1], gb, best, "Evolved optimum")]:
        im = ax.imshow(g[:, yslice, :].T, origin="lower", cmap="magma",
                       vmin=0, vmax=1, aspect="equal",
                       extent=[-0.36, 0.36, 0, 1.8])
        sel = np.abs(tree.pos[1:, 1]) < 0.06
        ax.plot(tree.pos[1:, 0][sel], tree.pos[1:, 2][sel], ".",
                ms=0.8, color="#00e5ff", alpha=0.5)
        ax.set_title(title)
        ax.set_axis_off()
    # hypoxic fraction comparison
    ax = axes[2]
    for g, c, lab in [(gr, REF_C, "reference"), (gb, EVOLVED_C, "evolved")]:
        vals = g[body.mask]
        ax.hist(vals, bins=40, histtype="step", color=c, label=lab,
                density=True, lw=1.5)
    ax.axvline(0.1, color="k", ls="--", lw=0.8)
    ax.text(0.11, 0.95, "hypoxia\nthreshold", transform=ax.transAxes,
            fontsize=7, va="top")
    ax.set_xlabel("normalized concentration C")
    ax.set_ylabel("tissue density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Tissue concentration distribution")
    fig.colorbar(im, cax=axes[3], label="C / C0")
    fig.suptitle("Steady-state nutrient concentration (coronal slice)",
                 y=0.98, fontsize=12)
    fig.savefig(FIGS / "fig2_concentration_fields.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


COMPARE_PROPS = [
    ("fractal_dimension", "Fractal dimension\n(box counting)"),
    ("murray_exponent", "Murray exponent γ"),
    ("bifurcation_ratio", "Strahler bifurcation\nratio R_b"),
    ("mean_daughters", "Mean daughters\nper junction"),
    ("radius_asymmetry", "Radius asymmetry\nr₂/r₁ at junctions"),
    ("length_to_radius", "Segment length /\nradius"),
    ("tortuosity", "Tortuosity\n(path/Euclid)"),
    ("total_length_m", "Total length (m)"),
    ("material_volume_L", "Luminal volume (L)"),
    ("n_terminals", "N terminals"),
    ("root_radius_mm", "Root radius (mm)"),
    ("max_strahler_order", "Max Strahler order"),
]


def fig3_properties(ref_m, best_m, div_metrics):
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.5))
    for ax, (key, label) in zip(axes.ravel(), COMPARE_PROPS):
        if key in LITERATURE:
            mid, (lo, hi) = LITERATURE[key]
            ax.axhspan(lo, hi, color=LIT_C, alpha=0.15, lw=0)
            ax.axhline(mid, color=LIT_C, lw=1, ls=":")
        vals = [m[key] for m in div_metrics if np.isfinite(m[key])]
        if vals:
            ax.scatter(np.full(len(vals), 2) +
                       np.linspace(-0.12, 0.12, len(vals)),
                       vals, s=14, color=EVOLVED_C, alpha=0.45, zorder=3,
                       label="diversity seeds")
        ax.bar([0], [ref_m[key]], width=0.55, color=REF_C, alpha=0.85)
        ax.bar([1], [best_m[key]], width=0.55, color=EVOLVED_C, alpha=0.85)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["human", "evolved", "seeds"], fontsize=7)
        ax.set_title(label, fontsize=8.5)
    handles = [Patch(color=REF_C, label="anatomical reference"),
               Patch(color=EVOLVED_C, label="evolved optimum"),
               Line2D([], [], marker="o", ls="", color=EVOLVED_C, alpha=0.45,
                      label="evolved (10 seeds)"),
               Patch(color=LIT_C, alpha=0.3, label="literature range")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Morphometric properties: human arterial tree vs. evolved optima",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(FIGS / "fig3_property_comparison.png", dpi=170)
    plt.close(fig)


def fig4_scaling(ref_m, best_m, ref, best):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    # box counting
    ax = axes[0]
    for m, c, lab in [(ref_m, REF_C, "human reference"),
                      (best_m, EVOLVED_C, "evolved")]:
        s = np.array(m["box_sizes"]); n = np.array(m["box_counts"])
        ax.loglog(1 / s, n, "o-", ms=3, color=c,
                  label=f"{lab} (D={m['fractal_dimension']:.2f})")
    ax.set_xlabel("1 / box size (1/m)")
    ax.set_ylabel("occupied boxes")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Box-counting scaling")
    # Strahler branch counts
    ax = axes[1]
    for m, c, lab in [(ref_m, REF_C, "human reference"),
                      (best_m, EVOLVED_C, "evolved")]:
        ax.semilogy(m["orders"], m["branches_per_order"], "o-", ms=3,
                    color=c, label=f"{lab} (R_b={m['bifurcation_ratio']:.2f})")
    ax.set_xlabel("Strahler order")
    ax.set_ylabel("number of branches")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Horton-Strahler branch counts")
    # radius vs order
    ax = axes[2]
    for tree, c, lab in [(ref, REF_C, "human reference"),
                         (best, EVOLVED_C, "evolved")]:
        order = strahler_orders(tree)
        omax = order.max()
        os_, rs = [], []
        for o in range(1, omax + 1):
            sel = (order == o)
            sel[0] = False
            if sel.any():
                os_.append(o)
                rs.append(np.mean(tree.radius[sel]) * 1e3)
        ax.semilogy(os_, rs, "o-", ms=3, color=c, label=lab)
    ax.set_xlabel("Strahler order")
    ax.set_ylabel("mean radius (mm)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Radius scaling across orders")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_scaling_laws.png", dpi=170)
    plt.close(fig)


def fig5_gallery(body, div_trees, div_summary):
    n = len(div_trees)
    fig = plt.figure(figsize=(11, 6.5))
    for i, tree in enumerate(div_trees):
        ax = fig.add_subplot(2, (n + 1) // 2, i + 1, projection="3d")
        draw_tree_3d(ax, tree, body, elev=8, azim=-88, body_alpha=0.015)
        s = div_summary[i]
        ax.set_title(f"seed {s['seed']}  F={s['fitness']:.2f}\n"
                     f"terms={s['metrics']['n_terminals']} "
                     f"D={s['metrics']['fractal_dimension']:.2f}",
                     fontsize=7.5, pad=0)
    fig.suptitle("Diversity of evolved optima (independent evolutionary runs)",
                 fontsize=12)
    fig.subplots_adjust(wspace=0, hspace=0.12, top=0.9, bottom=0.02)
    fig.savefig(FIGS / "fig5_diversity_gallery.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


DIVERSITY_PROPS = ["fractal_dimension", "murray_exponent",
                   "bifurcation_ratio", "mean_daughters",
                   "radius_asymmetry", "tortuosity", "total_length_m",
                   "material_volume_L", "n_terminals", "root_radius_mm",
                   "length_to_radius"]


def fig6_diversity_metrics(div_summary, ref_m):
    M = np.array([[s["metrics"][k] for k in DIVERSITY_PROPS]
                  for s in div_summary], dtype=float)
    labels = [k.replace("_", " ") for k in DIVERSITY_PROPS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [1.4, 1, 1]})
    # normalized spread (relative to ensemble mean)
    ax = axes[0]
    mean = np.nanmean(M, axis=0)
    for j in range(M.shape[1]):
        ax.scatter(M[:, j] / mean[j], np.full(M.shape[0], j), s=14,
                   color=EVOLVED_C, alpha=0.5)
        ax.plot(ref_m[DIVERSITY_PROPS[j]] / mean[j], j, "D", ms=5,
                color=REF_C, zorder=4)
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("value / ensemble mean")
    ax.set_title("Spread across 10 optima (red) vs human (blue)")
    ax.set_xscale("log")
    # coefficient of variation
    ax = axes[1]
    cv = np.nanstd(M, axis=0) / np.abs(np.nanmean(M, axis=0))
    ax.barh(range(len(labels)), cv, color=EVOLVED_C, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([])
    ax.set_xlabel("coefficient of variation")
    ax.set_title("How variable is each property?")
    # pairwise similarity of z-scored metric vectors
    ax = axes[2]
    Z = (M - np.nanmean(M, axis=0)) / (np.nanstd(M, axis=0) + 1e-12)
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    im = ax.imshow(D, cmap="viridis")
    ax.set_xlabel("seed")
    ax.set_ylabel("seed")
    ax.set_title("Pairwise distance between optima\n(z-scored property space)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_diversity_metrics.png", dpi=170)
    plt.close(fig)


SENS_RESPONSES = [
    ("eval", "coverage", "coverage"),
    ("metrics", "n_terminals", "N terminals"),
    ("metrics", "total_length_m", "total length (m)"),
    ("metrics", "material_volume_L", "luminal volume (L)"),
    ("metrics", "fractal_dimension", "fractal dimension"),
    ("genome", "murray_gamma", "evolved Murray γ"),
]

PARAM_LABELS = {
    "w_time": "delivery-speed weight w_time",
    "w_mat": "material-cost weight w_mat",
    "w_pow": "pumping-cost weight w_pow",
    "delta": "perfusion depth δ (m)",
}

BASELINES = {"w_time": 0.25, "w_mat": 0.1, "w_pow": 0.1, "delta": 0.035}


def fig7_sensitivity(sens_summary, baseline_entry, ref_m):
    params = ["w_time", "w_mat", "w_pow", "delta"]
    fig, axes = plt.subplots(len(params), len(SENS_RESPONSES),
                             figsize=(13, 9), sharex="row")
    gidx = {g: i for i, g in enumerate(GENE_NAMES)}
    for i, p in enumerate(params):
        entries = [e for e in sens_summary if e["param"] == p]
        entries += [dict(baseline_entry, param=p, value=BASELINES[p])]
        entries.sort(key=lambda e: e["value"])
        xs = [e["value"] for e in entries]
        for j, (group, key, label) in enumerate(SENS_RESPONSES):
            ax = axes[i, j]
            if group == "genome":
                ys = [e["genome"][gidx[key]] for e in entries]
            else:
                ys = [e[group][key] for e in entries]
            ax.plot(xs, ys, "o-", ms=4, color=EVOLVED_C)
            ax.axvline(BASELINES[p], color="k", lw=0.6, ls=":")
            if key in ref_m and group == "metrics":
                ax.axhline(ref_m[key], color=REF_C, lw=1, ls="--", alpha=0.7)
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=6.5)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(axis="x", which="minor", bottom=False)
            if i == 0:
                ax.set_title(label, fontsize=9)
            if j == 0:
                ax.set_ylabel(PARAM_LABELS[p], fontsize=8)
    fig.suptitle("Sensitivity of the evolved optimum to fitness weights and "
                 "perfusion physics (blue dashed = human reference)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "fig7_sensitivity.png", dpi=170)
    plt.close(fig)


def fig8_convergence(main_run, div_runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    h = main_run["result"]["history"]
    ax.plot([e["gen"] for e in h], [e["best_fitness"] for e in h],
            color=EVOLVED_C, label="best")
    ax.plot([e["gen"] for e in h], [e["mean_fitness"] for e in h],
            color=EVOLVED_C, alpha=0.4, label="population mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Main run convergence")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for r in div_runs:
        h = r["result"]["history"]
        ax.plot([e["gen"] for e in h], [e["best_fitness"] for e in h],
                alpha=0.6, lw=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title("Diversity ensemble (10 seeds)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_convergence.png", dpi=170)
    plt.close(fig)


def write_comparison_table(ref_m, best_m, div_metrics):
    lines = ["| Property | Human reference | Evolved optimum | "
             "Evolved 10-seed mean ± sd | Literature |",
             "|---|---|---|---|---|"]
    for key, label in COMPARE_PROPS:
        vals = np.array([m[key] for m in div_metrics], dtype=float)
        lit = ""
        if key in LITERATURE:
            mid, (lo, hi) = LITERATURE[key]
            lit = f"{mid} ({lo}–{hi})"
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {ref_m[key]:.3g} | "
            f"{best_m[key]:.3g} | {np.nanmean(vals):.3g} ± "
            f"{np.nanstd(vals):.2g} | {lit} |")
    (RESULTS / "comparison_table.md").write_text("\n".join(lines) + "\n")


def main():
    body = build_body()
    model = DiffusionModel(body)

    ref = load_tree(RESULTS / "reference_tree.npz")
    best = load_tree(RESULTS / "best_tree.npz")
    ref_data = load_json(RESULTS / "reference_metrics.json")
    main_run = load_json(RESULTS / "main_run.json")
    ref_m, best_m = ref_data["metrics"], main_run["metrics"]

    div_summary = load_json(RESULTS / "diversity" / "summary.json")
    div_trees = [load_tree(RESULTS / "diversity" / f"tree_seed{s['seed']}.npz")
                 for s in div_summary]
    div_runs = [load_json(RESULTS / "diversity" / f"run_seed{s['seed']}.json")
                for s in div_summary]
    div_metrics = [s["metrics"] for s in div_summary]

    sens_summary = load_json(RESULTS / "sensitivity" / "summary.json")
    baseline_entry = {"genome": main_run["result"]["best_genome"],
                      "metrics": best_m, "eval": main_run["eval"]}

    print("fig1 ..."); fig1_side_by_side(body, ref, best)
    print("fig2 ..."); fig2_concentration(body, model, ref, best)
    print("fig3 ..."); fig3_properties(ref_m, best_m, div_metrics)
    print("fig4 ..."); fig4_scaling(ref_m, best_m, ref, best)
    print("fig5 ..."); fig5_gallery(body, div_trees, div_summary)
    print("fig6 ..."); fig6_diversity_metrics(div_summary, ref_m)
    print("fig7 ..."); fig7_sensitivity(sens_summary, baseline_entry, ref_m)
    print("fig8 ..."); fig8_convergence(main_run, div_runs)
    write_comparison_table(ref_m, best_m, div_metrics)
    print("done; figures in", FIGS)


if __name__ == "__main__":
    main()
