"""Figures for the box control experiment: do the fractal, topological
and geometric properties of optimal networks coincide across domains?

Space-filling and fractal metrics for the body-domain champions and the
anatomical reference are recomputed here over the SAME box-counting
scale range used for the cube (1-12 cm), so the fractal-dimension
comparison is scale-matched.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vascular_evolution.body import build_body, build_box
from vascular_evolution.constants import LITERATURE
from vascular_evolution.io import load_json, load_tree, save_json
from vascular_evolution.metrics import (box_counting_dimension,
                                        compute_all_metrics,
                                        space_filling_metrics)
from vascular_evolution.viz import draw_tree_3d

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BOX_C = "#27ae60"
BODY_C = "#c0392b"
REF_C = "#2471a3"
LIT_C = "#7f8c8d"

FD_SIZES = np.geomspace(0.01, 0.12, 8)
BOX_LIMS = ((-0.21, 0.21), (-0.21, 0.21), (-0.01, 0.41))


def recompute(tree, domain):
    m = compute_all_metrics(tree, box_sizes=FD_SIZES)
    m.update(space_filling_metrics(tree, domain))
    return m


def fig9_box_gallery(box, box_summary):
    fig = plt.figure(figsize=(11, 5.4))
    for i, s in enumerate(box_summary):
        tree = load_tree(RESULTS / "box" / f"tree_seed{s['seed']}.npz")
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        draw_tree_3d(ax, tree, box, elev=14, azim=-60, body_alpha=0.012,
                     zoom=1.15, lims=BOX_LIMS, lw_scale=700)
        ax.set_title(f"seed {s['seed']}  F={s['fitness']:.2f}\n"
                     f"terms={s['metrics']['n_terminals']} "
                     f"D={s['metrics']['fractal_dimension']:.2f}",
                     fontsize=7.5, pad=0)
    fig.suptitle("Evolved optima in the uniform cube (64 L, same physics "
                 "and budget as the body runs)", fontsize=12)
    fig.subplots_adjust(wspace=0, hspace=0.16, top=0.88, bottom=0.02)
    fig.savefig(FIGS / "fig9_box_gallery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


BOX_PROPS = [
    ("fractal_dimension", "Fractal dimension\n(1-12 cm scale-matched)"),
    ("murray_exponent", "Murray exponent γ"),
    ("bifurcation_ratio", "Strahler bifurcation ratio"),
    ("mean_daughters", "Mean daughters/junction"),
    ("radius_asymmetry", "Radius asymmetry r₂/r₁"),
    ("length_ratio", "Strahler length ratio"),
    ("tortuosity", "Tortuosity"),
    ("length_to_radius", "Segment length/radius"),
    ("clark_evans", "Clark-Evans dispersion\n(terminals; 1 = random)"),
    ("mean_dist_over_ideal", "Tissue-vessel distance /\nideal territory radius"),
    ("length_density_m_per_L", "Length density (m/L)"),
    ("n_terminals", "N terminals"),
]


def fig10_box_vs_body(box_metrics, body_metrics, ref_m):
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8))
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes.ravel(), BOX_PROPS):
        if key in LITERATURE:
            mid, (lo, hi) = LITERATURE[key]
            ax.axhspan(lo, hi, color=LIT_C, alpha=0.15, lw=0)
            ax.axhline(mid, color=LIT_C, lw=1, ls=":")
        for x0, ms, c in [(0, box_metrics, BOX_C), (1, body_metrics, BODY_C)]:
            vals = [m[key] for m in ms if np.isfinite(m[key])]
            x = x0 + rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(x, vals, s=18, color=c, alpha=0.6, zorder=3)
            ax.hlines(np.mean(vals), x0 - 0.2, x0 + 0.2, color=c, lw=2)
        if key in ref_m and np.isfinite(ref_m[key]):
            ax.plot([2], [ref_m[key]], "D", ms=7, color=REF_C, zorder=4)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["cube", "body", "human\nref"], fontsize=7.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(label, fontsize=8.5)
    handles = [Line2D([], [], marker="o", ls="", color=BOX_C, label="cube optima (10 seeds)"),
               Line2D([], [], marker="o", ls="", color=BODY_C, label="body optima (10 seeds)"),
               Line2D([], [], marker="D", ls="", color=REF_C, label="anatomical reference"),
               plt.Rectangle((0, 0), 1, 1, color=LIT_C, alpha=0.3,
                             label="literature range")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Domain independence: network properties of optima evolved "
                 "in a cube vs. the human body", fontsize=12)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    fig.savefig(FIGS / "fig10_box_vs_body.png", dpi=170)
    plt.close(fig)


def fig11_scaling_overlay(box_trees, body_trees, ref):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    for trees, c, lab in [(box_trees, BOX_C, "cube optima"),
                          (body_trees, BODY_C, "body optima")]:
        for i, t in enumerate(trees):
            pts = t.polyline_points(spacing=0.004)
            _, s, n = box_counting_dimension(pts, sizes=FD_SIZES)
            ax.loglog(1 / s, n, "-", color=c, alpha=0.4, lw=1,
                      label=lab if i == 0 else None)
    pts = ref.polyline_points(spacing=0.004)
    _, s, n = box_counting_dimension(pts, sizes=FD_SIZES)
    ax.loglog(1 / s, n, "o--", color=REF_C, lw=1.5, ms=4,
              label="anatomical reference")
    # guide slopes anchored at the large-box end of the reference curve
    x = np.array([1 / 0.12, 1 / 0.01])
    for d, ls in [(2.0, ":"), (3.0, "--")]:
        ax.loglog(x, n[-1] * 0.7 * (x / x[0]) ** d, ls, color="k", lw=0.7,
                  label=f"slope {d:g}")
    ax.set_xlabel("1 / box size (1/m)")
    ax.set_ylabel("occupied boxes")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Box-counting curves (scale-matched)")

    ax = axes[1]
    for trees, c, lab in [(box_trees, BOX_C, "cube optima"),
                          (body_trees, BODY_C, "body optima")]:
        for i, t in enumerate(trees):
            m = compute_all_metrics(t)
            ax.semilogy(m["orders"], m["branches_per_order"], "-", color=c,
                        alpha=0.4, lw=1, label=lab if i == 0 else None)
    m = compute_all_metrics(ref)
    ax.semilogy(m["orders"], m["branches_per_order"], "o--", color=REF_C,
                lw=1.5, ms=4, label="anatomical reference")
    ax.set_xlabel("Strahler order")
    ax.set_ylabel("number of branches")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Horton-Strahler scaling")
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_scaling_overlay.png", dpi=170)
    plt.close(fig)


def write_box_table(box_metrics, body_metrics, ref_m):
    lines = ["| Property | Cube optima (10) | Body optima (10) | "
             "Human reference | Literature |",
             "|---|---|---|---|---|"]
    for key, label in BOX_PROPS:
        bx = np.array([m[key] for m in box_metrics], dtype=float)
        bd = np.array([m[key] for m in body_metrics], dtype=float)
        lit = ""
        if key in LITERATURE:
            mid, (lo, hi) = LITERATURE[key]
            lit = f"{mid} ({lo}–{hi})"
        rv = f"{ref_m[key]:.3g}" if key in ref_m and np.isfinite(ref_m[key]) else "—"
        lines.append(
            f"| {label.replace(chr(10), ' ')} | "
            f"{np.nanmean(bx):.3g} ± {np.nanstd(bx):.2g} | "
            f"{np.nanmean(bd):.3g} ± {np.nanstd(bd):.2g} | {rv} | {lit} |")
    (RESULTS / "box_comparison_table.md").write_text("\n".join(lines) + "\n")


def main():
    box = build_box()
    body = build_body()

    box_summary = load_json(RESULTS / "box" / "summary.json")
    box_trees = [load_tree(RESULTS / "box" / f"tree_seed{s['seed']}.npz")
                 for s in box_summary]
    box_metrics = [s["metrics"] for s in box_summary]

    div_summary = load_json(RESULTS / "diversity" / "summary.json")
    body_trees = [load_tree(RESULTS / "diversity" / f"tree_seed{s['seed']}.npz")
                  for s in div_summary]
    print("recomputing scale-matched metrics for body champions ...")
    body_metrics = [recompute(t, body) for t in body_trees]

    ref = load_tree(RESULTS / "reference_tree.npz")
    ref_m = recompute(ref, body)
    save_json(RESULTS / "box" / "scale_matched_body_metrics.json",
              {"body": body_metrics, "reference": ref_m})

    print("fig9 ..."); fig9_box_gallery(box, box_summary)
    print("fig10 ..."); fig10_box_vs_body(box_metrics, body_metrics, ref_m)
    print("fig11 ..."); fig11_scaling_overlay(box_trees, body_trees, ref)
    write_box_table(box_metrics, body_metrics, ref_m)
    print("done")


if __name__ == "__main__":
    main()
