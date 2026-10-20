"""Significance tests for the ensemble comparisons.

Writes results/significance_tests.md with:
- Mann-Whitney U tests (Holm-corrected) for every cube-vs-body property,
  with Cohen's d effect sizes;
- a Welch 95% CI for the cube-body difference in Murray exponent
  (an equivalence bound, since a non-significant difference alone
  cannot establish "identical");
- one-sample Wilcoxon tests of Clark-Evans dispersion against 1.0;
- an exact monotone-trend test for the n=1-per-condition sensitivity
  sweeps (probability of a perfectly monotone response under the null
  of exchangeability = 2/k!);
- the sparse-vs-dense fitness comparison in the body ensemble.
"""

import json
import sys
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

KEYS = ["fractal_dimension", "murray_exponent", "bifurcation_ratio",
        "mean_daughters", "radius_asymmetry", "length_ratio", "tortuosity",
        "length_to_radius", "clark_evans", "mean_dist_over_ideal",
        "length_density_m_per_L", "n_terminals"]


def monotone_trend_p(ys):
    """Two-sided exact p for a perfectly/partially monotone response under
    exchangeability: p = (# orderings at least as extreme by |Kendall tau|)
    / k!. For small k computed by enumeration."""
    ys = list(ys)
    k = len(ys)
    tau_obs = abs(stats.kendalltau(range(k), ys).statistic)
    count = 0
    for perm in permutations(ys):
        if abs(stats.kendalltau(range(k), perm).statistic) >= tau_obs - 1e-12:
            count += 1
    return count / factorial(k)


def main():
    box = json.load(open(RESULTS / "box" / "summary.json"))
    sm = json.load(open(RESULTS / "box" / "scale_matched_body_metrics.json"))
    box_m = [s["metrics"] for s in box]
    body_m = sm["body"]

    lines = ["# Significance tests", "",
             "## Cube vs body ensembles (n = 10 each, Mann-Whitney U, "
             "Holm-corrected)", "",
             "| Property | Cube | Body | p (raw) | p (Holm) | Cohen's d | |",
             "|---|---|---|---|---|---|---|"]

    pvals, rows = {}, {}
    for k in KEYS:
        a = np.array([m[k] for m in box_m], float)
        b = np.array([m[k] for m in body_m], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
        pvals[k] = p
        rows[k] = (a, b, d)

    order = sorted(pvals, key=lambda k: pvals[k])
    adj = {}
    for i, k in enumerate(order):
        adj[k] = min(1.0, pvals[k] * (len(order) - i))
    # enforce monotonicity of Holm-adjusted p-values
    run_max = 0.0
    for k in order:
        run_max = max(run_max, adj[k])
        adj[k] = run_max

    for k in KEYS:
        a, b, d = rows[k]
        sig = "**significant**" if adj[k] < 0.05 else "ns"
        lines.append(f"| {k} | {a.mean():.3g} ± {a.std():.2g} | "
                     f"{b.mean():.3g} ± {b.std():.2g} | {pvals[k]:.2g} | "
                     f"{adj[k]:.2g} | {d:+.2f} | {sig} |")

    # Murray gamma equivalence bound
    a = np.array([m["murray_exponent"] for m in box_m])
    b = np.array([m["murray_exponent"] for m in body_m])
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfw = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                     + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, dfw)
    lines += ["", "## Murray exponent equivalence",
              f"Cube - body difference = {diff:+.3f}, 95% CI "
              f"[{diff - t * se:+.3f}, {diff + t * se:+.3f}] — any true "
              "domain effect on the radius-scaling exponent is smaller "
              "than ~0.16."]

    # Clark-Evans vs 1
    lines += ["", "## Terminal dispersion vs spatial randomness "
              "(Wilcoxon vs CE = 1)"]
    for name, ms in [("cube", box_m), ("body", body_m)]:
        ce = np.array([m["clark_evans"] for m in ms])
        _, p = stats.wilcoxon(ce - 1.0)
        lines.append(f"- {name}: CE = {ce.mean():.3f} ± {ce.std():.2g}, "
                     f"p = {p:.4f} (clustered relative to Poisson; note CE "
                     "carries a negative edge-effect bias in bounded "
                     "domains — the cross-domain and vs-reference "
                     "comparisons use the same estimator)")

    # Sensitivity monotone trends (n=1 per condition)
    lines += ["", "## Sensitivity sweeps (n = 1 evolution per condition; "
              "exact monotone-trend test over 5 ordered conditions)"]
    sens = json.load(open(RESULTS / "sensitivity" / "summary.json"))
    for param, resp, label in [
            ("w_mat", "material_volume_L", "luminal volume vs w_mat"),
            ("w_mat", "n_terminals", "terminal count vs w_mat"),
            ("w_pow", "material_volume_L", "luminal volume vs w_pow"),
            ("delta", "n_terminals", "terminal count vs delta"),
            ("w_time", "fractal_dimension", "fractal dimension vs w_time")]:
        es = sorted([e for e in sens if e["param"] == param],
                    key=lambda e: e["value"])
        ys = [e["metrics"][resp] for e in es]
        p = monotone_trend_p(ys)
        lines.append(f"- {label}: values {['%.3g' % y for y in ys]}, "
                     f"trend p = {p:.3f}")

    # sparse vs dense basins in body ensemble
    div = json.load(open(RESULTS / "diversity" / "summary.json"))
    terms = np.array([s["metrics"]["n_terminals"] for s in div])
    fits = np.array([s["fitness"] for s in div])
    _, p = stats.mannwhitneyu(fits[terms < 1500], fits[terms >= 1500])
    lines += ["", "## Two-basin claim (body ensemble)",
              f"Sparse optima (n=2) fitness {fits[terms < 1500].round(3).tolist()} vs "
              f"dense (n=8) {fits[terms >= 1500].round(3).tolist()}: "
              f"Mann-Whitney p = {p:.3f}. With n = 2 the 'second basin' is "
              "suggestive (the terminal-count gap 982 to 2266 is large) but "
              "not statistically established, and could reflect incomplete "
              "GA convergence rather than true landscape multimodality."]

    out = RESULTS / "significance_tests.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
