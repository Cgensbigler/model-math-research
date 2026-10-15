# How Efficient Is the Human Vascular System?

A physics-based study: we evolve vascular networks inside a 3D model of the
human body to optimize a trade-off between **nutrient delivery by diffusion**
(coverage + speed) and **cost** (vessel material + pumping energy), and then
compare the evolved optima against the structure of the real human arterial
tree — fractal dimension, scaling-law exponents, branching statistics,
structural motifs, size, total length, and more.

![Side by side](figures/fig1_side_by_side_3d.png)

## The model

| Component | Implementation |
|---|---|
| **Body** | Voxelized 3D human (1.75 m, 63 L of tissue, 2 cm voxels) assembled from capsules/ellipsoids: head, neck, torso, arms, legs (`vascular_evolution/body.py`). A metabolic demand field places measured organ shares of resting VO₂ (brain 20%, heart 10%, liver+gut 27%, kidneys 7%, remainder uniform). |
| **Vascular trees** | Flat-array branching trees rooted at the heart. Radii follow a Murray-type rule r_p^γ = Σ r_c^γ applied bottom-up from a terminal radius; blood flows are assigned by demand-weighted Voronoi territories of the terminals and accumulated to the root (`tree.py`). |
| **Growth (genotype → phenotype)** | Trees are grown by a parameterized *space-colonization* program — the analogue of VEGF-guided angiogenesis. The 9-gene genome controls attractor density, attraction/kill radii, step size, directional momentum, noise, vertical tropism, the Murray exponent γ, and the terminal radius (`growth.py`). |
| **Diffusion physics** | Steady-state reaction–diffusion `D∇²C − M = 0` solved as a sparse linear system on ~7,900 tissue voxels, with C = 1 at terminal-tip voxels (perfusion sources) and zero-flux body boundary; concentrations are clamped at 0 where consumption extinguishes them (`physics.py`). The consumption/diffusivity ratio is calibrated analytically so a single source penetrates to the hypoxia threshold at exactly the perfusion depth δ = 3.5 cm in spherical geometry — the mesoscale reach of the microcirculatory bed each terminal artery feeds. |
| **Evolutionary algorithm** | Real-coded GA (tournament selection, BLX crossover, Gaussian mutation, elitism) with parallel fitness evaluation; ~2,000 diffusion solves per run (`evolve.py`). |

### Fitness function

```
F = coverage − 0.25·T_norm − 0.10·(V/V_ref) − 0.10·(P/P_ref)
```

* **coverage** — fraction of tissue above the hypoxia threshold (C ≥ 0.1·C₀,
  the Krogh venous-end criterion) in the solved diffusion field;
* **T_norm** — delivery speed: mean diffusion time to the nearest source,
  d²/2D, normalized by the time to cross one perfusion depth;
* **V/V_ref** — structural cost: total luminal volume over the blood volume
  held in real human arteries (V_ref = 0.5 L);
* **P/P_ref** — energetic cost: Poiseuille dissipation Σ 8μLQ²/πr⁴ at
  cardiac-output flows (5 L/min, blood viscosity 3.5 mPa·s), normalized by
  the hydraulic power the heart spends in the artery-level tree
  (P_ref ≈ 0.26 W, i.e. ~25% of total cardiac power).

The weights are physiologically anchored: a design with human-scale costs
pays ~0.2 of fitness for them, so costs matter but cannot dominate
delivery — mirroring a circulation that consumes only a few percent of basal
metabolism. All constants with sources are in `constants.py`, and the
**sensitivity analysis** below perturbs every weight 4–16× around baseline.

### The anatomical reference

The comparison target (`reference.py`) is built in two layers: 38 named
major arteries (aorta → carotids/subclavians/brachials/radials, celiac,
mesenteric, renals, iliacs → femorals → tibials) hand-placed from atlas
geometry with published radii, then extended distally to ~0.8 mm small
arteries with the same growth engine run at *literature-canonical*
parameters (γ = 2.7). Metric plots additionally show published literature
ranges as shaded bands.

## Results

### The evolved optimum vs. the human arterial tree

![Concentration fields](figures/fig2_concentration_fields.png)

The evolved champion oxygenates 99.8% of tissue (vs 92.5% for the anatomical
reference at the same physics) while spending **exactly the same vessel
volume (0.36 L)** — the evolved network is genuinely more efficient at this
mesoscale delivery task, which it achieves by spending its radius budget on
many fine terminals instead of thick trunks.

![Properties](figures/fig3_property_comparison.png)
![Scaling laws](figures/fig4_scaling_laws.png)

| Property | Human reference | Evolved | Literature |
|---|---|---|---|
| Fractal dimension (box counting) | 1.65 | 2.07 ± 0.10 | 2.2–2.9 (whole network) |
| Murray exponent γ | 2.7 | 3.2 ± 0.2 | 2.3–3.0 |
| Strahler bifurcation ratio | 4.1 | 3.9 ± 0.3 | 2.5–4.0 |
| Radius asymmetry at junctions | 0.84 | 0.81 ± 0.02 | 0.5–0.9 |
| Mean daughters per junction | 2.4 | 2.8 ± 0.1 | 2.0–2.3 |
| Luminal volume | 0.36 L | 0.36 ± 0.07 L | ~0.5 L (arteries) |
| N terminals | 364 | ~2,000 | — |
| Total length | 29 m | 160 ± 50 m | — |
| Root radius | 12.5 mm | 4–5 mm | 12.5 mm (aorta) |

(Full table: `results/comparison_table.md`.)

**What converges.** The *scaling* properties — the ones physics constrains —
match well: bifurcation ratio inside the measured human range, junction
radius asymmetry nearly identical (0.81 vs 0.84), Murray-type radius scaling
with an exponent in the physiological neighborhood, exponential
Horton–Strahler branch-count scaling (straight lines in fig. 4), and total
luminal volume indistinguishable from the reference. Evolution rediscovers
a hierarchical, asymmetric, space-filling bifurcating tree — nobody told the
genome to build one; a brush, a comb, or a fractal pom-pom were all
reachable phenotypes.

**What diverges.** The evolved optimum prefers ~2,000 half-millimetre
terminals over the anatomy's ~360 at this scale: a thinner root (4–5 mm vs
12.5 mm), 5× the total length, and a higher γ (≈3.2–3.5 vs 2.7–3.0,
i.e. thinner trunks relative to daughters). This is the fingerprint of what
the model omits: at resting flows Poiseuille dissipation in >0.5 mm vessels
is cheap, so there is little pressure to maintain thick conduits. Real
aortas are sized for exercise (×5 cardiac output → ×25 dissipation),
pulsatile flow, and turbulence avoidance. Consistently, raising the pumping
weight drags γ back toward Murray's 3 and thickens the tree (fig. 7), and
the human-like sparse architecture becomes optimal exactly when the
perfusion depth δ grows (see sensitivity).

### Diversity of the optima

![Gallery](figures/fig5_diversity_gallery.png)
![Diversity metrics](figures/fig6_diversity_metrics.png)

Ten independent evolutionary runs (different GA randomness *and* different
developmental noise) reveal a fitness landscape with **two basins**:

* 8/10 seeds converge on the *dense-fine* strategy — 2,270–2,390 terminals,
  fractal dimension 2.03–2.07, fitness 0.79–0.81;
* 2/10 seeds land in a *sparse* local optimum — 600–980 terminals, fractal
  dimension 1.79–1.83, fitness 0.75–0.78 (close, but measurably worse).

Within the dominant basin the optima are strikingly similar where it
matters: coefficients of variation are ~2% for junction asymmetry, ~4–5%
for fractal dimension, Murray exponent and mean daughters — while *size*
properties (terminal count, total length) vary by 30%+. **Scaling laws are
universal across solutions; size is contingent.** The pairwise-distance
heatmap (fig. 6, right) shows seeds 2 and 4 as the outlier pair. This is a
pleasing echo of real vasculature, where scaling exponents are conserved
across individuals and species while absolute network size varies.

### Sensitivity analysis

![Sensitivity](figures/fig7_sensitivity.png)

One-at-a-time sweeps (full evolution per condition, 18 conditions) around
the baseline:

* **Coverage is robust** (0.97–1.00 everywhere): delivery dominates the
  optimum regardless of weights — as in biology, the circulation exists to
  perfuse, and costs only sculpt *how*.
* **Material cost w_mat is the main structural lever.** ×16 increase drives
  luminal volume 0.70 → 0.11 L, halves the terminal count, and lowers the
  fractal dimension 2.09 → 1.89: expensive tissue buys a sparser, more
  anatomical tree.
* **Pumping cost w_pow mainly thickens vessels** (volume 0.28 → 0.49 L
  across the sweep) and pulls the radius exponent toward Murray's
  prediction; terminal count barely moves.
* **Perfusion depth δ is the strongest knob of all.** At δ = 2.5 cm full
  coverage becomes physically unreachable (0.69 at best); at δ = 6 cm the
  optimum collapses to 658 terminals, 0.14 L and FD 1.74 — closely
  resembling the sparse anatomical architecture. The structure of the
  optimal tree is set less by the economics than by **how far diffusion can
  reach**, which is why real bodies, whose true Krogh distance is ~100 µm,
  must continue the hierarchy far below this model's floor into billions of
  capillaries.

![Convergence](figures/fig8_convergence.png)

## Interpretation

Within this model's horizon the human-like architecture is *not* uniquely
optimal — a denser, finer, thinner-trunked network delivers the same
coverage for the same material. The constraints that make real anatomy
optimal are precisely the ones outside a resting-state, artery-level
diffusion model: exercise-range flows, pulsatility, and the four
orders of magnitude of scale separating the aorta from the capillaries.
Where the model and reality share constraints — space-filling delivery
against material/energy budgets — evolution in silico reproduces the same
structural grammar: hierarchical asymmetric bifurcations, exponential
Strahler scaling, Murray-type radius conservation, and a conserved set of
scaling exponents across independent evolutionary histories.

## Reproducing

```bash
pip install -r requirements.txt
python3 experiments/run_main.py          # ~3 min on 4 cores
python3 experiments/run_diversity.py     # ~15 min, 10 seeds
python3 experiments/run_sensitivity.py   # ~15 min, 18 conditions
python3 experiments/make_figures.py      # all figures + comparison table
```

All experiment outputs (champion trees, metrics, histories) are committed
under `results/`, figures under `figures/`.

## Key constants and sources

* O₂ diffusivity in tissue 2×10⁻⁹ m²/s; Krogh (1919) tissue-cylinder model.
* Whole-body VO₂ 250 mL/min; blood O₂ content 20 mL/dL; cardiac output
  5 L/min; mean arterial pressure 93 mmHg; blood viscosity 3.5 mPa·s.
* Arterial blood volume ~0.5 L (10–13% of 5 L); artery-level pressure drop
  ~25% of total.
* Murray (1926) — the cost-of-blood principle predicting γ = 3; measured
  γ ≈ 2.55–3.0 (Zamir; Sherman 1981).
* Strahler bifurcation ratios 2.5–4.0 from coronary and pulmonary casts
  (Kassab et al. 1993; Huang et al. 1996).
* Vascular fractal dimensions 2.2–2.9 (Lorthois & Cassot 2010 and refs).
* Major-artery radii/topology from standard 1D hemodynamic datasets
  (Westerhof et al.; Olufsen et al.).

## Limitations

* Mesoscale: the tree stops at ~0.5 mm vessels; the microcirculation is
  abstracted into a 3.5 cm effective perfusion depth per terminal.
* Resting hemodynamics only; no pulsatility, no exercise envelope, no
  turbulence constraint on the aorta.
* The reference tree's distal infill uses the same growth engine
  (literature-parameterized), so fine-scale reference statistics are
  model-informed; shaded literature bands in the plots are the
  measurement-grounded comparison.
* Single-objective scalarization; a Pareto-front study would separate the
  trade-off surface from the weight choice.
