# Significance tests

## Cube vs body ensembles (n = 10 each, Mann-Whitney U, Holm-corrected)

| Property | Cube | Body | p (raw) | p (Holm) | Cohen's d | |
|---|---|---|---|---|---|---|
| fractal_dimension | 2.23 ± 0.026 | 1.87 ± 0.13 | 0.00018 | 0.0022 | +3.59 | **significant** |
| murray_exponent | 3.17 ± 0.14 | 3.17 ± 0.17 | 0.88 | 1 | -0.01 | ns |
| bifurcation_ratio | 3.4 ± 0.16 | 3.85 ± 0.28 | 0.00077 | 0.0061 | -1.86 | **significant** |
| mean_daughters | 2.65 ± 0.023 | 2.76 ± 0.11 | 0.014 | 0.098 | -1.25 | ns |
| radius_asymmetry | 0.861 ± 0.014 | 0.809 ± 0.019 | 0.00044 | 0.0044 | +2.90 | **significant** |
| length_ratio | 0.982 ± 0.055 | 1.25 ± 0.19 | 0.00058 | 0.0052 | -1.81 | **significant** |
| tortuosity | 1.78 ± 0.039 | 2.06 ± 0.31 | 0.026 | 0.15 | -1.18 | ns |
| length_to_radius | 54.1 ± 3.6 | 67.5 ± 15 | 0.026 | 0.15 | -1.18 | ns |
| clark_evans | 0.816 ± 0.016 | 0.837 ± 0.087 | 0.62 | 1 | -0.32 | ns |
| mean_dist_over_ideal | 0.658 ± 0.011 | 0.772 ± 0.031 | 0.00018 | 0.0022 | -4.59 | **significant** |
| length_density_m_per_L | 2.11 ± 0.12 | 2.56 ± 0.85 | 0.026 | 0.15 | -0.70 | ns |
| n_terminals | 2.2e+03 ± 91 | 2.01e+03 ± 6.1e+02 | 0.12 | 0.36 | +0.42 | ns |

## Murray exponent equivalence
Cube - body difference = -0.002, 95% CI [-0.156, +0.152] — any true domain effect on the radius-scaling exponent is smaller than ~0.16.

## Terminal dispersion vs spatial randomness (Wilcoxon vs CE = 1)
- cube: CE = 0.816 ± 0.016, p = 0.0020 (clustered relative to Poisson; note CE carries a negative edge-effect bias in bounded domains — the cross-domain and vs-reference comparisons use the same estimator)
- body: CE = 0.837 ± 0.087, p = 0.0039 (clustered relative to Poisson; note CE carries a negative edge-effect bias in bounded domains — the cross-domain and vs-reference comparisons use the same estimator)

## Sensitivity sweeps (n = 1 evolution per condition; exact monotone-trend test over 5 ordered conditions)
- luminal volume vs w_mat: values ['0.705', '0.52', '0.364', '0.208', '0.11'], trend p = 0.017
- terminal count vs w_mat: values ['2.38e+03', '2.31e+03', '2.35e+03', '1.25e+03', '1.27e+03'], trend p = 0.233
- luminal volume vs w_pow: values ['0.277', '0.321', '0.364', '0.554', '0.49'], trend p = 0.083
- terminal count vs delta: values ['2.39e+03', '2.31e+03', '658'], trend p = 0.333
- fractal dimension vs w_time: values ['2.04', '2.03', '2.07', '2.08', '2.08'], trend p = 0.233

## Two-basin claim (body ensemble)
Sparse optima (n=2) fitness [0.78, 0.753] vs dense (n=8) [0.794, 0.796, 0.794, 0.794, 0.792, 0.751, 0.804, 0.807]: Mann-Whitney p = 0.178. With n = 2 the 'second basin' is suggestive (the terminal-count gap 982 to 2266 is large) but not statistically established, and could reflect incomplete GA convergence rather than true landscape multimodality.
