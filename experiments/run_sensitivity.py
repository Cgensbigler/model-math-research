"""Sensitivity analysis: how do the optimal structures change as the
fitness weights (delivery speed, material cost, pumping cost) and the
physical perfusion depth delta are varied around the baseline?

One-at-a-time sweeps on a log grid; every run is a full (smaller)
evolution. Champions and their morphometrics are saved per condition.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vascular_evolution.body import build_body
from vascular_evolution.evolve import run_evolution
from vascular_evolution.fitness import BASELINE_WEIGHTS, evaluate_tree
from vascular_evolution.growth import grow_tree
from vascular_evolution.io import save_json, save_tree
from vascular_evolution.metrics import compute_all_metrics
from vascular_evolution.physics import DiffusionModel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sensitivity"
RESULTS.mkdir(parents=True, exist_ok=True)

SWEEPS = {
    "w_time": [0.0625, 0.125, 0.25, 0.5, 1.0],
    "w_mat": [0.025, 0.05, 0.1, 0.2, 0.4],
    "w_pow": [0.025, 0.05, 0.1, 0.2, 0.4],
}
DELTA_SWEEP = [0.025, 0.045, 0.06]   # baseline is 0.035 m
POP, GENS, SEED = 24, 30, 0


def main():
    body = build_body()
    model = DiffusionModel(body)
    summary = []
    conditions = [(wname, v, None) for wname, values in SWEEPS.items()
                  for v in values]
    # physical sensitivity: effective microcirculatory perfusion depth
    conditions += [("delta", d, d) for d in DELTA_SWEEP]

    for wname, v, delta in conditions:
        tag = f"{wname}_{v:g}"
        t0 = time.time()
        weights = dict(BASELINE_WEIGHTS)
        if delta is None:
            weights[wname] = v
        print(f"=== sweep {tag} ===", flush=True)
        res = run_evolution(pop_size=POP, n_generations=GENS, seed=SEED,
                            weights=weights, delta=delta,
                            n_workers=4, verbose=False)
        genes = np.array(res["best_genome"])
        tree = grow_tree(genes, body,
                         np.random.default_rng(res["growth_seed"]))
        metrics = compute_all_metrics(tree)
        cond_model = model if delta is None else DiffusionModel(body, delta=delta)
        ev = evaluate_tree(tree, cond_model, weights)
        save_tree(RESULTS / f"tree_{tag}.npz", tree)
        save_json(RESULTS / f"run_{tag}.json",
                  {"result": res, "metrics": metrics, "eval": ev,
                   "weights": weights, "delta": delta})
        summary.append({"param": wname, "value": v,
                        "genome": res["best_genome"],
                        "metrics": metrics, "eval": ev})
        print(f"  fit={ev['fitness']:.4f} cov={ev['coverage']:.3f} "
              f"terms={metrics['n_terminals']} "
              f"V={ev['material_volume_L']:.2f}L "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_json(RESULTS / "summary.json", summary)


if __name__ == "__main__":
    main()
