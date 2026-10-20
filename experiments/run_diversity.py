"""Diversity experiment: repeat the evolution with independent seeds
(different GA randomness AND different growth-noise seeds) and save
every champion tree + its metrics for cross-solution comparison."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vascular_evolution.body import build_body
from vascular_evolution.evolve import run_evolution
from vascular_evolution.fitness import evaluate_tree
from vascular_evolution.growth import grow_tree
from vascular_evolution.io import save_json, save_tree
from vascular_evolution.metrics import compute_all_metrics
from vascular_evolution.physics import DiffusionModel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "diversity"
RESULTS.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10


def main():
    body = build_body()
    model = DiffusionModel(body)
    summary = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"=== diversity seed {seed} ===", flush=True)
        res = run_evolution(pop_size=28, n_generations=45, seed=seed,
                            n_workers=4, verbose=False)
        genes = np.array(res["best_genome"])
        tree = grow_tree(genes, body, np.random.default_rng(res["growth_seed"]))
        metrics = compute_all_metrics(tree)
        ev = evaluate_tree(tree, model)
        save_tree(RESULTS / f"tree_seed{seed}.npz", tree)
        save_json(RESULTS / f"run_seed{seed}.json",
                  {"result": res, "metrics": metrics, "eval": ev})
        summary.append({"seed": seed, "fitness": ev["fitness"],
                        "coverage": ev["coverage"],
                        "genome": res["best_genome"], "metrics": metrics,
                        "eval": ev})
        print(f"  fit={ev['fitness']:.4f} cov={ev['coverage']:.3f} "
              f"terms={metrics['n_terminals']} "
              f"FD={metrics['fractal_dimension']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_json(RESULTS / "summary.json", summary)


if __name__ == "__main__":
    main()
