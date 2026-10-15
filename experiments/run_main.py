"""Main experiment: evolve the baseline vascular tree and build the
anatomical reference; save trees, metrics and evolution history."""

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
from vascular_evolution.reference import build_reference_tree

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    body = build_body()
    model = DiffusionModel(body)

    print("=== Anatomical reference tree ===", flush=True)
    ref, info = build_reference_tree(body, seed=0)
    ref_metrics = compute_all_metrics(ref)
    ref_eval = evaluate_tree(ref, model)
    save_tree(RESULTS / "reference_tree.npz", ref)
    save_json(RESULTS / "reference_metrics.json",
              {"metrics": ref_metrics, "eval": ref_eval,
               "n_named": info["n_named"]})
    print(f"  nodes={ref.n_nodes} terminals={len(ref.terminals)} "
          f"coverage={ref_eval['coverage']:.3f} fitness={ref_eval['fitness']:.3f}")

    print("=== Main evolution (baseline weights) ===", flush=True)
    t0 = time.time()
    res = run_evolution(pop_size=36, n_generations=60, seed=0, n_workers=4)
    print(f"  evolution took {time.time() - t0:.0f}s")

    genes = np.array(res["best_genome"])
    tree = grow_tree(genes, body, np.random.default_rng(res["growth_seed"]))
    best_metrics = compute_all_metrics(tree)
    best_eval = evaluate_tree(tree, model)
    save_tree(RESULTS / "best_tree.npz", tree)
    save_json(RESULTS / "main_run.json",
              {"result": res, "metrics": best_metrics, "eval": best_eval})
    print(f"  best fitness={best_eval['fitness']:.4f} "
          f"coverage={best_eval['coverage']:.3f} "
          f"nodes={tree.n_nodes} terminals={len(tree.terminals)}")


if __name__ == "__main__":
    main()
