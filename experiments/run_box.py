"""Box control experiment: evolve vascular networks in a uniform-demand
cube with the same tissue volume (64 L vs 62.9 L), voxel size, physics
and evolutionary budget as the body ensemble. If the fractal,
topological and geometric properties coincide with those evolved in the
human-shaped domain, they are intrinsic to the delivery/cost physics
rather than imposed by anatomy."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vascular_evolution.body import build_box
from vascular_evolution.evolve import run_evolution
from vascular_evolution.fitness import evaluate_tree
from vascular_evolution.growth import grow_tree
from vascular_evolution.io import save_json, save_tree
from vascular_evolution.metrics import compute_all_metrics, space_filling_metrics
from vascular_evolution.physics import DiffusionModel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "box"
RESULTS.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
# Scale range over which box-counting is comparable across domains
# (well below the 0.40 m cube side).
FD_SIZES = np.geomspace(0.01, 0.12, 8)


def main():
    box = build_box()
    model = DiffusionModel(box)
    print(f"box domain: {box.shape} tissue={box.tissue_volume*1e3:.1f} L",
          flush=True)
    summary = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"=== box seed {seed} ===", flush=True)
        res = run_evolution(pop_size=28, n_generations=45, seed=seed,
                            domain="box", n_workers=4, verbose=False)
        genes = np.array(res["best_genome"])
        tree = grow_tree(genes, box, np.random.default_rng(res["growth_seed"]))
        metrics = compute_all_metrics(tree, box_sizes=FD_SIZES)
        metrics.update(space_filling_metrics(tree, box))
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
              f"CE={metrics['clark_evans']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_json(RESULTS / "summary.json", summary)


if __name__ == "__main__":
    main()
