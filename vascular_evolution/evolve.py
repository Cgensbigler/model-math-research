"""Evolutionary optimization of growth-program genomes.

A real-coded genetic algorithm: tournament selection, blend (BLX-alpha)
crossover, per-gene Gaussian mutation, elitism. Fitness evaluations run
in a multiprocessing pool; each worker builds the body and diffusion
model once. The growth RNG seed is fixed per run so the
genotype->phenotype map is deterministic and fitness is noise-free
within a run (different runs use different growth seeds, which is part
of the diversity experiment).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .body import build_body
from .fitness import evaluate_genome
from .growth import GENE_HIGH, GENE_LOW, GENE_NAMES, random_genome
from .physics import DiffusionModel

_WORKER = {}


def _init_worker(weights, growth_seed, delta):
    body = build_body()
    _WORKER["body"] = body
    _WORKER["model"] = (DiffusionModel(body) if delta is None
                        else DiffusionModel(body, delta=delta))
    _WORKER["weights"] = weights
    _WORKER["growth_seed"] = growth_seed


def _eval(genes):
    return evaluate_genome(genes, _WORKER["body"], _WORKER["model"],
                           _WORKER["weights"], seed=_WORKER["growth_seed"])


def run_evolution(pop_size: int = 28, n_generations: int = 40,
                  weights: dict | None = None, seed: int = 0,
                  n_workers: int = 4, elite: int = 2,
                  delta: float | None = None,
                  verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    growth_seed = 1000 + seed
    pop = np.array([random_genome(rng) for _ in range(pop_size)])
    span = GENE_HIGH - GENE_LOW

    history = []
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(weights, growth_seed, delta)) as pool:
        for gen in range(n_generations):
            results = pool.map(_eval, list(pop))
            fit = np.array([r["fitness"] for r in results])
            order = np.argsort(fit)[::-1]
            pop, fit = pop[order], fit[order]
            results = [results[i] for i in order]

            history.append({
                "gen": gen,
                "best_fitness": float(fit[0]),
                "mean_fitness": float(np.mean(fit[fit > -1e8])) if (fit > -1e8).any() else float("nan"),
                "best_eval": results[0],
                "best_genome": pop[0].tolist(),
            })
            if verbose:
                r0 = results[0]
                print(f"  gen {gen:3d}  best={fit[0]:+.4f} "
                      f"cov={r0.get('coverage', 0):.3f} "
                      f"T={r0.get('speed_penalty', 0):.2f} "
                      f"V={r0.get('material_norm', 0):.2f} "
                      f"P={r0.get('power_norm', 0):.2f} "
                      f"terms={r0.get('n_terminals', 0)}", flush=True)

            if gen == n_generations - 1:
                break

            # --- next generation ---
            new_pop = [pop[i].copy() for i in range(elite)]
            while len(new_pop) < pop_size:
                # tournament selection (k=3)
                def pick():
                    cand = rng.integers(0, pop_size, 3)
                    return pop[cand[np.argmax(fit[cand])]]
                p1, p2 = pick(), pick()
                # BLX-0.3 crossover
                lo = np.minimum(p1, p2)
                hi = np.maximum(p1, p2)
                d = hi - lo
                child = rng.uniform(lo - 0.3 * d, hi + 0.3 * d)
                # Gaussian mutation
                mut = rng.random(len(child)) < 0.35
                child[mut] += rng.normal(0, 0.12, mut.sum()) * span[mut]
                child = np.clip(child, GENE_LOW, GENE_HIGH)
                new_pop.append(child)
            pop = np.array(new_pop)

    return {
        "best_genome": pop[0].tolist(),
        "best_fitness": float(fit[0]),
        "best_eval": results[0],
        "history": history,
        "gene_names": GENE_NAMES,
        "seed": seed,
        "growth_seed": growth_seed,
        "weights": weights or {},
        "pop_size": pop_size,
        "n_generations": n_generations,
    }
