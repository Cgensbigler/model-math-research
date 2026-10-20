"""Fitness: benefit of nutrient delivery minus structural/energetic cost.

    F = coverage
        - w_time * T_norm        (delivery speed: mean diffusion time)
        - w_mat  * V / V_ref     (structural cost: vessel luminal volume)
        - w_pow  * P / P_ref     (energetic cost: Poiseuille pumping power)

Reference scales are physiological: V_ref is the blood volume held in
human arteries (~0.5 L) and P_ref the hydraulic power dissipated in the
arterial tree down to small arteries (~0.26 W). With the baseline
weights, a design matching human-scale costs pays ~0.2 of fitness for
them, so costs matter but cannot dominate the delivery benefit —
mirroring the biology, where the circulation consumes only a few
percent of basal metabolism.
"""

from __future__ import annotations

import numpy as np

from .constants import ARTERIAL_BLOOD_VOLUME, PUMP_POWER_REF
from .growth import grow_tree
from .physics import DiffusionModel

BASELINE_WEIGHTS = {
    "w_time": 0.25,
    "w_mat": 0.10,
    "w_pow": 0.10,
}


def evaluate_tree(tree, model: DiffusionModel, weights: dict | None = None) -> dict:
    w = dict(BASELINE_WEIGHTS)
    if weights:
        w.update(weights)
    phys = model.solve(tree)
    v_norm = tree.material_volume / ARTERIAL_BLOOD_VOLUME
    p_norm = tree.pumping_power() / PUMP_POWER_REF
    fitness = (phys["coverage"]
               - w["w_time"] * phys["speed_penalty"]
               - w["w_mat"] * v_norm
               - w["w_pow"] * p_norm)
    return {
        "fitness": float(fitness),
        "coverage": phys["coverage"],
        "speed_penalty": phys["speed_penalty"],
        "material_norm": float(v_norm),
        "power_norm": float(p_norm),
        "material_volume_L": float(tree.material_volume * 1e3),
        "pumping_power_W": float(tree.pumping_power()),
    }


def evaluate_genome(genes: np.ndarray, body, model: DiffusionModel,
                    weights: dict | None = None, seed: int = 0) -> dict:
    """Grow a tree from a genome and score it. Growth uses a fixed seed so
    the genotype-to-phenotype map is deterministic within one evolution."""
    rng = np.random.default_rng(seed)
    try:
        tree = grow_tree(genes, body, rng)
    except Exception:
        return {"fitness": -1e9}
    if tree.n_nodes < 10:
        return {"fitness": -1e9}
    out = evaluate_tree(tree, model, weights)
    out["n_nodes"] = tree.n_nodes
    out["n_terminals"] = int(len(tree.terminals))
    return out
