"""Parameterized space-colonization growth of vascular trees.

The evolutionary genome encodes the parameters of a developmental
growth program rather than raw geometry — analogous to how real
vasculature is built by local rules (VEGF gradients attract sprouting
vessels, perfused regions stop signalling). Attractor points are
sampled from the body's metabolic demand field; the tree grows from the
heart towards attractors; attractors are deleted once a vessel comes
close enough. Different parameter settings yield radically different
architectures (sparse trunks, dense brushes, straight or tortuous
paths), giving the evolutionary search a rich but well-behaved space.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .body import Body, HEART_POS
from .tree import VascularTree

# Genome definition: name -> (low, high). All lengths in metres.
GENOME = {
    "attractors": (200.0, 1600.0),     # number of demand-sampled targets
    "influence": (0.06, 0.45),         # attraction radius
    "kill": (0.015, 0.10),             # attractor satisfied within this range
    "step": (0.015, 0.06),             # growth step length
    "momentum": (0.0, 0.85),           # persistence of parent direction
    "jitter": (0.0, 0.5),              # random direction noise amplitude
    "down_bias": (-0.3, 0.3),          # tropism along -z (+ = downward pull)
    "murray_gamma": (2.0, 3.5),        # radius scaling exponent
    "r_terminal": (4e-4, 2.0e-3),      # terminal segment radius
}

GENE_NAMES = list(GENOME.keys())
GENE_LOW = np.array([GENOME[k][0] for k in GENE_NAMES])
GENE_HIGH = np.array([GENOME[k][1] for k in GENE_NAMES])


def random_genome(rng: np.random.Generator) -> np.ndarray:
    return GENE_LOW + rng.random(len(GENE_NAMES)) * (GENE_HIGH - GENE_LOW)


def genome_dict(g: np.ndarray) -> dict:
    return dict(zip(GENE_NAMES, g))


def sample_attractors(body: Body, n: int, rng: np.random.Generator) -> np.ndarray:
    demand = body.demand[tuple(body.tissue_idx.T)]
    p = demand / demand.sum()
    idx = rng.choice(len(p), size=n, replace=True, p=p)
    # jitter within the voxel so attractors are not on a lattice
    return body.tissue_pts[idx] + (rng.random((n, 3)) - 0.5) * body.voxel


def colonize(pos: list, parent: list, direction: list,
             attractors: np.ndarray, g: dict, body: Body,
             rng: np.random.Generator,
             max_nodes: int = 4000, max_iter: int = 400) -> None:
    """Run the space-colonization loop, extending pos/parent/direction
    in place. `g` needs keys: influence, kill, step, momentum, jitter,
    down_bias."""
    alive = np.ones(len(attractors), dtype=bool)
    kill2 = g["kill"] ** 2
    n_children = [0] * len(pos)
    for p in parent:
        if p >= 0:
            n_children[p] += 1

    for _ in range(max_iter):
        if not alive.any() or len(pos) >= max_nodes:
            break
        P = np.asarray(pos)
        A = attractors[alive]
        # Nodes saturated with children cannot sprout again (real junctions
        # are predominantly bifurcations, occasionally trifurcations).
        growable = np.flatnonzero(np.asarray(n_children) < 3)
        if len(growable) == 0:
            break
        tree_kd = cKDTree(P[growable])
        d, nearest = tree_kd.query(A, k=1, distance_upper_bound=g["influence"])
        valid = np.isfinite(d)
        if not valid.any():
            break

        # Accumulate normalized pull vectors per node
        node_ids = growable[nearest[valid]]
        pulls = A[valid] - P[node_ids]
        pulls /= np.linalg.norm(pulls, axis=1, keepdims=True) + 1e-12
        acc = np.zeros_like(P)
        cnt = np.zeros(len(P))
        np.add.at(acc, node_ids, pulls)
        np.add.at(cnt, node_ids, 1.0)
        growers = np.flatnonzero(cnt > 0)

        new_pts, new_parents, new_dirs = [], [], []
        for ni in growers:
            dvec = acc[ni] / cnt[ni]
            dvec = dvec + g["momentum"] * direction[ni]
            dvec = dvec + g["jitter"] * (rng.random(3) - 0.5)
            dvec[2] -= g["down_bias"]
            nrm = np.linalg.norm(dvec)
            if nrm < 1e-9:
                continue
            dvec /= nrm
            cand = P[ni] + dvec * g["step"]
            if not body.inside(cand[None])[0]:
                # try half-step, then give up for this node this round
                cand = P[ni] + dvec * (0.5 * g["step"])
                if not body.inside(cand[None])[0]:
                    continue
            new_pts.append(cand)
            new_parents.append(ni)
            new_dirs.append(dvec)

        if not new_pts:
            break
        pos.extend(new_pts)
        parent.extend(new_parents)
        direction.extend(new_dirs)
        for p in new_parents:
            n_children[p] += 1
        n_children.extend([0] * len(new_pts))

        # kill satisfied attractors (near any new node)
        NA = np.asarray(new_pts)
        live_idx = np.flatnonzero(alive)
        d2 = ((attractors[live_idx][:, None, :] - NA[None]) ** 2).sum(-1).min(1)
        alive[live_idx[d2 < kill2]] = False


def grow_tree(genes: np.ndarray, body: Body, rng: np.random.Generator,
              max_nodes: int = 4000, max_iter: int = 400) -> VascularTree:
    g = genome_dict(genes)
    attractors = sample_attractors(body, int(round(g["attractors"])), rng)

    pos = [HEART_POS.copy()]
    parent = [-1]
    direction = [np.array([0.0, 0.0, -1.0])]   # aorta initially descends

    colonize(pos, parent, direction, attractors, g, body, rng,
             max_nodes=max_nodes, max_iter=max_iter)

    tree = VascularTree(pos=np.asarray(pos), parent=np.asarray(parent, dtype=int))
    tree.assign_radii(g["murray_gamma"], g["r_terminal"])
    tree.assign_flows(body)
    return tree
