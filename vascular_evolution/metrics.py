"""Morphometric properties of vascular trees.

Implements the standard measures used to characterize real vascular
networks so evolved and anatomical trees can be compared quantitatively:

- box-counting fractal dimension of the embedded network
- Murray radius-scaling exponent fitted at junctions
- Strahler ordering: bifurcation, length and radius ratios
- branching number (mean daughters per junction) and motif census
- totals: length, luminal volume, segment counts, terminal counts
- segment geometry: length/radius ratio, tortuosity, asymmetry
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.stats import linregress

from .tree import VascularTree


def box_counting_dimension(points: np.ndarray,
                           sizes=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Fractal dimension by box counting over a dense point sample."""
    if sizes is None:
        sizes = np.geomspace(0.01, 0.32, 10)
    lo = points.min(axis=0)
    counts = []
    for s in sizes:
        ijk = np.floor((points - lo) / s).astype(np.int64)
        key = (ijk[:, 0] << 42) + (ijk[:, 1] << 21) + ijk[:, 2]
        counts.append(len(np.unique(key)))
    counts = np.asarray(counts, dtype=float)
    fit = linregress(np.log(1.0 / sizes), np.log(counts))
    return float(fit.slope), sizes, counts


def murray_exponent(tree: VascularTree) -> tuple[float, list]:
    """Per-junction exponent x solving r_p^x = sum r_c^x; returns median."""
    ch = tree.children
    exps = []
    for i in range(tree.n_nodes):
        kids = ch[i]
        if len(kids) < 2 or tree.parent[i] < 0:
            continue
        rp = tree.radius[i]
        rc = tree.radius[kids]
        if rp <= rc.max() + 1e-12:      # exponent undefined / degenerate
            continue

        def f(x):
            return np.sum((rc / rp) ** x) - 1.0
        try:
            if f(1.0) > 0 and f(8.0) < 0:
                exps.append(brentq(f, 1.0, 8.0))
        except ValueError:
            pass
    med = float(np.median(exps)) if exps else float("nan")
    return med, exps


def strahler_orders(tree: VascularTree) -> np.ndarray:
    order = np.zeros(tree.n_nodes, dtype=int)
    ch = tree.children
    for i in tree.topological_order()[::-1]:
        kids = ch[i]
        if not kids:
            order[i] = 1
        else:
            ko = order[kids]
            m = ko.max()
            order[i] = m + 1 if (ko == m).sum() >= 2 else m
    return order


def strahler_ratios(tree: VascularTree) -> dict:
    """Bifurcation, length and radius ratios from Strahler branch statistics."""
    order = strahler_orders(tree)
    ch = tree.children
    L = tree.lengths
    # Merge chains of same-order segments into Strahler "branches"
    branch_len, branch_rad, branch_ord = [], [], []
    visited = np.zeros(tree.n_nodes, dtype=bool)
    for i in tree.topological_order():
        if visited[i] or tree.parent[i] < 0:
            continue
        o = order[i]
        if order[tree.parent[i]] == o and tree.parent[i] != 0:
            continue   # interior of a chain; handled by its head
        # walk down the chain of equal order
        tot_len, rads, j = 0.0, [], i
        while True:
            visited[j] = True
            tot_len += L[j]
            rads.append(tree.radius[j])
            nxt = [k for k in ch[j] if order[k] == o]
            if len(nxt) != 1:
                break
            j = nxt[0]
        branch_len.append(tot_len)
        branch_rad.append(np.mean(rads))
        branch_ord.append(o)
    branch_ord = np.asarray(branch_ord)
    branch_len = np.asarray(branch_len)
    branch_rad = np.asarray(branch_rad)

    orders = np.arange(1, branch_ord.max() + 1)
    n_o = np.array([(branch_ord == o).sum() for o in orders], dtype=float)
    keep = n_o > 0
    res = {}
    if keep.sum() >= 2:
        res["bifurcation_ratio"] = float(np.exp(
            -linregress(orders[keep], np.log(n_o[keep])).slope))
        mean_l = np.array([branch_len[branch_ord == o].mean() for o in orders])
        mean_r = np.array([branch_rad[branch_ord == o].mean() for o in orders])
        res["length_ratio"] = float(np.exp(
            linregress(orders[keep], np.log(mean_l[keep])).slope))
        res["radius_ratio"] = float(np.exp(
            linregress(orders[keep], np.log(mean_r[keep])).slope))
    else:
        res["bifurcation_ratio"] = float("nan")
        res["length_ratio"] = float("nan")
        res["radius_ratio"] = float("nan")
    res["max_strahler_order"] = int(branch_ord.max())
    res["orders"] = orders.tolist()
    res["branches_per_order"] = n_o.tolist()
    return res


def motif_census(tree: VascularTree) -> dict:
    nch = tree.n_children
    junction = nch >= 2
    return {
        "n_bifurcations": int((nch == 2).sum()),
        "n_trifurcations": int((nch == 3).sum()),
        "n_higher_furcations": int((nch > 3).sum()),
        "n_continuations": int((nch == 1).sum()),
        "mean_daughters": float(nch[junction].mean()) if junction.any() else float("nan"),
    }


def tortuosity(tree: VascularTree) -> float:
    """Mean (path length / Euclidean distance) from root to terminals."""
    L = tree.lengths
    path = np.zeros(tree.n_nodes)
    for i in tree.topological_order():
        p = tree.parent[i]
        if p >= 0:
            path[i] = path[p] + L[i]
    term = tree.terminals
    eu = np.linalg.norm(tree.pos[term] - tree.pos[0], axis=1)
    ok = eu > 0.05
    return float(np.mean(path[term][ok] / eu[ok])) if ok.any() else float("nan")


def radius_asymmetry(tree: VascularTree) -> float:
    ch = tree.children
    ratios = []
    for i in range(tree.n_nodes):
        kids = ch[i]
        if len(kids) >= 2:
            r = np.sort(tree.radius[kids])
            ratios.append(r[-2] / r[-1])
    return float(np.mean(ratios)) if ratios else float("nan")


def clark_evans_index(tree: VascularTree, body) -> float:
    """Clark-Evans dispersion index of the terminal points in 3D: observed
    mean nearest-neighbour distance over the expectation for a Poisson
    process of the same intensity in the tissue volume. 1 = random,
    > 1 = dispersed towards uniform (space-filling), < 1 = clustered.
    E[d_NN] for 3D Poisson = Gamma(4/3) * (4*pi*lambda/3)^(-1/3)."""
    from math import gamma, pi
    pts = tree.pos[tree.terminals]
    if len(pts) < 3:
        return float("nan")
    kd = cKDTree(pts)
    d, _ = kd.query(pts, k=2)
    lam = len(pts) / body.tissue_volume
    expected = gamma(4.0 / 3.0) * (4.0 * pi * lam / 3.0) ** (-1.0 / 3.0)
    return float(d[:, 1].mean() / expected)


def tissue_vessel_distance(tree: VascularTree, body) -> dict:
    """Distance from tissue to the nearest vessel centreline: mean and p95
    (m), plus values normalized by the ideal radius of equal-volume
    spheres around the terminals (perfect space-filling ~ 0.75 ideal)."""
    pts = tree.polyline_points(spacing=0.01)
    kd = cKDTree(pts)
    d, _ = kd.query(body.tissue_pts, k=1)
    n_term = max(len(tree.terminals), 1)
    r_ideal = (3.0 * body.tissue_volume / (4.0 * np.pi * n_term)) ** (1.0 / 3.0)
    return {
        "mean_tissue_vessel_dist_cm": float(d.mean() * 100),
        "p95_tissue_vessel_dist_cm": float(np.percentile(d, 95) * 100),
        "mean_dist_over_ideal": float(d.mean() / r_ideal),
    }


def space_filling_metrics(tree: VascularTree, body) -> dict:
    out = {"clark_evans": clark_evans_index(tree, body),
           "length_density_m_per_L": float(
               tree.total_length / (body.tissue_volume * 1e3))}
    out.update(tissue_vessel_distance(tree, body))
    return out


def compute_all_metrics(tree: VascularTree, box_sizes=None) -> dict:
    pts = tree.polyline_points(spacing=0.004)
    fd, sizes, counts = box_counting_dimension(pts, sizes=box_sizes)
    mur, _ = murray_exponent(tree)
    sr = strahler_ratios(tree)
    mc = motif_census(tree)
    L = tree.lengths[1:]
    r = tree.radius[1:]
    span = tree.pos.max(axis=0) - tree.pos.min(axis=0)
    m = {
        "fractal_dimension": fd,
        "murray_exponent": mur,
        "mean_daughters": mc["mean_daughters"],
        "radius_asymmetry": radius_asymmetry(tree),
        "tortuosity": tortuosity(tree),
        "total_length_m": tree.total_length,
        "material_volume_L": tree.material_volume * 1e3,
        "n_segments": int(tree.n_nodes - 1),
        "n_terminals": int(len(tree.terminals)),
        "root_radius_mm": float(tree.radius[1:][np.argmax(r)] * 1e3),
        "mean_segment_length_cm": float(L.mean() * 100),
        "length_to_radius": float(np.mean(L / np.maximum(r, 1e-6))),
        "span_x_m": float(span[0]), "span_y_m": float(span[1]),
        "span_z_m": float(span[2]),
        "box_sizes": sizes.tolist(),
        "box_counts": counts.tolist(),
    }
    m.update({k: sr[k] for k in
              ("bifurcation_ratio", "length_ratio", "radius_ratio",
               "max_strahler_order", "orders", "branches_per_order")})
    m.update(mc)
    return m
