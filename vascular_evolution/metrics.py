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


def compute_all_metrics(tree: VascularTree) -> dict:
    pts = tree.polyline_points(spacing=0.004)
    fd, sizes, counts = box_counting_dimension(pts)
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
