"""Vascular tree data structure and hemodynamics.

A tree is stored as flat arrays: node positions, parent indices and
per-node radii (the radius of the segment joining the node to its
parent). Radii follow a Murray-type rule r_parent^g = sum r_child^g
applied bottom-up from a terminal radius. Flows are assigned to
terminals in proportion to the metabolic demand of the tissue territory
nearest to each terminal, then accumulated towards the root; pumping
power uses Poiseuille's law per segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .constants import BLOOD_VISCOSITY, BLOOD_O2_CONTENT


@dataclass
class VascularTree:
    pos: np.ndarray          # (N, 3) node positions, node 0 = root
    parent: np.ndarray       # (N,) int, parent[0] == -1
    radius: np.ndarray = field(default=None)   # (N,) segment radius to parent
    flow: np.ndarray = field(default=None)     # (N,) flow in segment to parent

    @property
    def n_nodes(self) -> int:
        return len(self.pos)

    @property
    def children(self) -> list:
        ch = [[] for _ in range(self.n_nodes)]
        for i in range(1, self.n_nodes):
            ch[self.parent[i]].append(i)
        return ch

    @property
    def n_children(self) -> np.ndarray:
        cnt = np.zeros(self.n_nodes, dtype=int)
        p = self.parent[1:]
        np.add.at(cnt, p, 1)
        return cnt

    @property
    def terminals(self) -> np.ndarray:
        return np.flatnonzero(self.n_children == 0)

    @property
    def lengths(self) -> np.ndarray:
        """Segment lengths; lengths[0] = 0 (root has no parent segment)."""
        L = np.zeros(self.n_nodes)
        L[1:] = np.linalg.norm(self.pos[1:] - self.pos[self.parent[1:]], axis=1)
        return L

    def topological_order(self) -> np.ndarray:
        """Nodes ordered root -> leaves (parents before children)."""
        order = np.empty(self.n_nodes, dtype=int)
        depth = np.zeros(self.n_nodes, dtype=int)
        for i in range(1, self.n_nodes):
            depth[i] = depth[self.parent[i]] + 1   # parents precede children
        order = np.argsort(depth, kind="stable")
        return order

    # ------------------------------------------------------------------
    # Radii and flows
    # ------------------------------------------------------------------
    def assign_radii(self, gamma: float, r_terminal: float) -> None:
        """Murray's law bottom-up: r^g = sum of child r^g."""
        rg = np.zeros(self.n_nodes)
        term = self.n_children == 0
        rg[term] = r_terminal ** gamma
        for i in self.topological_order()[::-1]:
            p = self.parent[i]
            if p >= 0:
                rg[p] += rg[i]
        # overwrite internal nodes that got accumulated values; terminals stay
        self.radius = rg ** (1.0 / gamma)
        self.radius[term] = r_terminal

    def assign_flows(self, body) -> None:
        """Terminal flows proportional to the demand each terminal serves.

        Tissue voxels are partitioned by nearest terminal (a Voronoi
        territory); terminal blood flow = territory O2 demand divided by
        deliverable O2 per unit blood (content x extraction ~ 1/3).
        """
        term = self.terminals
        tree_pts = cKDTree(self.pos[term])
        _, nearest = tree_pts.query(body.tissue_pts, k=1)
        demand_t = body.demand[tuple(body.tissue_idx.T)]
        served = np.bincount(nearest, weights=demand_t, minlength=len(term))
        q_term = served / (BLOOD_O2_CONTENT / 3.0)   # m^3 blood / s
        flow = np.zeros(self.n_nodes)
        flow[term] = q_term
        for i in self.topological_order()[::-1]:
            p = self.parent[i]
            if p >= 0:
                flow[p] += flow[i]
        self.flow = flow

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    @property
    def material_volume(self) -> float:
        """Total luminal volume of all segments [m^3]."""
        return float(np.sum(np.pi * self.radius[1:] ** 2 * self.lengths[1:]))

    @property
    def total_length(self) -> float:
        return float(self.lengths.sum())

    def pumping_power(self) -> float:
        """Poiseuille dissipation summed over segments [W]."""
        L = self.lengths[1:]
        r = np.maximum(self.radius[1:], 1e-6)
        Q = self.flow[1:]
        R_seg = 8.0 * BLOOD_VISCOSITY * L / (np.pi * r ** 4)
        return float(np.sum(R_seg * Q ** 2))

    # ------------------------------------------------------------------
    def polyline_points(self, spacing: float) -> np.ndarray:
        """Dense sample of points along all segments (for voxelization)."""
        pts = [self.pos[:1]]
        a = self.pos[self.parent[1:]]
        b = self.pos[1:]
        seg_len = np.linalg.norm(b - a, axis=1)
        for i in range(len(a)):
            n = max(2, int(np.ceil(seg_len[i] / spacing)) + 1)
            t = np.linspace(0.0, 1.0, n)[:, None]
            pts.append(a[i] + t * (b[i] - a[i]))
        return np.concatenate(pts, axis=0)
