"""Steady-state diffusion-consumption physics on the body voxel grid.

We solve the reaction-diffusion equation for normalized nutrient
concentration C in tissue,

    D_eff * laplacian(C) - M_eff = 0,    C = 1 on perfusion sources,

with zero-flux boundaries at the body surface (handled naturally by the
masked 7-point Laplacian). Sources are the voxels containing terminal
vessel tips: in this mesoscale model each terminal branch of the tree
feeds a microcirculatory bed with effective penetration depth delta.
The consumption/diffusivity ratio M_eff/D_eff is calibrated so that a
single voxel source penetrates to the hypoxia threshold at exactly
r = delta in spherical geometry (Krogh-type territory).
Concentrations are clamped at zero after the linear solve (zeroth-order
consumption switches off in anoxic tissue).

Outputs: coverage (tissue fraction above the hypoxia threshold) and a
delivery-speed penalty from the diffusion time d^2/(2 D), where d is
distance to the nearest source.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from .body import Body
from .constants import C0_NORMALIZED, CRITICAL_CONCENTRATION, DELTA_PERFUSION
from .tree import VascularTree


def calibrate_consumption(a: float, delta: float,
                          c_crit: float = CRITICAL_CONCENTRATION) -> float:
    """Consumption/diffusivity ratio k = M/D [1/m^2] such that a spherical
    source of radius `a` at C0=1 yields C(delta) = c_crit.

    Spherical steady state with zeroth-order consumption and a smooth
    extinction front at R (C(R)=0, C'(R)=0):
        C(r) = (k/6) r^2 - (k/2) R^2 + k R^3 / (3 r),
    with k fixed by C(a)=1. R is solved so that C(delta)=c_crit.
    """
    def k_of_R(R):
        return 1.0 / (a * a / 6.0 - R * R / 2.0 + R ** 3 / (3.0 * a))

    def c_at_delta(R):
        k = k_of_R(R)
        return k / 6.0 * delta ** 2 - k / 2.0 * R ** 2 + k * R ** 3 / (3.0 * delta)

    R = brentq(lambda R: c_at_delta(R) - c_crit, delta * 1.0001, delta * 50.0)
    return k_of_R(R)


class DiffusionModel:
    def __init__(self, body: Body, delta: float = DELTA_PERFUSION):
        self.body = body
        self.delta = delta
        nx, ny, nz = body.shape
        self.lin_index = -np.ones(body.shape, dtype=int)
        self.lin_index[body.mask] = np.arange(body.mask.sum())
        self.n = int(body.mask.sum())
        # volume-equivalent radius of one source voxel
        a_src = body.voxel * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.m_over_d = calibrate_consumption(a_src, delta)
        self._build_laplacian()

    def _build_laplacian(self):
        """Masked 7-point Laplacian (zero-flux at tissue boundary)."""
        b = self.body
        idx = self.lin_index
        rows, cols, vals = [], [], []
        diag = np.zeros(self.n)
        for axis in range(3):
            for shift in (1, -1):
                src = np.argwhere(b.mask)
                nb = src.copy()
                nb[:, axis] += shift
                ok = (nb[:, axis] >= 0) & (nb[:, axis] < b.shape[axis])
                src, nb = src[ok], nb[ok]
                ok2 = b.mask[nb[:, 0], nb[:, 1], nb[:, 2]]
                src, nb = src[ok2], nb[ok2]
                i = idx[src[:, 0], src[:, 1], src[:, 2]]
                j = idx[nb[:, 0], nb[:, 1], nb[:, 2]]
                rows.append(i)
                cols.append(j)
                vals.append(np.ones(len(i)))
                np.add.at(diag, i, -1.0)
        rows.append(np.arange(self.n))
        cols.append(np.arange(self.n))
        vals.append(diag)
        h2 = b.voxel ** 2
        self.L = sp.csr_matrix(
            (np.concatenate(vals) / h2,
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n))

    # ------------------------------------------------------------------
    def source_mask(self, tree: VascularTree) -> np.ndarray:
        """Linear-system indices of voxels containing terminal tips
        (plus the distal half of each terminal segment)."""
        b = self.body
        term = tree.terminals
        pts = [tree.pos[term]]
        par = tree.parent[term]
        ok = par >= 0
        mid = 0.5 * (tree.pos[term[ok]] + tree.pos[par[ok]])
        pts.append(mid)
        pts = np.concatenate(pts, axis=0)
        ijk = b.point_to_voxel(pts)
        ijk = np.clip(ijk, 0, np.array(b.shape) - 1)
        lin = self.lin_index[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return np.unique(lin[lin >= 0])

    def solve(self, tree: VascularTree) -> dict:
        src = self.source_mask(tree)
        if len(src) == 0:
            return {"coverage": 0.0, "speed_penalty": 10.0,
                    "concentration": np.zeros(self.n)}

        A = self.L.tolil(copy=True)
        rhs = np.full(self.n, self.m_over_d)
        for s in src:
            A.rows[s] = [s]
            A.data[s] = [1.0]
            rhs[s] = C0_NORMALIZED
        C = spla.spsolve(A.tocsr(), rhs)
        C = np.clip(C, 0.0, C0_NORMALIZED)

        covered = C >= CRITICAL_CONCENTRATION
        coverage = float(covered.mean())

        # Delivery-speed penalty: mean diffusion time to the nearest source,
        # normalized by the time to diffuse one penetration depth.
        src_grid = np.zeros(self.body.shape, dtype=bool)
        tidx = self.body.tissue_idx[src]
        src_grid[tidx[:, 0], tidx[:, 1], tidx[:, 2]] = True
        dist = ndi.distance_transform_edt(~src_grid, sampling=self.body.voxel)
        d_tissue = dist[self.body.mask]
        speed_penalty = float(np.mean((d_tissue / self.delta) ** 2))

        return {"coverage": coverage, "speed_penalty": speed_penalty,
                "concentration": C}
