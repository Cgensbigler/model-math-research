"""Voxelized 3D model of the human body with a metabolic demand field.

Coordinate system (metres): x = left-right, y = back-front, z = feet-head.
The body is assembled from capsules and ellipsoids approximating a
reference adult (1.75 m, ~66 L). Each tissue voxel carries a metabolic
demand weight; organ hotspots (brain, heart, liver, kidneys, gut) follow
measured organ shares of resting oxygen consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import VO2_REST

VOXEL = 0.02  # m

# Bounding box of the grid (m)
XLIM = (-0.36, 0.36)
YLIM = (-0.16, 0.20)
ZLIM = (0.0, 1.80)

HEART_POS = np.array([0.02, 0.0, 1.32])


def _capsule_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points to segment ab."""
    ab = b - a
    t = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def _ellipsoid_in(pts: np.ndarray, c: np.ndarray, semi: np.ndarray) -> np.ndarray:
    return (((pts - c) / semi) ** 2).sum(axis=1) <= 1.0


# (a, b, radius) capsules; symmetric limbs generated for both signs of x.
_CAPSULES = [
    # neck
    ((0.0, 0.0, 1.50), (0.0, 0.0, 1.60), 0.055),
    # upper arm, forearm, hand (left; mirrored below)
    ((0.20, 0.0, 1.44), (0.26, 0.0, 1.14), 0.045),
    ((0.26, 0.0, 1.14), (0.29, 0.0, 0.84), 0.038),
    ((0.29, 0.0, 0.84), (0.30, 0.0, 0.72), 0.042),
    # thigh, shank, foot
    ((0.09, 0.0, 0.92), (0.105, 0.0, 0.50), 0.072),
    ((0.105, 0.0, 0.50), (0.105, 0.0, 0.10), 0.052),
    ((0.105, 0.0, 0.06), (0.105, 0.135, 0.05), 0.045),
]

_ELLIPSOIDS = [
    # head
    ((0.0, 0.01, 1.66), (0.095, 0.11, 0.115)),
    # chest
    ((0.0, 0.0, 1.30), (0.165, 0.105, 0.19)),
    # abdomen
    ((0.0, 0.0, 1.06), (0.145, 0.10, 0.15)),
    # pelvis
    ((0.0, 0.0, 0.92), (0.16, 0.105, 0.11)),
]

# Organ demand hotspots: (center, semi-axes, share of whole-body VO2).
# Shares: brain ~20%, heart ~10%, liver+gut ~27%, kidneys ~7%; remainder
# (muscle, skin, bone, fat, ~36%) is spread uniformly over all tissue.
_ORGANS = [
    (np.array([0.0, 0.01, 1.66]), np.array([0.08, 0.09, 0.09]), 0.20),   # brain
    (np.array([0.02, 0.02, 1.32]), np.array([0.055, 0.05, 0.055]), 0.10), # heart
    (np.array([-0.07, 0.02, 1.16]), np.array([0.08, 0.07, 0.07]), 0.17),  # liver
    (np.array([0.05, -0.04, 1.08]), np.array([0.035, 0.03, 0.055]), 0.035),  # kidney R
    (np.array([-0.05, -0.04, 1.08]), np.array([0.035, 0.03, 0.055]), 0.035), # kidney L
    (np.array([0.0, 0.03, 1.00]), np.array([0.10, 0.07, 0.10]), 0.10),    # gut
]


@dataclass
class Body:
    voxel: float
    origin: np.ndarray                 # (3,) lower corner of grid
    shape: tuple                       # (nx, ny, nz)
    mask: np.ndarray                   # bool (nx, ny, nz) tissue voxels
    demand: np.ndarray                 # (nx, ny, nz) O2 demand per voxel [m^3 O2/s]
    tissue_idx: np.ndarray = field(default=None)   # (nt, 3) int coords of tissue voxels
    tissue_pts: np.ndarray = field(default=None)   # (nt, 3) centres in metres

    def __post_init__(self):
        self.tissue_idx = np.argwhere(self.mask)
        self.tissue_pts = self.origin + (self.tissue_idx + 0.5) * self.voxel

    def point_to_voxel(self, pts: np.ndarray) -> np.ndarray:
        return np.floor((pts - self.origin) / self.voxel).astype(int)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        ijk = self.point_to_voxel(np.atleast_2d(pts))
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        res = np.zeros(len(ijk), dtype=bool)
        res[ok] = self.mask[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return res

    @property
    def tissue_volume(self) -> float:
        return self.mask.sum() * self.voxel ** 3


def build_body(voxel: float = VOXEL) -> Body:
    nx = int(round((XLIM[1] - XLIM[0]) / voxel))
    ny = int(round((YLIM[1] - YLIM[0]) / voxel))
    nz = int(round((ZLIM[1] - ZLIM[0]) / voxel))
    origin = np.array([XLIM[0], YLIM[0], ZLIM[0]])

    xs = origin[0] + (np.arange(nx) + 0.5) * voxel
    ys = origin[1] + (np.arange(ny) + 0.5) * voxel
    zs = origin[2] + (np.arange(nz) + 0.5) * voxel
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    inside = np.zeros(len(pts), dtype=bool)
    for a, b, r in _CAPSULES:
        a, b = np.array(a), np.array(b)
        for sgn in (1.0, -1.0):
            s = np.array([sgn, 1.0, 1.0])
            inside |= _capsule_dist(pts, a * s, b * s) <= r
    for c, semi in _ELLIPSOIDS:
        inside |= _ellipsoid_in(pts, np.array(c), np.array(semi))

    mask = inside.reshape(nx, ny, nz)

    # Demand field: uniform background + organ hotspots, normalized to VO2.
    demand = np.zeros(len(pts))
    organ_total = sum(s for _, _, s in _ORGANS)
    background_share = 1.0 - organ_total
    demand[inside] = background_share
    for c, semi, share in _ORGANS:
        sel = _ellipsoid_in(pts, c, semi) & inside
        if sel.sum():
            demand[sel] += share * inside.sum() / sel.sum()
    demand *= VO2_REST / demand.sum()
    demand = demand.reshape(nx, ny, nz)

    return Body(voxel=voxel, origin=origin, shape=(nx, ny, nz),
                mask=mask, demand=demand)
