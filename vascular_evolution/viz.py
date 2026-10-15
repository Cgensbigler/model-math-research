"""3D and statistical visualization of vascular trees."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .body import Body
from .tree import VascularTree


def _segments(tree: VascularTree):
    a = tree.pos[tree.parent[1:]]
    b = tree.pos[1:]
    return np.stack([a, b], axis=1)


def draw_tree_3d(ax, tree: VascularTree, body: Body | None = None,
                 color_by: str = "radius", cmap: str = "turbo",
                 lw_scale: float = 900.0, body_alpha: float = 0.025,
                 elev: float = 10, azim: float = -75):
    segs = _segments(tree)
    r = tree.radius[1:]
    if body is not None:
        # light point-cloud silhouette of the body
        pts = body.tissue_pts[::7]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5,
                   c="#888888", alpha=body_alpha, linewidths=0)
    if color_by == "radius":
        cvals = np.log10(np.maximum(r, 1e-5) * 1e3)
    else:
        cvals = r
    lc = Line3DCollection(segs, cmap=cmap, linewidths=np.clip(r * lw_scale, 0.4, 7.0))
    lc.set_array(cvals)
    ax.add_collection3d(lc)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0.0, 1.8)
    ax.set_box_aspect((1, 1, 2))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return lc


def draw_tree_projection(ax, tree: VascularTree, body: Body | None = None,
                         plane=(0, 2), cmap: str = "turbo",
                         lw_scale: float = 700.0):
    i, j = plane
    segs = _segments(tree)[:, :, [i, j]]
    r = tree.radius[1:]
    if body is not None:
        pts = body.tissue_pts[::3]
        ax.scatter(pts[:, i], pts[:, j], s=1.0, c="#bbbbbb",
                   alpha=0.08, linewidths=0)
    lc = LineCollection(segs, cmap=cmap,
                        linewidths=np.clip(r * lw_scale, 0.3, 6.0))
    lc.set_array(np.log10(np.maximum(r, 1e-5) * 1e3))
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return lc
