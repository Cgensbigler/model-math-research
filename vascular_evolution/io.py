"""Saving/loading trees and experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tree import VascularTree


def save_tree(path: str | Path, tree: VascularTree) -> None:
    np.savez_compressed(path, pos=tree.pos, parent=tree.parent,
                        radius=tree.radius, flow=tree.flow)


def load_tree(path: str | Path) -> VascularTree:
    d = np.load(path)
    return VascularTree(pos=d["pos"], parent=d["parent"],
                        radius=d["radius"], flow=d["flow"])


def save_json(path: str | Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=default))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())
