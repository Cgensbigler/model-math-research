"""Anatomical reference model of the human arterial tree.

Two layers:

1. The named major arteries (aorta through tibial/radial level) are
   hand-placed from anatomical atlases, with published radii, inside the
   same voxel body used by the evolutionary runs.
2. Because measured metrics such as fractal dimension require several
   more generations of branching, the tree is extended distally from
   the major-artery endpoints with the same space-colonization engine,
   using *literature-canonical* parameters (Murray exponent 2.7,
   terminal radius 0.8 mm, demand-weighted targets). This infill is a
   model, not data; metric comparisons also show pure literature ranges
   (constants.LITERATURE) as shaded bands.

Radii of named segments keep their published anatomical values; the
infilled distal segments follow Murray's law from the terminal radius.
"""

from __future__ import annotations

import numpy as np

from .body import Body, HEART_POS
from .growth import colonize, sample_attractors
from .tree import VascularTree

# name: (parent_name, end_xyz, radius_m). Positions in body coordinates.
# Radii from standard tables (Gray's Anatomy; Westerhof et al., Snapshots
# of Hemodynamics; arterial 1D model datasets).
MAJOR_ARTERIES = {
    "aortic_root":        (None,              (0.02, 0.00, 1.36), 0.0125),
    "aortic_arch":        ("aortic_root",     (-0.01, -0.02, 1.42), 0.0110),
    "thoracic_aorta":     ("aortic_arch",     (-0.01, -0.04, 1.22), 0.0100),
    "abdominal_aorta":    ("thoracic_aorta",  (0.00, -0.03, 1.00), 0.0080),
    "aortic_bifurcation": ("abdominal_aorta", (0.00, -0.02, 0.94), 0.0075),

    # head & neck
    "brachiocephalic":    ("aortic_arch",     (0.03, -0.01, 1.47), 0.0060),
    "r_common_carotid":   ("brachiocephalic", (0.025, 0.00, 1.58), 0.0037),
    "r_carotid_head":     ("r_common_carotid", (0.035, 0.015, 1.68), 0.0025),
    "l_common_carotid":   ("aortic_arch",     (-0.025, 0.00, 1.58), 0.0037),
    "l_carotid_head":     ("l_common_carotid", (-0.035, 0.015, 1.68), 0.0025),

    # arms
    "r_subclavian":       ("brachiocephalic", (0.13, 0.00, 1.45), 0.0040),
    "r_axillary":         ("r_subclavian",    (0.21, 0.00, 1.42), 0.0030),
    "r_brachial":         ("r_axillary",      (0.255, 0.00, 1.15), 0.0021),
    "r_radial":           ("r_brachial",      (0.285, 0.01, 0.86), 0.0016),
    "r_ulnar":            ("r_brachial",      (0.275, -0.01, 0.86), 0.0017),
    "l_subclavian":       ("aortic_arch",     (-0.13, 0.00, 1.45), 0.0040),
    "l_axillary":         ("l_subclavian",    (-0.21, 0.00, 1.42), 0.0030),
    "l_brachial":         ("l_axillary",      (-0.255, 0.00, 1.15), 0.0021),
    "l_radial":           ("l_brachial",      (-0.285, 0.01, 0.86), 0.0016),
    "l_ulnar":            ("l_brachial",      (-0.275, -0.01, 0.86), 0.0017),

    # viscera
    "celiac":             ("thoracic_aorta",  (0.00, 0.01, 1.17), 0.0035),
    "hepatic":            ("celiac",          (-0.06, 0.02, 1.15), 0.0022),
    "splenic":            ("celiac",          (0.06, 0.01, 1.13), 0.0022),
    "sup_mesenteric":     ("abdominal_aorta", (0.00, 0.03, 1.02), 0.0040),
    "r_renal":            ("abdominal_aorta", (0.05, -0.04, 1.08), 0.0026),
    "l_renal":            ("abdominal_aorta", (-0.05, -0.04, 1.08), 0.0026),

    # legs
    "r_common_iliac":     ("aortic_bifurcation", (0.05, -0.01, 0.88), 0.0060),
    "r_femoral":          ("r_common_iliac",  (0.095, 0.00, 0.75), 0.0040),
    "r_sup_femoral":      ("r_femoral",       (0.105, -0.01, 0.52), 0.0033),
    "r_popliteal":        ("r_sup_femoral",   (0.105, -0.02, 0.42), 0.0025),
    "r_ant_tibial":       ("r_popliteal",     (0.105, 0.01, 0.12), 0.0015),
    "r_post_tibial":      ("r_popliteal",     (0.10, -0.02, 0.12), 0.0016),
    "l_common_iliac":     ("aortic_bifurcation", (-0.05, -0.01, 0.88), 0.0060),
    "l_femoral":          ("l_common_iliac",  (-0.095, 0.00, 0.75), 0.0040),
    "l_sup_femoral":      ("l_femoral",       (-0.105, -0.01, 0.52), 0.0033),
    "l_popliteal":        ("l_sup_femoral",   (-0.105, -0.02, 0.42), 0.0025),
    "l_ant_tibial":       ("l_popliteal",     (-0.105, 0.01, 0.12), 0.0015),
    "l_post_tibial":      ("l_popliteal",     (-0.10, -0.02, 0.12), 0.0016),
}

# Canonical (literature-tuned) growth parameters for the distal infill.
CANONICAL_INFILL = {
    "influence": 0.14,
    "kill": 0.030,
    "step": 0.025,
    "momentum": 0.25,
    "jitter": 0.08,
    "down_bias": 0.0,
    "murray_gamma": 2.7,
    "r_terminal": 8e-4,
}
INFILL_ATTRACTORS = 2600


def build_reference_tree(body: Body, seed: int = 0,
                         infill: bool = True) -> tuple[VascularTree, dict]:
    """Returns (tree, info) where info maps node index -> artery name for
    the named segments."""
    rng = np.random.default_rng(seed)
    names = list(MAJOR_ARTERIES.keys())
    name_to_idx = {}

    pos = [HEART_POS.copy()]
    parent = [-1]
    direction = [np.array([0.0, 0.0, 1.0])]
    named_radius = {0: 0.0125}

    for nm in names:
        pnm, end, r = MAJOR_ARTERIES[nm]
        pidx = 0 if pnm is None else name_to_idx[pnm]
        end = np.array(end)
        d = end - pos[pidx]
        d /= np.linalg.norm(d) + 1e-12
        pos.append(end)
        parent.append(pidx)
        direction.append(d)
        idx = len(pos) - 1
        name_to_idx[nm] = idx
        named_radius[idx] = r

    n_named = len(pos)

    if infill:
        g = dict(CANONICAL_INFILL)
        attractors = sample_attractors(body, INFILL_ATTRACTORS, rng)
        colonize(pos, parent, direction, attractors, g, body, rng,
                 max_nodes=6000, max_iter=400)

    tree = VascularTree(pos=np.asarray(pos), parent=np.asarray(parent, dtype=int))
    tree.assign_radii(CANONICAL_INFILL["murray_gamma"],
                      CANONICAL_INFILL["r_terminal"])
    # Named majors keep their published radii where larger than Murray value
    for idx, r in named_radius.items():
        tree.radius[idx] = max(tree.radius[idx], r)
    tree.assign_flows(body)

    info = {"n_named": n_named,
            "name_to_idx": name_to_idx}
    return tree, info
