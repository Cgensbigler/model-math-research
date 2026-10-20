"""Physiological and physical constants, with sources.

All values SI unless noted. Where a range exists in the literature the
mid-range value is used; the sensitivity analysis perturbs the fitness
weights derived from these numbers.
"""

# ---------------------------------------------------------------------------
# Blood / hemodynamics
# ---------------------------------------------------------------------------
BLOOD_VISCOSITY = 3.5e-3          # Pa*s, whole blood at high shear (3-4 mPa*s)
CARDIAC_OUTPUT = 5.0 / 60e3       # m^3/s  (5 L/min at rest)
MEAN_ARTERIAL_PRESSURE = 93 * 133.322   # Pa (~93 mmHg)
TOTAL_CARDIAC_POWER = CARDIAC_OUTPUT * MEAN_ARTERIAL_PRESSURE  # ~1.0 W

# Fraction of the total arterial pressure drop occurring in vessels larger
# than ~0.5 mm (the part of the tree we model explicitly). Most resistance
# lives in arterioles; large/medium arteries dissipate roughly 20-30%.
ARTERIAL_TREE_PRESSURE_FRACTION = 0.25
PUMP_POWER_REF = TOTAL_CARDIAC_POWER * ARTERIAL_TREE_PRESSURE_FRACTION  # ~0.26 W

# ---------------------------------------------------------------------------
# Oxygen transport
# ---------------------------------------------------------------------------
VO2_REST = 250e-6 / 60            # m^3 O2 / s  (250 mL/min whole-body)
BLOOD_O2_CONTENT = 0.20           # m^3 O2 per m^3 blood (20 mL O2/dL)
O2_DIFFUSIVITY_TISSUE = 2.0e-9    # m^2/s (Krogh; 1.5-2.4e-9)

# ---------------------------------------------------------------------------
# Anatomy (reference human, ~70 kg, 1.75 m)
# ---------------------------------------------------------------------------
BODY_HEIGHT = 1.75                # m
BODY_VOLUME = 0.066               # m^3 (~66 L, density ~1.05)
AORTIC_ROOT_RADIUS = 1.25e-2      # m (ascending aorta radius 1.2-1.5 cm)
ARTERIAL_BLOOD_VOLUME = 0.5e-3    # m^3 (arteries hold ~10-13% of 5 L blood)

# ---------------------------------------------------------------------------
# Mesoscale model bridge
# ---------------------------------------------------------------------------
# The simulated tree terminates at small arteries (~0.5-1 mm radius), each of
# which feeds an arteriolar/capillary bed perfusing a tissue territory of a
# few centimetres. We therefore use an *effective* diffusion model at the
# centimetre scale: terminal vessels act as nutrient sources with an
# effective penetration depth DELTA_PERFUSION, representing the reach of the
# microcirculation that a terminal branch supplies (small-artery supply
# territories in muscle/organ are typically 2-5 cm across).
DELTA_PERFUSION = 0.035           # m, effective source penetration depth

# Effective consumption / diffusivity ratio fixed by the penetration depth:
# for 1D diffusion with Dirichlet C0 at the source and zero-flux far field,
# concentration falls to zero at delta = sqrt(2 * D_eff * C0 / M_eff).
# We work with normalized concentration C0 = 1, so M_eff/D_eff = 2/delta^2.
C0_NORMALIZED = 1.0
CRITICAL_CONCENTRATION = 0.10     # fraction of C0 below which tissue is hypoxic
                                  # (venous-end PO2 ~ 1/10 arterial in Krogh model)

# ---------------------------------------------------------------------------
# Published morphometric values for the human arterial tree
# (used for comparison plots; see README for full citations)
# ---------------------------------------------------------------------------
LITERATURE = {
    # Fractal dimension of the arterial tree. Whole-organ vascular beds:
    # ~2.7 (theoretical space-filling), measured 2.2-2.9 across organs.
    "fractal_dimension": (2.7, (2.2, 2.9)),
    # Murray / radius-scaling exponent gamma in r_p^g = sum r_c^g.
    # ~3 for small arteries (Murray), ~2.55-2.76 measured for large arteries.
    "murray_exponent": (2.7, (2.3, 3.0)),
    # Strahler bifurcation ratio (coronary, pulmonary casts: 3.0-3.5).
    "bifurcation_ratio": (3.25, (2.5, 4.0)),
    # Mean daughters per junction (arterial trees are overwhelmingly binary).
    "mean_daughters": (2.05, (2.0, 2.3)),
    # Bifurcation asymmetry: mean r_small/r_large at junctions (~0.6-0.8).
    "radius_asymmetry": (0.7, (0.5, 0.9)),
    # Length/radius ratio of individual segments (arteries ~10-20).
    "length_to_radius": (14.0, (8.0, 25.0)),
}
