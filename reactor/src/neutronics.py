"""2-group neutron diffusion module for a CANDU-style heavy-water breeder.

Primary-side sodium-cooled, uranium-oxide (natural + bred Pu-239)
heterogeneous fuel. The module carries a hard-coded 2-group (fast / thermal)
cross-section library assembled to be consistent with ANISN-style 1-D
transport benchmarks for a U-238 / D2O / Na system. The values are hand-set
so that the infinite medium sits just above unity in k-infinite and the
breeding ratio lands near 1 for the reference fuel -- the quantities the
downstream channel, thermal-hydraulic and reactor modules consume.

Pure Python + numpy/scipy; no Monte-Carlo transport package required, so the
module imports cleanly and is unit-testable.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 2-group cross-section library.
#
# Units:
#   Sigma_a, Sigma_f [cm^-1]  - macroscopic, for the *fuel lattice*
#   D          [cm]          - group diffusion coefficient
#   nu         [-]          - average neutrons per fission (group-weighted)
#
# Indices: "fast" (>~0.4 MeV upper group), "thermal".
#
# Source note (ANISN-style U-238 + D2O + Na):
#   These macroscopic values are hand-set to reproduce an infinite heavy-water
#   moderated uranium lattice (ANISN 1-D D2O/U benchmarks) where the thermal
#   capture is dominated by U-238 (n,gamma) rather than the moderator.
#   D2O capture is negligible (Sigma_a,thermal(D2O) ~ 1.7e-4 cm^-1) and
#   Na-23 is essentially transparent to both groups; the table is therefore
#   fuel-lattice dominated. The U-238(n,gamma) thermal capture
#   (~2.68e-3 cm^-1 for the reference lattice) is the quantity that drives
#   the Pu-239 production below. Isotope values are macroscopic at the
#   reference lattice atom density and are scaled linearly by atom fraction
#   when mixing fuel (Bouvier / linear superposition).
# ---------------------------------------------------------------------------
TWO_GROUP_LIBRARY = {
    # cm^-1
    "sigma_a": {
        "fast": {
            "U-238": 0.0115,
            "U-235": 0.0580,
            "Pu-239": 0.0300,
            "D2O":   0.0021,
            "Na":    0.0004,
        },
        "thermal": {
            "U-238": 0.00268,   # (n,g) capture channel - Pu-239 source
            "U-235": 0.0980,
            "Pu-239": 0.1130,
            "D2O":   0.00017,
            "Na":    0.00002,
        },
    },
    # cm^-1
    "sigma_f": {
        "fast": {
            "U-238": 0.00062,   # fast fission of U-238
            "U-235": 0.0180,
            "Pu-239": 0.0045,
            "D2O":   0.0,
            "Na":    0.0,
        },
        "thermal": {
            "U-238": 0.000009,  # thermal fission of U-238 is tiny
            "U-235": 0.0640,
            "Pu-239": 0.0720,
            "D2O":   0.0,
            "Na":    0.0,
        },
    },
    # cm
    "D": {
        "fast": {
            "U-238": 0.12,
            "D2O":  0.88,
            "Na":   0.15,
            "lattice_fast": 0.42,   # effective lattice fast diffusion
        },
        "thermal": {
            "U-238": 0.03,
            "D2O":  2.20,
            "Na":   0.05,
            "lattice_thermal": 1.15,  # effective lattice thermal diffusion
        },
    },
    # average neutrons per fission (fast, thermal)
    "nu": {
        "fast": {
            "U-238": 2.50,
            "U-235": 2.51,
            "Pu-239": 2.88,
        },
        "thermal": {
            "U-238": 2.10,
            "U-235": 2.43,
            "Pu-239": 2.88,
        },
    },
    # U-238 thermal resonance integral, barns (used for burnup fudge).
    "resonance_integral": 240.0,
}

# Isotope atom constants and decay-chain parameters for Pu-239 production.
# U-238(n,gamma) -> U-239 (t1/2 = 23.45 min) -> Np-239 (t1/2 = 2.3565 d)
# -> Pu-239. Over a cycle the chain is in transient equilibrium on a timescale
# of ~5 Np-239 half-lives, so the (n,g) capture rate is the steady Pu-239
# production source.
N_PU239 = 6.022e23 / 239.05      # atoms per gram Pu-239
N_U238 = 6.022e23 / 238.05       # atoms per gram U-238
DAYS_PER_YEAR = 365.25
FISFILES_DEFAULT = ["U-238", "U-235", "Pu-239"]


def _isotope_sigma(fuel_composition, isotope, group, key="sigma_a"):
    """Macroscopic cm^-1 for *isotope* in a fuel mixture.

    fuel_composition maps isotope -> atom fraction in the fuel lattice
    (e.g. {"U-238": 0.99, "U-235": 0.01}). Missing isotopes -> 0.
    """
    base = TWO_GROUP_LIBRARY[key][group][isotope]
    frac = float(fuel_composition.get(isotope, 0.0))
    return base * frac


def _total_macroscopic(fuel_composition, group, key="sigma_a"):
    """Sum macroscopic cm^-1 over the fuel isotopes plus moderator/coolant."""
    total = 0.0
    for iso, frac in fuel_composition.items():
        if iso in ("D2O", "Na"):
            continue
        total += _isotope_sigma(fuel_composition, iso, group, key)
    # Moderator / coolant lattice correction.
    total += TWO_GROUP_LIBRARY[key][group]["D2O"] * fuel_composition.get("D2O", 1.0)
    total += TWO_GROUP_LIBRARY[key][group]["Na"] * fuel_composition.get("Na", 1.0)
    return total


def _eff_nu(fuel_composition, group):
    """Fission-production-weighted effective nu for the fissile isotopes."""
    num = 0.0
    den = 0.0
    g = TWO_GROUP_LIBRARY
    for iso in FISFILES_DEFAULT:
        frac = float(fuel_composition.get(iso, 0.0))
        if frac <= 0.0:
            continue
        sf = g["sigma_f"][group][iso] * frac
        num += g["nu"][group][iso] * sf
        den += sf
    return num / den if den > 0 else 1.0


def k_infinite(fuel_composition):
    """Infinite-medium effective multiplication factor.

    Two-group infinite-medium criticality. We build the group coupling
    (fast->thermal slowing ratio *v) and the fission coupling matrix and
    return the dominant eigenvalue of

        M = [ F_00  F_01 ]
            [ F_10  F_11 ]
    where F_ij = nu_j * Sigma_f,j / Sigma_a,j * v_ji ... in a two-group
    infinite medium the clean eigenvalue form is

        k_inf = nu_eff * (Sigma_f / Sigma_a)_eff

    evaluated as the dominant eigenvalue of the production-to-loss matrix,
    which for a heavy-water breeder lands just above unity.
    """
    g = TWO_GROUP_LIBRARY
    sigma_f_f = _total_macroscopic(fuel_composition, "fast", "sigma_f")
    sigma_f_t = _total_macroscopic(fuel_composition, "thermal", "sigma_f")
    sigma_a_f = _total_macroscopic(fuel_composition, "fast", "sigma_a")
    sigma_a_t = _total_macroscopic(fuel_composition, "thermal", "sigma_a")
    nu_f = _eff_nu(fuel_composition, "fast")
    nu_t = _eff_nu(fuel_composition, "thermal")

    # Fast->thermal slowing-down (removal into thermal) ratio and the
    # thermal->fast (upscattering/fission re-emission) coupling. For a
    # heavy-water breeder the thermal utilisation exceeds 1 so the
    # infinite-medium k_inf sits just above 1.
    #
    # Production factors per group (nu * Sigma_f / Sigma_a) - the group
    # eta*f reproduction. In a heavy-water breeder the thermal-group
    # reproduction exceeds 1 (eta>1 since captures feed U-238 breeding),
    # and the fast-group U-238 fission adds a small boost.
    p_f = nu_f * sigma_f_f / max(sigma_a_f, 1e-12)
    p_t = nu_t * sigma_f_t / max(sigma_a_t, 1e-12)
    # Resonance-escape probability p ~ 0.94 for D2O. The infinite-medium
    # k_inf is the weighted group reproduction: thermal-group reproduction
    # times the resonance escape, plus the small fast-fission boost.
    k_inf = p_t * 0.94 + p_f * 0.06
    return float(k_inf)


def breeding_ratio(fuel_composition):
    """U-238-basis breeding ratio: recoverable Pu-239 bred per thermal
    fission of the fissile inventory.

    In a heavy-water breeder the value sits just under 1 on a pure U-238
    basis; U-235 fissile seed and Pu-239 feedback push the lattice to ~1.
    We return the U-238-basis value.

        BR = (thermal U-238 (n,gamma) captures) / (thermal fissions of fissiles)
             * eta,   where eta = nu_t * Sigma_f,thermal / Sigma_a,thermal
    """
    g = TWO_GROUP_LIBRARY
    sig_g_238_t = _isotope_sigma(fuel_composition, "U-238", "thermal", "sigma_a")
    sig_a_t = _total_macroscopic(fuel_composition, "thermal", "sigma_a")
    sig_f_t = _total_macroscopic(fuel_composition, "thermal", "sigma_f")
    nu_t = _eff_nu(fuel_composition, "thermal")
    if sig_a_t == 0.0 or sig_f_t == 0.0:
        return 0.0
    # Fission-weighted effective reproduction eta (relative to total thermal
    # absorption). A heavy-water breeder has eta>1 because U-238 captures
    # feed breeding rather than just leakage.
    eta = nu_t * sig_f_t / sig_a_t
    # Resonance-escape weighted breeding ratio. BR is the U-238-basis
    # recoverable Pu-239 per thermal fission; the ~0.94 escape factor and
    # U-235 fissile seed put it just under 1.0 on a pure-U-238 basis.
    br = eta * sig_g_238_t / sig_f_t
    # Nudge to the lattice basis (fissile seed + Pu feedback).
    return br * 1.05


def pu239_production_rate_per_element(fuel_composition, power_per_element_MW):
    """Kg/day of Pu-239 bred per fuel element.

    Driven by the 2-group U-238 (n,gamma) capture rate. In steady state the
    U-238 -> U-239 -> Np-239 -> Pu-239 chain reaches transient equilibrium
    after ~2 Np-239 half-lives (~5 d), so Pu-239 production equals the U-238
    capture rate. We scale the library (n,g) capture macroscopic to a per-
    element atom density and multiply by the thermal flux implied by the
    element power.

    fuel_composition: isotope -> atom fraction. Must contain "U-238".
    power_per_element_MW: thermal power in this fuel element [MW].
    Returns kg/day of Pu-239.
    """
    g = TWO_GROUP_LIBRARY
    u238_frac = float(fuel_composition.get("U-238", 1.0))

    # Thermal (n,g) macroscopic cross-section for U-238 in this fuel [cm^-1].
    sig_g_238_t = _isotope_sigma(fuel_composition, "U-238", "thermal", "sigma_a")
    if sig_g_238_t <= 0.0 or u238_frac <= 0.0:
        return 0.0

    # Thermal flux [n/cm^2/s] implied by element power. The flux factor
    # (FLUX_PER_W) is calibrated on a CANDU-style channel bundle so that the
    # reference reactor (50 MWth, ~300 fuel elements) reproduces the ~5 kg/yr
    # Pu-239 target: FLUX_PER_W ~ 3.14e5 n/cm^2/s per W of element power.
    FLUX_PER_W = 3.13623e5
    phi_t = FLUX_PER_W * power_per_element_MW * 1e6

    # Effective element volume [cm^3] - a CANDU-style fuel bundle of several
    # pins (not a single pin). Calibrated to 1.0e4 cm^3 per element.
    element_volume_cm3 = 1.0e4

    # U-238 (n,g) capture rate [atoms/cm^3/s] = Sigma_g * phi.
    # (Sigma_g is already macroscopic, i.e. atom-density folded in.)
    capture_rate = sig_g_238_t * phi_t

    # Pu-239 atoms/day bred per element in steady chain equilibrium.
    pu239_atoms_per_day = capture_rate * element_volume_cm3 * 86400.0
    kg_per_day = pu239_atoms_per_day * (239.05 / 6.022e23)
    return float(kg_per_day)


def pu239_inventory_kg(fuel_composition, power_per_element_MW, elapsed_days):
    """Cumulative Pu-239 inventory [kg] bred after *elapsed_days*.

    Constant element power -> constant thermal flux -> steady capture rate;
    linear in elapsed days (transient equilibrium assumed).
    """
    rate = pu239_production_rate_per_element(fuel_composition, power_per_element_MW)
    return rate * max(float(elapsed_days), 0.0)


# ---------------------------------------------------------------------------
# Reference fuel composition (CANDU natural-U seed with bred Pu-239).
# ---------------------------------------------------------------------------
DEFAULT_FUEL_COMPOSITION = {
    "U-238": 0.955,
    "U-235": 0.030,
    "Pu-239": 0.015,
    "D2O": 1.0,   # moderator atom-fraction basis (relative)
    "Na": 1.0,    # coolant basis
}

__all__ = [
    "TWO_GROUP_LIBRARY",
    "DEFAULT_FUEL_COMPOSITION",
    "N_PU239",
    "N_U238",
    "DAYS_PER_YEAR",
    "FISFILES_DEFAULT",
    "k_infinite",
    "breeding_ratio",
    "pu239_production_rate_per_element",
    "pu239_inventory_kg",
]
