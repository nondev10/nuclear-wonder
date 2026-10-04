"""
yield.py - burst spectrum, temperature readout, and blast-zone scaling
======================================================================

The labeled-output layer of the device transient. Everything here is a
*report* quantity, not a fed-back control: the transient uses these to
show the reader what the burst looks like, and they are explicitly
labeled order-of-magnitude / illustrative, per the educational scope.

Reuses the Terrell LA-11807 spectrum constants and the LA-4208 densities
from sim.neutronics.materials; the blast-zone scaling is the cube-root
(yield) scaling of the historical "Documentation and Diagrams of the
Atomic Bomb" document's blast zones, labeled illustrative.
"""

import numpy as np

from ..neutronics import materials as M

# Fission-product heat per fission, attributed to the *fissile core* as a
# lumped temperature rise. Lumped specific heat (J/kg/K) of fissioned
# uranium/plutonium at the relevant temperatures (~labeled constant; real
# values are strongly T- and composition-dependent).
CORE_Cp_J_PER_KG_K = 2.0e3
# Specific yield: 1 kg of fissile material ~ 4.3e12 J (200 MeV/fission,
# A~236). ~1 kt TNT per kg is the standard rough conversion.
SPECIFIC_YIELD_J_PER_KG = 3.0e13


def prompt_spectrum(mat):
    """Prompt fission-neutron spectrum parameters (Terrell LA-11807).

    Returns a dict describing the Maxwellian fission spectrum N(E) =
    E*exp(-E/T) with T = (2/3)*E_avg, plus the prompt gamma yield.
    These are the *source* terms of a burst; the transient reports them
    as a diagnostic, not a transport solve.
    """
    eavg = M.FIS_NEUTRON_EAVG_MEV.get(mat, 2.0)
    T = M.MAXWELL_T_MEV.get(mat, (2.0 / 3.0) * eavg)
    return {
        "material": mat,
        "T_maxwellian_MeV": float(T),
        "E_avg_MeV": float(eavg),
        "nu_fast": M.NU_FAST.get(mat, 2.6),
        "gamma_per_fission": M.GAMMA_PER_FISSION.get(mat, 0.36),
        "gamma_E_avg_MeV": M.GAMMA_LEAK_EAVG_MEV.get(mat, 1.36),
        "neutron_leak_frac": M.NEUTRON_LEAK_FRAC.get(mat, 0.578),
    }


def prompt_gamma_yield(n_fissions, mat):
    """Prompt gamma photons & energy for n_fissions (Terrell source term)."""
    gp = M.GAMMA_PER_FISSION.get(mat, 0.36)
    eg = M.GAMMA_LEAK_EAVG_MEV.get(mat, 1.36) * M.MEV_TO_J
    return {"n_gamma": n_fissions * gp, "E_gamma_J": n_fissions * gp * eg}


def temperature_readout(Y_J, M_kg, mat):
    """Lumped core temperature rise (K) from fission-product heat.

    T rise = Y_fissile / (m_fissile * Cp). This is a *report* readout of
    how hot the core gets - it is NOT fed back into reactivity in this
    model (out of scope). Labeled lumped approximation.
    """
    Y_J = np.asarray(Y_J, dtype=float)
    m = max(M_kg, 1e-9)
    return Y_J / (m * CORE_Cp_J_PER_KG_K)


def yield_kt(Y_J):
    """Joules -> kilotons TNT (illustrative yield readout)."""
    return float(np.asarray(Y_J, dtype=float)) / M.TNT_J_PER_KT


# ----------------------------------------------------------------------
#  Blast-zone cube-root scaling (illustrative, from the reference doc)
# ----------------------------------------------------------------------
# Radii (m) for the ~10 kt reference (Hiroshima/Little-Boy-class) zones,
# from the "Documentation and Diagrams of the Atomic Bomb" blast-zone
# figures. Cube-root scaling: r(Y) = r_ref * (Y/10 kt)^(1/3). Labeled.
BLAST_ZONES_10KT_M = [
    ("vaporization",      800.0),   # ~0.5 mi radius total-vaporization
    ("total destruction", 1600.0),  # ~1 mi
    ("severe blast",      3200.0),  # ~2 mi
    ("fireball edge",     4800.0),  # ~3 mi
]


def blast_zones(yield_kt, ref_kt=10.0):
    """Cube-root-scaled blast-zone radii (m) for a yield of yield_kt.

    r_zone = r_10kt * (yield_kt / 10)^(1/3). Labeled illustrative - the
    zone radii are the reference-document order of magnitude, not a
    full blast/overpressure solve.
    """
    scale = (yield_kt / ref_kt) ** (1.0 / 3.0)
    return {name: {"radius_m": r * scale, "radius_km": r * scale / 1000.0}
            for name, r in BLAST_ZONES_10KT_M}


def all_outputs(res, yield_kt=None):
    """Diagnostic dump: spectrum, gamma, temperature, and blast zones."""
    mat = res["material"]
    M_max = res["M_kg"].max()
    if yield_kt is None:
        yield_kt = res["Y_max_kt"]
    # n_fissions = (fission fraction) * (available atoms in the assembled mass)
    m = M.get(mat)
    n_atoms = M_max * 1000.0 / m["A"] * M.NA          # kg -> g -> moles -> atoms
    n_fissions = int(res["f_disassembly"] * n_atoms)
    return {
        "material": mat,
        "prompt_spectrum": prompt_spectrum(mat),
        "n_fissions_est": n_fissions,
        "yield_kt_illustrative": float(yield_kt),
        "blast_zones": blast_zones(yield_kt),
        "T_peak_K": float(np.max(res["T_K"])),
    }
