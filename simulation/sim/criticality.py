"""
criticality.py - public facade over the multi-group neutronics engine
=====================================================================

Replaces the old curve-fitted one-group model. The physics now lives in
``sim.neutronics`` (calibrated 2-group sphere engine + point kinetics).
This module keeps the historical public API so the CLI, tests, and the
interactive blueprint keep working, re-parameterized on the *derived*
quantities:

    reactivity(R, mat)          -> rho from k_eff_sphere (derived, not assumed)
    fission_yield(R, mat, t..)  -> Terrell-grounded energy / fission fraction
    chain_growth(R, mat, t..)   -> point kinetics on the spatial model
    all_outputs(R, mat, t..)   -> full diagnostic dump

The old signature used mass M directly. Here the physical input is the
sphere radius R (cm); mass is recovered from R via the material density.
``M_BARE_CRIT`` is now the *calibrated* measured value, not an invention.
"""

import numpy as np

try:
    from .neutronics import sphere as S
    from .neutronics import kinetics as K
    from .neutronics import materials as M
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim.neutronics import sphere as S
    from sim.neutronics import kinetics as K
    from sim.neutronics import materials as M

# Calibrated bare critical masses (kg) - now measured LA-4208 values.
M_BARE_CRIT = M.MEASURED_MCRIT["u235"]["mcrit_kg"]  # 51.94 kg U-235


def mass_of_radius(R, material="u235"):
    """Fissile mass (kg) of a sphere of radius R (cm)."""
    m = M.get(material)
    return (4.0 / 3.0) * np.pi * m["rho"] * R**3 / 1000.0


def radius_of_mass(M_kg, material="u235"):
    """Sphere radius (cm) for a fissile mass M_kg.

    M_kg (kg) -> M_g (g); R^3 = 3*M_g/(4*pi*rho) with rho in g/cm^3, R in cm.
    """
    m = M.get(material)
    M_g = M_kg * 1000.0
    return (3.0 * M_g / (4.0 * np.pi * m["rho"])) ** (1.0 / 3.0)


def reactivity(R, material="u235", group="2g"):
    """Derived reactivity rho(R) = (k-1)/k for a sphere of radius R (cm).

    (The old API took a mass; pass radius, or use radius_of_mass.)
    """
    return S.rho_reactivity(R, material, group)


def critical_mass(material="u235", reflector_credit=0.0, group="2g"):
    """Bare critical mass (kg) - the *measured* calibrated value.

    ``reflector_credit`` is kept for API compatibility but, for a bare
    sphere, returns the calibrated bare mass regardless (the reflector
    effect is handled by the dedicated reflector module).
    """
    return S.Mcrit(material, group)


def runaway_time(R, material="u235", beta=0.0, group="2g"):
    """Prompt-generation timescale for a super-critical sphere of radius R."""
    rho = float(S.rho_reactivity(R, material, group)[0])
    if rho <= beta:
        return K.prompt_lifetime(material, R)
    return K.prompt_period(material, R, rho, beta)


def chain_growth(R, material="u235", t_array=None, t_max=2e-6, group="2g"):
    """Point-kinetics neutron population N(t). Returns the kinetics dict."""
    if t_array is None:
        t_array = np.linspace(0.0, t_max, 200)
    return K.chain_growth(material, R, t_array)


def fission_yield(R, material="u235", t_array=None, t_exp=None,
                  beta=0.0, group="2g"):
    """Cumulative fission-energy release and fission fraction for a sphere.

    Sub-critical -> steady power; super-critical -> explosive yield capped by
    the labeled disassembly time t_exp (hydrodynamic estimate, NOT a device
    yield).
    """
    if t_array is None:
        t_array = np.linspace(0.0, 2e-6, 200)
    return K.fission_yield(material, R, t_array, t_exp=t_exp, beta=beta)


def plot_criticality_curve(R_array=None, material="u235", group="2g"):
    """The (R, rho, k) curve used by the HTML what-if slider."""
    Rc = S.R_critical(material, group)
    if R_array is None:
        R_array = np.linspace(0.4 * Rc, 1.6 * Rc, 200)
    rho = S.rho_reactivity(R_array, material, group)
    k = S.k_eff_sphere(R_array, material, group)
    return {"R": R_array, "rho": rho, "k_eff": k}


def all_outputs(R, material="u235", t_array=None, t_max=2e-6, group="2g"):
    """Full diagnostic dump for a sphere of radius R (cm)."""
    if t_array is None:
        t_array = np.linspace(0.0, t_max, 200)
    rho = float(S.rho_reactivity(R, material, group)[0])
    return {
        "R": R,
        "material": material,
        "M_kg": mass_of_radius(R, material),
        "R_critical_cm": S.R_critical(material, group),
        "M_critical_kg": S.Mcrit(material, group),
        "k_eff": float(S.k_eff_sphere(R, material, group)[0]),
        "reactivity": rho,
        "runaway": runaway_time(R, material),
        "chain": K.chain_growth(material, R, t_array),
        "yield": K.fission_yield(material, R, t_array),
    }


if __name__ == "__main__":
    for mat in ("u235", "pu239", "u233"):
        Rc = S.R_critical(mat)
        print(f"{mat:6}: M_crit = {S.Mcrit(mat):6.2f} kg  (R_crit = {Rc:.2f} cm)")
    r = all_outputs(radius_of_mass(56.0, "u235"), "u235")
    print(f"\nU-235, M=56 kg (R={r['R']:.2f} cm): k={r['k_eff']:.4f}  "
          f"rho={r['reactivity']:+.4f}  chain={r['chain']['type']}")
