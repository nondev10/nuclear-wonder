"""
leakage.py — neutron & gamma leakage source terms
==================================================

Educational model of how much radiation leaks from a sub-critical assembly
before the neutron population reaches its sub-critical steady state.  This
is the physics that makes "leakage" a safety concept in fissile-mass
handling: a sub-critical mass still produces a neutron flux and a gamma
source from (n,γ) and spontaneous fission, and that source determines the
dose at a distance.

Grounding: LBL "Gamma-Ray and Neutron Leakage Spectra Calculated for
Unshielded Reactors" and "Reevaluated Critical Specifications of Some Los
Alamos Fast-Neutron Systems" — both treat leakage spectra for unshielded
fast-neutron assemblies.  We use a 1/r² inverse-square dose falloff with
an assembly-averaged source term, which is the right teaching-scale model.
"""
import numpy as np

# ----------------------------------------------------------------------
#  Source terms
# ----------------------------------------------------------------------
NEUTRONS_PER_FISSION = 2.9
SPONTANEOUS_FISSION_RATE = 30.0   # neutrons/s — thermal-equilibrium background
SPONTANEOUS_FISSION_PU = 4e4     # higher for Pu-239 (pedagogical)
GAMMA_PER_FISSION = 1.5          # (n,γ) + prompt gamma line source, J per fission
GAMMA_ENERGY_PER_FISSION = 3.0e-11  # ~1.5 MeV of gamma per fission in J


def steady_state_fission_rate(M, M_crit=52.0):
    """
    Fissions per second in a sub-critical assembly of mass M at steady state.

    For rho < 0 the point-kinetics steady state is:
        F_ss = N_source / |rho| * (1 - alpha)
    where N_source is the intrinsic source (spontaneous fission + (α,n))
    and alpha is the fission-neutron reproduction factor.  At M close to
    M_crit, |rho| → 0 so F_ss diverges — this is the "approach to
    criticality" that the LBL leakage-spectra papers track.
    """
    rho = _get_reactivity(M, M_crit=M_crit)
    if rho >= 0:
        return np.inf
    # Intrinsic source ~ spontaneous fission of the mass, scaled with M.
    S_eff = SPONTANEOUS_FISSION_RATE * (M / 52.0)
    return S_eff / abs(rho)


def _get_reactivity(M, M_crit=52.0):
    """Resolve reactivity(M) robustly whether imported as a package or run directly."""
    try:
        from . import criticality as _c
    except ImportError:
        import criticality as _c
    return float(_c.reactivity(M, M_crit=M_crit)[0])


def neutron_dose_at_distance(M, r, M_crit=52.0):
    """
    Neutron dose rate (arbitrary units, relative) at distance r from a
    sub-critical assembly of mass M.  Inverse-square falloff:
        D(r) ∝ F_ss / r²
    r is in meters; returns dose rate relative to a 1 m reference.
    """
    if r <= 0:
        return np.nan
    F = steady_state_fission_rate(M, M_crit=M_crit)
    if np.isinf(F):
        return np.inf
    return F * (1.0 / 1.0)**2 * 1.0 / (r**2)


def gamma_dose_at_distance(M, r, M_crit=52.0):
    """
    Gamma dose rate (relative units) from the fission + (n,γ) source.
    Same inverse-square structure; scaled by the gamma energy per fission.
    """
    if r <= 0:
        return np.nan
    F = steady_state_fission_rate(M, M_crit=M_crit)
    if np.isinf(F):
        return np.inf
    return F * GAMMA_ENERGY_PER_FISSION * GAMMA_PER_FISSION / (r**2)


def leakage_spectrum(r_array, M, M_crit=52.0):
    """
    (r, D_neutron, D_gamma) arrays for plotting the dose-vs-distance curve.
    """
    r_array = np.asarray(r_array, dtype=float)
    Dn = np.array([neutron_dose_at_distance(M, r, M_crit=M_crit) for r in r_array])
    Dg = np.array([gamma_dose_at_distance(M, r, M_crit=M_crit) for r in r_array])
    return {"r": r_array, "D_neutron": Dn, "D_gamma": Dg}


def dose_curve_over_mass(r_array, M_array, M_crit=52.0):
    """
    Dose at distance r_array as a function of assembly mass M_array.
    Useful for the "how dose rises as you approach criticality" plot.
    """
    M_array = np.asarray(M_array, dtype=float)
    r_array = np.asarray(r_array, dtype=float)
    Dn = np.array([[neutron_dose_at_distance(M, r, M_crit=M_crit)
                    for r in r_array] for M in M_array])
    return {"M": M_array, "r": r_array, "D_neutron": Dn}


if __name__ == "__main__":
    print(f"Sub-critical steady-state fission rate:")
    for M in [30, 40, 48, 51, 51.5]:
        F = steady_state_fission_rate(M)
        print(f"  M = {M:5.1f} kg : {F:.3e} fissions/s")
    print(f"\nNeutron dose @ 10 m for M = 51.5 kg: "
          f"{neutron_dose_at_distance(51.5, 10.0):.3e} (rel)")
