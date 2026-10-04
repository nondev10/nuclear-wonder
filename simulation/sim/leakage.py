"""
leakage.py - neutron & gamma leakage source terms (Terrell LA-11807 grounded)
=============================================================================

The educational leakage / dose layer. Source terms now come from the Terrell
MCNP leakage-spectra paper, NOT invented numbers:

* **Sub-critical fission rate.** For a sub-critical assembly the steady fission
  rate is multiplied by the neutron multiplication M = k/(1-k) (the LBL
  "M > 100" idiom, and the value 143->193 for the 6.8553-in shell assembly).
  k is the *model's* 2-group eigenvalue from the sphere engine, not an
  assumed number:
      F_ss = S_int * M          where M = k/(1-k),  k = k_eff_sphere(R)
  S_int is the intrinsic neutron source (spont. fission + (a,n)), set by the
  user per mass (default scales with fissile mass).

* **Neutron leakage.** Escaping neutrons per fission = NEUTRON_LEAK_FRAC
  (0.578 U-235, 0.676 Pu-239, Terrell). Inverse-square dose at distance r:
      D_n(r) = F_ss * leak_frac * E_avg_n / (4*pi*r^2)
  with the Maxwellian fission-spectrum average energy E_avg_n
  (2.047 / 2.159 MeV).

* **Gamma leakage.** Escaping prompt photons per fission = GAMMA_PER_FISSION
  (0.358 / 0.636), average energy ~1.36-1.39 MeV:
      D_g(r) = F_ss * gamma_per_fiss * E_avg_g / (4*pi*r^2)

Honest limit: this is a 1/r^2 point-source falloff with measured source
strengths - good for order-of-magnitude dose falloff, not a full photon
transport or detector spectrum. No shielding is modeled.

API-compatible with the old leakage.py (steady_state_fission_rate,
neutron_dose_at_distance, gamma_dose_at_distance, leakage_spectrum,
dose_curve_over_mass), re-parameterized on the new engine.
"""

import numpy as np

from .neutronics import materials as M
from .neutronics import sphere as S


def intrinsic_source(M_kg, material="u235"):
    """Default intrinsic neutron source (spont. fission + (a,n)), n/s.

    Scales with fissile mass. U-235 has a low spontaneous-fission
    background; Pu-239 (with 240Pu) has a much higher one. This is a
    teaching-scale value, not a measured source inventory.
    """
    base = {"u235": 3.0e2, "pu239": 4.0e4, "u233": 1.0e3}.get(material, 3.0e2)
    return base * (M_kg / 52.0)


def k_eff(R, material, group="2g"):
    """Model k_eff for a bare sphere of radius R (cm)."""
    return float(S.k_eff_sphere(R, material, group)[0])


def steady_state_fission_rate(R, material="u235", S_int=None, group="2g"):
    """Steady sub-critical fission rate (fissions/s) for a sphere of radius R.

    F_ss = S_int * M,  M = k/(1-k). Diverges as k->1 (approach to criticality).
    """
    M_kg = (4.0 / 3.0) * np.pi * M.get(material)["rho"] * R**3 / 1000.0
    if S_int is None:
        S_int = intrinsic_source(M_kg, material)
    k = k_eff(R, material, group)
    if k >= 1.0:
        return np.inf
    M_mult = k / (1.0 - k)
    return S_int * M_mult


def _gamma_energy_J(material):
    return M.GAMMA_LEAK_EAVG_MEV[material] * M.MEV_TO_J


def _neutron_energy_J(material):
    return M.FIS_NEUTRON_EAVG_MEV[material] * M.MEV_TO_J


def neutron_dose_at_distance(R, r_m, material="u235", S_int=None, group="2g"):
    """Neutron energy fluence (J/cm^2*s scale, relative) at distance r_m (m)."""
    if r_m <= 0:
        return np.nan
    F = steady_state_fission_rate(R, material, S_int=S_int, group=group)
    if np.isinf(F):
        return np.inf
    leak = M.NEUTRON_LEAK_FRAC.get(material, 0.578)
    E = _neutron_energy_J(material)
    # 4*pi*r^2 geometric falloff; r in cm
    return F * leak * E / (4.0 * np.pi * (r_m * 100.0)**2)


def gamma_dose_at_distance(R, r_m, material="u235", S_int=None, group="2g"):
    """Gamma energy fluence at distance r_m (m)."""
    if r_m <= 0:
        return np.nan
    F = steady_state_fission_rate(R, material, S_int=S_int, group=group)
    if np.isinf(F):
        return np.inf
    gp = M.GAMMA_PER_FISSION.get(material, 0.358)
    E = _gamma_energy_J(material)
    return F * gp * E / (4.0 * np.pi * (r_m * 100.0)**2)


def leakage_spectrum(r_array, R, material="u235", S_int=None, group="2g"):
    """(r, D_neutron, D_gamma) arrays for plotting the dose-vs-distance curve."""
    r_array = np.asarray(r_array, dtype=float)
    Dn = np.array([neutron_dose_at_distance(R, r, material, S_int, group)
                   for r in r_array])
    Dg = np.array([gamma_dose_at_distance(R, r, material, S_int, group)
                   for r in r_array])
    return {"r": r_array, "D_neutron": Dn, "D_gamma": Dg}


def dose_curve_over_mass(R_array, r_m, material="u235", S_int=None, group="2g"):
    """Neutron dose at a fixed distance r_m as a function of radius R_array."""
    R_array = np.asarray(R_array, dtype=float)
    Dn = np.array([neutron_dose_at_distance(R, r_m, material, S_int, group)
                   for R in R_array])
    return {"R": R_array, "r_m": r_m, "D_neutron": Dn}


def all_outputs(R, material="u235", r_array=None, S_int=None, group="2g"):
    """Diagnostic dump for leakage at a sphere of radius R (cm)."""
    if r_array is None:
        r_array = np.logspace(0.5, 3.0, 40)  # 3 m .. 1000 m
    Fss = steady_state_fission_rate(R, material, S_int=S_int, group=group)
    return {
        "R_cm": R,
        "material": material,
        "k_eff": k_eff(R, material, group),
        "steady_fission_rate": Fss,
        "leakage": leakage_spectrum(r_array, R, material, S_int=S_int, group=group),
    }


if __name__ == "__main__":
    print("=== leakage @ U-235, R=8.0 cm (sub-critical) ===")
    a = all_outputs(8.0, "u235", r_array=[1, 10, 100])
    print(f"  k_eff = {a['k_eff']:.4f}   F_ss = {a['steady_fission_rate']:.3e} fissions/s")
    for r, dn, dg in zip(a["leakage"]["r"],
                         a["leakage"]["D_neutron"],
                         a["leakage"]["D_gamma"]):
        print(f"  r={r:6.0f} m : Dn={dn:.3e}  Dgamma={dg:.3e} (rel J/cm^2*s)")
