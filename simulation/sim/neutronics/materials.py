"""
materials.py - isotope data, calibration targets, and calibrated constants
==========================================================================

The single source of truth for every number the engine uses.

* **Calibration targets** are the *measured* Los Alamos critical masses from
  Hansen & Paxton, "Reevaluated Critical Specifications of Some Los Alamos
  Fast-Neutron Systems", LA-4208 (1969). These are the fixed ground-truth
  anchors the model reproduces.
* **Leakage / kinetics constants** come from Terrell, "Gamma-Ray and Neutron
  Leakage Spectra Calculated for Unshielded Reactors", LA-11807 (MCNP), plus
  the 1967 LRL experiment report's generation-time relation.
* **Calibrated constants** (`CALIBRATED`) are produced by
  `sim.neutronics.calibrate` and committed here so the runtime never fits
  anything. Each entry is annotated with the target it was fit to.

Physical framing (be honest about it): the diffusion model is calibrated
**per material**. A single global diffusion-length set cannot fit U-235,
Pu-239 and U-233 together - U-235 is barely super-critical (k-inf ~1.03)
while plutonium is strongly so (k-inf ~2.2), so their leakage lengths differ
too much for one set. Per-material calibration is what makes the 2-group
structure defensible physics rather than three unrelated fits; the model
interpolates across the three calibrated isotopes and should not be
extrapolated to new materials.
"""

# ----------------------------------------------------------------------
#  Universal constants
# ----------------------------------------------------------------------
NA = 6.022e23                  # Avogadro (1/mol)
FISSION_ENERGY_J = 3.204e-11   # ~200 MeV per fission, in J
TNT_J_PER_KT = 4.184e12        # J per kiloton TNT
V_FAST = 2.0e5                 # m/s - 1 MeV neutron speed (diffusion flight)
V_EXP = 1.0e5                  # m/s (~1e7 cm/s) - post-critical expansion est.
BETA_FAST = 0.002              # fast-reactor effective delayed-neutron fraction
EXTRAP_FRACTION = 0.7104       # spherical extrapolation distance d = 0.7104*Ld


# ----------------------------------------------------------------------
#  Measured bare critical masses (calibration targets, LA-4208)
#  key -> (M_crit kg, rho g/cm^3, measured uncertainty as a fraction)
# ----------------------------------------------------------------------
MEASURED_MCRIT = {
    # U-235:  "51.94 +/- 0.16 kg U(93.86) at a density of 18.8" (+/-0.3%)
    "u235":  dict(mcrit_kg=51.94, rho=18.806, A=235.0, err=0.003),
    # Pu-239 (4.5 % 240Pu, 1 wt% Ga):  "mc = 17.02 kg +/- 0.6% at 15.61 g/cm^3"
    "pu239": dict(mcrit_kg=17.02, rho=15.61,  A=239.0, err=0.006),
    # U-233 (98.13 %):  "mc = 16.53 kg +/- 0.4% at 18.424 g/cm^3"
    "u233":  dict(mcrit_kg=16.53, rho=18.424, A=233.0, err=0.004),
}

# k-infinity seeds (fast-data order of magnitude; calibrate.py refines Lf/Ls).
# U-235 barely super-critical; Pu/U-233 strongly so.
KINF_SEED = {"u235": 1.03, "pu239": 2.25, "u233": 2.23}


# ----------------------------------------------------------------------
#  Reflected critical-mass targets (U-235 core + reflector), LA-4208
#  Reflector thickness in cm and measured core mass in kg, +/-0.5 %.
# ----------------------------------------------------------------------
REFLECTOR_TARGETS = {
    # natural-uranium reflectors (close-fitting)
    "u235+3.93in natU": dict(mcrit_kg=19.82, rho=18.69, refl="natU",
                              refl_t_cm=3.93 * 2.54, refl_rho=19.00),
    "u235+1.742in natU": dict(mcrit_kg=26.56, rho=18.67, refl="natU",
                               refl_t_cm=1.742 * 2.54, refl_rho=18.67),
    "u235+0.683in natU": dict(mcrit_kg=36.53, rho=18.70, refl="natU",
                               refl_t_cm=0.683 * 2.54, refl_rho=19.00),
    "u235+2.0in Cu":     dict(mcrit_kg=28.14, rho=18.38, refl="cu",
                               refl_t_cm=1.980 * 2.54, refl_rho=8.88),
}
REFLECTOR_ERR = 0.005  # +/-0.5 %

# Reflector material densities (g/cm^3), LBL / LANS values.
REFLECTOR_DENS = {"natU": 19.0, "cu": 8.88, "be": 1.848}


# ----------------------------------------------------------------------
#  Leakage / kinetics constants (Terrell LA-11807 + 1967 LRL)
#  Used only in the leakage and kinetics layers, NOT in the k_eff closure.
# ----------------------------------------------------------------------
NU_FAST = {"u235": 2.616, "pu239": 3.181}          # prompt neutrons/fission (fast)
GAMMA_PER_FISSION = {"u235": 0.358, "pu239": 0.636}  # escaping prompt photons/fission
NEUTRON_LEAK_FRAC = {"u235": 0.578, "pu239": 0.676}  # escaping neutrons / fission n
FIS_NEUTRON_EAVG_MEV = {"u235": 2.047, "pu239": 2.159}  # initial fission-neutron E
GAMMA_LEAK_EAVG_MEV = {"u235": 1.364, "pu239": 1.387}   # avg escaping photon energy
# Maxwellian fission spectrum N(E) = E*exp(-E/T), T = (2/3)*E_avg
MAXWELL_T_MEV = {k: (2.0 / 3.0) * v for k, v in FIS_NEUTRON_EAVG_MEV.items()}
MEV_TO_J = 1.602e-13


# ----------------------------------------------------------------------
#  Calibrated per-material constants (written by calibrate.py)
#
#  For the 1-group lean path:   Ld_cm  (the single diffusion length)
#  For the 2-group engine:      Lf_cm, Ls_cm (fast / slow leakage lengths)
#  kinf                          (the infinite reproduction factor)
#
#  These make k_eff(R_crit) = 1 exactly and reproduce the measured bare mass.
#  See calibrate.py for how they were obtained; the "target" note on each
#  records what it was fit to.
# ----------------------------------------------------------------------
CALIBRATED = {
    "u235": dict(
        kinf=1.03, Ld_cm=1.041, Lf_cm=0.573, Ls_cm=1.037,
        target="bare U-235 51.94 kg @ 18.806 g/cm^3 (R_crit=8.70 cm; "
               "Godiva radius 8.74 cm -> Terrell MCNP k=0.997)"),
    "pu239": dict(
        kinf=2.25, Ld_cm=9.193, Lf_cm=5.056, Ls_cm=4.799,
        target="bare Pu-239 17.02 kg @ 15.61 g/cm^3 (R_crit=6.39 cm)"),
    "u233": dict(
        kinf=2.23, Ld_cm=8.476, Lf_cm=4.662, Ls_cm=4.470,
        target="bare U-233 16.53 kg @ 18.424 g/cm^3 (R_crit=5.98 cm)"),
}

# Reflectors (secondary fit, U-235 reflected cases): Ld_ref, Sigma_a_ref
REFLECTOR_CALIB = {
    "natU": dict(Ld_cm=9.0, Sigma_a=0.020),
    "cu":   dict(Ld_cm=6.0, Sigma_a=0.060),
}


def get(name):
    """Return MEASURED_MCRIT for *name*; raise if unknown."""
    if name not in MEASURED_MCRIT:
        raise KeyError(f"unknown material {name!r}; "
                       f"known: {sorted(MEASURED_MCRIT)}")
    return MEASURED_MCRIT[name]


def calibrated(name):
    """Return the CALIBRATED constant set for *name*."""
    if name not in CALIBRATED:
        raise KeyError(f"no calibrated constants for {name!r}; "
                       f"known: {sorted(CALIBRATED)}")
    return CALIBRATED[name]
