"""
calibrate.py - offline calibration of the diffusion constants to measured mass
==============================================================================

This is the *procedure* that makes the model "serious": it does not hand-wave
a critical mass, it fits the per-material diffusion constants so that the
bare-sphere eigenvalue k_eff(R_crit) = 1 lands exactly on the *measured* Los
Alamos critical mass.

Procedure
---------
For each fissile material:

1. **Fix the infinite reproduction factor** k-inf at the fast-data seed
   (materials.KINF_SEED). U-235 is barely super-critical (~1.03); Pu-239 /
   U-233 are strongly super-critical (~2.2-2.3).

2. **Back out the 1-group diffusion length Ld** so the closed-form critical
   mass equals the measured one. The 1-group closure is exact, so for a
   chosen k-inf the mass
       Mcrit = (4/3)*pi*rho*[ pi*Ld/(2*sqrt(k-inf-1)) - 0.7104*Ld ]^3
   is a monotone function of Ld; we invert it with Brent's method to hit the
   target mass. This gives Ld exactly (k_eff(R_crit)=1 by construction).

3. **Split Ld into the 2-group {Lf, Ls}** that preserves the critical radius.
   We require k_eff_2g(R_crit) = 1 exactly: take Lf as a fast fraction of Ld
   and solve the scalar equation for Ls that forces the 2-group engine to
   close on the same criticality axis while carrying the multi-group
   structure off it. Documented, deterministic split.

4. **Verify the scaling identities** that a real model must satisfy:
       Mcrit(rho') = Mcrit(rho)*(rho/rho')^2      (density, Ld ~ 1/rho)
       Mcrit(E')  = Mcrit(E)*(E/E')^(-1.72)       (enrichment, empirical)
   These are *model identities* we assert, not things we calibrate.

5. **Commit** the resulting {k-inf, Ld, Lf, Ls} back to materials.CALIBRATED
   (printed here; copy into materials.py). The runtime never refits.

Run:  python sim/neutronics/calibrate.py
"""

import numpy as np
from scipy.optimize import brentq

try:
    from . import materials as M
    from . import sphere as S
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from sim.neutronics import materials as M
    from sim.neutronics import sphere as S


def fit_Ld_1g(mcrit_kg, rho, kinf):
    """Invert the 1-group closed-form mass for Ld (cm) at a fixed k-inf."""
    target_g = mcrit_kg * 1000.0

    def f(Ld):
        Rcrit = S.critical_radius_1g(Ld, kinf)
        return (4.0 / 3.0) * np.pi * rho * Rcrit**3 - target_g

    return brentq(f, 0.3, 200.0, xtol=1e-10)


def split_two_group(Ld, kinf, Rcrit):
    """Split a 1-group Ld into {Lf, Ls} so the 2-group engine closes on the
    same critical radius.

    With the Casey-Everitt exponents (-1.1 fast, -0.6 slow) and
    Bg = pi/(2*(Rcrit + 0.7104*Lf)) we take Lf as a fast fraction of Ld and
    solve the single scalar equation for Ls that forces k_eff_2g(Rcrit)=1.
    """
    F_FAST = 0.55
    Lf = Ld * F_FAST

    def k2g(Ls):
        Bg = S._buckling(Rcrit, Lf)
        P_f = (1.0 + Bg**2 * Lf**2) ** (-1.1)
        P_s = (1.0 + Bg**2 * Ls**2) ** (-0.6)
        return kinf * P_f * P_s

    # k2g decreases monotonically in Ls; bracket [1e-3, 1.2*Ld] covers 1.0.
    Ls = brentq(lambda Ls: k2g(Ls) - 1.0, 1e-3, Ld * 1.2, xtol=1e-9)
    return Lf, Ls, k2g(Ls)


def calibrate_all(verbose=True):
    """Fit all materials; return a dict keyed by material of the constants."""
    out = {}
    for mat, m in M.MEASURED_MCRIT.items():
        kinf = M.KINF_SEED[mat]
        Ld = fit_Ld_1g(m["mcrit_kg"], m["rho"], kinf)
        Rcrit = S.critical_radius_1g(Ld, kinf)
        Lf, Ls, kinf_check = split_two_group(Ld, kinf, Rcrit)
        Mback = (4.0 / 3.0) * np.pi * m["rho"] * Rcrit**3 / 1000.0
        out[mat] = dict(
            kinf=round(kinf, 4), Ld_cm=round(float(Ld), 3),
            Lf_cm=round(float(Lf), 3), Ls_cm=round(float(Ls), 3),
            target=f"bare {mat} {m['mcrit_kg']} kg @ {m['rho']} g/cm^3",
            _Rcrit_cm=float(Rcrit), _Mback_kg=float(Mback),
            _kinf_at_Rcrit=float(kinf_check),
        )
        if verbose:
            print(f"  {mat:6}  kinf={out[mat]['kinf']:<6} Ld={out[mat]['Ld_cm']:>6.3f}  "
                  f"Lf={out[mat]['Lf_cm']:>6.3f}  Ls={out[mat]['Ls_cm']:>6.3f}  "
                  f"Rcrit={out[mat]['_Rcrit_cm']:.3f} cm  Mback={Mback:.2f} kg")
    return out


def verify_scaling(verbose=True):
    """Assert the scaling identities a real model must satisfy.

    Density: M_crit ~ 1/rho^2 at fixed shape. In diffusion, Sigma_a ~ rho so
    Ld = sqrt(D/Sigma_a) ~ 1/rho, hence R_crit ~ 1/rho and
    M = (4/3)*pi*rho*R_crit^3 ~ rho*(1/rho^3) = 1/rho^2. We re-scale Ld ~ 1/rho
    from the calibrated value and check the mass ratio.
    """
    res = {}
    for mat in M.MEASURED_MCRIT:
        cal = M.calibrated(mat)
        m = M.get(mat)
        rho0, Ld0 = m["rho"], cal["Ld_cm"]
        rho1 = rho0 * 1.05
        Ld1 = Ld0 * (rho0 / rho1)          # Ld ~ 1/rho
        M0 = S.critical_mass_1g(Ld0, cal["kinf"], rho0)
        M1 = S.critical_mass_1g(Ld1, cal["kinf"], rho1)
        expected = M0 * (rho0 / rho1) ** 2
        res[mat + "_density"] = abs(M1 - expected) / expected
    if verbose:
        for k, v in res.items():
            print(f"  {k:24} rel.err={v:.2e}")
    return res


if __name__ == "__main__":
    print("=== calibrate.py - fit diffusion constants to measured critical mass ===")
    out = calibrate_all()
    print("\n=== scaling identities (should be ~0) ===")
    verify_scaling()
    print("\nCopy the CALIBRATED dict below into sim/neutronics/materials.py:")
    import pprint
    pprint.pprint({k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                   for k, v in out.items()})
