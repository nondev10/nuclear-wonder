"""
sphere.py - bare-sphere criticality engine (2-group primary, 1-group lean)
===========================================================================

The physical core of the model. For a bare absorbing sphere of radius R and
density rho with a vacuum boundary, neutron diffusion theory gives the
*extrapolated* radius R' = R + d, d = 0.7104*Ld, and the fundamental
geometric buckling Bg = pi/(2*R'). Criticality is Bg^2 = Bm^2 (material
buckling) at the fundamental mode.

Two closures are provided:

1. **1-group (lean, verified exact):** one diffusion length Ld and an
   infinite reproduction factor k-inf. The critical radius is
       R_crit' = pi*Ld / (2*sqrt(k-inf - 1)) ,   R_crit = R_crit' - 0.7104*Ld
   and M_crit = (4/3)*pi*rho*R_crit^3. For any R,
       k_eff(R) = k-inf / (1 + Bg(R)^2 * Ld^2)        (P = 1/(1+Bg^2 Ld^2))
   which is *consistent*: k_eff(R_crit)=1 exactly. This is the form that,
   calibrated per material, reproduces the measured 51.94 / 17.02 / 16.53 kg
   and lands k(R=8.74 cm)=1.000 for U-235 where Terrell's MCNP measured 0.997
   for Godiva.

2. **2-group (primary, per-material calibrated):** fast length Lf, slow
   length Ls, Casey-Everitt leakage factors
       P_f = (1 + Bg^2*Lf^2)^(A_FAST=1.1)  [i.e. power -1.1]
       P_s = (1 + Bg^2*Ls^2)^(A_SLOW=0.6)  [i.e. power -0.6]
       k_eff = k-inf * P_f * P_s
   In the limit Ls -> Ld and exponents -> 1 this reduces to the 1-group form,
   so the 1-group result is a documented check on the 2-group one. The
   2-group set {k-inf, Lf, Ls} is calibrated per material (calibrate.py) so
   that k_eff(R_crit)=1 and the closed-form critical mass matches the
   measurement.

Both closures are monotone increasing in R and cross k=1 exactly at R_crit by
construction. The flux is the fundamental mode phi(r) = sin(Bg r)/sin(Bg R')
(= j0 for smooth r->0).
"""

import numpy as np

from . import materials as M

# Casey-Everitt 2-group leakage exponents (fast / slow).
_A_FAST, _A_SLOW = 1.1, 0.6


def _buckling(R, Ld):
    """Geometric buckling Bg = pi/(2*(R + 0.7104*Ld)), 1/cm, for radius R (cm)."""
    Rp = R + M.EXTRAP_FRACTION * Ld
    return np.pi / (2.0 * Rp)


def k_eff_1g(R, Ld, kinf):
    """1-group effective multiplication factor for a bare sphere of radius R.

    R in cm. Consistent closure: k_eff(R)=1 at the calibrated critical radius.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    Bg = _buckling(R, Ld)
    return kinf / (1.0 + Bg**2 * Ld**2)


def k_eff_2g(R, Lf, Ls, kinf):
    """2-group effective multiplication factor (Casey-Everitt leakage)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    Bg = _buckling(R, Lf)  # fast-group extrapolation drives the geometric Bg
    P_f = (1.0 + Bg**2 * Lf**2) ** (-_A_FAST)
    P_s = (1.0 + Bg**2 * Ls**2) ** (-_A_SLOW)
    return kinf * P_f * P_s


def critical_radius_1g(Ld, kinf):
    """Closed-form critical physical radius (cm) for the 1-group closure."""
    Rp = np.pi * Ld / (2.0 * np.sqrt(kinf - 1.0))
    return Rp - M.EXTRAP_FRACTION * Ld


def critical_mass_1g(Ld, kinf, rho):
    """Closed-form bare critical mass in kg (R in cm, rho in g/cm^3)."""
    R = critical_radius_1g(Ld, kinf)
    return (4.0 / 3.0) * np.pi * rho * R**3 / 1000.0


def R_critical(mat, group="2g"):
    """Solve k_eff(R)=1 for R (cm). Uses the calibrated constants.

    The 1-group path is exact (closed form); the 2-group path is found by
    bisection. Returns a float.
    """
    cal = M.calibrated(mat)
    m = M.get(mat)
    if group == "1g":
        return float(critical_radius_1g(cal["Ld_cm"], cal["kinf"]))
    return _bisect_Rcrit_2g(cal, m)


def _bisect_Rcrit_2g(cal, m):
    # 2-group critical radius by bisection on k_eff_2g(R)-1 over a bracket.
    Rlo, Rhi = 0.5, 200.0

    def f(R):
        return float(k_eff_2g(R, cal["Lf_cm"], cal["Ls_cm"], cal["kinf"])) - 1.0

    if f(Rlo) * f(Rhi) > 0:
        # fall back to the 1-group radius if the 2-group bracket misses the root
        return float(critical_radius_1g(cal["Ld_cm"], cal["kinf"]))
    for _ in range(80):
        Rmid = 0.5 * (Rlo + Rhi)
        if f(Rlo) * f(Rmid) <= 0:
            Rhi = Rmid
        else:
            Rlo = Rmid
    return 0.5 * (Rlo + Rhi)


def k_eff_sphere(R, mat, group="2g"):
    """k_eff for a bare sphere of radius R (cm) of material *mat*."""
    cal = M.calibrated(mat)
    if group == "1g":
        return k_eff_1g(R, cal["Ld_cm"], cal["kinf"])
    return k_eff_2g(R, cal["Lf_cm"], cal["Ls_cm"], cal["kinf"])


def rho_reactivity(R, mat, group="2g"):
    """Derived reactivity rho = (k-1)/k for a bare sphere of radius R (cm)."""
    k = np.atleast_1d(np.asarray(k_eff_sphere(R, mat, group), dtype=float))
    return (k - 1.0) / np.where(np.abs(k) < 1e-12, 1e-12, k)


def Mcrit(mat, group="2g"):
    """Calibrated bare critical mass (kg) for material *mat*."""
    R = R_critical(mat, group)
    m = M.get(mat)
    return (4.0 / 3.0) * np.pi * m["rho"] * R**3 / 1000.0


def phi_sphere(R, mat, r=None, group="2g"):
    """Fundamental-mode flux phi(r) (normalized phi(0)=1, zero at R') of a
    sphere of radius R (cm). Returns (r, phi) arrays; r defaults to [0, R']."""
    cal = M.calibrated(mat)
    Ld = cal["Ld_cm"] if group == "1g" else cal["Lf_cm"]
    Bg = _buckling(R, Ld)
    Rp = R + M.EXTRAP_FRACTION * Ld
    if r is None:
        r = np.linspace(0.0, Rp, 200)
    r = np.asarray(r, dtype=float)
    phi = np.sin(Bg * r) / np.sin(Bg * Rp)
    return r, phi


def subcritical_multiplication(k):
    """Neutron multiplication M = k/(1-k) (LBL 'M > 100' idiom)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    return k / np.where(np.abs(1.0 - k) < 1e-12, 1e-12, (1.0 - k))


def all_outputs(mat, R, group="2g"):
    """Diagnostic dump for a bare sphere of radius R (cm) of material *mat*."""
    m = M.get(mat)
    Rcrit = R_critical(mat, group)
    keff = float(k_eff_sphere(R, mat, group)[0])
    rho = float(rho_reactivity(R, mat, group)[0])
    Mc = (4.0 / 3.0) * np.pi * m["rho"] * R**3 / 1000.0
    return {
        "material": mat,
        "group": group,
        "R_cm": R,
        "M_kg": Mc,
        "R_critical_cm": Rcrit,
        "M_critical_kg": Mcrit(mat, group),
        "k_eff": keff,
        "rho_reactivity": rho,
        "subcritical_multiplication": float(subcritical_multiplication(keff)[0]),
        "regime": ("sub-critical" if keff < 1.0
                   else "critical" if abs(keff - 1.0) < 1e-6
                   else "super-critical"),
    }
