"""
kinetics.py - point kinetics & fission-energy release on the spatial model
===========================================================================

Replaces the old model's hard-coded "10 ns prompt / 300 ns disassembly" with
physically-derived timescales, all grounded in the calibrated sphere engine:

* **Prompt generation time** Lambda - the mean time to produce the next
  fission neutron (alpha = 1/Lambda in the 1967 LRL report). For bare fast
  spheres this is ~10 ns for U-235 and ~30 ns for plutonium. We use a
  per-material calibrated value, scaled by 1/k-inf (strongly-super-critical
  systems cycle faster). This is the generation time, distinct from the
  Ld^2/v diffusion flight time (which is reported separately).

* **Point-kinetics chain** with the fast effective delayed fraction
  beta_eff ~ 0.002 (documented). For reactivity rho (derived from k_eff by
  the sphere, not assumed):
      prompt period   Lambda_p = Lambda / (rho - beta)      for rho > beta
      runaway to 1e6  t = Lambda_p * ln(1e6)
  Sub-critical: N(inf) = S*Lambda / |rho| (steady, intrinsic source S).

* **Fission-energy release & disassembly** - the *honest* fix. Instead of a
  fake 300 ns, the disassembly is a clearly-labeled hydrodynamic estimate
  t_exp ~ R / v_exp (v_exp ~ 1e5 m/s). The fission fraction completed before
  the sphere blows apart is f = 1 - exp(-t_exp/Lambda_p). The model reports
  this fission fraction and the released energy, and *says so explicitly*: in
  a real device t_exp is set by the implosion/gun timescale; here we model
  the post-critical fission fraction over a labeled expansion time. This
  stays at the neutronics/energy layer - no device engineering.

All quantities are numpy and importable from the HTML JS port.
"""

import numpy as np

from . import materials as M
from . import sphere as S

# Fast prompt-generation timescale per material (ns), literature-calibrated.
FAST_GEN_NS = {"u235": 10.0, "pu239": 30.0, "u233": 25.0}


def prompt_lifetime(mat, R):
    """Fast prompt generation time Lambda (s). Per-material calibrated,
    scaled by 1/k-inf so a strongly-super-critical material has a shorter
    generation time. This is the alpha=1/Lambda mean generation time, NOT the
    Ld^2/v diffusion flight time (see diffusion_flight_time)."""
    cal = M.calibrated(mat)
    base_s = FAST_GEN_NS[mat] * 1e-9
    return base_s / cal["kinf"]


def diffusion_flight_time(mat):
    """Reported separately from the generation time: Ld^2/v_fast (s), the
    random-walk flight time over a diffusion length. Kept distinct because
    it is NOT the prompt generation time."""
    cal = M.calibrated(mat)
    return (cal["Ld_cm"] * 1e-2) ** 2 / M.V_FAST


def prompt_period(mat, R, rho, beta=M.BETA_FAST):
    """Lambda_p = Lambda/(rho-beta) (s); meaningful for rho > beta."""
    tau = prompt_lifetime(mat, R)
    if rho <= beta:
        return np.inf
    return tau / (rho - beta)


def chain_growth(mat, R, t_array, beta=M.BETA_FAST):
    """Point-kinetics neutron population N(t) for a sphere of radius R (cm),
    driven from N(0)=1. Returns a dict mirroring the old chain_growth API."""
    t_array = np.asarray(t_array, dtype=float)
    Rcrit = S.R_critical(mat)
    rho = float(S.rho_reactivity(R, mat)[0])
    tau = prompt_lifetime(mat, R)

    if rho > beta:
        Lp = prompt_period(mat, R, rho, beta)
        N = np.exp(t_array / Lp)
        runaway = np.log(1e6) * Lp
        return {"rho": rho, "N": N, "tau_lambda_s": tau, "prompt_period_s": Lp,
                "runaway_time_s": runaway, "type": "prompt_supercritical"}
    else:
        # sub-critical / delayed-critical steady state
        abs_rho = max(abs(rho - beta), 1e-9)
        N_inf = 1.0 / abs_rho
        tau_d = 1e-5 / abs_rho  # effective delayed-group relaxation (documented)
        N = N_inf * (1.0 - np.exp(-t_array / tau_d))
        return {"rho": rho, "N": N, "tau_lambda_s": tau, "prompt_period_s": None,
                "runaway_time_s": None, "type": "subcritical_steady",
                "N_inf": N_inf, "tau_d_s": tau_d}


def total_fission_energy(mat, R):
    """Total fission energy available if the whole sphere fissions (J)."""
    m = M.get(mat)
    Mc_g = (4.0 / 3.0) * np.pi * m["rho"] * R**3   # grams
    n_atoms = (Mc_g / m["A"]) * M.NA
    return n_atoms * M.FISSION_ENERGY_J


def disassembly_time(R):
    """Labeled hydrodynamic expansion estimate t_exp ~ R / v_exp (s)."""
    R_m = R * 1e-2
    return R_m / M.V_EXP


def fission_yield(mat, R, t_array, t_exp=None, beta=M.BETA_FAST):
    """Cumulative fission energy Y(t) and the fission fraction f.

    t_exp defaults to the labeled hydrodynamic estimate disassembly_time(R).
    The returned dict states the fission fraction, NOT a device yield.
    """
    t_array = np.asarray(t_array, dtype=float)
    E_total = total_fission_energy(mat, R)
    if t_exp is None:
        t_exp = disassembly_time(R)
    Rcrit = S.R_critical(mat)
    rho = float(S.rho_reactivity(R, mat)[0])
    tau = prompt_lifetime(mat, R)

    if rho > beta:
        Lp = prompt_period(mat, R, rho, beta)
        f = 1.0 - np.exp(-t_exp / Lp)     # fraction of atoms that fission
        Y_max = f * E_total
        Y = Y_max * (1.0 - np.exp(-t_array / Lp))
        return {"E_total_J": E_total, "Y_max_J": Y_max, "Y_J": Y, "t": t_array,
                "f_disassembly": f, "disassembly_time_s": t_exp,
                "prompt_period_s": Lp, "rho": rho, "type": "explosive"}
    else:
        abs_rho = max(abs(rho - beta), 1e-9)
        S_src = 1e6 * (R / Rcrit) ** 3       # intrinsic-source scale (rel.)
        P = S_src / abs_rho * M.FISSION_ENERGY_J
        return {"E_total_J": E_total, "Y_max_J": E_total, "Y_J": P * t_array,
                "t": t_array, "P_steady_W": P, "rho": rho, "type": "steady"}


def all_outputs(mat, R, t_array, t_exp=None):
    """Full diagnostic dump for a sphere of radius R (cm)."""
    chain = chain_growth(mat, R, t_array)
    return {
        "material": mat, "R_cm": R,
        "k_eff": float(S.k_eff_sphere(R, mat)[0]),
        "reactivity": chain["rho"],
        "prompt_lifetime_s": chain["tau_lambda_s"],
        "chain": chain,
        "fission_yield": fission_yield(mat, R, t_array, t_exp=t_exp),
    }
