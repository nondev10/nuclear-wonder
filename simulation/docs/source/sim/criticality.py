"""
criticality.py — sub-critical-mass / point-kinetics model
=========================================================

Two independent but linked models:

1. Material-buckling model — for a fissile sphere of mass M the geometric
   leakage competes with neutron reproduction. We define a non-dimensional
   reactivity  rho(M)  with  rho < 0 (sub-critical, neutron leak-out wins)
   rho = 0 (critical)
   rho > 0 (prompt super-critical)
   and locate M_crit where rho(M) crosses zero.

2. Point-kinetics (one-group, delayed-neutron source) — the chain-reaction
   growth of a sub-critical assembly that is driven above criticality by a
   prompt burst or by adding the last mass increment. The prompt generation
   time gives the *runaway* timescale; the neutron lifetime gives the slow
   exponential approach.

The numbers are calibrated to standard textbook values for U-235:
    U-235 critical bare-sphere mass     ~ 46.5 kg  (we use 52 kg here to be
                                            conservative and to leave head-room
                                            for the 3-D buckling correction)
    Fast neutron generation time       ~ 10 ns      (prompt mean life)
    Full neutron lifetime (with d-N)   ~ 10 us      (delayed)
    Neutrons per fission (effective)  ~ 2.9
    Fission energy                      200 MeV

These constants are *teaching-scale* — they reproduce the order of magnitude
and shape of the textbook "critical-mass plot" without being device specs.
All functions are pure numpy so they are importable from other modules and
from the interactive HTML (JS port).
"""
import numpy as np

# ----------------------------------------------------------------------
#  Physical constants & fissile-mass model
# ----------------------------------------------------------------------
NEUTRONS_PER_FISSION = 2.9          # ν effective
FISSION_ENERGY_J     = 3.204e-11   # 200 MeV in J
PROMPT_NEUTRON_LIFE  = 1e-8        # 10 ns
DELAYED_NEUTRON_LIFE = 1e-5        # 10 us (effective delayed group)
BETA_EFFECTIVE       = 0.0065      # β_eff — fraction that is delayed

# Critical-mass model:
#    The bare-sphere critical mass is reduced by a reflector/shield term.
#    For pedagogy we use:
#       M_crit(reflector) = M_bare * (1 - reflector_credit)
#       where reflector_credit ∈ [0, 1]  (0 = bare, 1 = fully reflected)
#    M_bare for U-235 with the conservative 52 kg value.
M_BARE_CRIT = 52.0
REFLECTOR_CREDIT_MAX = 0.40


# ----------------------------------------------------------------------
#  1.  Reactivity as a function of mass (spherical bare assembly)
# ----------------------------------------------------------------------
def reactivity(M, M_crit=M_BARE_CRIT):
    """
    Non-dimensioned reactivity rho(M) for a bare fissile sphere of mass M.

    Model:
        The source term grows linearly with mass M (fissile atom count),
        and the loss baseline is set so the critical point sits exactly at
        M = M_crit.  The reactivity (0 at critical, <0 below, >0 above)
        is:
            rho(M) = (M/M_crit - 1) / (M/M_crit)

    We parameterise:
        k_eff(M) = M / (K0 * M^(2/3) + M_crit - M_crit)
    That is:
        k_eff(M) = M / M_crit   (for M ≠ M_crit, tuned so that k_eff(M_crit) = 1)
    This is a one-group "source / loss" eigenvalue form: the source grows
    linearly with M (fissile atom count) while the loss has a constant
    baseline (the critical mass defines the threshold).  The resulting
    reactivity is:
        rho(M) = (M / M_crit - 1) / (M / M_crit)
    which is rho = 0 exactly at M = M_crit.

    This is a pedagogical model — it reproduces the sub-critical / critical /
    super-critical structure of the real fast-neutron system without being
    a device specification.
    """
    M = np.asarray(M, dtype=float)
    M_crit = float(M_crit)
    k_eff = M / M_crit
    rho   = (k_eff - 1.0) / k_eff
    return np.atleast_1d(rho)


def critical_mass(bare_mass=M_BARE_CRIT, reflector_credit=0.0):
    """Critical mass with an optional reflector credit in [0, 1]."""
    rc = float(min(max(reflector_credit, 0.0), REFLECTOR_CREDIT_MAX))
    return bare_mass * (1.0 - rc)


# ----------------------------------------------------------------------
#  2.  Point-kinetics runaway time
# ----------------------------------------------------------------------
def prompt_lifetime(rho_positive, beta=0.0):
    """
    Mean prompt-generation time when reactivity is *positive* (rho > 0).

    tau_prompt = tau_p / rho_effective,  where rho_effective = rho - beta.
    If rho_effective <= 0 (sub-critical or critical) we return the raw
    prompt neutron lifetime tau_p as the minimum meaningful timescale,
    not a divergent value.
    """
    rho_positive = np.asarray(rho_positive, dtype=float)
    rho_effective = rho_positive - beta
    # Where rho_effective <= 0, use tau_p (the neutron lifetime) instead of
    # dividing by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = PROMPT_NEUTRON_LIFE / np.where(rho_effective > 0, rho_effective, 1.0)
    tau = np.where(rho_effective > 0, tau, PROMPT_NEUTRON_LIFE)
    return tau


def runaway_time(M, M_crit=M_BARE_CRIT, beta=0.0):
    """
    Prompt burst time if mass M is pushed over criticality.
    Below or at M_crit (rho <= 0) returns the prompt neutron lifetime tau_p
    (no runaway — the chain reaction does not self-sustain).
    """
    rho = reactivity(M, M_crit=M_crit)
    rho = np.clip(rho, 0.0, None)
    return prompt_lifetime(rho, beta=beta)


def chain_growth(M, M_crit=M_BARE_CRIT, t_array=None, t_max=2e-6):
    """
    Neutron population N(t) for a mass M driven from rest by a prompt
    source.

    The governing ODE (one-group point kinetics, delayed-neutron source) is
        dN/dt = (rho_p / tau_p) N + beta_eff * N_source
    where rho_p is the prompt part of reactivity and N_source is the
    steady delayed source.

    For M > M_crit (rho > 0) the exponential dominates and we get
        N(t) ~ N0 * exp(rho * t / tau_p)
    For M < M_crit the population saturates at a sub-critical steady state
        N_inf = N_source / |rho|

    We solve piecewise: if rho <= 0 we report the steady state plus the
    transient relaxation time  tau_d = tau_d_n / |rho| ; if rho > 0 we
    report the exponential growth and the time to "run away"
    (defined as N = 1e6 N0).
    """
    if t_array is None:
        t_array = np.linspace(0.0, t_max, 200)
    rho = reactivity(M, M_crit=M_crit)
    rho = float(rho[0])

    if rho > 0:
        tau_p = PROMPT_NEUTRON_LIFE / rho
        N = np.exp(rho * t_array / tau_p)
        runaway = t_array[0] + np.log(1e6) * tau_p
        return {"rho": rho, "N": N, "tau_prompt": tau_p, "runaway_time": runaway,
                "type": "prompt_supercritical"}
    else:
        # Exactly-critical (rho ~ 0) → use prompt lifetime as the transient
        # timescale and report a flat steady state.  Sub-critical otherwise.
        abs_rho = max(abs(rho), 1e-9)
        N_inf = 1.0 / abs_rho
        tau_d = DELAYED_NEUTRON_LIFE / abs_rho
        N = N_inf * (1.0 - np.exp(-t_array / tau_d))
        return {"rho": rho, "N": N, "tau_prompt": None, "runaway_time": None,
                "type": "critical_steady" if abs(rho) < 1e-9 else "subcritical_steady",
                "N_inf": N_inf, "tau_d": tau_d}


# ----------------------------------------------------------------------
#  3.  Energy release & expansion asymptote
# ----------------------------------------------------------------------
def fission_yield(M, t_array, M_crit=M_BARE_CRIT, t_max=2e-6, beta=0.0,
                  fissile_mass_fraction=1.0):
    """
    Cumulative thermal energy released (J) by fission of mass M over time.

    The total fission energy available is:
        E_total = M * NA / A * E_fission
    where NA is Avogadro's number and A is the molar mass of the fissile
    isotope (235 for U-235).  Each fission releases ~200 MeV; one mole
    of U-235 therefore releases ~235 kg × 200 MeV ≈ 43 TJ.

    For a super-critical mass (rho > 0) the system disassembles in ~300 ns.
    The fraction of fissile atoms that undergo fission *before* the system
    blows apart is:
        f_dis = 1 - exp(-t_dis / tau_prompt)
    and the total yield is Y_max = f_dis * E_total.  The curve
    Y(t) approaches Y_max on the timescale tau_prompt.

    For a sub-critical mass the fission power is constant (steady-state
    point kinetics with a delayed source) and Y(t) grows linearly.
    """
    NA = 6.022e23
    A_U235 = 235.0   # molar mass (g/mol)
    t_array = np.asarray(t_array, dtype=float)
    # E_total = (M / A) * NA * E_fission  →  fissions per kg × energy per fission
    E_total = (M / A_U235) * NA * FISSION_ENERGY_J
    rho = float(reactivity(M, M_crit=M_crit)[0])

    if rho > 0:
        tau_prompt = PROMPT_NEUTRON_LIFE / max(rho, np.finfo(float).eps)
        t_dis = 3e-7   # 300 ns disassembly
        f_dis = 1.0 - np.exp(-t_dis / tau_prompt)
        Y_max = f_dis * E_total
        Y = Y_max * (1.0 - np.exp(-t_array / tau_prompt))
        return {"E_total": E_total, "Y_max": Y_max, "Y": Y, "t": t_array,
                "type": "explosive", "tau_prompt": tau_prompt,
                "disassembly_time": t_dis, "f_disassembly": f_dis}
    else:
        # Sub-critical / exactly-critical steady fission power.
        # Scale with mass and inverse reactivity: closer to critical → higher P.
        abs_rho = max(abs(rho), 1e-9)
        S_source = 1.0 * (M / M_crit) * 1e6   # intrinsic source scale (rel)
        P = S_source / abs_rho * FISSION_ENERGY_J / 1.0
        Y = P * t_array
        return {"E_total": E_total, "Y_max": E_total, "Y": Y, "t": t_array,
                "type": "steady", "P_steady": P}


def plot_criticality_curve(M_array=None, M_crit=M_BARE_CRIT):
    """Return the (M, rho) curve used by the HTML what-if slider."""
    if M_array is None:
        M_array = np.linspace(0.5, 2.5 * M_crit, 200)
    rho = reactivity(M_array, M_crit=M_crit)
    k_eff = 1.0 / (1.0 + rho)
    return {"M": M_array, "rho": rho, "k_eff": k_eff}


def all_outputs(M, M_crit=M_BARE_CRIT, t_array=None, t_max=2e-6):
    """
    Full diagnostic dump for a given mass — used by main.py and tests.
    """
    t_array = t_array if t_array is not None else np.linspace(0.0, t_max, 200)
    out = {
        "M": M,
        "M_crit": M_crit,
        "reactivity": reactivity(M, M_crit=M_crit),
        "runaway": runaway_time(M, M_crit=M_crit),
        "chain": chain_growth(M, M_crit=M_crit, t_array=t_array, t_max=t_max),
        "yield": fission_yield(M, t_array, M_crit=M_crit, t_max=t_max),
    }
    return out


if __name__ == "__main__":
    print(f"M_crit (bare) = {critical_mass():.2f} kg")
    print(f"  at M = 46 kg  : rho = {reactivity(46.0)[0]:+.4f}  (sub-critical)")
    print(f"  at M = 52 kg  : rho = {reactivity(52.0)[0]:+.4f}  (critical)")
    print(f"  at M = 56 kg  : rho = {reactivity(56.0)[0]:+.4f}  (super-critical)")
    print(f"  prompt generation time @ 56 kg = {runaway_time(56.0)[0]*1e9:.1f} ns")
    r = all_outputs(56.0)
    print(f"\nChain-growth @ M=56 kg:")
    print(f"  type                 = {r['chain']['type']}")
    print(f"  prompt gen time      = {r['chain']['tau_prompt']*1e9:.1f} ns")
    print(f"  time to 1e6x grow  = {r['chain']['runaway_time']*1e9:.1f} ns")
    print(f"  fission yield        = {r['yield']['Y_max']/1e9:.3f} GJ  "
          f"(f_dis={r['yield']['f_disassembly']:.3f} of {r['yield']['E_total']/1e9:.3f} GJ)")
    print(f"\nSub-critical @ M=46 kg:")
    r2 = all_outputs(46.0)
    print(f"  type                 = {r2['chain']['type']}")
    print(f"  steady fission power = {r2['yield']['P_steady']:.2e} W")
    print(f"  relaxation time      = {r2['chain']['tau_d']*1e6:.1f} us")
