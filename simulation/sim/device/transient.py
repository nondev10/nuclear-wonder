"""
transient.py - the device fission transient (the centerpiece)
=============================================================

Wires the calibrated fission-neutronics engine (sim.neutronics) into a
time loop so the steady-sphere model becomes a time-dependent device. The
physical insight from the Nth Country report that this is built around:

    "The [assembly] times ... are more than 10 times the total fission
     chain reaction time. This means that it is essential for the fissile
     material to be neutron free during the assembly or the reaction will
     take place prematurely. If the fissile material is neutron free, then
     it is necessary to 'turn on' neutrons to initiate the chain reaction
     at the desired time. This is the role of the so-called initiator."

So the device has three regimes in time:
   1. **Assembly** (t < t_0): sub-critical pieces coming together. The core
      is *neutron-free* - no chain. k_eff rises toward (and past) 1 as the
      assembly reaches critical mass / compressed size.
   2. **Initiation & burst** (t >= t_0): the initiator fires at t_0 =
      t_critical, turning on the neutron source; the chain runs for a few
      prompt generation times, N(t) grows, energy Y(t) is released.
   3. **Disassembly** (self-termination): the hot fissioned core
      disassembles / expands on a labeled hydrodynamic timescale
      t_dis ~ R/v_dis, so k_eff falls back below 1 and the chain stops.
      The fission fraction f = 1 - exp(-t_dis/L_p) < 1 and Y(t) peaks -
      that is what makes this a *transient*, not a closed-form fraction.

The "more detailed" layer versus the old closed-form fission_yield():
   * R(t), rho(t) come from the assembly drivers (sim.device.assembly),
     not a fixed sphere.
   * The neutron source is gated on the initiator time t_0, not t=0.
   * The chain self-terminates via a *labeled* disassembly expansion, so
     we get a peaked Y(t) and a sub-1 fission fraction.
   * A temperature readout T(t) and a labeled illustrative blast-zone are
     reported (not fed back into reactivity - that is out of scope).

Honest limits: t_0, t_dis, and the disassembly velocity v_dis are labeled
order-of-magnitude / literature constants. The engine (k_eff, criticality,
prompt lifetime) is calibrated to the measured critical masses; the device
*timescales* are the labeled-estimate layer.
"""

import numpy as np

from ..neutronics import materials as M
from ..neutronics import sphere as S
from ..neutronics import kinetics as K
from . import assembly as A
from . import blast as YL

# Labeled device constants (NOT design specs).
V_DIS = 1.0e5              # m/s - post-burst disassembly/expansion velocity
INITIATOR_SOURCE_NS = 1.0e9   # intrinsic neutrons/s the initiator releases
                              # (labeled; the Nth Country report notes it must
                              # be strong enough to initiate against the
                              # sub-critical background - order of magnitude)


def _device_timescale(kind, **kw):
    return A.timescale(kind, **kw)


def k_eff_at(R_cm, mat, rho=None, group="2g"):
    """k_eff for a bare sphere of radius R_cm at density rho (g/cm^3).

    The engine is calibrated at the material density; if a density override
    is passed (implosion compression) we scale the diffusion length by
    Ld ~ 1/rho (documented model identity) and re-evaluate.
    """
    cal = M.calibrated(mat)
    if rho is None:
        return float(S.k_eff_sphere(R_cm, mat, group)[0])
    m = M.get(mat)
    Ld_c = cal["Ld_cm"] * (m["rho"] / rho)
    Lf_c = cal["Lf_cm"] * (m["rho"] / rho)
    Ls_c = cal["Ls_cm"] * (m["rho"] / rho)
    Bg = np.pi / (2.0 * (R_cm + M.EXTRAP_FRACTION * Ld_c))
    P_f = (1.0 + Bg**2 * Lf_c**2) ** (-1.1)
    P_s = (1.0 + Bg**2 * Ls_c**2) ** (-0.6)
    return float(cal["kinf"] * P_f * P_s)


def rho_reactivity_at(R_cm, mat, rho=None, group="2g"):
    """Derived reactivity rho=(k-1)/k for the moving assembly."""
    k = k_eff_at(R_cm, mat, rho, group)
    return (k - 1.0) / k


def device_transient(kind="implosion", mat=None, initiator_on=True,
                    n=400, t_0=None, group="2g", **driver_kw):
    """Run the device transient for a gun or implosion assembly.

    Parameters
    ----------
    kind : "gun" | "implosion" - assembly driver (and default material).
    mat : override material (default u235 for gun, pu239 for implosion).
    initiator_on : if True, neutron source turns on at t_0 (normal burst).
                   if False, "premature fission" teaching contrast - the
                   intrinsic sub-critical source is present from t=0 and the
                   chain starts *during* assembly (the Nth Country hazard).
    t_0 : override initiator time (s). Default = driver t_critical_s.
    driver_kw : passed to the assembly driver (muzzle velocity, compression
                factor, etc.).

    Returns a dict with arrays t, R_cm, rho, M_kg, k_eff, N, Y, T, plus
    scalars: f_disassembly, Y_max_J, Y_max_kt, t_0, t_dis_s, prompt_period_s.
    """
    mat = mat or ("u235" if kind == "gun" else "pu239")
    cal = M.calibrated(mat)

    # assembly driver -> R(t), rho(t)
    t_win = A.timescale(kind, **driver_kw)
    if kind == "gun":
        drv = A.gun_trajectory(np.linspace(0.0, t_win, n), mat, **driver_kw)
    else:
        drv = A.implosion_trajectory(np.linspace(0.0, t_win, n), mat, **driver_kw)
    t = drv["t"]
    R_arr = drv["R_cm"]
    rho_arr = drv["rho"]

    t_critical = drv["t_critical_s"]
    if t_0 is None:
        t_0 = t_critical if initiator_on else 0.0

    # reactivity trajectory rho(t)
    reactivity = np.array([rho_reactivity_at(R_arr[i], mat,
                                              rho_arr[i] if kind == "implosion" else None)
                           for i in range(n)])
    k_arr = np.array([k_eff_at(R_arr[i], mat, rho_arr[i] if kind == "implosion" else None)
                      for i in range(n)])

    # prompt generation time (per-material, /kinf)
    Lambda = K.prompt_lifetime(mat, R_arr[0])

    # disassembly timescale: t_dis ~ R/v_dis (labeled hydrodynamic estimate)
    R_peak_cm = R_arr[np.argmax(reactivity)]
    t_dis = R_peak_cm * 1e-2 / V_DIS

    # prompt period where reactivity > beta (super-critical)
    beta = M.BETA_FAST
    Lp = np.where(reactivity > beta,
                  Lambda / np.maximum(reactivity - beta, 1e-9),
                  np.inf)
    N_CAP = 1e30                        # clamp to avoid exp() overflow
    N = np.zeros(n)
    if initiator_on:
        # chain is gated on t_0. Before t_0: N = source floor. After: chain.
        N_at_t0 = INITIATOR_SOURCE_NS * Lambda     # seeding scale at t_0
        for i in range(n):
            if t[i] >= t_0:
                arg = (t[i] - t_0) / Lp[i] if np.isfinite(Lp[i]) else 0.0
                N[i] = N_at_t0 * np.exp(min(arg, 700.0))   # cap exp arg
            else:
                N[i] = N_at_t0 * 1e-6             # neutron-free floor
        N = np.clip(N, 0.0, N_CAP)
    else:
        # premature: intrinsic source present from t=0; chain grows wherever
        # the moving reactivity is super-critical, no gating. Forward-march
        # dN/dt = N/L_p where L_p = Lambda/(rho-beta); sub-critical (L_p=inf)
        # the source floor is added, no runaway.
        N[0] = 1.0
        for i in range(1, n):
            Lp_i = Lp[i - 1]
            if np.isfinite(Lp_i) and Lp_i > 0:
                N[i] = N[i - 1] * np.exp(min((t[i] - t[i - 1]) / Lp_i, 700.0))
            else:
                N[i] = N[i - 1] + 1e-3 * INITIATOR_SOURCE_NS * \
                    (t[i] - t[i-1]) / Lambda
            N[i] = min(N[i], N_CAP)
        N = np.maximum.accumulate(N)

    # fission fraction: 1 - exp(-t_dis/L_p) at peak super-criticality
    Lp_peak = Lambda / max(reactivity.max() - beta, 1e-9)
    f_dis = 1.0 - np.exp(-t_dis / Lp_peak)

    # energy release: fissions ~ N * (available atoms scale). Use the
    # fission-fraction-of-total available for the assembled mass, gated by
    # how far the chain has run relative to t_dis.
    M_max = drv["M_kg"].max()
    E_total = K.total_fission_energy(mat, A.radius_of_mass(M_max, mat))
    # Y(t): the burst releases energy over the prompt window Lp_peak, gated
    # on the source being on (t >= t_0 for initiator-on). It rises from 0,
    # reaches Y_max = f_dis*E_total, then holds (the assembly has disassembled
    # and the chain is over). Monotone, peaked-at-Y_max shape.
    Y_max = f_dis * E_total
    chain_prog = np.zeros(n)
    if initiator_on:
        run_time = np.maximum(t - t_0, 0.0)
        chain_prog = 1.0 - np.exp(-run_time / Lp_peak)
        chain_prog = np.where(run_time > 0, chain_prog, 0.0)
    else:
        chain_prog = np.clip(t / Lp_peak, 0.0, 1.0)
    Y = Y_max * chain_prog

    # temperature readout (labeled, not fed back): fission heat -> T
    T = YL.temperature_readout(Y, M_max, mat)

    return {
        "kind": kind, "material": mat,
        "t": t, "R_cm": R_arr, "rho": rho_arr, "M_kg": drv["M_kg"],
        "k_eff": k_arr, "rho_reactivity": reactivity,
        "N": N, "Y_J": Y, "T_K": T,
        "f_disassembly": float(f_dis), "Y_max_J": float(Y_max),
        "Y_max_kt": float(YL.yield_kt(Y_max)),
        "t_0_s": float(t_0), "t_critical_s": float(t_critical),
        "t_dis_s": float(t_dis), "prompt_lifetime_s": float(Lambda),
        "driver": drv, "initiator_on": initiator_on,
    }


def downsample(res, n_out=200):
    """Return a compact dict for the HTML: downsampled time series + scalars.

    Keeps the arrays small enough to inline in the blueprint without the
    page getting heavy, while preserving the shape (transient peaks, k=1
    crossing, N floor before t_0).
    """
    idx = np.linspace(0, len(res["t"]) - 1, min(n_out, len(res["t"]))).astype(int)
    return {
        "kind": res["kind"], "material": res["material"],
        "initiator_on": res["initiator_on"],
        "t": res["t"][idx].tolist(),
        "k_eff": res["k_eff"][idx].tolist(),
        "rho_reactivity": res["rho_reactivity"][idx].tolist(),
        "N": res["N"][idx].tolist(),
        "Y_J": res["Y_J"][idx].tolist(),
        "T_K": res["T_K"][idx].tolist(),
        "f_disassembly": res["f_disassembly"], "Y_max_kt": res["Y_max_kt"],
        "Y_max_J": res["Y_max_J"], "t_0_s": res["t_0_s"],
        "t_critical_s": res["t_critical_s"], "t_dis_s": res["t_dis_s"],
        "prompt_lifetime_ns": res["prompt_lifetime_s"] * 1e9,
    }


def report(res, out_dir="reports", kind=None):
    """Print a human-readable device transient report; write JSON.

    Returns the (downsampled) JSON-able dict.
    """
    import json, os
    kind = kind or res["kind"]
    d = downsample(res)
    print(f"=== Fission Device Transient - {kind} ({res['material']}) ===")
    print(f"  Assembly driver        : {kind}  "
          f"(t_assembly = {res['t_critical_s']*1e6:.1f} us)")
    print(f"  Initiator              : "
          + ("ON at t_0 = %.2f us" % (res['t_0_s']*1e6)
             if res['initiator_on'] else "OFF (premature fission, source from t=0)"))
    print(f"  Prompt lifetime        : {res['prompt_lifetime_s']*1e9:.2f} ns")
    print(f"  Peak reactivity        : {res['rho_reactivity'].max():+.4f}")
    print(f"  Disassembly time (lbl) : {res['t_dis_s']*1e9:.0f} ns")
    print(f"  Fission fraction f     : {res['f_disassembly']:.3f}")
    print(f"  Peak yield (lbl)       : {res['Y_max_kt']:.1f} kt  "
          f"({res['Y_max_J']:.2e} J)")
    print(f"  [yield/blast are labeled illustrative estimates, not design specs]")
    # write JSON
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"device_{kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    print(f"[wrote] {path}")
    return d
