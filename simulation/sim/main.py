"""
main.py - CLI report generator for the calibrated fission-neutronics model
==========================================================================

    python sim/main.py --material u235 --mass 56
    python sim/main.py --material pu239            # at calibrated critical mass
    python sim/main.py --material u235 --reflected 3.93 --refl natU
    python sim/main.py --material u235 --sweep
    python sim/main.py --material u235 --verify    # run the MC cross-check

Prints a physics report (k_eff, reactivity, prompt lifetime, fission energy,
leakage) and writes a JSON results file into --out. All numbers come from
the calibrated multi-group engine (sim.neutronics), not the old curve-fit.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.neutronics import sphere as S
from sim.neutronics import kinetics as K
from sim.neutronics import reflector as Rf
from sim.neutronics import materials as M


def kt(J):
    """Joules -> kilotons TNT."""
    return J / M.TNT_J_PER_KT


def report(mat, R, group="2g", case="default"):
    m = M.get(mat)
    Rc = S.R_critical(mat, group)
    Mc = S.Mcrit(mat, group)
    k = float(S.k_eff_sphere(R, mat, group)[0])
    rho = float(S.rho_reactivity(R, mat, group)[0])
    regime = ("sub-critical" if k < 1.0
              else "critical" if abs(k - 1.0) < 1e-6
              else "super-critical")
    t = np.linspace(0.0, 2e-6, 200)
    fy = K.fission_yield(mat, R, t)

    print(f"=== Fission Neutronics Report - {mat}, R={R:.2f} cm ({case}) ===")
    print(f"  Material             : {mat}")
    print(f"  Sphere radius R      : {R:.3f} cm   M = {(4/3)*np.pi*m['rho']*R**3/1000:.2f} kg @ {m['rho']} g/cm^3")
    print(f"  R_crit (model)       : {Rc:.3f} cm   M_crit = {Mc:.2f} kg  "
          f"(measured {m['mcrit_kg']} kg)")
    print(f"  k_eff                : {k:.5f}   rho = {rho:+.5f}   [{regime}]")
    print(f"  Sub-crit multiplier  : M=k/(1-k) = "
          + (f"{k/(1-k):.1f}" if k < 1.0 else "diverges (>= critical)"))
    print(f"  Prompt gen. time     : {K.prompt_lifetime(mat, R)*1e9:.1f} ns")

    if fy["type"] == "explosive":
        print(f"  Total fission energy : {fy['E_total_J']:.3e} J  "
              f"({kt(fy['E_total_J']):.1f} kt if 100% fissioned)")
        print(f"  Fission fraction f   : {fy['f_disassembly']:.3f}  "
              f"(t_exp = {fy['disassembly_time_s']*1e9:.1f} ns, labeled estimate)")
        print(f"  Yield @ t_exp        : {fy['Y_max_J']:.3e} J  ({kt(fy['Y_max_J']):.1f} kt)")
    else:
        print(f"  Steady fission power : {fy['P_steady_W']:.3e} W")
    return {
        "material": mat, "R_cm": R, "M_kg": (4/3)*np.pi*m['rho']*R**3/1000,
        "R_critical_cm": Rc, "M_critical_kg": Mc, "k_eff": k,
        "rho_reactivity": rho, "regime": regime,
        "subcritical_multiplier": (k/(1-k)) if k < 1.0 else None,
        "prompt_lifetime_ns": K.prompt_lifetime(mat, R)*1e9,
        "fission_yield": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                          for kk, vv in fy.items() if kk not in ("t",)},
    }


def report_reflected(mat, t_cm, refl="natU", group="2g"):
    Rcrit = Rf.R_crit_reflected(t_cm, mat, refl)
    Mc = Rf.Mcrit_reflected(t_cm, mat, refl)
    print(f"=== Reflected {mat}, {t_cm/2.54:.3f}-in {refl} reflector ===")
    print(f"  Core R_crit          : {Rcrit:.3f} cm")
    print(f"  Reflected M_crit     : {Mc:.2f} kg")
    return {"material": mat, "reflector": refl, "refl_t_cm": t_cm,
            "R_core_crit_cm": Rcrit, "M_crit_reflected_kg": Mc}


def run_sweep(mat, group="2g"):
    Rc = S.R_critical(mat, group)
    print(f"=== {mat} radius sweep (R_crit={Rc:.2f} cm) ===")
    rows = []
    for frac in [0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5]:
        R = frac * Rc
        k = float(S.k_eff_sphere(R, mat, group)[0])
        rho = float(S.rho_reactivity(R, mat, group)[0])
        Mkg = (4/3)*np.pi*M.get(mat)["rho"]*R**3/1000.0
        Mmult = k/(1-k) if k < 1.0 else float("inf")
        tag = "SUB" if k < 1.0 else "SUP"
        extra = (f"M=k/(1-k)={Mmult:.1f}" if k < 1.0
                 else f"gen={K.prompt_period(mat, R, rho):.1e}s")
        print(f"  R={R:6.2f} cm ({frac:4.2f}x) [{tag}]  M={Mkg:7.2f}kg  "
              f"k={k:.4f}  rho={rho:+.4f}  {extra}")
        rows.append({"R_cm": R, "frac": frac, "M_kg": Mkg, "k_eff": k,
                     "rho": rho, "regime": tag})
    return {"R_critical_cm": Rc, "rows": rows}


def write_json(out_dir, data, name="criticality_results.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)
    print(f"[wrote] {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Calibrated fission-neutronics report")
    ap.add_argument("--material", default="u235",
                    choices=sorted(M.MEASURED_MCRIT), help="fissile material")
    ap.add_argument("--mass", type=float, default=None,
                    help="fissile mass in kg (default: calibrated critical mass)")
    ap.add_argument("--radius", type=float, default=None,
                    help="sphere radius in cm (overrides --mass)")
    ap.add_argument("--reflected", type=float, default=None,
                    help="reflector thickness in INCHES (LA-4208 convention, "
                         "e.g. 3.93 for a 3.93-in natU reflector)")
    ap.add_argument("--refl", default="natU", help="reflector material")
    ap.add_argument("--group", default="2g", choices=["1g", "2g"],
                    help="diffusion group closure")
    ap.add_argument("--sweep", action="store_true", help="radius sweep")
    ap.add_argument("--verify", action="store_true",
                    help="run the MC k-eff cross-check")
    ap.add_argument("--device", choices=["gun", "implosion"], default=None,
                    help="run the fission-DEVICE transient (educational): "
                         "gun-assembly U-235 or implosion Pu-239")
    ap.add_argument("--premature", action="store_true",
                    help="with --device: the initiator-off / premature-fission "
                         "teaching contrast (source present from t=0)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    mat = args.material
    # --- device-transient mode (the 'more detailed' device layer) ---
    if args.device:
        from sim import device
        if args.premature:
            res = device.run_premature(args.device)
        else:
            res = (device.run_gun() if args.device == "gun"
                   else device.run_implosion())
        device.report(res, out_dir=args.out)
        return
    # resolve radius: --radius > --mass > calibrated critical
    if args.radius is not None:
        R = args.radius
    elif args.mass is not None:
        m = M.get(mat)
        R = (3.0*(args.mass*1000.0)/(4.0*np.pi*m["rho"]))**(1/3.0)
    else:
        R = S.R_critical(mat, args.group)

    result = {}
    if args.reflected is not None:
        t_cm = args.reflected * 2.54   # inches (LA-4208 convention) -> cm
        result = report_reflected(mat, t_cm, args.refl, args.group)
        write_json(args.out, result, "reflected_results.json")
    elif args.sweep:
        result = run_sweep(mat, args.group)
        write_json(args.out, {"sweep": result}, "criticality_curve.json")
    else:
        result = report(mat, R, args.group)
        write_json(args.out, result)

    if args.verify:
        from sim import mc_check
        Rv = args.radius if args.radius is not None else S.R_critical(mat, args.group)
        comp = mc_check.compare_to_diffusion(Rv, mat, n_source=2000,
                                              n_batches=200, seed=0,
                                              group=args.group)
        print(f"\n=== MC cross-check @ R={Rv:.2f} cm ({mat}) ===")
        print(f"  MC k_eff        : {comp['k_eff_mc']:.4f} +/- {comp['k_eff_mc_sigma']:.4f}")
        print(f"  Diffusion k_eff : {comp['k_eff_diffusion']:.4f}")
        print(f"  Agreement       : {comp['agreement_pct']:.1f} %")
        write_json(args.out, comp, "mc_check.json")


if __name__ == "__main__":
    main()
