"""
main.py — CLI report generator for the fission criticality simulation
======================================================================

    python sim/main.py --mass 56
    python sim/main.py --mass 46 --subcritical
    python sim/main.py --sweep            # sweep M over [0.5..2.5]*M_crit
    python sim/main.py --out reports/

Prints a physics report (reactivity, runaway time, energy, leakage) and
writes a JSON results file into the output directory.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim import criticality as c
from sim import leakage as l
from sim import diffusion as d


def kt_joules(J):
    """Joules -> kilotons TNT (1 kt = 4.184e12 J)."""
    return J / 4.184e12


def format_report(M, M_crit, case="default"):
    t = np.linspace(0.0, 2e-6, 200)
    out = c.all_outputs(M, M_crit=M_crit, t_array=t)
    rho = out["reactivity"][0]
    ch = out["chain"]
    y  = out["yield"]

    print(f"=== Fission Criticality Report — case '{case}' ===")
    print(f"  Assembly mass          : {M:.1f} kg U-235")
    print(f"  Critical mass (bare)   : {M_crit:.1f} kg")
    print(f"  Reactivity rho         : {rho:+.4f}")
    print(f"  Regime                 : "
          + ("sub-critical" if rho < 0 else "critical" if abs(rho) < 1e-9
             else "super-critical (prompt)"))

    if rho > 0:
        print(f"  Prompt gen time        : {ch['tau_prompt']*1e9:.1f} ns")
        print(f"  Time to 1e6x growth  : {ch['runaway_time']*1e9:.1f} ns")
        print(f"  Disassembly time     : {y['disassembly_time']*1e9:.1f} ns")
        print(f"  Fiss. energy avail.  : {y['E_total']/1e9:.2f} GJ  "
              f"({kt_joules(y['E_total']):.4f} kt TNT)")
        print(f"  Yield @ disassembly  : {y['Y_max']/1e9:.2f} GJ  "
              f"({kt_joules(y['Y_max']):.4f} kt TNT)")
        print(f"  Fission fraction     : {y['f_disassembly']:.3f}")
    else:
        print(f"  Steady fission power : {y['P_steady']:.3e} W")
        print(f"  Relaxation time      : {ch['tau_d']*1e6:.1f} us")

    # Leakage (sub-critical only — for rho>0 the source is transient)
    Fss = l.steady_state_fission_rate(M, M_crit=M_crit)
    print(f"  Sub-crit steady rate : "
          + ("divergent (at/above critical)" if np.isinf(Fss)
             else f"{Fss:.3e} fissions/s"))
    if rho < 0:
        Dn = l.neutron_dose_at_distance(M, 10.0, M_crit=M_crit)
        Dg = l.gamma_dose_at_distance(M, 10.0, M_crit=M_crit)
        print(f"  Neutron dose @10 m   : {Dn:.3e} (rel)")
        print(f"  Gamma dose @10 m     : {Dg:.3e} (rel)")

    # Spatial (diffusion) note
    Lc = d.critical_half_length()
    print(f"  [spatial] Diff. length : {d.diffusion_length():.2f} cm, "
          f"crit. half-length {Lc:.2f} cm")
    print()
    return out


def run_sweep(M_crit=c.M_BARE_CRIT):
    M_arr = np.linspace(0.5, 2.5 * M_crit, 100)
    rho = c.reactivity(M_arr, M_crit=M_crit)
    # Find the critical crossing
    idx = np.argmin(np.abs(rho))
    print(f"=== Mass sweep (M_crit = {M_crit:.1f} kg) ===")
    for frac in [0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5]:
        M = frac * M_crit
        r = c.all_outputs(M, M_crit=M_crit)
        tag = "SUB" if r["reactivity"][0] < 0 else "SUP"
        if r["reactivity"][0] > 0:
            extra = f"gen={r['chain']['tau_prompt']*1e9:.0f}ns  " \
                    f"yield={r['yield']['Y_max']/1e9:.1f}GJ"
        else:
            extra = f"P={r['yield']['P_steady']:.1e}W"
        print(f"  M={M:6.1f} kg ({frac:4.2f}x) [{tag}]  "
              f"rho={r['reactivity'][0]:+.4f}  {extra}")
    return {"M": M_arr, "rho": rho, "critical_index": int(idx),
            "M_at_crossing": float(M_arr[idx])}


def write_json(out_dir, data):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "criticality_results.json")
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)
    print(f"[wrote] {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fission criticality simulation")
    ap.add_argument("--mass", type=float, default=56.0,
                    help="assembly mass in kg U-235 (default 56)")
    ap.add_argument("--mcrit", type=float, default=c.M_BARE_CRIT,
                    help="critical mass kg (default 52)")
    ap.add_argument("--sweep", action="store_true",
                    help="run a mass sweep instead of a single point")
    ap.add_argument("--out", default="reports", help="output directory")
    args = ap.parse_args()

    if args.sweep:
        sweep = run_sweep(args.mcrit)
        write_json(args.out, {"sweep": sweep})
    else:
        result = format_report(args.mass, args.mcrit)
        # Serialize a compact results dict (drop the large arrays)
        compact = {
            "M": args.mass,
            "M_crit": args.mcrit,
            "reactivity": float(result["reactivity"][0]),
            "prompt_gen_time_s": result["chain"].get("tau_prompt"),
            "runaway_time_s": result["chain"].get("runaway_time"),
            "fission_energy_J": result["yield"]["E_total"],
            "yield_max_J": result["yield"]["Y_max"],
            "f_disassembly": result["yield"].get("f_disassembly"),
            "steady_power_W": result["yield"].get("P_steady"),
        }
        write_json(args.out, compact)
