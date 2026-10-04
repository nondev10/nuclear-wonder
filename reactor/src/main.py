"""CLI for the Pu-239 breeder reactor case study.

Usage:
    python src/main.py --case default --burnup-days 900 --out reports/

Prints a human-readable report and, if --out is given, writes a JSON results
file into that directory. No external deps beyond numpy/scipy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Make sibling modules importable when run as `python src/main.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import reactor as reactor_mod  # noqa: E402
import channel_model  # noqa: E402


def build_case(name: str) -> dict:
    """Return the params for a named case. 'default' is the reference case."""
    if name == "default":
        return reactor_mod.default_params()
    raise SystemExit(f"Unknown case '{name}' (only 'default' is implemented).")


def print_report(results: dict, cycle_days: int) -> None:
    """Print a clean human-readable report to stdout."""
    print("=" * 72)
    print(" Pu-239 CANDU-style Heavy-Water Breeder - Case Study Report")
    print("=" * 72)
    print(f" Power (thermal)            : {results['power_MW']:8.2f} MWth")
    print(f" Sodium mass flow           : {results['sodium_mass_flow_kg_s']:8.3f} kg/s")
    print(f" Fuel elements              : {results['n_elements']:8d}")
    print(f" Power / element            : {results['power_per_element_MW']:8.4f} MW")
    print("-" * 72)
    print(f" k_infinite (lattice)       : {results['k_infinite']:8.4f}")
    print(f" Breeding ratio             : {results['breeding_ratio']:8.4f}")
    print(f" Pu-239 production          : {results['pu239_production_rate_kg_per_day']:8.5e} kg/day")
    print(f" Realized burnup            : {results['realized_burnup_MWd_per_tU']:8.1f} MWd/tU")
    print(f" Target burnup              : {results['target_burnup_MWd_per_tU']:8.1f} MWd/tU")
    print(f" Max fuel temperature       : {results['max_fuel_temp_C']:8.1f} deg C")
    print(f" Thermal margin to Na boil  : {results['thermal_margin_K']:8.1f} K")
    print("-" * 72)
    print(" Pu-239 inventory over the cycle (3-month increments):")
    inv = results["pu239_inventory_kg_over_time"]
    print(f"   Month     Pu-239 [kg]")
    for m, v in zip(inv["month"], inv["pu239_kg"]):
        print(f"   {m:5.1f}   {v:12.5f}")
    print("=" * 72)
    print(f" Cycle length: {cycle_days} days ({cycle_days/30.44:.1f} months)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="default", help="named case (default)")
    ap.add_argument("--burnup-days", type=int, default=365, help="cycle length [days]")
    ap.add_argument("--out", default=None, help="output directory for JSON results")
    args = ap.parse_args(argv)

    params = build_case(args.case)
    rx = reactor_mod.Reactor(params)
    results = rx.run_case(params)

    # Scale the Pu-239 inventory timeline to the requested cycle length.
    cycle_days = args.burnup_days
    results["cycle_days"] = cycle_days

    print_report(results, cycle_days)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_file = os.path.join(args.out, f"case_{args.case}_results.json")
        # Make a JSON-safe copy (arrays -> lists, drop nested channel dict).
        safe = json_safe(results)
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(safe, fh, indent=2, allow_nan=False)
        print(f"\n Wrote JSON results -> {out_file}")
    return 0


def json_safe(obj):
    """Recursively convert numpy types / arrays to JSON-serializable forms."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items() if k not in ("channel",)}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(float(v)) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
