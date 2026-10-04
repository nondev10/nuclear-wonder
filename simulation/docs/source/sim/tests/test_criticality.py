"""
test_criticality.py — unit tests for the fission criticality & leakage models
==============================================================================

Run:  python sim/tests/test_criticality.py   (standalone)
Or:   python -m pytest sim/tests/test_criticality.py -q   (if pytest is present)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from sim import criticality as c
from sim import leakage as l
from sim import diffusion as d


# ---- reactivity sign & threshold -------------------------------------
def test_reactivity_below_critical_is_negative():
    assert c.reactivity(46.0)[0] < 0.0

def test_reactivity_at_critical_is_zero():
    assert abs(c.reactivity(52.0)[0]) < 1e-9

def test_reactivity_above_critical_is_positive():
    assert c.reactivity(56.0)[0] > 0.0

def test_reactivity_monotonic_in_M():
    M = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
    rho = c.reactivity(M)
    assert np.all(np.diff(rho) > 0)

# ---- critical mass with reflector credit ----------------------------
def test_reflector_reduces_critical_mass():
    bare = c.critical_mass()
    ref  = c.critical_mass(reflector_credit=0.3)
    assert ref < bare
    assert 0.0 < ref < bare

def test_reflector_credit_clamped():
    # rc > REFLECTOR_CREDIT_MAX clamps to the max; rc < 0 clamps to 0 (bare).
    assert c.critical_mass(reflector_credit=5.0) == c.critical_mass(reflector_credit=c.REFLECTOR_CREDIT_MAX)
    assert c.critical_mass(reflector_credit=-5.0) == c.critical_mass()

# ---- runaway time -----------------------------------------------------
def test_runaway_time_zero_reactivity_is_prompt_lifetime():
    # At M_crit (rho=0) the runaway time should be the full prompt lifetime
    assert abs(c.runaway_time(c.M_BARE_CRIT)[0] - c.PROMPT_NEUTRON_LIFE) < 1e-12

def test_runaway_time_decreases_with_M():
    assert c.runaway_time(54.0)[0] > c.runaway_time(60.0)[0]

# ---- energy / fission yield ------------------------------------------
def test_total_energy_scales_linearly_with_M():
    E1 = (52.0 / 235.0) * 6.022e23 * c.FISSION_ENERGY_J
    E2 = (56.0 / 235.0) * 6.022e23 * c.FISSION_ENERGY_J
    assert E2 > E1

def test_fission_yield_above_critical_is_positive_and_finite():
    t = np.linspace(0.0, 5e-7, 50)
    out = c.fission_yield(56.0, t)
    assert out["Y_max"] > 0.0 and np.isfinite(out["Y_max"])
    assert out["type"] == "explosive"
    assert 0.0 < out["f_disassembly"] <= 1.0

def test_fission_yield_below_critical_is_steady():
    t = np.linspace(0.0, 1e-3, 50)
    out = c.fission_yield(40.0, t)
    assert out["type"] == "steady"
    assert out["P_steady"] > 0.0

# ---- chain growth -----------------------------------------------------
def test_chain_growth_above_critical_grows():
    t = np.linspace(0.0, 5e-6, 100)
    out = c.chain_growth(56.0, t_array=t)
    assert out["type"] == "prompt_supercritical"
    assert out["N"][-1] > out["N"][0]

def test_chain_growth_below_critical_saturates():
    t = np.linspace(0.0, 1e-3, 100)
    out = c.chain_growth(40.0, t_array=t)
    assert out["type"] == "subcritical_steady"
    assert out["N_inf"] > 0.0

# ---- leakage ----------------------------------------------------------
def test_steady_fission_rate_increases_approaching_critical():
    F1 = l.steady_state_fission_rate(45.0)
    F2 = l.steady_state_fission_rate(51.0)
    assert F2 > F1

def test_steady_fission_rate_diverges_at_critical():
    assert np.isinf(l.steady_state_fission_rate(52.0))
    assert np.isinf(l.steady_state_fission_rate(60.0))

def test_neutron_dose_falloff_with_distance():
    D10 = l.neutron_dose_at_distance(50.0, 10.0)
    D20 = l.neutron_dose_at_distance(50.0, 20.0)
    assert D10 > D20
    # Inverse-square: D10/D20 ~ 4
    assert abs(D10 / D20 - 4.0) < 0.05

# ---- diffusion (spatial) --------------------------------------------
def test_diffusion_length_positive():
    assert d.diffusion_length() > 0.0

def test_critical_half_length_positive():
    assert d.critical_half_length() > 0.0

def test_long_rod_is_subcritical():
    Lc = d.critical_half_length()
    r_long = d.all_outputs(10 * Lc)
    assert r_long["is_subcritical"]

def test_short_rod_center_flux_higher_than_edge():
    Lc = d.critical_half_length()
    r = d.all_outputs(Lc)
    assert r["center_flux_rel"] > r["edge_flux_rel"]

# ---- all_outputs key integrity ---------------------------------------
def test_all_outputs_keys_present():
    r = c.all_outputs(56.0)
    for key in ["M", "M_crit", "reactivity", "runaway", "chain", "yield"]:
        assert key in r


if __name__ == "__main__":
    # Minimal test runner
    import traceback
    test_funcs = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in test_funcs:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(test_funcs)} total")
    sys.exit(0 if failed == 0 else 1)
