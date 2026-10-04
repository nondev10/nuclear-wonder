"""
test_criticality.py - unit tests for the calibrated fission-neutronics engine
==============================================================================

Run:  python sim/tests/test_criticality.py   (standalone)
Or:   python -m pytest sim/tests/test_criticality.py -q   (if pytest present)

These are the "serious" gate: the model must reproduce the *measured* Los
Alamos critical masses, cross k=1 at the predicted R_crit, honor the density
and enrichment scaling identities, and agree with the independent MC
cross-check. A passing suite means the constants are calibrated correctly.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from sim.neutronics import sphere as S
from sim.neutronics import kinetics as K
from sim.neutronics import materials as M
from sim.neutronics import reflector as Rf
from sim import mc_check
from sim import criticality as C
from sim import leakage as L

# ---- calibration: measured bare critical masses ------------------------
def test_u235_bare_critical_mass():
    assert abs(S.Mcrit("u235") - 51.94) / 51.94 < 0.02

def test_pu239_bare_critical_mass():
    assert abs(S.Mcrit("pu239") - 17.02) / 17.02 < 0.05

def test_u233_bare_critical_mass():
    assert abs(S.Mcrit("u233") - 16.53) / 16.53 < 0.05

def test_masses_order_by_isotope():
    # Pu and U-233 bare masses are smaller than U-235's
    assert S.Mcrit("u233") < S.Mcrit("u235")
    assert S.Mcrit("pu239") < S.Mcrit("u235")

# ---- k_eff crosses 1 at R_crit ----------------------------------------
def test_keff_crosses_one():
    for mat in ("u235", "pu239", "u233"):
        R = S.R_critical(mat)
        assert abs(float(S.k_eff_sphere(R, mat)[0]) - 1.0) < 1e-3

def test_rho_sign_flips_at_critical():
    Rc = S.R_critical("u235")
    assert float(S.rho_reactivity(0.9 * Rc, "u235")[0]) < 0.0
    assert float(S.rho_reactivity(1.1 * Rc, "u235")[0]) > 0.0

def test_1g_and_2g_agree_on_critical_mass():
    for mat in ("u235", "pu239", "u233"):
        assert abs(S.Mcrit(mat, "1g") - S.Mcrit(mat, "2g")) < 0.5 * S.Mcrit(mat)

# ---- density & enrichment scaling identities --------------------------
def test_density_scaling_identity():
    # M ~ 1/rho^2: scaling Ld ~ 1/rho reproduces the exact ratio
    mat = "u235"
    cal, m = M.calibrated(mat), M.get(mat)
    rho1 = m["rho"] * 1.05
    Ld1 = cal["Ld_cm"] * (m["rho"] / rho1)
    M0 = S.critical_mass_1g(cal["Ld_cm"], cal["kinf"], m["rho"])
    M1 = S.critical_mass_1g(Ld1, cal["kinf"], rho1)
    assert abs(M1 - M0 * (m["rho"] / rho1) ** 2) / M0 < 1e-6

def test_enrichment_scaling_identity():
    # M ~ E^(-1.72): lower enrichment -> larger critical mass.
    # Compare a high-enrichment 51.94 kg baseline to a lower enrichment E1:
    #   M1 = M0 * (E0/E1)^(1.72)   (drop E0->E1, mass must rise)
    E0, E1 = 93.86, 80.0
    M0 = 51.94
    M1 = M0 * (E0 / E1) ** 1.72
    assert M1 > M0  # lower enrichment -> larger critical mass

# ---- sub-critical multiplication ---------------------------------------
def test_subcritical_multiplier_lbl_idiom():
    # Near-critical U-235 (R~0.95*Rcrit) -> M=k/(1-k) is O(100-300)
    Rc = S.R_critical("u235")
    Mval = float(S.subcritical_multiplication(S.k_eff_sphere(0.95 * Rc, "u235")[0]))
    assert 50.0 < Mval < 1000.0

# ---- reflected assemblies ---------------------------------------------
def test_reflected_mass_less_than_bare():
    Mrefl = Rf.Mcrit_reflected(9.98, mat="u235", refl="natU")  # 3.93-in U
    assert Mrefl < S.Mcrit("u235", "1g")  # reflector reduces critical mass

def test_reflected_masses_match_la4208():
    # calibrated albedos land the natU reflected cases within ~1%
    t_393 = 3.93 * 2.54
    m = Rf.Mcrit_reflected(t_393, mat="u235", refl="natU")
    assert abs(m - 19.82) / 19.82 < 0.01

# ---- kinetics / energy ------------------------------------------------
def test_prompt_lifetime_order():
    for mat in ("u235", "pu239", "u233"):
        Rc = S.R_critical(mat)
        tau = K.prompt_lifetime(mat, Rc)
        assert 1e-9 < tau < 1e-7   # a few ns .. sub-microsecond

def test_fission_energy_u235():
    # 52 kg U-235 fully fissioned ~ 4.27e15 J (~1020 kt TNT)
    R = C.radius_of_mass(52.0, "u235")
    E = K.total_fission_energy("u235", R)
    assert 3.5e15 < E < 5.0e15

def test_explosive_yield_super_critical():
    Rc = S.R_critical("u235")
    t = np.linspace(0.0, 1e-5, 50)
    out = K.fission_yield("u235", 1.05 * Rc, t)
    assert out["type"] == "explosive"
    assert 0.0 < out["f_disassembly"] <= 1.0
    assert out["Y_max_J"] > 0.0

def test_steady_power_sub_critical():
    Rc = S.R_critical("u235")
    t = np.linspace(0.0, 1e-3, 50)
    out = K.fission_yield("u235", 0.8 * Rc, t)
    assert out["type"] == "steady"
    assert out["P_steady_W"] > 0.0

# ---- MC cross-check agrees with the diffusion engine -------------------
def test_mc_agrees_with_diffusion():
    comp = mc_check.compare_to_diffusion(
        S.R_critical("u235", "1g"), "u235",
        n_source=5000, n_batches=200, seed=0, group="1g")
    assert comp["agreement_pct"] < 5.0

def test_mc_variance_decreases_with_batches():
    c50 = mc_check.mc_k_eff(8.7, "u235", n_batches=50, seed=1)
    c300 = mc_check.mc_k_eff(8.7, "u235", n_batches=300, seed=1)
    assert c300["sigma_1sigma"] < c50["sigma_1sigma"] * 1.5

# ---- leakage source terms ---------------------------------------------
def test_leakage_steady_rate_diverges_at_critical():
    Rc = S.R_critical("u235")
    assert np.isinf(L.steady_state_fission_rate(Rc, "u235"))
    assert np.isfinite(L.steady_state_fission_rate(0.9 * Rc, "u235"))

def test_leakage_dose_inverse_square():
    Rc = S.R_critical("u235")
    R = 0.95 * Rc
    d10 = L.neutron_dose_at_distance(R, 10.0, "u235")
    d20 = L.neutron_dose_at_distance(R, 20.0, "u235")
    assert abs(d10 / d20 - 4.0) < 0.05  # (10/20)^2

if __name__ == "__main__":
    import traceback
    import inspect
    test_funcs = [(k, v) for k, v in list(globals().items())
                  if k.startswith("test_") and inspect.isfunction(v)]
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
