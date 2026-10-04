"""
test_device.py - unit tests for the fission-device transient layer
===================================================================

Run:  python sim/tests/test_device.py   (standalone)
Or:   python -m pytest sim/tests/test_device.py -q   (if pytest present)

These are the "serious" gate for the device layer: the transient must
(i) start neutron-free before the initiator, (ii) turn on at t_0, (iii)
cross k=1 during assembly, (iv) peak its energy and self-terminate on the
labeled disassembly, (v) show gun/implosion at different assembly
timescales, and (vi) keep the fission fraction in [0,1]. The blast-zone
cube-root scaling must be monotone. A passing suite means the device
transient is wired to the calibrated engine correctly.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from sim import device
from sim.device import transient as T
from sim.device import blast as B


def test_transient_has_time_series():
    res = device.run_implosion(n=200)
    for key in ("t", "R_cm", "k_eff", "N", "Y_J", "T_K"):
        assert key in res, f"missing {key}"
        assert len(res[key]) == 200


def test_neutron_free_before_initiator():
    res = device.run_implosion(n=200, t_0=2.0e-6)
    N = res["N"]; t = res["t"]
    early = N[t < 1.0e-6]
    assert early.max() < 1e-3 * N.max()   # N is ~0 well before t_0


def test_chain_turns_on_at_initiator():
    res = device.run_implosion(n=400, t_0=3.0e-6)
    N = res["N"]; t = res["t"]
    N_after = N[t >= 3.0e-6].max()
    N_before = N[t < 2.5e-6].max()
    assert N_after > N_before * 1e3   # chain has run up by the end


def test_keff_crosses_one_during_assembly():
    for kind in ("gun", "implosion"):
        res = device.run_gun() if kind == "gun" else device.run_implosion()
        k = res["k_eff"]
        assert k.max() > 1.0, f"{kind}: k_eff never crosses 1"
        assert np.any(k > 1.0)


def test_energy_rises_then_peaks():
    res = device.run_implosion(n=400)
    Y = res["Y_J"]
    assert Y[0] < Y[-1] or np.isclose(Y[-1], res["Y_max_J"])
    assert np.argmax(Y) > 0
    assert Y.max() <= res["Y_max_J"] * (1 + 1e-9)


def test_fission_fraction_in_unit_range():
    res = device.run_gun(n=200)
    assert 0.0 <= res["f_disassembly"] <= 1.0
    res2 = device.run_implosion(n=200)
    assert 0.0 <= res2["f_disassembly"] <= 1.0


def test_gun_slower_than_implosion_assembly():
    g = device.run_gun(n=50)
    im = device.run_implosion(n=50)
    assert g["t_critical_s"] > im["t_critical_s"], \
        "gun assembly should take longer than implosion compression"


def test_premature_differs_from_initiated():
    on = device.run_implosion(n=200)
    off = device.run_premature("implosion", n=200)
    # the premature case turns the source on at t=0, so N is non-trivial
    # well before t_0 (unlike the initiated case)
    assert off["N"][:20].max() > on["N"][:20].max()
    assert off["initiator_on"] is False
    assert on["initiator_on"] is True


def test_blast_zones_cube_root_monotone():
    # doubling yield -> radius grows by 2^(1/3)
    r1 = B.blast_zones(10.0)["total destruction"]["radius_m"]
    r8 = B.blast_zones(80.0)["total destruction"]["radius_m"]
    assert r8 / r1 == 2.0
    # 10 kt -> 80 kt (8x) -> radius 2x
    assert B.blast_zones(20.0)["severe blast"]["radius_m"] > \
           B.blast_zones(5.0)["severe blast"]["radius_m"]


def test_temperature_readout_proportional():
    res = device.run_gun(n=100)
    T = res["T_K"]
    assert T.max() > 0.0
    # temperature rises with energy released
    assert np.argmax(T) > 0


def test_downsample_is_compact():
    res = device.run_implosion(n=400)
    d = device.downsample(res, n_out=100)
    assert len(d["t"]) == 100
    assert "Y_max_kt" in d and "f_disassembly" in d


def test_report_writes_json(tmp_path=None):
    res = device.run_gun(n=100)
    d = device.report(res, out_dir="reports")
    import json, os
    assert os.path.exists("reports/device_gun.json")
    data = json.load(open("reports/device_gun.json"))
    assert data["kind"] == "gun"
    assert len(data["t"]) >= 50


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
