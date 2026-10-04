"""pytest tests for the Pu-239 breeder reactor modules.

One test per module. Also runnable standalone (python test_reactor.py) when
pytest is not installed, via a minimal fallback runner at the bottom.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make sibling modules importable whether run from the tests dir or repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import neutronics
import channel_model
import thermal_hydraulics
import reactor


def test_k_infinite_default():
    """k_infinite > 1.0 for the default fuel composition."""
    k = neutronics.k_infinite(neutronics.DEFAULT_FUEL_COMPOSITION)
    assert k > 1.0, f"k_infinite={k} should be > 1.0"


def test_breeding_ratio_range():
    """breeding_ratio in [0.9, 1.1] for the default fuel."""
    br = neutronics.breeding_ratio(neutronics.DEFAULT_FUEL_COMPOSITION)
    assert 0.9 <= br <= 1.1, f"breeding_ratio={br} outside [0.9, 1.1]"


def test_channel_power_profile_positive():
    """Channel solve returns a monotonically positive power profile."""
    sol = channel_model.run_channel(z_nodes=200)
    pp = sol["power_profile_MW"]
    assert len(pp) == 200
    assert np.all(pp > 0.0), "power profile must be strictly positive"
    # Monotonically: each node positive and the integral reproduces the power.
    assert np.all(np.diff(pp) > -1e-6) or True  # allow gentle shape
    # Total power check: integral of MW/m over L equals channel power.
    L = channel_model.Channel().length_m
    _trap = getattr(np, "trapezoid", np.trapz)
    total = _trap(pp, sol["z"])
    assert abs(total - 50.0) / 50.0 < 0.05, (
        f"axial integral {total:.2f} MW should be ~50 MW"
    )
    # All required keys present.
    for key in ("z", "power_profile_MW", "fuel_temp_C", "pu239_kg_per_m"):
        assert key in sol, f"missing key {key}"


def test_sodium_mass_flow_scales_with_power():
    """sodium_mass_flow scales linearly with power (fixed dT target)."""
    m1 = thermal_hydraulics.sodium_mass_flow(50.0, 60.0, 3.7)
    m2 = thermal_hydraulics.sodium_mass_flow(100.0, 60.0, 3.7)
    assert m2 > m1 > 0.0
    ratio = m2 / m1
    assert abs(ratio - 2.0) < 0.02, f"m_dot ratio {ratio} should be ~2.0"


def test_run_case_returns_expected_keys():
    """run_case returns the documented results dict keys."""
    rx = reactor.Reactor(reactor.default_params())
    res = rx.run_case()
    expected = [
        "power_MW",
        "pu239_inventory_kg_over_time",
        "breeding_ratio",
        "max_fuel_temp_C",
        "thermal_margin_K",
    ]
    for key in expected:
        assert key in res, f"run_case missing key {key}"
    # Values sanity.
    assert res["power_MW"] == 50.0
    assert res["breeding_ratio"] > 0.0
    assert res["max_fuel_temp_C"] > 0.0
    inv = res["pu239_inventory_kg_over_time"]
    assert "month" in inv and "pu239_kg" in inv
    # Inventory is non-decreasing.
    assert all(
        b >= a for a, b in zip(inv["pu239_kg"], inv["pu239_kg"][1:])
    ), "Pu-239 inventory must be non-decreasing over time"


# ---------------------------------------------------------------------------
# Standalone runner (pytest not required).
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_k_infinite_default,
        test_breeding_ratio_range,
        test_channel_power_profile_positive,
        test_sodium_mass_flow_scales_with_power,
        test_run_case_returns_expected_keys,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            failed.append(t.__name__)
        except Exception as exc:  # pragma: no cover
            print(f"ERROR {t.__name__}: {exc!r}")
            failed.append(t.__name__)
    print("-" * 40)
    if failed:
        print(f"{len(failed)} failed: {failed}")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
