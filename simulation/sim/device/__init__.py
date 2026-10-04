"""
sim.device - the fission-device transient layer (educational)
============================================================

Turns the calibrated fission-neutronics engine (sim.neutronics) into a
time-dependent device model: an assembly driver (gun or implosion) feeds
R(t)/rho(t) into a time-marched point-kinetics transient that is
neutron-free during assembly and turns on at the initiator, bursts for a
few prompt generation times, then self-terminates on a labeled
hydrodynamic disassembly.

    assembly    - gun / implosion R(t), rho(t), M(t) drivers
    transient   - the time-marched device transient (the centerpiece)
    yield       - burst spectrum, temperature readout, blast-zone scaling

Public facade:
    run_gun(**kw)        -> device transient dict, initiator ON
    run_implosion(**kw)  -> device transient dict, initiator ON
    run_premature(kind)  -> device transient dict, initiator OFF (contrast)
    blast_zones(yield_kt) -> cube-root-scaled blast-zone radii
    report(res)          -> print + write reports/device_<kind>.json

Everything device-timescale is a labeled order-of-magnitude estimate, not a
design spec. The neutronics engine behind it is calibrated to the measured
Los Alamos critical masses. See docs/README.md and the module docstrings.
"""

import numpy as np

from . import assembly
from . import transient
from . import blast as yld

__all__ = ["assembly", "transient", "blast",
           "run_gun", "run_implosion", "run_premature",
           "blast_zones", "report", "downsample"]


def run_gun(n=400, **kw):
    """Gun-assembly device transient (U-235), initiator ON.

    Extra kwargs (labeled device constants, overridable): m_s, barrel_cm,
    overlap_cm, light_frac, t_0.
    """
    return transient.device_transient("gun", "u235", n=n, **kw)


def run_implosion(n=400, **kw):
    """Implosion device transient (Pu-239), initiator ON.

    Extra kwargs (labeled device constants, overridable): t_comp_s,
    compr_factor, t_0.
    """
    return transient.device_transient("implosion", "pu239", n=n, **kw)


def run_premature(kind="implosion", n=400, **kw):
    """The 'initiator off / premature fission' teaching contrast.

    The neutron source is present from t=0, so the chain starts *during*
    assembly rather than at t_0 - exactly the hazard the Nth Country
    report calls out.
    """
    mat = ("u235" if kind == "gun" else "pu239")
    return transient.device_transient(kind, mat, initiator_on=False, n=n, **kw)


def blast_zones(yield_kt, ref_kt=10.0):
    """Cube-root-scaled blast-zone radii (m). Labeled illustrative."""
    return yld.blast_zones(yield_kt, ref_kt)


def report(res, out_dir="reports"):
    """Print + write a device transient report. Returns the JSON dict."""
    return transient.report(res, out_dir=out_dir)


def downsample(res, n_out=200):
    """Compact JSON-able time series for the HTML blueprint."""
    return transient.downsample(res, n_out=n_out)
