"""
sim.neutronics - the "serious" fission-neutronics engine
=========================================================

A multi-group neutron-diffusion criticality + point-kinetics + leakage model
of a bare (or reflected) fissile sphere, calibrated to *measured* Los Alamos
critical masses (Hansen & Paxton, LA-4208) and grounded in the Terrell MCNP
leakage spectra (LA-11807). Strictly nuclear-physics layer: criticality,
chain-reaction kinetics, energy release, and leakage. No device engineering.

Modules
-------
    materials   - isotope table, calibration targets, calibrated constants
    sphere      - 2-group (and 1-group) bare-sphere k_eff(R), R_crit, flux
    reflector   - two-region core + reflector match
    calibrate   - offline procedure that fits the constants to measured masses
    kinetics    - point kinetics on the spatial model

This package is the engine. The public facade (`sim.criticality`), the CLI
(`sim.main`), and the leakage/dose layer (`sim.leakage`) re-export it so the
existing entry points and the interactive blueprint keep working.
"""

from . import materials
from . import sphere
from . import reflector
from . import kinetics

__all__ = ["materials", "sphere", "reflector", "kinetics"]
