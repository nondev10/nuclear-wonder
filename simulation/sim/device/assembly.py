"""
assembly.py - device assembly drivers: R(t), rho(t), M_assembled(t)
===================================================================

The *new physical input* to the device transient. The fission-neutronics
engine (sim.neutronics) is a steady-sphere model: it answers "for a sphere
of radius R and density rho, what is k_eff?". A bomb is not a sphere that
magically appears - it is *assembled* from subcritical pieces, and the
assembly itself is time-dependent. This module produces that time
dependence: two drivers, each returning R(t) / rho(t) / M_assembled(t) so
the transient (sim.device.transient) can march criticality forward in time.

Both drivers are grounded in the 1967 Nth Country Experiment report
(UCRL-50239, "The Nth Country Fission Weapon Design"), which is the
device-physics backbone for this package:

    "There are two general techniques for assembling a supercritical
     mass ... the gun method where subcritical masses are 'shot' together,
     and the implosion method where a subcritical mass is made
     supercritical by compression."
        - Nth Country report, I. Basic Concepts

And, crucially:

    "The most important feature of the assembly times ... is that they are
     more than 10 times the total fission chain reaction time. This means
     that it is essential for the fissile material to be neutron free
     during the assembly or the reaction will take place prematurely."

The *ordering* t_assembly >> t_chain is the physical reason the initiator
exists (the transient turns the neutron source on at t_0 = t_assembly, not
at t=0). The absolute device timescales are labeled order-of-magnitude /
literature constants, NOT design specs - the device package is explicitly
educational and does not model weapon engineering.

Honest limits (declared, not hidden):
  * Gun muzzle / projectile velocity, implosion compression time & factor,
    and the initiator source strength are labeled estimates, not measured
    design values.
  * The "effective radius" of the combined gun assembly is a 1-D teaching
    proxy (the radius of the assembled mass treated as a bare sphere), not
    a full 3-D gun-geometry solve.
"""

import numpy as np

from ..neutronics import materials as M
from ..neutronics import sphere as S

# ----------------------------------------------------------------------
#  Labeled order-of-magnitude device constants (NOT design specs)
# ----------------------------------------------------------------------
GUN_PROJECTILE_M_S = 2.0e3    # ~2 km/s light-assembly muzzle velocity
GUN_LIGHT_MASS_FRAC = 0.36    # light sub-critical hemisphere fraction of M_crit
GUN_BARREL_LENGTH_CM = 12.0   # light-assembly flight path before overlap
GUN_OVERLAP_CM = 6.0         # overlap distance over which it goes critical
# The fully-seated gun assembly (light + heavy + U-238 tamper credit) lands
# *above* the bare critical mass: the light plug (light_frac) plus the heavy
# barrel plus the reflecting U-238 tamper make the assembled mass
# GUN_ASSEMBLY_OVERSHOT * M_crit, i.e. meaningfully super-critical. This is
# the physical point of a gun device: two sub-critical pieces plus a tamper
# that together exceed criticality.
GUN_ASSEMBLY_OVERSHOT = 1.20

IMPL_COMPR_TIME_S = 3.0e-6    # ~3 us implosion compression window
# Implosion drives the core to a radius meaningfully BELOW the critical
# radius: the compressed core is ~ IMPL_SUBCRIT_FRACTION * R_crit, a strong
# super-critical compression (this is the physical point of an implosion
# device - compression, not mass addition). R0 starts at 1.35*R_crit
# (sub-critical margin) and is compressed to that target.
IMPL_SUBCRIT_FRACTION = 0.62   # R_compressed / R_crit  (strong compression)
IMPL_COMPR_FACTOR = None       # legacy default replaced by IMPL_SUBCRIT_FRACTION


def radius_of_mass(M_kg, mat):
    """Sphere radius (cm) of a fissile mass M_kg (kg) at material density.

    cm = (3000*M_kg / (4*pi*rho[g/cm^3]))^(1/3)  -> cm
    """
    m = M.get(mat)
    return (3000.0 * M_kg / (4.0 * np.pi * m["rho"])) ** (1.0 / 3.0)


def mass_of_radius(R_cm, mat, rho=None):
    """Fissile mass (kg) of a sphere of radius R_cm. rho override (g/cm^3)."""
    m = M.get(mat)
    if rho is None:
        rho = m["rho"]
    return (4.0 / 3.0) * np.pi * rho * R_cm ** 3 / 1000.0


def gun_trajectory(t, mat="u235", m_s=GUN_PROJECTILE_M_S,
                   barrel_cm=GUN_BARREL_LENGTH_CM,
                   overlap_cm=GUN_OVERLAP_CM, light_frac=GUN_LIGHT_MASS_FRAC,
                   n=400):
    """Gun-assembly driver: an assembled effective radius R(t), rho(t), M(t).

    Picture (Little Boy): a light sub-critical hemisphere ("the plug") is
    shot through a heavy sub-critical U-235 barrel (the two pieces are
    individually sub-critical; only together do they exceed M_crit). The
    light plug flies the barrel length in flight time t_flight = L/v, then
    overlaps the heavy end over overlap_cm, and the *combined* assembled
    mass ramps up to the full M_crit as it seats. The "effective radius"
    is the radius of the assembled mass treated as a bare sphere - a 1-D
    teaching proxy, not a 3-D gun solve.

    t is an array (s). Returns a dict with keys
    t, R_cm, rho, M_kg, t_critical_s, t_flight_s.
    """
    m = M.get(mat)
    Rc = S.R_critical(mat)
    M_crit = S.Mcrit(mat)

    # The fully-seated gun assembly (light + heavy + U-238 tamper credit)
    # lands above the bare critical mass, so the device is super-critical.
    M_assembled = GUN_ASSEMBLY_OVERSHOT * M_crit

    # light plug mass = light_frac of the assembled mass; the rest is the
    # heavy end + tamper credit, assembled as the plug seats.
    M_light = light_frac * M_assembled
    M_heavy = M_assembled - M_light

    # flight: light plug travels barrel_cm.
    t_flight = (barrel_cm * 1e-2) / m_s          # s

    # overlap: t_overlap = overlap_cm / m_s, during which assembled mass ramps
    # from M_light (seated plug) to M_light + M_heavy = M_assembled.
    t_overlap = (overlap_cm * 1e-2) / m_s
    t_critical = t_flight + t_overlap             # fully assembled / critical

    t = np.asarray(t, dtype=float)
    # assembled mass(t): 0 before flight ends, M_light -> M_assembled during
    # the overlap (when the heavy end + tamper come into the assembly).
    def _mass_of_tt(tt):
        tt = np.asarray(tt, dtype=float)
        # before the plug has reached the heavy end: no overlap yet.
        in_overlap = (tt >= t_flight) & (tt < t_critical)
        after = tt >= t_critical
        frac = np.clip((tt - t_flight) / t_overlap, 0.0, 1.0)
        Mas = np.where(after, M_assembled,
                       np.where(in_overlap, M_light + frac * M_heavy, 0.0))
        return Mas

    M_as = _mass_of_tt(t)
    R = np.where(M_as > 0, radius_of_mass(np.where(M_as > 0, M_as, 1e-9), mat), 0.0)
    return {"t": t, "R_cm": R, "rho": np.full_like(t, m["rho"]),
            "M_kg": M_as, "t_critical_s": t_critical,
            "t_flight_s": t_flight, "light_frac": light_frac,
            "M_assembled_kg": M_assembled, "M_crit_kg": M_crit, "R_crit_cm": Rc}


def implosion_trajectory(t, mat="pu239", t_comp_s=IMPL_COMPR_TIME_S,
                        subcrit_frac=IMPL_SUBCRIT_FRACTION, n=400):
    """Implosion driver: core radius R(t) and mass-conserving density rho(t).

    Picture (Fat Man): a Pu-239 sub-critical core of radius R0 is compressed
    by a spherically converging detonation wave to radius
    R_crit*subcrit_frac over t_comp_s. Compression reduces R (raises k, since
    leakage drops) and raises density (rho' = rho0*(R0/R)^3, mass-conserving)
    - BOTH drive the core past critical to a strongly super-critical state.
    The core mass is fixed; only R and rho change.

    t is an array (s). Returns a dict with keys
    t, R_cm, rho, M_kg, t_critical_s, R0_cm, R_comp_cm, subcrit_frac.
    """
    m = M.get(mat)
    Rc = S.R_critical(mat)
    M_crit = S.Mcrit(mat)

    # initial sub-critical core radius: start above critical (larger R),
    # compress to a radius well below R_crit. R0 = 1.35*R_crit (sub-critical
    # margin), R_compressed = Rc*subcrit_frac.
    R0 = 1.35 * Rc
    R_comp = Rc * subcrit_frac
    M_core = mass_of_radius(R0, mat, rho=m["rho"])   # fixed mass

    t = np.asarray(t, dtype=float)
    # compression progress p(t) in [0,1]; smoothstep so wavefront eases in.
    p = np.clip(t / t_comp_s, 0.0, 1.0)
    p = p * p * (3.0 - 2.0 * p)                       # smoothstep
    R = R0 + (R_comp - R0) * p
    # mass-conserving density (g/cm^3):  rho' = rho0*(R0/R)^3
    rho = m["rho"] * (R0 / R) ** 3
    # critical instant: first t where k_eff >= 1 (recomputed in transient);
    # here label the compression end-point as the reference.
    t_critical = t_comp_s
    return {"t": t, "R_cm": R, "rho": rho, "M_kg": np.full_like(t, M_core),
            "t_critical_s": t_critical, "R0_cm": R0, "R_comp_cm": R_comp,
            "subcrit_frac": subcrit_frac, "M_core_kg": M_core,
            "M_crit_kg": M_crit, "R_crit_cm": Rc}


def default_trajectory(kind, mat=None, n=400, **kwargs):
    """Dispatch on kind in {"gun","implosion"}; default material per kind."""
    mat = mat or ("u235" if kind == "gun" else "pu239")
    t = np.linspace(0.0, _device_timescale(kind, **kwargs), n)
    if kind == "gun":
        return gun_trajectory(t, mat, **kwargs)
    if kind == "implosion":
        return implosion_trajectory(t, mat, **kwargs)
    raise ValueError(f"unknown device kind {kind!r} (want 'gun' or 'implosion')")


def timescale(kind="gun", **kwargs):
    """Public accessor for the device wall-clock window (used by transient)."""
    return _device_timescale(kind, **kwargs)


def _device_timescale(kind, **kwargs):
    """Device wall-clock window (s): assembly time + a tail for the burst."""
    if kind == "gun":
        m_s = kwargs.get("m_s", GUN_PROJECTILE_M_S)
        t_assem = ((GUN_BARREL_LENGTH_CM + GUN_OVERLAP_CM) * 1e-2) / m_s
        return t_assem * 1.5 + 2.0e-5      # assembly + chain + tail
    if kind == "implosion":
        tc = kwargs.get("t_comp_s", IMPL_COMPR_TIME_S)
        return tc * 1.5 + 2.0e-5
    raise ValueError(f"unknown device kind {kind!r}")
