"""
Monte Carlo verification of the Pu-239 production estimate.

Physics
-------
The 2-group diffusion model in ``src/channel_model.py`` (driven by the
neutron library in ``src/neutronics.py``) predicts a Pu-239 production
rate for a CANDU-type channel with U-238/THU fuel plates in a 1-D
channel geometry.  We cross-check it here with an OpenMC Monte Carlo
run over the same 1-D geometry.

Why expect agreement within ~5%:

* The channel is a 1-D, slab-like core (r = 2.5 m, h = 3 m) that is
  essentially well-thermalised by heavy water.  The dominant reaction
  (n,gamma) on U-238 is a capture reaction, and capture cross-sections
  are well tabulated in evaluated nuclear data.
* For a core this small (a few hundred cm in the axial direction,
  ~2.5 m radially) the diffusion approximation (flux ~ 1/r, smooth
  1-D) tracks the transport solution closely: neither group has a
  large migration length relative to the core dimension.
* The Np-239 beta decay to Pu-239 (t1/2 = 2.36 d) is a slow
  chemistry step; the (n,gamma) capture rate on U-238 measured in
  OpenMC directly bounds the Pu-239 production when Np-239 is
  stable on the time scale of interest.

If the gap is larger than 5%:

1. Bump the particle count to 1e7 and re-run -- the default 1e6
   histories give ~0.3% relative statistical uncertainty on a well-
   converged (n,gamma) rate; going to 1e7 cuts that by another sqrt(10)
   and the residual bias is statistical, not systematic.
2. Confirm the cross-sections used in ``src/neutronics.py`` are the
   same library family as those loaded by OpenMC here (ACE vs. JEFF).
   Re-tune the 2-group constants in ``src/neutronics.py`` if they
   diverge.
3. Confirm the U-238 / THU ratio in the fuel plates is the same in
   both models -- the capture rate scales linearly with U-238 atom
   density, so a 10% mass discrepancy propagates directly.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------
#  OpenMC availability check -- run as a plain file with a useful
#  install hint when the library is missing.
# ---------------------------------------------------------------------
try:
    import openmc
    import openmc.lib  # noqa: F401 -- keep this out if lib isn't needed
except ImportError:
    print(
        "OpenMC not installed.  Run:\n"
        "    pip install openmc\n"
        "or:  conda install -c conda-forge openmc\n",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------
#  Local project modules
# ---------------------------------------------------------------------
# ``src/channel_model`` exposes the 1-D channel geometry dimensions.
# ``src/neutronics`` exposes pu239_production_rate_per_element, which
# returns the 2-group estimate in kg/day for a given fuel composition.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.channel_model import (  # type: ignore[attr-defined]
    CORE_RADIUS_M,
    CORE_ACTIVE_HEIGHT_M,
)
from src.neutronics import pu239_production_rate_per_element  # type: ignore[import-not-found]


# =====================================================================
#  Fuel / material definitions matching channel_model.py
# =====================================================================
# Core fuel composition: U-238 in a heavy-uranium-matrix (THU) plate.
# U-238 mass fraction in the fuel plates: 0.55; the rest is Th-232
# matrix.  (These constants mirror src/channel_model.py; if the channel
# model uses different values, keep them in sync here.)
FUEL_U238_MASS_FRAC = 0.55
FUEL_TH232_MASS_FRAC = 0.45
MODERATOR_DENSITY_G_CM3 = 1.11   # D2O
COOLANT_NA_DENSITY_G_CM3 = 0.97  # liquid Na at ~300 C


def _openmc_fuel_plate() -> "openmc.Material":
    """Fuel plate: U-238 + Th-232 in the same volume fraction as
    channel_model.py.  Enrichment is low -- this is a CANDU-type
    breeder channel where the U-238 is the fuel."""
    m = openmc.Material(name="U238_THU_fuel")
    m.add_element("U", 1.0, enriched_element="U-238")
    m.add_element("Th", FUEL_TH232_MASS_FRAC, enriched_element="Th-232")
    m.depletionable = True
    m.temperature = 300.0  # C
    return m


def _openmc_heavy_water() -> "openmc.Material":
    m = openmc.Material(name="D2O_moderator")
    m.add_element("D", 1.0, 10000.0)
    m.add_element("O", 0.5, 10000.0)
    m.set_density(density=MODERATOR_DENSITY_G_CM3, units="g/cm3")
    return m


def _openmc_sodium() -> "openmc.Material":
    m = openmc.Material(name="Na_coolant")
    m.add_element("Na", 1.0)
    m.set_density(density=COOLANT_NA_DENSITY_G_CM3, units="g/cm3")
    return m


def _openmc_calandria() -> "openmc.Material":
    m = openmc.Material(name="calandria_zircaloy")
    m.add_element("Zr", 0.01)
    m.add_element("Nb", 0.99)
    m.set_density(density=8.0, units="g/cm3")
    return m


# =====================================================================
#  Geometry -- 1-D channel, same envelope as channel_model.py
# =====================================================================
def build_geometry(fuel, mod, na, cal):
    """Return (cells, materials, universe) for the 1-D channel.

    The channel is a cylinder of radius CORE_RADIUS_M (=2.5 m) and
    height CORE_ACTIVE_HEIGHT_M (=3 m).  It is surrounded by heavy
    water moderator on all four sides (outer infinite moderator region)
    and capped with calandria zircaloy sheets on the top and bottom
    active-ends.  Sodium fills the gap between the fuel bundle and the
    calandria shell -- a thin annular coolant channel.

    This is a *reasonable CAD*, not a full plant model; we keep the
    1-D channel character so that the transport flux profile is
    comparable to the 1-D diffusion flux.
    """
    #  Geometry constants (in cm) so we can use openmc built-in shapes
    #  conveniently; openmc works in cm internally.
    R_fuel_bundle = 25.0      # cm, radius of the fuel bundle
    R_calandria    = 40.0     # cm, inner calandria radius
    R_outer        = 250.0    # cm, outer moderator boundary = CORE_RADIUS
    H_active       = CORE_ACTIVE_HEIGHT_M * 100.0
    H_total        = H_active + 2.0 * R_calandria  # top & bottom sheets

    # ----- Regions -----
    # Fuel: solid cylinder, radius R_fuel_bundle, height H_active
    fuel_cell = openmc.Cell(name="U238_THU_fuel")
    fuel_cell.fill = fuel
    fuel_cell.region = (openmc.z_max < H_active / 2) & \
                       (openmc.z_min > -H_active / 2) & \
                       (openmc.r_max < R_fuel_bundle)

    # Sodium annulus between fuel bundle and calandria
    na_cell = openmc.Cell(name="Na_coolant")
    na_cell.fill = na
    na_cell.region = (openmc.r_max < R_calandria) & \
                     (openmc.z_max < H_total / 2) & \
                     (openmc.z_min > -H_total / 2) & \
                     (openmc.r_min > R_fuel_bundle)

    # Calandria sheet (top + bottom)
    cal_cell = openmc.Cell(name="calandria")
    cal_cell.fill = cal
    cal_cell.region = ((openmc.z_max > H_active / 2) |
                       (openmc.z_min < -H_active / 2)) & \
                      (openmc.z_max < H_total / 2) & \
                      (openmc.z_min > -H_total / 2) & \
                      (openmc.r_max < R_calandria)

    # Moderator: fills the outer annulus and the top/bottom outside
    mod_cell = openmc.Cell(name="D2O_moderator")
    mod_cell.fill = mod
    mod_cell.region = (openmc.r_max < R_outer) & \
                      (openmc.z_max < H_total / 2) & \
                      (openmc.z_min > -H_total / 2) & \
                      (openmc.r_min > R_calandria)

    # Infinite void (background)
    bg_cell = openmc.Cell(name="background_void")
    bg_cell.fill = None
    bg_cell.region = ~mod_cell.region & ~cal_cell.region & \
                    ~na_cell.region & ~fuel_cell.region

    # ----- Universe / geometry -----
    universe = openmc.Universe(cells=[fuel_cell, na_cell, cal_cell,
                                       mod_cell, bg_cell])
    model = openmc.Model(geometry=universe)
    model.settings.particles = 100_000   # placeholder; overridden below
    model.settings.run_mode = "fixed source"
    model.source = openmc.Source(
        space=openmc.stats.Box([0, 0, 0], [0, 0, H_active]),
        angle=openmc.stats.Isotropic(),
        nuclide="U-238" if False else "D",  # D-source for D2O
    )
    return fuel_cell, model, [fuel, mod, na, cal]


# =====================================================================
#  Tallies
# =====================================================================
def add_pu239_tallies(model, fuel_cell):
    """Attach a (n,gamma) capture tally on U-238 in the fuel region
    and a Np-239 (n,2n) rate, then integrate to a production estimate.

    Practical shortcut for Pu-239 production: the slowest step in the
    U-238 -> Np-239 -> Pu-239 chain is the Np-239 beta decay
    (t1/2 = 2.36 d).  In steady state Np-239 is in equilibrium with
    U-238 capture, so dP/dt = (n,gamma) capture rate on U-238.
    We therefore score U-238 capture (n,gamma) in the fuel cell and
    convert to kg/day at the end.
    """
    # --- Tally 1: (n,gamma) capture rate on U-238 in the fuel plate
    u238_capture = openmc.Tally(name="U-238_capture_rate")
    u238_capture.nuclides = ["U-238"]
    u238_capture.reactions = ["capture"]
    u238_capture.filters = [openmc.CellFilter(fuel_cell)]
    u238_capture.output = {"rate": True}
    model.tallies.append(u238_capture)

    # --- Tally 2: total absorption in the fuel (sanity)
    u238_abs = openmc.Tally(name="U-238_absorption")
    u238_abs.nuclides = ["U-238"]
    u238_abs.reactions = ["abs"]
    u238_abs.filters = [openmc.CellFilter(fuel_cell)]
    u238_abs.output = {"rate": True}
    model.tallies.append(u238_abs)

    # --- Tally 3: Np-239 (n,gamma) in the same cell (decay product)
    #     Scored for cross-check; Np-239 is transient so this is a
    #     lower bound, not the dominant channel.
    np239_cap = openmc.Tally(name="Np-239_capture_rate")
    np239_cap.nuclides = ["Np-239"]
    np239_cap.reactions = ["capture"]
    np239_cap.filters = [openmc.CellFilter(fuel_cell)]
    np239_cap.output = {"rate": True}
    model.tallies.append(np239_cap)

    return [u238_capture, u238_abs, np239_cap]


# =====================================================================
#  Cross-section / material setup
# =====================================================================
def configure_materials(fuel, mod, na, cal):
    """Load ACERENDU / endf6 / jeff data so the capture rate is
    evaluated against the real evaluated data.  All four materials
    get cross_sections so that transport uses real data, not just
    the thermal group."""
    for m in (fuel, mod, na, cal):
        m.add_cross_sections("ENDFB7.1")   # or "JEFF3.3"; keep in sync
    openmc.mg_library  # unused reference; kept for style
    openmc.add_cross_sections("ENDFB7.1")


# =====================================================================
#  Runner
# =====================================================================
def run_mc(particles: int) -> dict:
    """Build the model, run OpenMC, and return tallies + run stats.

    Returns a dict with keys:
        'u238_capture' : openmc tally result
        'particles'    : int
        'est_pu239_kg_day' : float, in kg/day
    """
    fuel = _openmc_fuel_plate()
    mod  = _openmc_heavy_water()
    na   = _openmc_sodium()
    cal  = _openmc_calandria()
    configure_materials(fuel, mod, na, cal)

    fuel_cell, model, materials = build_geometry(fuel, mod, na, cal)
    add_pu239_tallies(model, fuel_cell)

    # ----- Settings -----
    model.settings.particles = int(particles)
    model.settings.batches   = 400
    model.settings.inactive = 400
    model.settings.run_mode = "fixed source"
    model.settings.seeded    = True
    model.settings.max_history = 1.0e6

    # ----- Run -----
    model.run()
    results = openmc.export_assemblies() if False else None

    # ----- Read back the U-238 capture rate from the tally ---
    t = model.tallies[0]
    rate = t.get_values()[0]    # captures / cm^3 / s in the fuel cell

    # ----- Convert to Pu-239 production in kg/day ------
    # 1 capture = 1 Np-239 atom -> (after ~2.36 d beta decay) 1 Pu-239
    # atom.  For a steady-state channel, kg/day = rate * V * 24 * 3600
    #   where V = fuel volume in cm^3.
    # Pu-239 molar mass = 239.05 g/mol;  N_A = 6.022e23.
    R_cm = 25.0
    H_cm = CORE_ACTIVE_HEIGHT_M * 100.0
    V_fuel_cm3 = 3.14159 * R_cm ** 2 * H_cm / 2.0
    pu_molar_mass = 239.05      # g/mol
    N_A = 6.022e23
    kg_day = rate * V_fuel_cm3 * 86400.0 * pu_molar_mass / N_A / 1000.0
    return {
        "particles": particles,
        "u238_capture": rate,
        "est_pu239_kg_day": kg_day,
        "fuel_volume_cm3": V_fuel_cm3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--particles", type=int, default=1_000_000,
        help="Number of OpenMC histories (default 1e6).",
    )
    args = ap.parse_args(argv)

    print(f"[mc_verification] Running OpenMC with {args.particles:,} "
          f"histories...")
    res = run_mc(args.particles)
    est = res["est_pu239_kg_day"]
    print(f"[mc_verification] MC estimate: {est:.6g} kg/day Pu-239")
    print(f"[mc_verification]   U-238 capture rate: "
          f"{res['u238_capture']:.4e} cap/cm3/s")

    # ----- 2-group reference -----
    # Mirrors the fuel composition in channel_model.py / _openmc_fuel_plate.
    fuel_composition = {
        "U-238_frac": FUEL_U238_MASS_FRAC,
        "Th-232_frac": FUEL_TH232_MASS_FRAC,
        "D2O": True,
        "Na": True,
    }
    ref_kg_day = pu239_production_rate_per_element(fuel_composition)
    print(f"[mc_verification] 2-group reference: {ref_kg_day:.6g} kg/day")

    pct_diff = 100.0 * abs(est - ref_kg_day) / max(abs(ref_kg_day), 1e-30)
    print(f"[mc_verification] Percent difference: {pct_diff:.2f}%")

    threshold = 5.0
    if pct_diff < threshold:
        print(f"[mc_verification] PASS -- within {threshold}% "
              f"(pct_diff = {pct_diff:.2f}%)")
        return 0
    else:
        print(
            f"[mc_verification] FAIL -- pct_diff {pct_diff:.2f}% >= "
            f"threshold {threshold}%.\n"
            "  Try: (1) bump --particles to 1e7; "
            "  (2) verify cross-section library in src/neutronics.py "
            "  matches the one used by OpenMC here; "
            "  (3) re-check the U-238 mass fraction.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
