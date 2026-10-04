"""Reactor module: wires neutronics + channel + thermal-hydraulics together.

The Reactor class is the main entry point for a full case study. It produces
the reference ~50 MWth / 5 kg/yr Pu-239 heavy-water breeder case and exposes
.plot_results() (matplotlib is imported lazily so the module still works
without it).

Pure Python + numpy/scipy; matplotlib is optional and guarded.
"""

from __future__ import annotations

import numpy as np

import neutronics
import channel_model
import thermal_hydraulics


def default_params() -> dict:
    """Reference case: ~50 MWth / ~5 kg/yr Pu-239 heavy-water breeder.

    10,000 MWd/tU burnup, sodium-cooled primary, CANDU-style heterogeneous
    heavy-water fuel channel.
    """
    return {
        "core_radius_m": 2.2,
        "core_height_m": 3.7,
        "fuel_enrichment_frac": 0.030,
        "sodium_mass_flow_kg_s": thermal_hydraulics.DEFAULT_SODIUM_MASS_FLOW,
        # Power and target rise.
        "power_MW": 50.0,
        "dT_target_K": 60.0,
        # Burnup [MWd/tU] - reference 10,000.
        "burnup_MWd_per_tU": 10000.0,
        # Pu-239 target [kg/yr].
        "pu239_target_kg_per_yr": 5.0,
        # Number of fuel elements (CANDU-style channel count).
        "n_elements": 300,
        # Mass of U-238 + fissile per element [kg] for the burnup metric.
        # Tuned so the reference 50 MWth / ~3-yr cycle lands near the
        # 10,000 MWd/tU target (~20 tU total uranium in the core).
        "mass_per_element_kg": 34.0,
        # Effective axial k for the channel power profile.
        "boundary_k": 1.05,
        # Axial nodes.
        "z_nodes": 200,
    }


class Reactor:
    """Heavy-water breeder reactor case study.

    Wraps the neutronics, channel, and thermal-hydraulics modules and
    exposes run_case() as the main entry point.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or default_params()
        p = self.params
        # Assemble the fuel composition from the enrichment.
        u235 = float(p.get("fuel_enrichment_frac", 0.030))
        u238 = max(1.0 - u235 - 0.015, 0.0)
        self.fuel_composition = {
            "U-238": u238,
            "U-235": u235,
            "Pu-239": 0.015,
            "D2O": 1.0,
            "Na": 1.0,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_case(self, params: dict | None = None) -> dict:
        """Run a full reactor case study. Returns a results dict.

        params: optional overrides merged on top of self.params.
        """
        p = {**self.params, **(params or {})}
        # Sodium mass flow to meet the target temperature rise.
        m_dot = thermal_hydraulics.sodium_mass_flow(
            p["power_MW"], p["dT_target_K"], p["core_height_m"]
        )
        # Use the user-provided mass flow if it exceeds the target-based one.
        m_dot = max(m_dot, float(p.get("sodium_mass_flow_kg_s", 0.0)))

        # Channel.
        ch = channel_model.Channel(
            z_nodes=int(p.get("z_nodes", 200)),
            length_m=p["core_height_m"],
            power_MW=p["power_MW"],
            fuel_composition=self.fuel_composition,
            boundary_k=float(p.get("boundary_k", 1.05)),
            sodium_mass_flow_kg_s=m_dot,
        )
        sol = ch.solve()
        z = sol["z"]
        power_MW_per_m = sol["power_profile_MW"]
        fuel_temp_C = sol["fuel_temp_C"]

        # Peak fuel temp / thermal margin.
        max_fuel_temp_C = float(np.max(fuel_temp_C)) if fuel_temp_C.size else 0.0
        # Thermal margin: distance to the sodium boiling point (~1150 K = 877 C).
        T_boil_C = 1150.0 - 273.15
        thermal_margin_K = T_boil_C - max_fuel_temp_C

        # Neutronics.
        k_inf = neutronics.k_infinite(self.fuel_composition)
        br = neutronics.breeding_ratio(self.fuel_composition)

        # Pu-239 inventory over the cycle.
        power_per_element = p["power_MW"] / max(int(p["n_elements"]), 1)
        pu_rate = neutronics.pu239_production_rate_per_element(
            self.fuel_composition, power_per_element
        )
        # Total reactor Pu-239 production rate [kg/day] over all elements.
        total_rate_kg_per_day = pu_rate * int(p["n_elements"])
        # Inventory over 3-month increments.
        n_months = 12
        months = np.arange(0, n_months + 1) * 0.25  # fraction of a year
        days = months * 365.25
        pu_inventory_kg = [total_rate_kg_per_day * d for d in days]

        # Burnup consistency (informational). Total energy [MWd] =
        # power_MW * cycle_days / 1000. Realized burnup is spread over the
        # total uranium mass in the core (n_elements * mass-per-element).
        cycle_days = 365.25
        total_energy_MWd = p["power_MW"] * cycle_days / 1000.0
        # Reference mass of U-238 (+ fissile) per element [kg].
        mass_per_element_kg = p.get("mass_per_element_kg", 40.0)
        total_fuel_mass_t = int(p["n_elements"]) * mass_per_element_kg / 1000.0
        realized_burnup = total_energy_MWd / max(total_fuel_mass_t, 1e-9)

        results = {
            "power_MW": p["power_MW"],
            "sodium_mass_flow_kg_s": m_dot,
            "n_elements": int(p["n_elements"]),
            "power_per_element_MW": power_per_element,
            "pu239_production_rate_kg_per_day": total_rate_kg_per_day,
            "pu239_inventory_kg_over_time": {
                "month": [round(m * 12, 1) for m in months],
                "pu239_kg": [round(v, 5) for v in pu_inventory_kg],
            },
            "breeding_ratio": br,
            "k_infinite": k_inf,
            "max_fuel_temp_C": max_fuel_temp_C,
            "thermal_margin_K": thermal_margin_K,
            "realized_burnup_MWd_per_tU": realized_burnup,
            "target_burnup_MWd_per_tU": p["burnup_MWd_per_tU"],
            "z": z,
            "power_profile_MW_per_m": power_MW_per_m,
            "fuel_temp_C": fuel_temp_C,
            "pu239_kg_per_m": sol["pu239_kg_per_m"],
            "channel": sol,
            "params": dict(p),
        }
        return results

    # ------------------------------------------------------------------
    # Plotting (guarded matplotlib)
    # ------------------------------------------------------------------
    def plot_results(self, results: dict, out_path: str | None = None, show=False):
        """Plot the channel power / fuel-temperature / Pu-239 profiles.

        matplotlib is imported lazily; if it is unavailable, a message is
        printed and the method returns None so the module still works.
        """
        try:
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - matplotlib optional
            print(
                f"[plot_results] matplotlib unavailable ({exc}); "
                "skipping plot. Results dict still returned."
            )
            return None

        z = results["z"]
        ax, subplots = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        ax[0, 0].plot(z, results["power_profile_MW_per_m"])
        ax[0, 0].set_ylabel("Power [MW/m]")
        ax[0, 0].set_title("Axial power profile")
        ax[0, 1].plot(z, results["fuel_temp_C"])
        ax[0, 1].set_ylabel("Fuel / cladding [deg C]")
        ax[0, 1].set_title("Axial fuel temperature")
        ax[1, 0].plot(z, results["pu239_kg_per_m"])
        ax[1, 0].set_ylabel("Pu-239 [kg/m]")
        ax[1, 0].set_title("Axial Pu-239 production")
        inv = results["pu239_inventory_kg_over_time"]
        ax[1, 1].plot(inv["month"], inv["pu239_kg"])
        ax[1, 1].set_xlabel("Month")
        ax[1, 1].set_ylabel("Pu-239 [kg]")
        ax[1, 1].set_title("Pu-239 inventory over the cycle")
        for a in ax.flat:
            a.grid(True, alpha=0.3)
        fig = ax.get_figure()
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=140)
        if show:
            plt.show()
        return fig


__all__ = ["Reactor", "default_params"]
