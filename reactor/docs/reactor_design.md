# Pu-239 CANDU-style Heavy-Water Breeder - Design Parameter Table

Reference (blueprint) case: a sodium-cooled, heavy-water moderated,
CANDU-style heterogeneous net breeder sized for ~5 kg/yr Pu-239 production.
Every value below is derived from the physics modules in `src/`
(`neutronics`, `channel_model`, `thermal_hydraulics`, `reactor`).

## Core geometry

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Core radius | R | 2.5 m | blueprint (blueprint legend) radius; reference default case R=2.2 m |
| Core height | H | 3.0 m | blueprint height; reference default case H=3.7 m |
| Thermal power | P | 50 MWth | reference `power_MW` |
| Number of fuel elements | N | 300 | CANDU-style channel count (`n_elements`) |
| Power per element | - | 0.1667 MW | P / N |
| Mass per fuel element (U-238 + fissile) | - | 34 kg | ~20 tU total in core |
| U-238 seed enrichment | f | 3.0 % | `fuel_enrichment_frac` |
| Pu-239 initial fraction | - | 1.5 % | bred-Pu seed |
| U-235 fraction | - | 3.0 % | fissile seed |
| U-238 fraction | - | 95.5 % | 1 - enrichment - Pu seed |

## Calandria cell / channel lattice

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Calandria tube outer diameter (OD) | - | 190 mm | CANDU-style calandria vessel |
| Calandria tube inner diameter (ID) | - | 180 mm | thin-wall Na-490 tube |
| Calandria cell pitch | p | 90 mm | honeycomb lattice pitch |
| Fuel channel (U-238/THU plate) diameter | d | 54 mm | 37/44 mm family, plate-bundle |
| Channel count per calandria | - | 300 total in core | heterogeneous CANDU bundle |

## Fuel (U-238 / THU plates)

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Fuel form | - | U-238 + bred Pu-239 in THU plate | CANDU heterogeneous fuel |
| Element volume (calibrated) | V | 1.0e4 cm3 | per CANDU fuel bundle |
| Burnup target | - | 10,000 MWd/tU | reference burnup metric |
| Realized burnup (ref) | - | ~1.8 MWd/tU @ 1 yr cycle | informational, see main.py |

## Heavy-water moderator

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Moderator | - | D2O pool | moderator + moderator-irrigated |
| D2O thermal capture | sigma_a | 1.7e-4 cm-1 | negligible capture (ANISN D2O/U) |
| D2O thermal diffusion | D | 2.20 cm | 2-group lattice value |
| Scheme-2 heavy-water mass | - | ~100 t D2O | ~$2.2 M, ~15-17 % of capex |

## Sodium coolant loop

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Coolant | - | liquid Na (primary) | two-loop Na system |
| Reference sodium mass flow | m_dot | 446.8 kg/s | from `sodium_mass_flow` at dT_target=60 K |
| Reference sodium dT | dT | ~350 K rise | outlet - inlet over core |
| Sodium inlet temperature | T_in | 550 K (~277 C) | `T_IN_K_DEFAULT` |
| Sodium specific heat | cp | ~1,890 J/kg/K | at 550 K |
| S-HX (sodium heat exchanger) | - | primary/secondary Na Na | primary->secondary Na heat exchanger |
| Two legs: hot / cold | - | hot Na leg (outlet), cold Na leg (inlet) | accented hot leg in cross-section |

## Shielding and reflector

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Steel shield (biological) | - | ~2-3 m Na-steel | MWe-class biological + thermal shield |
| Thermal shield | - | Na-490 | between core and biological shield |
| Top reflector | - | heavy-water + steel | axial reflector, top |
| Bottom reflector | - | heavy-water + steel | axial reflector, bottom |

## Control rods

| Parameter | Symbol | Value | Notes |
| --- | --- | --- | --- |
| Control rod assemblies | - | 8-10 drives | Na-490 + boron-carbide absorber |
| Drive mechanism | - | CANDU-style in-channel | fast shutdown rods |

## Neutronics reference values (from `neutronics.py`)

| Parameter | Symbol | Value |
| --- | --- | --- |
| k-infinite (lattice) | k_inf | 1.0037 |
| Breeding ratio (U-238 basis) | BR | 0.9428 |
| U-238 (n,gamma) thermal capture | sigma_a | 2.68e-3 cm-1 |
| Pu-239 production | - | 1.3765e-2 kg/day (ref) |

## Thermal margins

| Parameter | Symbol | Value |
| --- | --- | --- |
| Max fuel temperature (ref) | T_fuel | ~432 C |
| Thermal margin to Na boil (1150 K) | - | ~444 K |
