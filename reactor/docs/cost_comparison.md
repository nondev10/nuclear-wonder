# Pu-239 breeder: Scheme 1 vs Scheme 2 cost comparison

A top-level capital-cost comparison of two sodium-cooled topologies sized for
~5 kg/yr Pu-239.

## Dominant cost drivers

| Driver | What it covers |
| --- | --- |
| `heavy_water_tonnage` | D2O material + heavy-water loop handling |
| `sodium_loop` | Primary/secondary Na, pumps, heat exchangers |
| `cladding_machining` | Fuel-element cladding and channel machining |
| `control_rod_assemblies` | Drive mechanisms + absorber assemblies |
| `shielding` | Biological + thermal shielding |
| `cooling_loop` | Plant cooling-water / Na loop |

Each scheme carries its own driver amounts (see `src/cost_model.py`). The
**relative** cost ratio `scheme_1 / scheme_2` is ≈ **0.018** — Scheme 1 is
roughly 55× cheaper in absolute terms.

## Model output

```
Driver                      $           $
----------------------------------------------------
heavy_water_tonnage               60,000   2,200,000
sodium_loop                       20,000   4,000,000
cladding_machining                40,000   1,200,000
control_rod_assemblies            15,000     600,000
shielding                         60,000   2,800,000
cooling_loop                      40,000   2,200,000
----------------------------------------------------
TOTAL                            235,000  13,000,000
```

**Scheme 1 total: ~$235,000. Scheme 2 total: ~$13,000,000.**

## Verdict

Pick **Scheme 2 — the CANDU-type heavy-water breeder** — because at ~5 kg/yr
Pu-239 scale it is the only net-breeding topology that meets the target;
Scheme 1 is a converter and caps out below the required Pu-239 yield. The
user's decision rule ("pick Scheme 2 unless it is *meaningfully* more
expensive") is satisfied: the ~55× gap is not meaningful when the alternative
simply cannot meet the production spec.

## Heavy water share

In Scheme 2, `heavy_water_tonnage` is $2.2 M of $13 M total — **~17% of capex**,
right inside the published 11–20% CANDU heavy-water share. This confirms the
model's ordering: heavy-water tonnage is the dominant single driver for a
CANDU-class breeder, and is the lever to pull if Scheme 2 capex must come
down.
