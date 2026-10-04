# Pu-239 CANDU Breeder — Project Index

A simulation-ready design + blueprint for a **sodium-cooled, heavy-water
moderated, CANDU-style heterogeneous net breeder** sized for ~5 kg/yr Pu-239
production (~50 MWth, 300 fuel channels, U-238/THU plates).

Scheme 2 (CANDU breeder) was chosen over Scheme 1 (pool converter) via a
cost model: Scheme 2 is ~55× more expensive (~$13 M vs ~$235 k capex) but is
the **only net-breeding topology** that meets the Pu-239 yield — so the
"pick the cheaper unless it can't do the job" rule lands on Scheme 2.

## Layout

```
Nuclear Wonder/
├── docs/
│   ├── cost_comparison.md      # Scheme 1 vs Scheme 2 capex, drivers, verdict
│   ├── reactor_design.md       # full design parameter table (all values)
│   └── README.md               # this index
├── src/
│   ├── cost_model.py           # runnable capex model (python src/cost_model.py)
│   ├── neutronics.py           # 2-group diffusion, k_inf, breeding ratio, Pu-239 rate
│   ├── channel_model.py        # 1-D axial P-1 channel solve
│   ├── thermal_hydraulics.py   # Dittus-Boelter + fuel-plate conduction, Na loop
│   ├── reactor.py              # Reactor / run_case() top-level wrapper
│   ├── main.py                 # CLI report generator
│   ├── mc_verification.py      # OpenMC 1e6-particle cross-check (exits 2 if OpenMC missing)
│   └── tests/
│       ├── test_reactor.py     # 5 unit tests (runnable standalone or via pytest)
│       └── pytest              # offline pytest shim (used when pip pytest is absent)
├── blueprints/
│   └── cross-section.svg       # CAD cross-section (labeled parts + dimensions)
├── blueprint.html              # interactive blueprint: SVG + params + what-if sliders + charts
└── reports/
    └── case_default_results.json   # last main.py run (default case)
```

## Run it

```bash
python src/main.py --case default            # Pu-239 inventory, breeding ratio, margins
python src/cost_model.py                     # Scheme 1 vs Scheme 2 capex table
python src/tests/pytest src/tests/ -q        # 5 unit tests
python src/mc_verification.py                # OpenMC cross-check (needs `pip install openmc`)
```

Open `blueprint.html` in a browser for the interactive what-if sliders and
charts. The Pu-239 time series embedded in the page matches
`reports/case_default_results.json` byte-for-byte.

## Key reference numbers (default case)

| Quantity | Value |
| --- | --- |
| Thermal power | 50 MWth |
| Core radius × height | 2.5 m × 3.0 m |
| Fuel channels | 300 (U-238/THU, Na-490 clad) |
| U-238 seed enrichment | 3.0 % |
| k-infinite (lattice) | 1.0037 |
| Breeding ratio | 0.9428 |
| Pu-239 production | 1.377 × 10⁻² kg/day → **15.08 kg over 36 mo** |
| Max fuel temperature | ~432 °C |
| Thermal margin to Na boil | ~444 K |
| Sodium mass flow | 446.8 kg/s |
