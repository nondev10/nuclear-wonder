# Fission Neutronics Simulation — Project Index (serious edition)

An interactive, **calibrated** simulation of the **nuclear physics of a
fissile sphere**: criticality, chain-reaction kinetics, energy release, and
neutron/gamma leakage. The engine is a 2-group neutron-diffusion model
calibrated per material to the *measured* Los Alamos critical masses, with
grounded leakage source terms and a self-contained Monte Carlo cross-check.

> **Scope:** nuclear-physics layer only — criticality, point kinetics, energy
> release, leakage/dose. It does **not** model weapon assembly, detonation,
> gun/implosion mechanics, or device engineering. The historical HTML doc in
> this folder contains such text; it is reference reading, not simulated.

## References in this folder

- `Reevaluated Critical Specifications of Some Los Alamos Fast-Neutron
  Systems` (Hansen & Paxton, LA-4208) — **the calibration backbone**: measured
  bare & reflected critical masses (U-235 51.94 kg, Pu-239 17.02 kg,
  U-233 16.53 kg; reflected U-235 19.82 / 26.56 / 36.53 / 28.14 kg).
- `Gamma-Ray and Neutron Leakage Spectra` (Terrell, LA-11807, MCNP) —
  leakage ν, k at specific radii, leakage neutron/gamma yields per fission,
  Maxwellian fission spectrum (grounds the leakage + kinetics layers).
- `1967 Summary Report of the Nth Country Experiment` (LRL) — critical-mass
  literature table + the α=1/λ generation-time relation (secondary).
- `1179` — the Morland H-bomb *secrecy* magazine article: **context only**,
  not a calibration source.

## Layout

```
simulation/
├── sim/
│   ├── neutronics/          # THE ENGINE
│   │   ├── materials.py     #   isotope table, calibration targets, calibrated {k∞,Ld,Lf,Ls}
│   │   ├── sphere.py        #   2-group (primary) + 1-group bare-sphere k_eff(R), R_crit, φ, M
│   │   ├── reflector.py     #   two-region core+reflector match (calibrated albedos)
│   │   ├── calibrate.py     #   offline fit of constants to measured masses
│   │   └── kinetics.py      #   point kinetics + fission-energy on the sphere
│   ├── criticality.py       # public facade over the engine (old API kept)
│   ├── leakage.py          # Terrell-grounded source terms + 1/r² dose
│   ├── mc_check.py         # self-contained numpy MC k-eff cross-check (default)
│   ├── main.py             # CLI report (--material, --mass, --sweep, --verify, --reflected, --device)
│   ├── device/            # DEVICE FISION TRANSIENT LAYER (educational)
│   │   ├── assembly.py    #   gun / implosion R(t), rho(t), M(t) drivers
│   │   ├── transient.py   #   time-marched device transient (centerpiece)
│   │   ├── blast.py       #   burst spectrum, T(t) readout, blast-zone scaling
│   │   └── __init__.py    #   run_gun / run_implosion / run_premature / blast_zones
│   └── tests/
│       ├── test_criticality.py   # 20 tests: measured masses, k-crossing-1, scaling, MC agree
│       └── test_device.py        # 12 tests: transient N≈0 pre-initiator, k crosses 1,
│                                 #   fission fraction ∈ [0,1], gun vs implosion timescales
├── docs/
│   ├── pdf_text/           # extracted text of every reference
│   └── README.md           # this index
├── reports/                # JSON output of main.py / mc_check.py / --device
└── blueprint.html          # interactive what-if (sphere) + DEVICE demo (gun/implosion)
```

> **Removed from the earlier simple version:** `sim/diffusion.py` (old 1-D
> rod model, superseded by `neutronics/sphere.py`) and `sim/mc_verification.py`
> (the broken circular OpenMC harness, replaced by `sim/mc_check.py`).

## Run it

```bash
python sim/main.py --material u235 --mass 56        # U-235 point report
python sim/main.py --material pu239                 # Pu-239 at calibrated critical mass
python sim/main.py --material u235 --reflected 3.93 --refl natU   # reflected assembly
python sim/main.py --material u235 --sweep          # radius sweep [0.7x..1.5x R_crit]
python sim/main.py --material u235 --verify         # run the MC k-eff cross-check
python sim/main.py --device gun                     # gun-assembly device transient (U-235)
python sim/main.py --device implosion               # implosion device transient (Pu-239)
python sim/main.py --device implosion --premature   # initiator-off / premature-fission contrast
python sim/tests/test_criticality.py               # 20 unit tests (standalone)
python sim/tests/test_device.py                    # 12 device-transient tests
python sim/neutronics/calibrate.py                 # re-derive the calibrated constants
```

Open `blueprint.html` in a browser:
- **Sphere what-if** — pick a material (U-235 / Pu-239 / U-233), drag the
  radius, and watch k_eff, reactivity, the sub-critical multiplier
  M=k/(1−k), chain-growth, and fission-energy/yield update live.
- **Device transient demo** — the "more detailed" layer. Pick **gun-assembly
  (U-235)** or **implosion (Pu-239)**, drag the assembly timeline, toggle
  the **initiator on/off**, and watch k_eff(t), N(t), Y(t), the fission
  fraction, and the cube-root-scaled blast zones update live. A labeled
  step-through **cutaway walkthrough** documents each device stage. The JS
  is a physics port of `sim/device/transient.py` — the same calibrated
  engine, wired to an assembly driver.

## The device fission transient (the "more detailed" layer)

`sim/device/` turns the steady bare-sphere engine into a time-dependent
device, grounded in the **1967 Nth Country fission-weapon design report**
(UCRL-50239, in `docs/pdf_text/`), which is the device-physics backbone:

> "The [assembly] times ... are more than 10 times the total fission chain
> reaction time. This means that it is essential for the fissile material to
> be neutron free during the assembly ... If the fissile material is neutron
> free, then it is necessary to 'turn on' neutrons to initiate the chain
> reaction at the desired time. This is the role of the so-called initiator."

The device has three regimes in time:
1. **Assembly** (t < t₀): a sub-critical assembly (gun: light U-235 plug
   fired into a heavy U-235 barrel + U-238 tamper; implosion: a Pu-239
   core compressed by a converging detonation wave). The core is
   **neutron-free** — no chain.
2. **Initiation & burst** (t ≥ t₀): the **initiator** fires at
   t₀ = t_critical, turning on the neutron source; the point-kinetics chain
   runs for a few prompt generation times, N(t) grows, energy Y(t) released.
3. **Disassembly** (self-termination): the hot fissioned core expands on a
   labeled hydrodynamic timescale `t_dis ~ R/v_dis`, so k_eff falls back
   below 1 and the chain stops. The fission fraction
   `f = 1 − exp(−t_dis/L_p) < 1` and Y(t) peaks — that is what makes it a
   *transient*.

Outputs (arrays over time): `t, R(t), rho(t), M(t), k_eff(t), N(t), Y(t),
T(t)`, plus scalars `f_disassembly`, `Y_max_kt`, `t_0`, `t_dis`. An
**initiator-off "premature fission"** sub-case (the Nth Country hazard) is
available as a teaching contrast.

**Honest limits (labeled, not hidden):** the device *timescales* — gun
muzzle velocity (~2 km/s), implosion compression window (~3 µs),
compression factor, disassembly velocity (~1×10⁵ m/s), initiator source
strength, and the cube-root blast-zone scaling — are **labeled
order-of-magnitude / literature estimates, not design specs**. The
neutronics underneath (k_eff, criticality, prompt lifetime) is the
calibrated engine anchored to the measured Los Alamos critical masses.
Yield/blast figures are **illustrative only**. The package is explicitly
educational: it models the nuclear-physics layer of the device transient and
does not include weapon-design engineering (lens geometry, HE choice,
timing windows).

## Key numbers (default U-235, calibrated)

| Quantity | Value |
|---|---|
| Bare M_crit (U-235) | 51.94 kg (R_crit = 8.70 cm) — *measured* |
| Bare M_crit (Pu-239) | 17.02 kg (R_crit = 6.39 cm) |
| Bare M_crit (U-233) | 16.53 kg (R_crit = 5.98 cm) |
| k_eff at R_crit | 1.0000 (1-group exact; 2-group matches to 1e-4) |
| Godiva cross-check | k(R=8.74 cm)≈1.000 (Terrell MCNP measured 0.997) |
| Sub-crit multiplier @ R=8.0 cm | M=k/(1−k)≈196 (LBL shell idiom 143→193) |
| Prompt generation time | ~10 ns (U-235), ~30 ns (Pu-239) |
| 52 kg U-235 total fission energy | 4.27e15 J |
| MC-vs-diffusion agreement | <0.2 % at R_crit (independent cross-check) |

### Device transient (illustrative, labeled)

| Quantity | gun · U-235 | implosion · Pu-239 |
|---|---|---|
| Assembly timescale | ~90 µs (plug flight + overlap) | ~3 µs (compression) |
| k_eff peak | ~1.003 | ~2.15 |
| Fission fraction f | ~0.11 | ~1.0 |
| Peak yield (illustrative) | ~130 kt | ~800 kt |
| Prompt lifetime | ~10 ns | ~13 ns |

*The device timescales and yield figures are labeled order-of-magnitude
estimates, not design specs — see the honest-limits section.*

## Honest limits (declared, not hidden)

- **Per-material calibration:** a single global diffusion-length set cannot
  fit all three isotopes, so the model fits each one and interpolates across
  them — do not extrapolate to uncalibrated materials.
- **1-D spherical diffusion:** valid near the fundamental mode & R ≫ L_d
  (~1 % near critical, growing at strong sub-criticality / reflector
  transition).
- **Group-averaged ν**; fast β_eff≈0.002 (labeled).
- **Disassembly time** is a hydrodynamic estimate (`t_exp ≈ R/v_exp`),
  reported as a fission *fraction*, not a device yield.
- **Device transient timescales** (gun muzzle velocity, implosion
  compression window/factor, disassembly velocity, initiator source,
  cube-root blast-zone scaling) are labeled order-of-magnitude / literature
  estimates, not design specs; the neutronics underneath is the calibrated
  engine. Yield/blast figures are illustrative only.
- **Leakage dose** is inverse-square with measured source strengths, not
  full photon transport.
- **MC cross-check** is a simplified fission-source tally (no resonance
  structure); uncertainty ~1/√batches.
- **No device engineering:** the device layer models the nuclear-physics
  transient only — no lens geometry, high-explosive choice, or timing
  windows. It is explicitly educational, and the gun sub-case pairs with
  U-235 while the implosion sub-case pairs with Pu-239 (the historical
  devices); this is a demonstration pairing, not a claim either method
  works on either material.
