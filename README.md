# Nuclear Wonder

A self-contained suite of **educational nuclear-physics models** built with the
same methodology end to end: a Python physics engine, unit tests, a Monte-Carlo /
cross-check harness, a CAD-style interactive `blueprint.html` with live
what-if sliders, and grounded documentation. Two sub-projects live here:

| Sub-project | What it is | Start at |
| --- | --- | --- |
| [`reactor/`](reactor/) | **Pu-239 CANDU net breeder** — sodium-cooled, heavy-water moderated, CANDU-style heterogeneous design sized for ~5 kg/yr Pu-239. Neutronics + channel + thermal-hydraulics + capex cost model. | [`reactor/docs/README.md`](reactor/docs/README.md) |
| [`simulation/`](simulation/) | **Fission criticality & device transient** — a calibrated 2-group neutronics engine (criticality, point kinetics, energy release, leakage) with a time-marched **device-fission transient layer** (gun / implosion) and an interactive demo. | [`simulation/docs/README.md`](simulation/docs/README.md) |

## How to read it

- **One shared toolkit, two applications.** Each sub-project is: a
  `src`/`sim` Python package + a standalone `blueprint.html` + `docs/` +
  generated `reports/*.json`. `simulation` builds a deeper core (the
  `sim/neutronics` engine) and layers the device transient on top of it;
  `reactor` applies the same modeling style to a breeder design.
- **The two connect physically.** `reactor` shows a net breeder turning
  U-238 into Pu-239; `simulation` studies what that fissile material does
  at the criticality / transient level. Read them in either order — the
  methodology is the point, not the specific device.

## Repo layout

```
.
├── README.md                  # this file — top-level index
├── reactor/                   # Pu-239 CANDU net breeder (design + sim)
│   ├── src/                   #   Python: neutronics, channel, thermal-hydraulics, cost
│   ├── docs/                  #   cost_comparison.md, reactor_design.md, README index
│   ├── blueprints/            #   CAD cross-section SVG
│   ├── blueprint.html         #   interactive what-if (sliders + charts)
│   └── reports/               #   generated case JSON
└── simulation/                # fission criticality + device transient
    ├── sim/                   #   Python: neutronics engine, device layer, tests
    ├── docs/                  #   README index + extracted reference PDFs
    ├── blueprint.html         #   interactive sphere + device demo
    └── reports/               #   generated case JSON
```

## Run

Each sub-project is independent — no top-level install. Quick start:

```bash
# Reactor (Pu-239 CANDU breeder)
cd reactor
python src/main.py --case default          # inventory, breeding ratio, margins
python src/cost_model.py                  # Scheme 1 vs Scheme 2 capex
python src/tests/pytest src/tests/ -q     # 5 unit tests
# open blueprint.html in a browser for the interactive what-if

# Simulation (fission criticality + device transient)
cd simulation
python sim/main.py --material u235 --mass 56    # point report
python sim/main.py --device gun                 # gun-assembly transient
python sim/main.py --device implosion           # implosion transient
python sim/main.py --device implosion --premature  # initiator-off contrast
python sim/tests/test_criticality.py           # 20 tests
python sim/tests/test_device.py                # 12 tests
# open blueprint.html in a browser for the interactive demo
```

Optional: `pip install openmc` to close the Monte-Carlo cross-check loop in
both sub-projects (each degrades gracefully with an install hint when it's
absent).

## Scope & honesty

Both projects are **educational / criticality-safety physics models**.
`simulation`'s device transient is grounded in declassified public-domain
references (UCRL-50239, Terrell LA-11807, LA-4208) — the device
*timescales* and yield/blast figures are **labeled order-of-magnitude
estimates, not design specs**, and the package explicitly excludes
weapon-design engineering (lens geometry, HE choice, timing windows).
See the per-project "Honest limits" sections in their `docs/README.md`.
