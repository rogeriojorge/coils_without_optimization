# Coil design without optimisation

This repository collects two pedagogical examples for designing coil
geometries without the overhead of a full stellarator optimisation
framework.  Both examples leverage the [`ESSOS`](https://github.com/uwplasma/ESSOS)
library and its dependencies to produce coil centre‐lines, optimise
the resulting shapes against a simple `B·n` objective, trace field
lines and render 3D figures suitable for publication.

## What’s inside?

* **`Boozer→Coils` example** – starting from a VMEC `wout_*.nc` file,
  the code performs a Boozer transformation, fits Fourier–series
  coils to the constant–phi surface, runs an optimisation and then
  traces field lines.  It produces an interactive Plotly figure
  showing the VMEC surface, the initial guess and the optimised
  coils.

* **Near–Axis example** – starting from a near–axis expansion defined
  by simple parameters (`rc`, `zs`, `etabar`, `nfp`), the code
  constructs surfaces and coils directly in Boozer coordinates,
  fits Fourier–series curves, traces field lines and produces both
  interactive and 2D Poincaré plots.

Both drivers share a set of utility modules for coil fitting,
optimisation, tracing and plotting.  They are written with clarity
in mind and include simple logging to illustrate the flow of
computation.

## Installation

It is recommended to use a virtual environment.  After checking out
this repository, run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The heavy dependencies (`essos`, `simsopt`, `booz_xform`) require
a working JAX CPU installation.  GPU support is optional but will
accelerate tracing and optimisation.

## Running an example

Configuration is specified via a simple [TOML](https://toml.io) file.
At minimum you must set the `example` key to either `"booz"` or
`"nearaxis"`.  The other fields override defaults defined in the
corresponding data classes; any unknown keys are ignored.  A minimal
configuration for the Boozer example might look like

```toml
example = "booz"
output_name = "my_boozer_case"
file_to_use = "LandremanPaul2021_QH_reactorScale_lowres"
ncoils = 4
ntheta  = 41
use_circular_coils = false
plot_fieldlines = true
```

and for the near–axis example

```toml
example = "nearaxis"
output_name = "my_nearaxis_case"
rc    = [1.0, 0.045]
zs    = [0.0, -0.045]
etabar= -0.9
nfp   = 3
ntheta= 41
ncoils= 4
```

Run the solver by pointing the package at your configuration file:

```bash
python -m coils_without_optimization --config path/to/my_config.toml
```

Outputs are written into `output_files/<output_name>/` and include
the generated Boozer harmonics (`boozmn_*.nc`) for the Boozer case,
interactive HTML visualisations (`coils.html`) and high–resolution
PNG images (`coils.png`).

## Directory layout

```
coils_without_optimization/
├── coils_without_optimization/   # importable library code
│   ├── __main__.py                # command–line entry point
│   ├── booz_driver.py             # Boozer→Coils workflow
│   ├── nearaxis_driver.py         # Near–Axis workflow
│   ├── booz_config.py             # dataclass for Boozer config
│   ├── nearaxis_config.py         # dataclass for near–axis config
│   ├── toml_config.py             # TOML loading utilities
│   ├── log_utils.py, ...          # small helpers
│   └── …
├── input_files/                  # put your VMEC files here
├── output_files/                 # per–run output directories
├── examples/                     # sample configuration files
├── requirements.txt              # Python dependencies
└── pyproject.toml                # package metadata
```

## Licence

This repository is released under the MIT licence.  See `LICENSE` for
details.