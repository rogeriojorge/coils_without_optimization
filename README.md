Love it. Here’s a clean CLI that lets you override the important parameters from the command line (especially the `wout`), while still keeping the driver runnable as a regular script with inline tweaks.

Below are the **full contents of every file** you need (only the *new/changed* ones compared to the previous layout). You can run either:

* as a module: `python -m boozcoils ...`
* or the driver directly: `python scripts/coils_from_BOOZ_XFORM.py ...`
* or with no flags at all (it will use the defaults in `Config`)

---

# 🔧 New/updated files

## `README.md` (updated)

````markdown
# Boozer → Coils: Optimization + Fieldlines + Publication-Ready Plots

This repository computes Boozer-space surfaces from a VMEC `wout_*.nc`, fits coil curves,
optimizes them against a `B·n` objective, traces fieldlines, and renders interactive 3D coil
visualizations (as tubes with lighting) alongside Poincaré and |B| plots.

- **Fast, memory-aware Plotly rendering** (`Mesh3d` tubes by default)
- **Modular structure** for geometry, coils, optimization, tracing, and plotting
- **Pedagogic driver** that reads like a story: _load → build → optimize → trace → plot_
- **Command-line interface** to override parameters (including the `wout` file)

## Quick start

```bash
# 1) Create env (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) Put your VMEC file here:
#    input_files/wout_LandremanPaul2021_QH_reactorScale_lowres.nc

# 4) Run via CLI (module)
python -m boozcoils \
  --wout input_files/wout_LandremanPaul2021_QH_reactorScale_lowres.nc --ncoils 4

# Or run the driver directly
python scripts/coils_from_BOOZ_XFORM.py \
  --wout input_files/wout_LandremanPaul2021_QH_reactorScale_lowres.nc
````

### Common flags

```bash
python -m boozcoils \
  --wout input_files/wout_MYCASE.nc \
  --ntheta 61 \
  --ncoils 6 \
  --s-surface 0.92 \
  --use-circular-coils \
  --plot-fieldlines \
  --refine-nphi-surface 4 \
  --tube-radius 0.12 \
  --tube-theta 12 \
  --decimate 2 \
  --max-fun-evals 150
```

See `python -m boozcoils --help` for the full list.

### Fast vs publish plots

The CLI defaults to fast **Mesh3d** tubes (lower memory & latency).
For publication PNGs, consider switching to `Surface` tubes in `plot_helpers.py` or saving:

```python
fig.write_image("coils.png", scale=4)  # requires kaleido
```

## Project layout

* `boozcoils/` — importable library code + CLI
* `scripts/coils_from_BOOZ_XFORM.py` — main driver (also CLI-aware)
* `input_files/` — your `wout_*.nc`
* `output_files/` — generated results

## License

MIT — see `LICENSE`.