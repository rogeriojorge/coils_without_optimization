from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class Config:
    file_to_use: str = "LandremanPaul2021_QH_reactorScale_lowres"
    ntheta: int = 41
    ncoils: int = 4
    tmax: float = 1100.0
    nfieldlines_per_core: int = 1
    trace_tol: float = 1e-9
    num_steps: int = 22000
    order_Fourier_coils: int = 4
    current_on_each_coil: float = 2e8
    refine_nphi_surface: int = 4
    radial_extension: float = 1e-3
    max_len_amp: float = 3.0
    max_curv_amp: float = 0.5
    min_distance_cc: float = 0.2
    max_fun_evals: int = 100
    tol_opt: float = 1e-6
    s_surface: float = 0.95
    use_circular_coils: bool = False
    plot_fieldlines: bool = True
    show_coils_fitted_to_fourier: bool = False
    poincare_phi_shift: float = 3.141592653589793

    # Plot knobs (fast defaults)
    tube_radius: float = 0.15
    tube_theta: int = 12
    decimate: int = 1
    tube_opacity: float = 1.0

    # Paths (computed lazily)
    def paths(self, here: str | Path, wout_path_override: Optional[str] = None) -> dict:
        here = Path(here).resolve()
        root = here.parent if (here.name == "__main__.py" or here.suffix == ".py") else here
        input_dir = root / "input_files"
        output_dir = root / "output_files"
        output_dir.mkdir(parents=True, exist_ok=True)
        if wout_path_override:
            wout = Path(wout_path_override)
        else:
            wout = input_dir / f"wout_{self.file_to_use}.nc"
        boozmn = output_dir / f"boozmn_{self.file_to_use}.nc"
        return {"root": root, "input_dir": input_dir, "output_dir": output_dir, "wout": wout, "boozmn": boozmn}

def config_from_args(args, base: Config | None = None) -> Config:
    """Create a Config by selectively overriding fields from argparse args."""
    cfg = base or Config()
    updates = {}

    # mappings: argparse dest -> Config field
    mapping = {
        "file_to_use": "file_to_use",
        "ntheta": "ntheta",
        "ncoils": "ncoils",
        "tmax": "tmax",
        "nfieldlines_per_core": "nfieldlines_per_core",
        "trace_tolerance": "trace_tol",
        "num_steps": "num_steps",
        "order_Fourier_coils": "order_Fourier_coils",
        "current_on_each_coil": "current_on_each_coil",
        "refine_nphi_surface": "refine_nphi_surface",
        "radial_extension": "radial_extension",
        "max_len_amp": "max_len_amp",
        "max_curv_amp": "max_curv_amp",
        "min_distance_cc": "min_distance_cc",
        "max_fun_evals": "max_fun_evals",
        "tol_opt": "tol_opt",
        "s_surface": "s_surface",
        "use_circular_coils": "use_circular_coils",
        "plot_fieldlines": "plot_fieldlines",
        "show_coils_fitted_to_fourier": "show_coils_fitted_to_fourier",
        "poincare_phi_shift": "poincare_phi_shift",
        # plot knobs
        "tube_radius": "tube_radius",
        "tube_theta": "tube_theta",
        "decimate": "decimate",
        "tube_opacity": "tube_opacity",
    }

    for k_cli, k_cfg in mapping.items():
        v = getattr(args, k_cli, None)
        if v is not None:
            updates[k_cfg] = v

    return replace(cfg, **updates)
