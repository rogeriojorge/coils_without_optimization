"""Configuration dataclass for the Boozer→Coils example.

The Boozer driver reads most of its parameters from an instance of
this dataclass.  Values here reflect the defaults used in the
original `essos` example.  The ``output_name`` field determines
the subdirectory under ``output_files/`` into which results are
written.  A convenience method ``paths`` computes absolute paths
to input and output files relative to the current module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


@dataclass(frozen=True)
class BoozConfig:
    # Name of the VMEC file *without* the ``wout_`` prefix and ``.nc``
    file_to_use: str = "LandremanPaul2021_QH_reactorScale_lowres"
    # Number of poloidal points along each coil
    ntheta: int = 41
    # Number of coils
    ncoils: int = 4
    # Maximum tracing time for fieldline tracing
    tmax: float = 1100.0
    # Number of field lines per device for tracing
    nfieldlines_per_core: int = 1
    # Trace tolerance (atol=rtol)
    trace_tol: float = 1e-9
    # Number of time steps in field line tracing
    num_steps: int = 22000
    # Order of the Fourier representation of the coils
    order_Fourier_coils: int = 4
    # Nominal current per coil (amperes)
    current_on_each_coil: float = 2e8
    # Refinement factor for the toroidal resolution of the surface
    refine_nphi_surface: int = 4
    # Radial offset to push the Boozer surface outward (m)
    radial_extension: float = 1e-3
    # Maximum length amplification relative to the initial fit
    max_len_amp: float = 3.0
    # Maximum curvature amplification relative to the initial max
    max_curv_amp: float = 0.5
    # Minimum coil–coil distance (m)
    min_distance_cc: float = 0.2
    # Maximum number of function evaluations in the optimiser
    max_fun_evals: int = 100
    # Optimisation tolerance
    tol_opt: float = 1e-6
    # VMEC surface index ``s`` at which to evaluate B·n (0 < s ≤ 1)
    s_surface: float = 0.95
    # Use circular initial coil guess instead of fitting to the Boozer fit
    use_circular_coils: bool = False
    # Whether to trace and plot field lines
    plot_fieldlines: bool = True
    # Whether to plot Fourier–fitted coils in addition to the initial fit
    show_coils_fitted_to_fourier: bool = False
    # Phase shift for the Poincaré plot (unused here)
    poincare_phi_shift: float = 3.141592653589793
    # Tube visualisation parameters
    tube_radius: float = 0.15
    tube_theta: int = 12
    decimate: int = 1
    tube_opacity: float = 1.0
    # Name of the output directory within ``output_files/``
    output_name: str = "booz_example"
    # Number of XLA devices to expose to JAX (affects parallelisation)
    devices: int = 4

    def paths(self, here: str | Path, wout_path_override: Optional[str] = None) -> Dict[str, Path]:
        """Compute absolute paths for input and output files.

        The returned dictionary contains ``root``, ``input_dir``,
        ``output_dir``, ``wout`` and ``boozmn``.  The ``output_dir``
        directory is created if necessary.
        """
        here = Path(here).resolve()
        # When called from inside the package, ``here`` is a file under
        # ``coils_without_optimization/``.  We want to locate the project
        # root one level up.
        root = here.parent if (here.name == "__main__.py" or here.suffix == ".py") else here
        input_dir = root / "input_files"
        output_root = root / "output_files" / self.output_name
        output_root.mkdir(parents=True, exist_ok=True)
        # Determine VMEC wout file
        if wout_path_override:
            wout = Path(wout_path_override)
        else:
            wout = input_dir / f"wout_{self.file_to_use}.nc"
        boozmn = output_root / f"boozmn_{self.file_to_use}.nc"
        return {
            "root": root,
            "input_dir": input_dir,
            "output_dir": output_root,
            "wout": wout,
            "boozmn": boozmn,
        }