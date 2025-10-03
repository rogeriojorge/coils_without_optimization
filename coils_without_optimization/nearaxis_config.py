"""Configuration dataclass for the near–axis coil example.

This dataclass captures the geometric parameters of a near–axis magnetic
field and the numerical parameters controlling surface generation,
coil fitting, tracing and plotting.  All values have sensible
defaults derived from the original ESSOS example.  The ``output_name``
field selects a subdirectory within ``output_files/`` for storing
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence


@dataclass(frozen=True)
class NearAxisConfig:
    # Directory name under ``output_files``
    output_name: str = "nearaxis_example"
    # Near–axis parameters: centreline radius and z–shifts
    rc: Sequence[float] = (1.0, 0.045)
    zs: Sequence[float] = (0.0, -0.045)
    # Eta_bar controlling helical twist
    etabar: float = -0.9
    # Number of field periods
    nfp: int = 3
    # Internal resolution for the near–axis field
    nphi_internal_pyQSC: int = 51
    # Radial coordinate of the coils (m)
    r_coils: float = 0.4
    # Radial coordinate of the last evaluated surface (m)
    r_surface: float = 0.2
    # Number of poloidal points around each curve
    ntheta: int = 41
    # Number of coils per half period
    ncoils: int = 4
    # Maximum tracing time
    tmax: float = 800.0
    # Number of field lines per device
    nfieldlines_per_core: int = 1
    # Absolute/relative tolerance for tracing
    trace_tolerance: float = 1e-8
    # Number of integration steps
    num_steps: int = 22000
    # Order of the Fourier representation of the coils
    order: int = 4
    # Nominal current on each coil (amperes)
    current_on_each_coil: float = 2e8
    # Plotting flags
    plot_coils_without_fourier_fit: bool = False
    plot_coils_on_2d: bool = False
    plot_difference_varphi_phi: bool = False
    plot_fieldlines: bool = True
    # Visualisation parameters
    tube_radius: float = 0.15
    tube_theta: int = 12
    decimate: int = 1
    tube_opacity: float = 1.0
    # JAX device count
    devices: int = 4

    def paths(self, here: str | Path) -> Dict[str, Path]:
        """Return absolute paths for output directories.

        The near–axis example does not read VMEC input; therefore only
        ``root`` and ``output_dir`` are returned.  The ``output_dir``
        directory is created if necessary.
        """
        here = Path(here).resolve()
        root = here.parent if (here.name == "__main__.py" or here.suffix == ".py") else here
        output_root = root / "output_files" / self.output_name
        output_root.mkdir(parents=True, exist_ok=True)
        return {"root": root, "output_dir": output_root}