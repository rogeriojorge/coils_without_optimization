"""Driver for the Boozer→Coils workflow.

This module contains a single function, :func:`run_booz_example`, which
performs the following steps:

1. Read a VMEC equilibrium file and compute the Boozer harmonics using
   `simsopt.mhd.Boozer`.
2. Construct radial and poloidal grids and evaluate the harmonic sums
   to obtain the Boozer coordinates (R, Z, nu).
3. Build an initial coil set either by fitting Fourier coefficients to
   the Boozer centre lines or by creating an equal–spaced circular
   guess.
4. Compute coil length and curvature constraints and run the coil
   optimisation from :mod:`essos.optimization`.
5. Trace magnetic field lines on the resulting coil set (optional).
6. Produce an interactive Plotly figure showing the VMEC surface,
   the initial guess and the optimised coils; save it to HTML/PNG.

The function accepts a :class:`~coils_without_optimization.booz_config.BoozConfig`
instance to override default parameters and writes all output into
``output_files/<output_name>/``.  It prints informative messages
throughout the computation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
from simsopt.mhd import Vmec, Boozer
from essos.fields import BiotSavart, Vmec as VmecESSOS

from .booz_config import BoozConfig
from .log_utils import log, time_block
from .geom_boozer import (
    make_grids,
    accum_RZnu_derivs,
    push_off_surface,
    cyl_xyz_from_RphiZ,
)
from .coils_utils import (
    gamma_from_xyz_columns,
    circular_guess,
    fit_curves_from_gamma,
    build_coils,
)
from .opt_runner import run_optimization
from .trace_utils import initial_conditions_from_vmec, trace_fieldlines
from .plot_helpers import (
    surface_trace_from_RZ_phi,
    tubes_mesh3d_from_gammas,
    add_polyline_trajs,
    npf,
)


def run_booz_example(cfg: BoozConfig, wout_override: Optional[str] = None) -> None:
    """Execute the Boozer→Coils example with the given configuration.

    Parameters
    ----------
    cfg : BoozConfig
        Configuration parameters for the run.
    wout_override : str, optional
        Absolute path to a VMEC `wout_*.nc` file.  When provided, this
        path overrides ``cfg.file_to_use``.
    """
    # Expose the requested number of devices to JAX.  On a CPU machine
    # this will result in parallel execution across threads/processes.
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={cfg.devices}"
    )

    # Compute paths for input and output
    here = Path(__file__)
    paths = cfg.paths(here, wout_path_override=wout_override)
    log(f"Using VMEC file: {paths['wout']}")

    # Step 1: Boozer transform
    vmec = Vmec(str(paths["wout"]), verbose=False)
    b = Boozer(vmec, mpol=64, ntor=64, verbose=True)
    with time_block("Boozer transform"):
        b.register([1])
        b.run()
    b = b.bx
    # Save the Boozer harmonics to file for later reuse
    try:
        b.write_boozmn(str(paths["boozmn"]))
        log(f"Saved Boozer harmonics to {paths['boozmn']}")
    except Exception as e:
        # Not all versions of booz_xform support this method
        log(f"Warning: could not write boozmn file: {e}")

    # Derived quantities
    current = (
        cfg.current_on_each_coil
        / cfg.ncoils
        * vmec.wout.Aminor_p**2
        / 1.7**2
    )
    nphi = cfg.ncoils * 2 * b.nfp
    vmec_essos = VmecESSOS(
        str(paths["wout"]),
        ntheta=cfg.ntheta,
        nphi=int(nphi * cfg.refine_nphi_surface),
        range_torus="half period",
        s=cfg.s_surface,
    )

    # Step 2: construct grids and accumulate Boozer harmonics
    theta1D, (varphi, theta), (varphi_s, theta_s), phi1D, phi1D_surface = make_grids(
        cfg.ntheta, nphi, int(nphi * cfg.refine_nphi_surface)
    )
    (R, Z, nu, dR, dZ), (R_s, Z_s, dR_s, dZ_s) = accum_RZnu_derivs(
        b, theta, varphi, theta_s, varphi_s, js=None
    )
    R_s, Z_s = push_off_surface(R_s, Z_s, dR_s, dZ_s, cfg.radial_extension)

    # Step 3: build coils
    phi = varphi - nu
    X, Y, ZZ = cyl_xyz_from_RphiZ(R, phi, Z)
    coils_gamma = gamma_from_xyz_columns(X, Y, ZZ, ncurves=cfg.ncoils)
    Rmaj = vmec_essos.r_axis
    rmin = (np.max(coils_gamma[:, :, 0]) - Rmaj) * 1.5
    # Choose initial guess: either circular or fitted
    if cfg.use_circular_coils:
        curves_guess = circular_guess(
            cfg.ncoils,
            cfg.order_Fourier_coils,
            Rmaj,
            rmin,
            cfg.ntheta,
            vmec_essos.nfp,
        )
    else:
        curves_guess = fit_curves_from_gamma(
            coils_gamma,
            cfg.order_Fourier_coils,
            cfg.ntheta,
            b.nfp,
        )
    coils_initial = build_coils(curves_guess, current, cfg.ncoils)

    # Constraint caps from Boozer fit
    field_fit = BiotSavart(
        build_coils(
            fit_curves_from_gamma(
                coils_gamma,
                cfg.order_Fourier_coils,
                cfg.ntheta,
                b.nfp,
            ),
            current,
            cfg.ncoils,
        )
    )
    max_len = float(np.sum(field_fit.coils.length)) * cfg.max_len_amp
    max_curv = float(np.max(field_fit.coils.curvature)) * cfg.max_curv_amp

    # Step 4: optimisation
    log(
        f"Optimising {cfg.ncoils} coils with at most {cfg.max_fun_evals} evaluations…"
    )
    with time_block("Optimisation"):
        opt = run_optimization(
            coils_initial,
            vmec_essos,
            max_len,
            max_curv,
            cfg.min_distance_cc,
            cfg.tol_opt,
            cfg.max_fun_evals,
        )

    # Print summary
    log(
        f"Max length: {opt.length0:.2f} → {opt.length_opt:.2f} m; "
        f"Mean curvature: {opt.curvature0:.2f} → {opt.curvature_opt:.2f} m⁻¹; "
        f"max(B·n/B): {jnp.max(opt.bdotn_initial):.2e} → {jnp.max(opt.bdotn_opt):.2e}"
    )

    # Step 5: field line tracing
    nfieldlines = cfg.devices * cfg.nfieldlines_per_core
    init_xyz = initial_conditions_from_vmec(vmec, nfieldlines)
    trajs = None
    if cfg.plot_fieldlines:
        with time_block("Tracing field lines"):
            trajs = trace_fieldlines(
                opt.coils,
                init_xyz,
                cfg.tmax,
                cfg.num_steps,
                cfg.trace_tol,
                cfg.trace_tol,
            )

    # Step 6: plotting
    data: list = []
    # VMEC surface
    data.append(
        surface_trace_from_RZ_phi(
            R_s,
            Z_s,
            phi1D_surface,
            color="#C5B6A7",
            opacity=0.4,
        )
    )
    # Original centre lines (decimated)
    coils_orig = [
        np.column_stack([npf(X)[:, i], npf(Y)[:, i], npf(ZZ)[:, i]])[:: cfg.decimate]
        for i in range(cfg.ncoils)
    ]
    data.append(
        tubes_mesh3d_from_gammas(
            coils_orig,
            radius=cfg.tube_radius,
            n_theta=cfg.tube_theta,
            color="#BA4444",
            opacity=cfg.tube_opacity,
        )
    )
    # Optimised coils
    coils_opt = [npf(P)[:: cfg.decimate] for P in npf(opt.coils.gamma)]
    data.append(
        tubes_mesh3d_from_gammas(
            coils_opt,
            radius=cfg.tube_radius,
            n_theta=cfg.tube_theta,
            color="#CD9B3F",
            opacity=cfg.tube_opacity,
        )
    )
    # Field lines
    if cfg.plot_fieldlines and trajs is not None:
        add_polyline_trajs(
            data,
            trajs,
            color="black",
            width=0.2,
            every=max(1, cfg.decimate),
        )

    fig = go.Figure(data=data)
    # Turn off surface contours
    fig.update_traces(
        contours_x_highlight=False,
        contours_y_highlight=False,
        contours_z_highlight=False,
        selector={"type": "surface"},
    )
    # Equal aspect ratio and minimal axes
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        ),
        hovermode=False,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Boozer→Coils: {cfg.output_name}", x=0.5),
    )

    # Save outputs
    outdir = paths["output_dir"]
    html_path = outdir / "coils.html"
    png_path = outdir / "coils.png"
    try:
        fig.write_html(str(html_path))
        log(f"Wrote interactive figure to {html_path}")
    except Exception as e:
        log(f"Warning: could not write HTML figure: {e}")
    try:
        fig.write_image(str(png_path), scale=4)
        log(f"Wrote PNG figure to {png_path}")
    except Exception as e:
        log(f"Warning: could not write PNG figure: {e}")