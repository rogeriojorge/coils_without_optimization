"""Driver for the near–axis coil construction example.

This workflow uses the near–axis module from `essos.fields` to
generate magnetic field surfaces directly from geometric parameters
(`rc`, `zs`, `etabar`, etc.).  It then constructs coil centre lines
on these surfaces, fits Fourier–series representations to obtain
analytic coils, and traces field lines.  Results are rendered with
Plotly in both 3D (surface and coils) and 2D (Poincaré section).

The primary entry point is :func:`run_nearaxis_example` which accepts a
:class:`~coils_without_optimization.nearaxis_config.NearAxisConfig` and
writes all outputs into ``output_files/<output_name>/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from essos.fields import near_axis, BiotSavart, BiotSavart_from_gamma
from essos.coils import Curves, Coils, fit_dofs_from_coils
from essos.dynamics import Tracing
from jax import block_until_ready, vmap

from .nearaxis_config import NearAxisConfig
from .log_utils import log, time_block
from .coils_utils import build_coils
from .plot_helpers import (
    tubes_mesh3d_from_gammas,
    add_polyline_trajs,
    npf,
)


def run_nearaxis_example(cfg: NearAxisConfig) -> None:
    """Execute the near–axis coil example with the given configuration."""
    # Expose devices to JAX
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={cfg.devices}"
    )

    here = Path(__file__)
    paths = cfg.paths(here)
    outdir = paths["output_dir"]
    log("Running near–axis coil example…")

    # Step 1: initialise the near–axis field
    field_nearaxis = near_axis(
        rc=jnp.array(cfg.rc),
        zs=jnp.array(cfg.zs),
        etabar=cfg.etabar,
        nfp=cfg.nfp,
        nphi=cfg.nphi_internal_pyQSC,
    )

    # Compute number of field lines
    nfieldlines = cfg.devices * cfg.nfieldlines_per_core

    # Step 2: build surfaces of constant cylindrical angle for tracing
    log(
        f"Creating {nfieldlines} surfaces at r ≤ {cfg.r_surface:.3f} m using {cfg.ntheta}×{cfg.nphi_internal_pyQSC} grid…"
    )
    with time_block("Surface generation"):
        r_array = jnp.linspace(1e-5, cfg.r_surface, nfieldlines)
        results = [
            field_nearaxis.get_boundary(
                r=r, ntheta=cfg.ntheta, nphi=cfg.nphi_internal_pyQSC
            )
            for r in r_array
        ]
        # Each element of results is (x,y,z,R); pack into arrays
        x_2D_surface_array, y_2D_surface_array, z_2D_surface_array, R_2D_surface_array = map(
            lambda arr: jnp.stack(arr), zip(*results)
        )

    # Step 3: build coils on a surface of constant Boozer angle (varphi)
    nphi = cfg.ncoils * 2 * cfg.nfp
    log(
        f"Building coil loops on surface r={cfg.r_coils:.3f} m with nphi={nphi}…"
    )
    with time_block("Coil loop generation"):
        x_2D_coils, y_2D_coils, z_2D_coils, R_2D_coils = field_nearaxis.get_boundary(
            r=cfg.r_coils,
            ntheta=cfg.ntheta,
            nphi=nphi,
            phi_is_varphi=True,
            phi_offset=2 * jnp.pi / nphi / 2,
        )
        # Assemble the full set of 2*nfp*ncoils loops
        coils_gamma = jnp.zeros((cfg.ncoils * 2 * cfg.nfp, cfg.ntheta, 3))
        coil_i = 0
        for n in range(2 * cfg.nfp):
            # Choose one coil in each half period with an offset
            phi_vals = (
                (jnp.arange(cfg.ncoils) + 0.5)
                * (2 * jnp.pi)
                / (2 * cfg.nfp * cfg.ncoils)
                + 2 * jnp.pi / (2 * cfg.nfp) * n
            )
            phi_idx = (
                (phi_vals / (2 * jnp.pi) * nphi).astype(int) % nphi
            )
            for idx in phi_idx:
                loop = jnp.stack(
                    [
                        x_2D_coils[:, idx],
                        y_2D_coils[:, idx],
                        z_2D_coils[:, idx],
                    ],
                    axis=-1,
                )
                coils_gamma = coils_gamma.at[coil_i].set(loop)
                coil_i += 1

    # Step 4: fit Fourier–series coils
    log("Fitting Fourier–series representation of coils…")
    with time_block("Curve fitting"):
        dofs, _ = fit_dofs_from_coils(
            coils_gamma[: cfg.ncoils],
            order=cfg.order,
            n_segments=cfg.ntheta,
            assume_uniform=True,
        )
        curves = Curves(
            dofs=dofs,
            n_segments=cfg.ntheta,
            nfp=cfg.nfp,
            stellsym=True,
        )
        # Scale current by the same factor as the Boozer example: r_surface**2/1.7**2
        current_scale = cfg.current_on_each_coil / cfg.ncoils * cfg.r_surface ** 2 / 1.7 ** 2
        coils = Coils(
            curves=curves,
            currents=[-current_scale] * cfg.ncoils,
        )

    # Step 5: trace field lines
    log("Tracing field lines on fitted coils…")
    with time_block("Field line tracing"):
        R0 = R_2D_surface_array[:, 0, 0]
        Z0 = jnp.zeros_like(R0)
        phi0 = jnp.zeros_like(R0)
        initial_xyz = jnp.stack(
            [
                R0 * jnp.cos(phi0),
                R0 * jnp.sin(phi0),
                Z0,
            ],
            axis=1,
        )
        trajectories_coils = None
        trajectories_coils_gamma = None
        if cfg.plot_coils_without_fourier_fit:
            # Use gamma representation to build a BiotSavart field and trace
            field_coils_gamma = BiotSavart_from_gamma(
                coils_gamma,
                currents=cfg.current_on_each_coil * jnp.ones(len(coils_gamma)),
            )
            tracing_coils_gamma = block_until_ready(
                Tracing(
                    field=field_coils_gamma,
                    model="FieldLineAdaptative",
                    initial_conditions=initial_xyz,
                    maxtime=cfg.tmax,
                    times_to_trace=cfg.num_steps,
                    atol=cfg.trace_tolerance,
                    rtol=cfg.trace_tolerance,
                )
            )
            trajectories_coils_gamma = tracing_coils_gamma.trajectories
        if cfg.plot_fieldlines:
            tracing_coils_fitted = block_until_ready(
                Tracing(
                    field=BiotSavart(coils),
                    model="FieldLineAdaptative",
                    initial_conditions=initial_xyz,
                    maxtime=cfg.tmax,
                    times_to_trace=cfg.num_steps,
                    atol=cfg.trace_tolerance,
                    rtol=cfg.trace_tolerance,
                )
            )
            trajectories_coils = tracing_coils_fitted.trajectories

    # Step 6: 3D Plotly visualisation
    log("Rendering 3D visualisation…")
    data: list = []
    # Plot the outermost surface (last entry of the surface array) using Plotly.Surface directly
    # Colourscale chosen to be similar to Boozer example
    colorscale = [[0, "#C5B6A7"], [1, "#C5B6A7"]]
    # Use the last surface (largest r) for visualisation
    Xsurf = npf(x_2D_surface_array[-1])
    Ysurf = npf(y_2D_surface_array[-1])
    Zsurf = npf(z_2D_surface_array[-1])
    data.append(
        go.Surface(
            x=Xsurf,
            y=Ysurf,
            z=Zsurf,
            colorscale=colorscale,
            showscale=False,
            opacity=0.4,
            lighting={"specular": 0.3, "diffuse": 0.9},
            hoverinfo="skip",
        )
    )
    # Optionally plot the raw coils from the near–axis loops
    if cfg.plot_coils_without_fourier_fit:
        gammas_raw = [npf(g) for g in coils_gamma[: cfg.ncoils]]
        data.append(
            tubes_mesh3d_from_gammas(
                gammas_raw,
                radius=cfg.tube_radius,
                n_theta=cfg.tube_theta,
                color="#b87333",
                opacity=cfg.tube_opacity,
            )
        )
    # Plot the fitted coils
    gammas_fitted = [npf(P)[:: cfg.decimate] for P in npf(curves.gamma)]
    data.append(
        tubes_mesh3d_from_gammas(
            gammas_fitted,
            radius=cfg.tube_radius,
            n_theta=cfg.tube_theta,
            color="#93785A",
            opacity=cfg.tube_opacity,
        )
    )
    # Plot trajectories
    if cfg.plot_coils_without_fourier_fit and trajectories_coils_gamma is not None:
        add_polyline_trajs(
            data,
            trajectories_coils_gamma,
            color="black",
            width=2.0,
            every=max(1, cfg.decimate),
        )
    if cfg.plot_fieldlines and trajectories_coils is not None:
        add_polyline_trajs(
            data,
            trajectories_coils,
            color="black",
            width=0.2,
            every=max(1, cfg.decimate),
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        ),
        hovermode=False,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Near–Axis coils: {cfg.output_name}", x=0.5),
    )
    # Save interactive and PNG versions
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

    # Step 7: 2D Poincaré plot using Matplotlib
    import matplotlib.pyplot as plt
    log("Rendering 2D Poincaré plot…")
    fig2 = plt.figure(figsize=(6, 5))
    ax = fig2.add_subplot(111)
    # Plot field lines
    if cfg.plot_coils_without_fourier_fit and trajectories_coils_gamma is not None:
        for traj in trajectories_coils_gamma:
            ax.plot(
                npf(traj[:, 0]),
                npf(traj[:, 2]),
                color="black",
                linewidth=2,
                alpha=0.4,
            )
    if cfg.plot_fieldlines and trajectories_coils is not None:
        for traj in trajectories_coils:
            ax.plot(
                npf(traj[:, 0]),
                npf(traj[:, 2]),
                color="blue",
                linewidth=1,
                alpha=0.4,
            )
    # Overplot surfaces of constant angle
    for i in range(nfieldlines):
        ax.plot(
            npf(R_2D_surface_array[i, :, 0]),
            npf(z_2D_surface_array[i, :, 0]),
            "r--",
            linewidth=1.5,
        )
    # Add a representative coil surface trace in 2D
    x_2D_at_coil, y_2D_at_coil, z_2D_at_coil, R_2D_at_coil = field_nearaxis.get_boundary(
        r=cfg.r_coils,
        ntheta=cfg.ntheta,
        nphi=nphi,
        phi_is_varphi=True,
    )
    ax.plot(
        npf(R_2D_at_coil[:, 0]),
        npf(z_2D_at_coil[:, 0]),
        "r--",
        linewidth=1.5,
    )
    # Plot the fitted coils in R–Z space
    if cfg.plot_coils_on_2d:
        for coil_number in range(cfg.ncoils):
            R_curve = jnp.sqrt(
                curves.gamma[coil_number, :, 0] ** 2
                + curves.gamma[coil_number, :, 1] ** 2
            )
            ax.plot(
                npf(R_curve),
                npf(curves.gamma[coil_number, :, 2]),
                "b-",
                linewidth=2,
            )
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Poincaré plot: {cfg.output_name}")
    ax.set_aspect("equal", "box")
    fig2.tight_layout()
    png2_path = outdir / "poincare.png"
    try:
        fig2.savefig(str(png2_path), dpi=300)
        log(f"Wrote Poincaré plot to {png2_path}")
    except Exception as e:
        log(f"Warning: could not save Poincaré plot: {e}")
    plt.close(fig2)