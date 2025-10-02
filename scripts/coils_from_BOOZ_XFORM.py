#!/usr/bin/env python3.11
import os
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
from simsopt.mhd import Vmec, Boozer
from essos.fields import BiotSavart, Vmec as VmecESSOS

from boozcoils.cli import parse_args_to_config
from boozcoils.log_utils import log, time_block
from boozcoils.geom_boozer import make_grids, accum_RZnu_derivs, push_off_surface, cyl_xyz_from_RphiZ
from boozcoils.coils_utils import gamma_from_xyz_columns, circular_guess, fit_curves_from_gamma, build_coils
from boozcoils.opt_runner import run_optimization
from boozcoils.trace_utils import initial_conditions_from_vmec, trace_fieldlines
from boozcoils.plot_helpers import surface_trace_from_RZ_phi, tubes_mesh3d_from_gammas, add_polyline_trajs, npf

def main():
    args, cfg = parse_args_to_config(None)
    os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={args.devices}'

    # Paths
    here = __file__
    paths = cfg.paths(here, wout_path_override=args.wout)
    log(f"Using wout: {paths['wout']}")

    # Boozer
    vmec = Vmec(str(paths["wout"]), verbose=False)
    b = Boozer(vmec, mpol=64, ntor=64, verbose=True)
    with time_block("Boozer transform"):
        b.register([1]); b.run()
    b = b.bx

    # Derived
    current = cfg.current_on_each_coil / cfg.ncoils * vmec.wout.Aminor_p**2 / 1.7**2
    nphi   = cfg.ncoils * 2 * b.nfp
    vmec_essos = VmecESSOS(str(paths["wout"]), ntheta=cfg.ntheta,
                           nphi=nphi*cfg.refine_nphi_surface,
                           range_torus='half period', s=cfg.s_surface)

    # Grids
    theta1D, (varphi, theta), (varphi_s, theta_s), phi1D, phi1D_surface = make_grids(
        cfg.ntheta, nphi, nphi*cfg.refine_nphi_surface
    )

    # Accumulate
    (R, Z, nu, dR, dZ), (R_s, Z_s, dR_s, dZ_s) = accum_RZnu_derivs(
        b, theta, varphi, theta_s, varphi_s, js=None
    )
    R_s, Z_s = push_off_surface(R_s, Z_s, dR_s, dZ_s, cfg.radial_extension)

    # Centerlines
    phi = varphi - nu
    X, Y, Z = cyl_xyz_from_RphiZ(R, phi, Z)

    # Coils
    coils_gamma = gamma_from_xyz_columns(X, Y, Z, ncurves=cfg.ncoils)
    Rmaj = vmec_essos.r_axis
    rmin = (np.max(coils_gamma[:, :, 0]) - Rmaj) * 1.5
    curves_guess = (circular_guess(cfg.ncoils, cfg.order_Fourier_coils, Rmaj, rmin, cfg.ntheta, vmec_essos.nfp)
                    if cfg.use_circular_coils else
                    fit_curves_from_gamma(coils_gamma, cfg.order_Fourier_coils, cfg.ntheta, b.nfp))
    coils_initial = build_coils(curves_guess, current, cfg.ncoils)

    # Caps
    field_fit = BiotSavart(build_coils(
        fit_curves_from_gamma(coils_gamma, cfg.order_Fourier_coils, cfg.ntheta, b.nfp),
        current, cfg.ncoils))
    max_len  = float(np.sum(field_fit.coils.length)) * cfg.max_len_amp
    max_curv = float(np.max(field_fit.coils.curvature)) * cfg.max_curv_amp

    # Optimize
    log(f"Optimizing coils with {cfg.max_fun_evals} function evaluations.")
    with time_block("Optimization"):
        opt = run_optimization(coils_initial, vmec_essos, max_len, max_curv,
                               cfg.min_distance_cc, cfg.tol_opt, cfg.max_fun_evals)
    log(f"Max length: {opt.length0:.2f} → {opt.length_opt:.2f} m")
    log(f"Mean curvature: {opt.curvature0:.2f} → {opt.curvature_opt:.2f} m^-1")
    log(f"max(B·n/B): {jnp.max(opt.bdotn_initial):.2e} → {jnp.max(opt.bdotn_opt):.2e}")

    # Trace
    nfieldlines = args.devices * cfg.nfieldlines_per_core
    init_xyz = initial_conditions_from_vmec(vmec, nfieldlines)
    trajs = None
    if cfg.plot_fieldlines:
        with time_block("Tracing fieldlines"):
            trajs = trace_fieldlines(opt.coils, init_xyz, cfg.tmax, cfg.num_steps, cfg.trace_tol, cfg.trace_tol)

    # Plotly
    data = []
    data.append(surface_trace_from_RZ_phi(R_s, Z_s, phi1D_surface, color="#C5B6A7", opacity=0.4))
    coils_orig = [np.column_stack([npf(X)[:,i], npf(Y)[:,i], npf(Z)[:,i]])[::cfg.decimate] for i in range(cfg.ncoils)]
    data.append(tubes_mesh3d_from_gammas(coils_orig, radius=cfg.tube_radius, n_theta=cfg.tube_theta,
                                         color="#BA4444", opacity=cfg.tube_opacity))
    coils_opt = [npf(P)[::cfg.decimate] for P in npf(opt.coils.gamma)]
    data.append(tubes_mesh3d_from_gammas(coils_opt, radius=cfg.tube_radius, n_theta=cfg.tube_theta,
                                         color="#CD9B3F", opacity=cfg.tube_opacity))
    if cfg.plot_fieldlines and trajs is not None:
        add_polyline_trajs(data, trajs, color="black", width=0.2, every=max(1,cfg.decimate), name="Fieldlines")
    fig = go.Figure(data=data)
    fig.update_traces(contours_x_highlight=False, contours_y_highlight=False, contours_z_highlight=False,
                      selector={"type": "surface"})
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      hovermode=False, margin=dict(l=0, r=0, t=25, b=0))
    fig.show()

if __name__ == "__main__":
    main()
