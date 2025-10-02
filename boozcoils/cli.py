import argparse
from .config import config_from_args, Config

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boozcoils",
        description="Boozer→Coils optimization, tracing, and plotting"
    )
    # Paths / I/O
    p.add_argument("--wout", type=str, help="Path to VMEC wout_*.nc (overrides file_to_use)")
    p.add_argument("--file-to-use", type=str, help="Name (without prefix) for wout_*.nc in input_files/")
    # Physics / geometry
    p.add_argument("--ntheta", type=int, help="Number of poloidal points along coils")
    p.add_argument("--ncoils", type=int, help="Number of coils")
    p.add_argument("--s-surface", dest="s_surface", type=float, help="VMEC s for plotting surface")
    p.add_argument("--refine-nphi-surface", type=int, dest="refine_nphi_surface",
                   help="Refinement factor for toroidal resolution of surface")
    p.add_argument("--order-Fourier-coils", type=int, help="Order of Fourier representation for coils")
    # Optimization
    p.add_argument("--tmax", type=float, help="Max time for tracing")
    p.add_argument("--num-steps", type=int, help="Number of steps for tracing")
    p.add_argument("--trace-tolerance", type=float, help="Trace tol (atol=rtol)")
    p.add_argument("--current-on-each-coil", type=float, help="Nominal current per coil")
    p.add_argument("--max-len-amp", type=float, help="Max length multiplier vs initial")
    p.add_argument("--max-curv-amp", type=float, help="Max curvature multiplier vs initial max")
    p.add_argument("--min-distance-cc", type=float, help="Minimum coil-coil distance")
    p.add_argument("--max-fun-evals", type=int, help="Max optimization function evaluations")
    p.add_argument("--tol-opt", type=float, help="Optimization tolerance")
    # Modes
    p.add_argument("--use-circular-coils", action="store_true", help="Use circular initial coil guess")
    p.add_argument("--no-plot-fieldlines", dest="plot_fieldlines", action="store_false",
                   help="Do not trace or plot fieldlines")
    p.add_argument("--show-coils-fitted-to-fourier", action="store_true", help="Also plot Fourier-fitted coils")
    # Plot knobs
    p.add_argument("--tube-radius", type=float, help="Tube radius")
    p.add_argument("--tube-theta", type=int, dest="tube_theta", help="Angular samples for tubes")
    p.add_argument("--tube-opacity", type=float, dest="tube_opacity", help="Opacity of tube meshes")
    p.add_argument("--decimate", type=int, help="Take every-k points along centerlines")
    # System
    p.add_argument("--devices", type=int, default=5, help="XLA host device count for JAX (env flag)")
    return p

def parse_args_to_config(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args, Config())
    return args, cfg
