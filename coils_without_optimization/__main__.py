"""Command–line interface for the coil examples.

This module provides a very small CLI that takes a single argument,
``--config``, pointing at a TOML file.  The file must contain at least
``example = "booz"`` or ``example = "nearaxis"``.  Based on this
flag the appropriate workflow is executed.  Additional keys in the
TOML file override default parameters of the corresponding
configuration dataclass.

Usage:

    python -m coils_without_optimization --config path/to/my_config.toml

You can optionally override the VMEC file used in the Boozer example
by providing ``--wout /path/to/wout_mycase.nc`` on the command line.
This override takes precedence over ``file_to_use`` in the TOML file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .toml_config import load_config
from .booz_driver import run_booz_example
from .nearaxis_driver import run_nearaxis_example
from .log_utils import log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="coils_without_optimization",
        description="Run coil design examples from a TOML configuration",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "--wout",
        type=str,
        default=None,
        help=(
            "Override path to VMEC wout_*.nc file (Boozer example only). "
            "If provided, this takes precedence over file_to_use in the config."
        ),
    )
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    # Dispatch based on type of config
    if cfg.__class__.__name__ == "BoozConfig":
        log(f"Selected Boozer→Coils example with output name '{cfg.output_name}'.")
        run_booz_example(cfg, wout_override=args.wout)
    elif cfg.__class__.__name__ == "NearAxisConfig":
        log(f"Selected near–axis example with output name '{cfg.output_name}'.")
        run_nearaxis_example(cfg)
    else:
        raise RuntimeError(f"Unknown config type: {cfg}")


if __name__ == "__main__":  # pragma: no cover
    main()