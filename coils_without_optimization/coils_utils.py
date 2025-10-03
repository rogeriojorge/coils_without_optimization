"""Utilities for constructing and fitting coil curves.

This module wraps common functions from the `essos.coils` namespace and
adds a few convenience helpers for packing arrays into the `gamma`
format and creating simple initial guesses.  The functions defined
here are used by both the Boozer and near–axis examples.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import numpy as np
import jax.numpy as jnp
from essos.coils import (
    CreateEquallySpacedCurves,
    Curves,
    Coils,
    fit_dofs_from_coils,
)


def gamma_from_xyz_columns(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cols: Optional[Iterable[int]] = None,
    ncurves: Optional[int] = None,
) -> np.ndarray:
    """Pack columnar coordinate arrays into a 3D array of centre lines.

    The inputs ``X``, ``Y`` and ``Z`` should each be two–dimensional
    arrays with shape ``(ntheta, nphi)``.  The returned array has
    shape ``(ncurves, ntheta, 3)``.  You can select a subset of
    columns via ``cols`` or specify ``ncurves`` to take the first
    ``ncurves`` columns.  If neither is provided, all columns are
    packed (legacy behaviour).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    ntheta, ncols = X.shape
    if cols is None:
        if ncurves is not None:
            cols = range(int(ncurves))
        else:
            cols = range(ncols)
    cols = list(cols)
    gam = np.zeros((len(cols), ntheta, 3), dtype=X.dtype)
    for ii, c in enumerate(cols):
        gam[ii, :, 0] = X[:, c]
        gam[ii, :, 1] = Y[:, c]
        gam[ii, :, 2] = Z[:, c]
    return gam


def circular_guess(
    ncoils: int,
    order: int,
    R_major: float,
    r_minor: float,
    ntheta: int,
    nfp: int,
    stellsym: bool = True,
) -> Curves:
    """Create an equal–spaced circular coil guess.

    This is a thin wrapper around `essos.coils.CreateEquallySpacedCurves`.
    """
    return CreateEquallySpacedCurves(
        n_curves=ncoils,
        order=order,
        R=R_major,
        r=r_minor,
        n_segments=ntheta,
        nfp=nfp,
        stellsym=stellsym,
    )


def fit_curves_from_gamma(
    coils_gamma: Sequence[np.ndarray] | np.ndarray,
    order: int,
    ntheta: int,
    nfp: int,
    stellsym: bool = True,
) -> Curves:
    """Fit Fourier–series curves to an array of sample points.

    The first dimension of ``coils_gamma`` indexes individual curves,
    the second dimension indexes the poloidal angle, and the last
    dimension holds ``(x,y,z)`` Cartesian coordinates.  The returned
    object is an instance of :class:`essos.coils.Curves`.
    """
    dofs, _ = fit_dofs_from_coils(
        coils_gamma, order=order, n_segments=ntheta, assume_uniform=True
    )
    return Curves(dofs=dofs, n_segments=ntheta, nfp=nfp, stellsym=stellsym)


def build_coils(curves: Curves, current: float, ncoils: int) -> Coils:
    """Construct a :class:`essos.coils.Coils` with uniform current."""
    return Coils(curves=curves, currents=[current] * ncoils)