import numpy as np
import jax.numpy as jnp
from essos.coils import CreateEquallySpacedCurves, Curves, Coils, fit_dofs_from_coils
from typing import Iterable, Optional

def gamma_from_xyz_columns(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cols: Optional[Iterable[int]] = None,
    ncurves: Optional[int] = None,
):
    """
    Pack (ntheta, ncols) X,Y,Z into gamma (n_curves, ntheta, 3).

    By default this used to pack *all* columns (ncols), which mismatched the
    number of currents. Now you can:
      - pass 'cols' explicitly (indices of columns to use), or
      - pass 'ncurves' to take the first ncurves columns, or
      - if both None, fall back to ALL columns (legacy behavior).
    """
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
    ntheta, ncols = X.shape

    if cols is None:
        if ncurves is not None:
            cols = range(int(ncurves))  # take the first ncurves columns
        else:
            cols = range(ncols)         # legacy: take all columns

    cols = list(cols)
    gam = np.zeros((len(cols), ntheta, 3), dtype=X.dtype)
    for ii, c in enumerate(cols):
        gam[ii, :, 0] = X[:, c]
        gam[ii, :, 1] = Y[:, c]
        gam[ii, :, 2] = Z[:, c]
    return gam

def circular_guess(ncoils, order, R_major, r_minor, ntheta, nfp, stellsym=True):
    return CreateEquallySpacedCurves(n_curves=ncoils, order=order, R=R_major, r=r_minor,
                                     n_segments=ntheta, nfp=nfp, stellsym=stellsym)

def fit_curves_from_gamma(coils_gamma, order, ntheta, nfp, stellsym=True):
    dofs, _ = fit_dofs_from_coils(coils_gamma, order=order, n_segments=ntheta, assume_uniform=True)
    return Curves(dofs=dofs, n_segments=ntheta, nfp=nfp, stellsym=stellsym)

def build_coils(curves, current, ncoils):
    # Ensure currents length matches number of curves
    return Coils(curves=curves, currents=[current]*ncoils)
