"""Tracing helpers for magnetic field lines.

These routines wrap functionality from the `essos.dynamics` and
`essos.fields` modules to provide convenient field line tracing for a
given coil set.  They avoid recomputing large arrays on the JAX
tracing graph by converting VMEC data to NumPy prior to constructing
initial conditions.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from jax import block_until_ready


def initial_conditions_from_vmec(vmec, nfieldlines: int) -> jnp.ndarray:
    """Construct a ring of initial conditions based on the VMEC radius profile.

    The VMEC output `wout.rmnc` is summed over (m,n) to obtain an
    approximate radial coordinate as a function of the VMEC poloidal
    index `s`.  This is then sampled uniformly to build initial
    conditions on a circle in the midplane.  The returned array has
    shape ``(nfieldlines, 3)`` and type `jax.numpy.ndarray`.

    Parameters
    ----------
    vmec : `simsopt.mhd.vmec.Vmec`
        A VMEC equilibrium from which to draw the radial profile.
    nfieldlines : int
        Number of field lines to trace.

    Returns
    -------
    jax.numpy.ndarray
        Initial positions in Cartesian coordinates.
    """
    rmnc = np.asarray(vmec.wout.rmnc)
    rmnc_sum = rmnc.sum(axis=0)
    rmin = float(rmnc_sum[0])
    rmax = float(rmnc_sum[-1])
    R0 = np.linspace(rmin, rmax, int(nfieldlines) + 1, endpoint=True)[:-1]
    Z0 = np.zeros_like(R0)
    phi0 = np.zeros_like(R0)
    xyz0 = np.stack(
        [R0 * np.cos(phi0), R0 * np.sin(phi0), Z0], axis=1
    ).astype(np.float32)
    return jnp.asarray(xyz0)


def trace_fieldlines(coils, init_xyz, tmax, num_steps, atol, rtol):
    """Trace magnetic field lines through a coil-generated field.

    Parameters
    ----------
    coils : `essos.coils.Coils`
        The coil set defining the magnetic field.
    init_xyz : array‐like
        Initial conditions of shape `(nfieldlines, 3)`.
    tmax : float
        Maximum tracing time.
    num_steps : int
        Number of integration steps.
    atol, rtol : float
        Absolute and relative tolerances for the adaptive integrator.

    Returns
    -------
    list of jax.numpy.ndarray
        A list of field line trajectories; each entry is a 2D array
        containing the Cartesian coordinates along the trajectory.
    """
    traj = block_until_ready(
        Tracing(
            field=BiotSavart(coils),
            model="FieldLineAdaptative",
            initial_conditions=init_xyz,
            maxtime=tmax,
            times_to_trace=num_steps,
            atol=atol,
            rtol=rtol,
        )
    )
    return traj.trajectories