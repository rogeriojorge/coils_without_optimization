# boozcoils/trace_utils.py
import numpy as np
import jax.numpy as jnp
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from jax import block_until_ready

def initial_conditions_from_vmec(vmec, nfieldlines: int):
    """
    Build initial conditions on a ring of radii spanning the min/max cylindrical R
    implied by VMEC rmnc at s=0..1. We use NumPy for VMEC arrays (to avoid JAX
    tracing issues) and convert to JAX arrays at the end.
    """
    # vmec.wout.rmnc shape: (mnmax, ns) NumPy array
    rmnc = np.asarray(vmec.wout.rmnc)
    # Sum over mn to get R(s) at theta=0 approximation (matches your original usage):
    rmnc_sum = rmnc.sum(axis=0)  # shape (ns,)
    rmin = float(rmnc_sum[0])
    rmax = float(rmnc_sum[-1])

    # Create a ring of starting points in Cartesian coords
    R0 = np.linspace(rmin, rmax, int(nfieldlines) + 1, endpoint=True)[:-1]
    Z0 = np.zeros_like(R0)
    phi0 = np.zeros_like(R0)

    xyz0 = np.stack([R0 * np.cos(phi0), R0 * np.sin(phi0), Z0], axis=1).astype(np.float32)
    return jnp.asarray(xyz0)  # JAX array, fine for Tracing

def trace_fieldlines(coils, init_xyz, tmax, num_steps, atol, rtol):
    traj = block_until_ready(Tracing(
        field=BiotSavart(coils),
        model='FieldLineAdaptative',
        initial_conditions=init_xyz,
        maxtime=tmax,
        times_to_trace=num_steps,
        atol=atol, rtol=rtol
    ))
    return traj.trajectories
