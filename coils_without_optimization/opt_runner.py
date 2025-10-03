"""Wrapper for coil optimisation with ESSOS.

This module defines a light wrapper around
``essos.optimization.optimize_loss_function`` to compute the
optimised coil set along with summary metrics.  The resulting
``OptResult`` namedtuple contains the optimised coils and diagnostic
information which is printed by the drivers.
"""

from __future__ import annotations

from collections import namedtuple
import jax.numpy as jnp
from essos.fields import BiotSavart
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_BdotN
from essos.surfaces import BdotN_over_B


# Result type for optimisation output
OptResult = namedtuple(
    "OptResult",
    "coils bdotn_initial bdotn_opt length0 curvature0 length_opt curvature_opt",
)


def run_optimization(
    coils_initial,
    vmec_essos,
    max_len: float,
    max_curv: float,
    min_d_cc: float,
    tol: float,
    max_evals: int,
) -> OptResult:
    """Optimise coil DOFs against the B·n objective with constraints.

    Parameters
    ----------
    coils_initial : essos.coils.Coils
        Initial coil set.
    vmec_essos : essos.fields.Vmec
        The VMEC field object on which B·n is evaluated.
    max_len : float
        Maximum allowed coil length.
    max_curv : float
        Maximum allowed coil curvature.
    min_d_cc : float
        Minimum coil–coil distance.
    tol : float
        Optimisation tolerance.
    max_evals : int
        Maximum number of objective function evaluations.

    Returns
    -------
    OptResult
        Named tuple containing the optimised coils and summary metrics.
    """
    coils_opt = optimize_loss_function(
        loss_BdotN,
        initial_dofs=coils_initial.x,
        coils=coils_initial,
        tolerance_optimization=tol,
        maximum_function_evaluations=max_evals,
        vmec=vmec_essos,
        max_coil_length=max_len,
        max_coil_curvature=max_curv,
        min_distance_cc=min_d_cc,
    )
    field0 = BiotSavart(coils_initial)
    field1 = BiotSavart(coils_opt)
    b0 = BdotN_over_B(vmec_essos.surface, field0)
    b1 = BdotN_over_B(vmec_essos.surface, field1)
    length0 = jnp.max(field0.coils.length)
    curv0 = jnp.mean(field0.coils.curvature)
    length1 = jnp.max(field1.coils.length)
    curv1 = jnp.mean(field1.coils.curvature)
    return OptResult(
        coils_opt,
        b0,
        b1,
        length0,
        curv0,
        length1,
        curv1,
    )