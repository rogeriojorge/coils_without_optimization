from collections import namedtuple
from essos.fields import BiotSavart
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_BdotN
from essos.surfaces import BdotN_over_B
import jax.numpy as jnp

OptResult = namedtuple("OptResult", "coils bdotn_initial bdotn_opt length0 curvature0 length_opt curvature_opt")

def run_optimization(coils_initial, vmec_essos, max_len, max_curv, min_d_cc, tol, max_evals):
    coils_opt = optimize_loss_function(
        loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial,
        tolerance_optimization=tol, maximum_function_evaluations=max_evals,
        vmec=vmec_essos, max_coil_length=max_len, max_coil_curvature=max_curv,
        min_distance_cc=min_d_cc)

    field0 = BiotSavart(coils_initial); field1 = BiotSavart(coils_opt)
    b0 = BdotN_over_B(vmec_essos.surface, field0)
    b1 = BdotN_over_B(vmec_essos.surface, field1)

    length0 = jnp.max(field0.coils.length)
    curv0   = jnp.mean(field0.coils.curvature)
    length1 = jnp.max(field1.coils.length)
    curv1   = jnp.mean(field1.coils.curvature)

    return OptResult(coils_opt, b0, b1, length0, curv0, length1, curv1)
