
from jax.example_libraries import optimizers as jax_opt
from jax import ( 
    jit, device_put, value_and_grad, vmap
)
from ..transformation_function import _personalize, projection_params
from functools import partial
from .utils import l2_loss
import jax.numpy as jnp
import jax.lax as lax


@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def _IPU(
    X, Phi, Z, A,
    step_size:float, nb_steps:int,
    D:int, W:int, L:int
):
    """
    Parameters update step.

    Inputs:
        - X: S x N
        - Phi: K x L
        - Z: S x K x N
        - A: S x K x M
    """
    
    @jit
    def _loss(A_current):
        D_perso = _personalize(Phi_, A_current, D, W, L)
        return l2_loss(X_, Z_, D_perso)    
    
    @jit
    def _step(nb_step, opt_state):
        _, grads = value_and_grad(_loss)(get_params(opt_state))
        opt_state = opt_update(nb_step, grads, opt_state)

        # Projection step
        opt_state = opt_init(
            vmap(
                lambda x_S: vmap(lambda x: proj(x, D, W))(x_S) 
            )(get_params(opt_state))
        )
        return opt_state
    
    # JAX acceleration
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_ = device_put(Phi)
    A_init = device_put(A)
    
    # Initialize optimizer
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(A_init)
    
    # Optimize A
    opt_final = lax.fori_loop(0, nb_steps, _step, opt_state)
    final_params=get_params(opt_final)
    
    # Return new A
    return final_params

@partial(jit, static_argnames=["D", "W"])
def proj(params, D:int, W:int):
    return projection_params(jnp.reshape(params, (D, W)))