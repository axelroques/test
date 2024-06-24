
from jax.example_libraries import optimizers as jax_opt
from jax import ( 
    jit, device_put, value_and_grad
)

from ..transformation_function import _personalize
from functools import partial
from .utils import l2_loss
import jax.numpy as jnp
import jax.lax as lax

@partial(jit, static_argnames=["step_size", "nb_steps"])
def _CDU(
    X, Phi, Z,
    step_size: float,
    nb_steps: int
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """

    @jit
    def _loss(Phi_current):
        D = jnp.repeat(jnp.expand_dims(Phi_current, axis=1), X_.shape[0], axis=1)
        return l2_loss(X_, Z_, D)

    @jit
    def _step(nb_step, opt_state):
        _, grads = value_and_grad(_loss)(get_params(opt_state))
        opt_state = opt_update(nb_step, grads, opt_state)
        return opt_state
    
    # JAX acceleration
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_init = device_put(Phi)

    # Initialize optimizer
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(Phi_init)

    # Optimize Phi 
    opt_final = lax.fori_loop(0, nb_steps, _step, opt_state)
    
    # Return new Phi
    return get_params(opt_final)



@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def _CDU_perso(
    X, Phi, Z, A,
    step_size:float,
    nb_steps:int,
    D:int,
    W:int,
    L:int
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """

    @jit
    def _loss(Phi_current):
        Phi_perso = _personalize(Phi_current, A_, D, W, L)
        return l2_loss(X_, Z_, Phi_perso)

    @jit
    def _step(nb_step, opt_state):
        _, grads = value_and_grad(_loss)(get_params(opt_state))
        opt_state = opt_update(nb_step, grads, opt_state)
        return opt_state
    
    # JAX acceleration
    A_ = device_put(A) 
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_init = device_put(Phi)

    # Initialize optimizer
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(Phi_init)

    # Optimize Phi 
    opt_final = lax.fori_loop(0, nb_steps, _step, opt_state)
    
    # Return new Phi
    return get_params(opt_final)