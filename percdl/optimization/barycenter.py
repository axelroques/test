
from ..transformation_function.transformation import *
from jax.example_libraries import optimizers as jax_opt
from jax import value_and_grad, jit, vmap
from functools import partial
import jax.numpy as jnp


@partial(jit,static_argnames=["nb_layers", "width", "L"])
def recenter_Phi(Phi, A, nb_layers:int, width:int, L:int):

    Psi_mean = vmap(
        lambda x: Psi_k_mean(x, nb_layers, width, L)
    )(A.transpose(1, 0, 2)) 
    
    ### A.T because A (S,M,K) #A.transpose(2,0,1)=(K,S,M)
    PhiT = vmap(
        lambda x, psi: transform_x_by_psi(x, psi, L)
    )(Phi, Psi_mean)
    return PhiT


@partial(jit,static_argnames=["nb_layers", "width", "L"])
def relearn_A(Phi_new, Phi_old, A, nb_layers:int, width:int, L:int):

    nb_steps = 10
    step_size = 0.01
    A_init = A

    D_personalised_old = vmap(
        lambda x, alpha: transform_x_from_all_params(x, alpha, nb_layers, width, L)
    )(Phi_old, A.transpose(1, 0, 2)) # We put K on the first axis
    # size (K, S, L)

    @jit
    def loss_to_opt(A_new):
        D_personalised_new = vmap(
            lambda x, alpha: transform_x_from_all_params(x, alpha, nb_layers, width, L)
        )(Phi_new, A_new.transpose(1, 0, 2))
        return jnp.linalg.norm(D_personalised_old-D_personalised_new)**2
    
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(A_init)

    def step(step, opt_state):
        value, grads = value_and_grad(loss_to_opt)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        state_params = get_params(opt_state)
        params_to_normalized_K_S = jnp.reshape(state_params, (state_params.shape[0], -1, nb_layers, width))
        opt_state = opt_init(
            vmap(
                lambda x_S: vmap(projection_params)(x_S)
            )(params_to_normalized_K_S) # Change here
        )
        return value, opt_state

    for i in range(nb_steps):
        _, opt_state = step(i, opt_state)

    return get_params(opt_state)


@partial(jit,static_argnames=["nb_layers", "width", "L"])
def Psi_k_mean(A_k, nb_layers:int, width:int, L:int):
    Psi_k_vec = vmap(
        lambda x: multiple_layer(x, nb_layers, width, L)
    )(A_k)
    Psi_k_mean_output = Psi_k_vec.mean(axis=0)
    return Psi_k_mean_output





