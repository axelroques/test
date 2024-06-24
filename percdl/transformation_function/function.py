
from .transformation import (
    transform_x_from_params,
    projection_params,
    multiple_layer,
    _personalize, 
    jac_D_base
)
from jax import device_put
import numpy as np


class TransformationFunction:
    
    def __init__(self, L, D, W) -> None:
        """
        Transformation function.
        Acts as a reparameterization on discrete samples, with a 
        neural networks like structure (composition of sinus functions).

        Inputs:
            - L: length of the atoms
            - D: depth
            - W: width
        """
        
        self.L = L 
        self.D = D
        self.W = W

        self.M = self.D * self.W 
        self.arange_pi = np.arange(self.W + 1)[1:] * np.pi
    
    def transform(self, x, params):
        """
        Apply function f to x.
        x=(x_jnp,param_flatten)
        and return x\circ psi where psi is defined with params
        """
        x_ = device_put(x)
        params_ = device_put(params)
        return transform_x_from_params(x_, params_, self.D, self.W, self.L)
    
    def personalize(self, Phi, A):
        """
        Personalize a common dictionary.
        """
        return _personalize(Phi, A, self.D, self.W, self.L)
    
    def derive_transform(self, x, params, m):
        """
        Apply function df (w.r.t. param m) to x.
        """
        x_ = device_put(x)
        param_ = device_put(params)
        return self.derive_partial_transform_x_from_params_m(
            x_, param_, self.D, self.W, self.L, m
        )
    
    def proj(self, params):
        """
        Take the parameter (numpy array) and renormalize it 
        to ensure that psi is a diffeomorphism.
        """
        params_ = device_put(params)
        params_ = np.reshape(params_, (self.D, self.W))
        return projection_params(params_)
    
    def _find_psi(self,param_flatten):
        return multiple_layer(param_flatten, self.D, self.W, self.L)
    
    def _jac_D_base(self,atom):
        return jac_D_base(atom, self.D, self.W, self.L)

    
    
