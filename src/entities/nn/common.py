import autograd.numpy as np  # type: ignore
import autograd.numpy.random as random  # type: ignore
import numpy as onp

np: onp  # type: ignore
random: onp.random  # type: ignore


class Layer:
    def __init__(self):
        self.width = 0

    def init_fn(self, input_dim, randkey=None):
        self.width = input_dim
        return ()

    def apply_fn(self, params, x):
        pass
