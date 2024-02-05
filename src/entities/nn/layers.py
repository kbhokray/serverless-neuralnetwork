from entities.nn.common import *


class Dense(Layer):
    def __init__(self, width: int):
        self.width = width

    def init_fn(self, input_dim, randkey=None):
        self.input_dim = input_dim
        # based on default initialization of pytorch linear layer
        # https://github.com/pytorch/pytorch/blob/f8e14f3b46e68a5271a8c57ce749ad8057d77ddd/torch/nn/modules/linear.py#L105
        gain = np.sqrt(2.0 / 6.0)
        w_bound = gain * np.sqrt(3 / input_dim)
        b_bound = np.sqrt(1 / input_dim)
        if randkey is not None:
            rng = np.random.RandomState(randkey)
            W = rng.uniform(low=-w_bound, high=w_bound, size=(input_dim, self.width))
            b = rng.uniform(low=-b_bound, high=b_bound, size=(self.width, 1))
        else:
            W = random.uniform(low=-w_bound, high=w_bound, size=(input_dim, self.width))
            b = random.uniform(low=-b_bound, high=b_bound, size=(self.width, 1))

        return (W, b)

    def apply_fn(self, params, x):
        W, b = params

        outputs = np.dot(W.T, x) + b

        return outputs
