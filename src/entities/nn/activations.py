from enum import Enum
from entities.nn.common import Layer
from entities.nn.common import *


# activation functions
class ActivationFn(Enum):
    LEAKYRELU = "LEAKYRELU"
    TANH = "TANH"
    SIGMOID = "SIGMOID"


class LeakyRelu(Layer):
    def __init__(self, negative_slope=1e-2):
        super().__init__()
        self.negative_slope = negative_slope

    def apply_fn(self, params, x):
        return np.where(x >= 0, x, self.negative_slope * x)


class Relu(Layer):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def apply_fn(self, params, x):
        return np.maximum(x, 0)


class Celu(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def apply_fn(self, params, x):
        return np.maximum(x, 0.0) + self.alpha * np.expm1(
            np.minimum(x, 0.0) / self.alpha
        )


class Elu(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def apply_fn(self, params, x):
        return np.where(x > 0, x, self.alpha * np.expm1(np.where(x > 0, 0.0, x)))


class Selu(Layer):
    def apply_fn(self, params, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        return scale * Elu(alpha).apply_fn(params, x)


class SoftPlus(Layer):
    def apply_fn(self, params, x):
        return np.logaddexp(x, 0)


class LogSigmoid(Layer):
    def apply_fn(self, params, x):
        return -SoftPlus().apply_fn(params, -x)


class Sigmoid(Layer):
    def apply_fn(self, params, x):
        return 1 / (1 + np.exp(-x))


class Silu(Layer):
    def apply_fn(self, params, x):
        return x * Sigmoid().apply_fn(params, x)


class SoftSign(Layer):
    def apply_fn(self, params, x):
        return x / (np.abs(x) + 1)


class HardTanh(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def apply_fn(self, params, x):
        return np.where(x > 1, 1, np.where(x < -1, -1, x))


class Tanh(Layer):
    def apply_fn(self, params, x):
        return np.tanh(x)
