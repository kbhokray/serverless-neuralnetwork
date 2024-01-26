# %%
import os
import autograd.numpy as np  # type: ignore
import numpy as onp
import autograd.numpy.random as random  # type: ignore
from autograd.misc.optimizers import sgd
from autograd import grad
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from bson import ObjectId, Binary
import pickle
from typing import Union
from entities.dbdocument import Document

np: onp  # type: ignore
random: onp.random  # type: ignore


# activation functions
class ActivationFn(Enum):
    LEAKYRELU = "LEAKYRELU"


class Layer:
    def __init__(self):
        self.width = 0

    def init_fn(self, input_dim, randkey=None):
        pass

    def apply_fn(self, params, x):
        pass


class Dense(Layer):
    def __init__(
        self,
        width: int,
        W_std: float = 1.5,
        b_std: float = 0.05,
    ):
        self.width = width
        self.W_std = W_std
        self.b_std = b_std

    def init_fn(self, input_dim, randkey=None):
        self.input_dim = input_dim

        if randkey is not None:
            rng = np.random.RandomState(randkey)
            W = rng.normal(size=(input_dim, self.width))
            b = rng.normal(size=(self.width, 1))
        else:
            W = random.normal(size=(input_dim, self.width))
            b = random.normal(size=(self.width, 1))

        return (
            W * self.W_std / (input_dim**0.5),
            b * self.b_std,
        )

    def apply_fn(self, params, x):
        W, b = params

        outputs = np.dot(W.T, x) + b

        return outputs


class LeakyRelu(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def init_fn(self, input_dim, randkey):
        self.width = input_dim
        return ()

    def apply_fn(self, params, x):
        return np.maximum(self.alpha * x, x)


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def init_fn(self, input_dim, randkey):
        self.width = input_dim
        return ()

    def apply_fn(self, params, x):
        mean = np.mean(x, axis=0, keepdims=True)
        variance = np.var(x, axis=0, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        return x_normalized


def serial(layers: list[Layer]):
    init_params = []

    def init_fn(input_dim, randkey):
        _input_dim = input_dim
        for layer in layers:
            init_params.append(layer.init_fn(_input_dim, randkey))
            _input_dim = layer.width

        return init_params

    def apply_fn(params, input):
        x = input
        for i, layer in enumerate(layers):
            x = layer.apply_fn(params[i], x)
        return x

    return init_fn, apply_fn


@dataclass
class NeuralNetwork(Document):
    learning_rate: float
    activation_fn: ActivationFn
    width: int
    depth: int
    training_steps: int
    params_seed: int
    input_dim: int
    W_std: float = 1.5
    b_std: float = 0.05
    relu_alpha: float = 0.1
    init_params: list = field(default_factory=list)
    final_params: list = field(default_factory=list)
    params_data: Union[Binary, None] = None
    docCreatedOn: int = int(datetime.now().timestamp())
    _id: ObjectId = ObjectId()

    __col_name__ = "neuralnetwork"

    _NO_DB_FIELDS = ["init_params", "final_params"]

    @classmethod
    def serialize_params(cls, params):
        serialized = []
        for layer in params:
            if layer:
                serialized.append(
                    {"Weights": layer[0].tolist(), "Bias": layer[1].tolist()}
                )
            else:
                serialized.append({})

        return serialized

    @classmethod
    def deserialize_params(cls, params):
        deserialized = []
        for layer in params:
            if layer:
                deserialized.append(
                    (np.array(layer.get("Weights")), np.array(layer.get("Bias")))
                )
            else:
                deserialized.append(())

        return deserialized

    def model(self):
        layers_list: list[Layer] = [
            Dense(
                width=self.width,
                W_std=self.W_std,
                b_std=self.b_std,
            ),
        ]
        for _ in range(self.depth):
            layers_list += [
                LeakyRelu(alpha=self.relu_alpha),
                Dense(
                    width=self.width,
                    W_std=self.W_std,
                    b_std=self.b_std,
                ),
            ]

        layers_list += [
            Dense(
                width=1,
                W_std=self.W_std,
                b_std=self.b_std,
            )
        ]

        return serial(layers_list)

    def __post_init__(self):
        init_fn, predict_fn = self.model()
        self.init_params = init_fn(input_dim=self.input_dim, randkey=self.params_seed)
        self.predict_fn = predict_fn

        if self.final_params and not self.params_data:
            self.params_data = Binary(pickle.dumps(self.final_params, protocol=2))

        if self.params_data and not self.final_params:
            self.final_params = pickle.loads(self.params_data)

    def train(self, train):
        def mse_loss(params, x, y):
            preds = self.predict_fn(params, x)
            return np.mean((preds - y) ** 2)

        X, Y = train

        def loss(params, i):
            loss = 0.5 * mse_loss(params, X, Y)  # type: ignore
            if i % 100 == 0:
                print(f"loss={loss._value}")

            return loss

        grad_loss = grad(loss, 0)

        def print_perf(params, iter, gradient):
            # print("{:15}".format(iter // 10))
            pass

        optimized_params = sgd(
            grad_loss,
            self.init_params,
            num_iters=self.training_steps,
            callback=print_perf,
        )

        self.final_params = optimized_params
        self.params_data = Binary(pickle.dumps(optimized_params, protocol=2))

        return optimized_params


if __name__ == "__main__":
    xs = np.array(
        [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    ).T

    ys = np.array([[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]])

    seed = random.randint(0, 1000)
    print(f"Using {seed=}")
    NeuralNetwork(
        learning_rate=0.001,
        activation_fn=ActivationFn.LEAKYRELU,
        width=3,
        depth=1,
        training_steps=1000,
        params_seed=seed,
        input_dim=3,
    ).train((xs, ys))
# %%
