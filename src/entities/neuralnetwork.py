# %%
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from autograd.misc.optimizers import adam
from autograd import grad
from dataclasses import dataclass, field
from datetime import datetime
from bson import ObjectId, Binary
import pickle
from typing import Union
from entities.nn.activations import *
from entities.nn.layers import *
from entities.nn.common import *
from entities.dbdocument import Document
from utils import load_data


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
    relu_alpha: float = 0.1
    params: list = field(default_factory=list)
    params_data: Union[Binary, None] = None
    docCreatedOn: int = int(datetime.now().timestamp())
    _id: ObjectId = ObjectId()

    __col_name__ = "neuralnetwork"

    _NO_DB_FIELDS = ["params"]

    def serialize_params(self):
        serialized = []
        for layer in self.params:
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
            Dense(width=self.width),
            LeakyRelu(negative_slope=self.relu_alpha),
        ]
        for _ in range(self.depth):
            layers_list += [
                Dense(width=self.width),
                LeakyRelu(negative_slope=self.relu_alpha),
            ]

        layers_list += [Dense(width=1)]

        return serial(layers_list)

    def __post_init__(self):
        init_fn, predict_fn = self.model()
        self.params = init_fn(input_dim=self.input_dim, randkey=self.params_seed)
        self.predict_fn = predict_fn

        if self.params and not self.params_data:
            self.params_data = Binary(pickle.dumps(self.params, protocol=2))

        if self.params_data and not self.params:
            self.params = pickle.loads(self.params_data)

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

        optimized_params = adam(
            grad_loss,
            self.params,
            num_iters=self.training_steps,
            callback=print_perf,
        )

        self.params = optimized_params
        self.params_data = Binary(pickle.dumps(optimized_params, protocol=2))

        return optimized_params


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    xs = X_train[0:1, :].T
    ys = np.array([y_train[0:1]])

    seed = random.randint(0, 1000)
    print(f"Using {seed=}")
    NeuralNetwork(
        learning_rate=0.01,
        activation_fn=ActivationFn.LEAKYRELU,
        width=3,
        depth=3,
        training_steps=1000,
        params_seed=seed,
        input_dim=len(xs),
    ).train((xs, ys))
# %%
