import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import logging

from entities.sls import Response, ResponseStatusCode
from entities.neuralnetwork import NeuralNetwork, ActivationFn
from entities.data import DataPoint
import numpy as np

logging.basicConfig(
    format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(event, context):
    logger.info("Starting Neural Network Training")

    datapoints: list[DataPoint] = DataPoint.find_all()

    if not datapoints:
        logger.warning("No datapoints found in DB; Returning")
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"message": "No datapoints found"},
        )
        return str(response)

    logger.info(f"Training neural net with {len(datapoints)} datapoints")

    xs = np.array([d.x for d in datapoints]).T
    ys = np.array([[d.y for d in datapoints]])

    network = NeuralNetwork(
        learning_rate=0.001,
        activation_fn=ActivationFn.LEAKYRELU,
        width=3,
        depth=1,
        training_steps=1000,
        params_seed=np.random.randint(3),
        input_dim=len(xs),
    )

    network.train((xs, ys))
    network.save()

    logger.info(f"Network saved to DB; _id:{network._id}")

    response = Response(
        statusCode=ResponseStatusCode.SUCCESS, result={"Network Id": str(network._id)}
    )

    return str(response)


if __name__ == "__main__":
    train(None, None)
