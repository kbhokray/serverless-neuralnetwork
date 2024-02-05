import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import logging
from typing import Union

from entities.sls import Response, ResponseStatusCode, AppException
from entities.neuralnetwork import NeuralNetwork, ActivationFn
from entities.datapoint import DataPoint
import numpy as np


logging.basicConfig(
    format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

is_input = (
    lambda x: isinstance(x, list)
    and all(isinstance(ft, float) or isinstance(ft, int) for ft in x)
    and len(x) > 0
)

is_list_of_inputs = lambda x: isinstance(x, list) and all(
    isinstance(input, list)
    and all(isinstance(ft, float) for ft in input)
    and len(input) == len(x[0])
    and len(input) > 0
    for input in x
)


class ServerlessNeuralNetwork:
    @staticmethod
    def save_datapoint(
        datapoint: dict[str, Union[list[Union[float, int]], float, int]]
    ) -> Response:
        if not datapoint:
            logger.error("No data passed")
            raise AppException("No data passed")

        x = datapoint.get("x")
        if not isinstance(x, list) or not is_input(x):
            logger.error("Illegal data format: Input (x) should be a list of floats")
            raise AppException(
                "Illegal data format: Input (x) should be a list of floats"
            )

        y = datapoint.get("y")
        if not isinstance(y, float) and not isinstance(y, int):
            logger.error("Illegal data format: Target (y) should be a float")
            raise AppException("Illegal data format: Target (y) should be a float")

        logger.info(f"Saving datapoint with {len(x)} features")

        datapoint_id = None
        try:
            dp = DataPoint(x=x, y=y).save()
            datapoint_id = dp._id
        except:
            logger.exception("Error saving datapoint to DB")
            raise AppException("Error saving datapoint to DB")

        logger.info(f"Datapoint saved to DB; _id:{datapoint_id}")

        return Response(
            statusCode=ResponseStatusCode.SUCCESS,
            result={"DatapointId": str(datapoint_id)},
        )

    @staticmethod
    def train() -> Response:
        logger.info("Starting Neural Network Training")

        datapoints: list[DataPoint] = DataPoint.find_all()

        if not datapoints:
            logger.warning("No datapoints found in DB; Returning")
            response = Response(
                statusCode=ResponseStatusCode.ERROR,
                result={"message": "No datapoints found"},
            )
            return response

        logger.info(f"Training neural net with {len(datapoints)} datapoints")

        xs = np.array([d.x for d in datapoints]).T
        ys = np.array([[d.y for d in datapoints]])

        network = NeuralNetwork(
            learning_rate=0.001,
            activation_fn=ActivationFn.LEAKYRELU,
            width=3,
            depth=3,
            training_steps=1000,
            params_seed=np.random.randint(3),
            input_dim=len(xs),
        )

        network.train((xs, ys))

        try:
            network.save()
        except:
            logger.exception("Error saving network to DB")
            raise AppException("Error saving network to DB")

        logger.info(f"Network saved to DB; _id:{network._id}")

        return Response(
            statusCode=ResponseStatusCode.SUCCESS,
            result={"NetworkId": str(network._id)},
        )

    @staticmethod
    def infer(
        x: Union[list[Union[float, int]], list[list[Union[float, int]]]],
        network_id: Union[str, None] = None,
    ):
        if not is_input(x) and not is_list_of_inputs(x):
            logger.error(
                "Illegal data format: Input (x) should be a list of floats or list of list of floats"
            )
            raise AppException(
                "Illegal data format: Input (x) should be a list of floats or list of list of floats"
            )

        network: Union[NeuralNetwork, None] = NeuralNetwork.find_one(
            {"_id": network_id} if network_id else {}
        )

        if not network:
            logger.error("Cannot find network in DB")
            raise AppException("Cannot find network in DB")

        input = np.array(x if isinstance(x[0], list) else [x]).T

        preds = network.predict_fn(params=network.params, input=input)

        ret_val = preds.tolist() if preds is not None else None

        return Response(
            statusCode=ResponseStatusCode.SUCCESS,
            result={"Predictions": ret_val},
        )

    @staticmethod
    def get_network(
        network_id: str,
    ):
        if not network_id:
            logger.error("Illegal data: network_id cannot be empty")
            raise AppException("Illegal data: network_id cannot be empty")

        network: Union[NeuralNetwork, None] = NeuralNetwork.find_one(
            {"_id": network_id}
        )

        if not network:
            logger.error("Cannot find network in DB")
            raise AppException("Cannot find network in DB")

        return network.serialize_params()
