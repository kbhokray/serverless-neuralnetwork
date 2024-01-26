import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import logging
import json
from typing import Union

from entities.sls import Response, ResponseStatusCode, AppException
from services.core import ServerlessNeuralNetwork

logging.basicConfig(
    format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_datapoint(event, context):
    datapoint: dict[str, Union[list[Union[float, int]], float, int]] = json.loads(
        event.get("body", "{}")
    )
    try:
        response = ServerlessNeuralNetwork.save_datapoint(datapoint=datapoint)
        return str(response)
    except AppException as ae:
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": str(ae)},
        )
        return str(response)
    except Exception:
        logger.exception(f"Unknown Exception")
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": "Internal Error"},
        )
        return str(response)


def train(event, context):
    try:
        response = ServerlessNeuralNetwork.train()
        return str(response)
    except AppException as ae:
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": str(ae)},
        )
        return str(response)
    except Exception:
        logger.exception(f"Unknown Exception")
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": "Internal Error"},
        )
        return str(response)


def infer(event, context):
    datapoint: dict[str, str] = json.loads(event.get("body", "{}"))
    x: list = datapoint.get("x", [])  # type: ignore
    network_id = datapoint.get("network_id")
    try:
        response = ServerlessNeuralNetwork.infer(x, network_id)
        return str(response)
    except AppException as ae:
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": str(ae)},
        )
        return str(response)
    except Exception:
        logger.exception(f"Unknown Exception")
        response = Response(
            statusCode=ResponseStatusCode.ERROR,
            result={"error": "Internal Error"},
        )
        return str(response)


if __name__ == "__main__":
    save_datapoint(
        event={"body": json.dumps({"x": [2.0, 3.0, -1.0], "y": 1})}, context=None
    )

    train(None, None)

    infer(event={"body": json.dumps({"x": [2.0, 3.0, -1.0]})}, context=None)
