from enum import Enum
from dataclasses import dataclass, asdict, is_dataclass
import json


class ResponseStatusCode(Enum):
    ERROR = 400
    SUCCESS = 200


class AppException(Exception):
    pass


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


@dataclass(init=False)
class Response:
    statusCode: int
    headers: dict
    body: dict

    def __init__(self, statusCode: ResponseStatusCode, result: dict):
        self.statusCode = statusCode.value
        self.headers = {"Content-Type": "application/json"}
        self.body = result

    def __str__(self):
        return json.dumps(asdict(self), cls=EnumEncoder)


# %%
