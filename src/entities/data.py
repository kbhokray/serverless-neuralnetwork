from __future__ import annotations
import os
from datetime import datetime
from bson import ObjectId
from dataclasses import dataclass
from pymongo import MongoClient
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

URL_DB = os.environ.get("URL_DB", "")
client = MongoClient(URL_DB)
db = client[URL_DB.split("/")[-1].split("?")[0]]


@dataclass
class DataPoint:
    x: list[float]
    y: float

    docCreatedOn: int = int(datetime.now().timestamp())
    _id: ObjectId = ObjectId()

    __col_name__ = "datapoints"

    _NO_DB_FIELDS = []

    def save(self) -> DataPoint:
        collection = db[self.__col_name__]
        result = collection.insert_one(
            {
                key: value.value if isinstance(value, Enum) else value
                for key, value in self.__dict__.items()
                if key not in set(self._NO_DB_FIELDS + ["_id"])
            }
        )

        self._id = result.inserted_id
        return self

    @classmethod
    def find_all(cls) -> list[DataPoint]:
        collection = db[cls.__col_name__]
        kwargs: dict[str, dict | list | int] = {"filter": {}}

        response = collection.find(**kwargs)

        return [DataPoint(x=r.get("x", []), y=r.get("y", 0)) for r in response]
