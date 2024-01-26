from __future__ import annotations
import os
from enum import Enum
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from typing import Type, TypeVar, Union

T = TypeVar("T", bound="Document")

load_dotenv()

URL_DB = os.environ.get("URL_DB", "")
client = MongoClient(URL_DB)
db = client[URL_DB.split("/")[-1].split("?")[0]]


class Document:
    _id: ObjectId = ObjectId()

    __col_name__ = "documents"

    _NO_DB_FIELDS = []

    def __init__(self, _id=ObjectId()):
        self._id = _id

    @classmethod
    def _get_collection(cls):
        return db[cls.__col_name__]

    def save(self):
        collection = db[self.__col_name__]
        result = collection.insert_one(
            {
                key: value.value if isinstance(value, Enum) else value
                for key, value in self.__dict__.items()
                if key not in set(self._NO_DB_FIELDS + ["_id"])
                and not callable(self.__dict__[key])
            }
        )

        self._id = result.inserted_id
        return self

    @classmethod
    def insert_many(cls, objs: list):
        collection = db[cls.__col_name__]
        result = collection.insert_many(
            [
                {
                    key: value.value if isinstance(value, Enum) else value
                    for key, value in o.__dict__.items()
                    if key not in set(cls._NO_DB_FIELDS + ["_id"])
                    and not callable(o.__dict__[key])
                }
                for o in objs
            ]
        )

        return result.inserted_ids

    @classmethod
    def find_one(cls: Type[T], filter={}) -> Union[T, None]:
        collection = db[cls.__col_name__]
        kwargs: dict[str, dict] = {"filter": filter}

        response = collection.find_one(**kwargs)

        if not response:
            return None

        return cls(**response)

    @classmethod
    def find_all(cls: Type[T]) -> list[T]:
        collection = db[cls.__col_name__]
        kwargs: dict[str, dict] = {"filter": {}}

        response = collection.find(**kwargs)

        return [cls(**r) for r in response]
