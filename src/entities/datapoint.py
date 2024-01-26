from __future__ import annotations
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from datetime import datetime
from bson import ObjectId
from dataclasses import dataclass
from entities.dbdocument import Document


@dataclass
class DataPoint(Document):
    x: list[float]
    y: float

    docCreatedOn: int = int(datetime.now().timestamp())
    _id: ObjectId = ObjectId()

    __col_name__ = "datapoints"

    _NO_DB_FIELDS = []


if __name__ == "__main__":
    dps = DataPoint.find_all()
