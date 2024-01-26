import logging
from entities.datapoint import DataPoint

logging.basicConfig(
    format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [1.0, 1.0, -1.0],
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

    logger.info(f"Loading {len(xs)} datapoints to database")

    datapoints = []
    for x, y in zip(xs, ys):
        datapoints.append(DataPoint(x=x, y=y))

    DataPoint.insert_many(datapoints)

    logger.info(f"All datapoints loaded!")


if __name__ == "__main__":
    run()
