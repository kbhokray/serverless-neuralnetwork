import logging
from entities.data import DataPoint

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
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

    logger.info(f"Loading {len(xs)} datapoints to database")

    for x, y in zip(xs, ys):
        DataPoint(x=x, y=y).save()

    logger.info(f"All datapoints loaded!")


if __name__ == "__main__":
    run()
