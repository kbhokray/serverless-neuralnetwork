import numpy as np
from pathlib import Path


def load_data():
    curr_dir = Path(__file__).resolve().parent
    relative_path = "../data/winequality-red.csv"

    raw_data = []
    with open(curr_dir / relative_path, "r") as file:
        raw_data = file.readlines()

    data = np.array([d.strip().split(";") for d in raw_data[1:]], dtype=np.float64)

    np.random.shuffle(data)

    split_index = int(0.8 * len(data))
    train_data, test_data = data[:split_index, :], data[split_index:, :]

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    return (X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    load_data()
