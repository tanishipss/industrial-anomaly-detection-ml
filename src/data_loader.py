import os
import pandas as pd

def load_data(data_path="data"):
    train_path = os.path.join(data_path, "train.parquet")
    test_path  = os.path.join(data_path, "test.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Dataset not found. Place train.parquet and test.parquet inside the data/ directory."
        )

    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)

    train["target"] = train["target"].astype(int)

    return train, test
