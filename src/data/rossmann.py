from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from features.rossmann import add_rossmann_features


def load_rossmann_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    store = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.store)
    data = pd.merge(data, store, on="Store")
    data = data[data["Open"] != 0]
    data = add_rossmann_features(data)

    train = data[data["year"] < 2015]
    test = data[data["year"] == 2015]

    train_x = train.drop(columns=[*config.data.drop_features, config.data.target])
    train_y = np.log1p(train[config.data.target])
    test_x = test.drop(columns=[*config.data.drop_features, config.data.target])
    test_y = np.log1p(test[config.data.target])

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed
    )

    return X_train, X_valid, y_train, y_valid, test_x, test_y
