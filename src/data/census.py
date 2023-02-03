from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from features.census import add_census_features


def load_census_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    train = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    train[config.data.target] = train[config.data.target].map({"<=50K": 0, ">50K": 1})
    train = add_census_features(train)

    for col in config.data.cat_features:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])

    train_x = train.drop(columns=config.data.target)
    train_y = train[config.data.target]

    return train_x, train_y
