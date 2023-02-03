from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features.census import add_census_features


def load_census_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    train = pd.read_csv(Path(config.data.path) / config.data.train)
    train[config.data.target] = train[config.data.target].map({"<=50K": 0, ">50K": 1})
    train = add_census_features(train)

    for col in config.data.cat_features:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])

    train_x = train.drop(columns=config.data.target)
    train_y = train[config.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed, stratify=train_y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=config.data.test_size, random_state=config.data.seed, stratify=y_train
    )

    return X_train, X_valid, y_train, y_valid, X_test, y_test
