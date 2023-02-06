from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from features.engineering import categorize_train_features
from features.psychometrics import add_psychometrics_features


def load_psychometrics_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    train = pd.read_csv(Path(config.data.path) / config.data.train)
    train = add_psychometrics_features(train)
    train_x = train.drop(columns=[*config.data.drop_features, config.data.target])
    train_y = 2 - train[config.data.target]
    train_x = categorize_train_features(config, train_x)

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed, stratify=train_y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=config.data.test_size, random_state=config.data.seed, stratify=y_train
    )

    return X_train, X_valid, y_train, y_valid, X_test, y_test
