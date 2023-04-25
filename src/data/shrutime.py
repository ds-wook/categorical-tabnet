from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from features.engineering import change_numeric2category


def load_shrutime_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    train = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    test = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.test)

    train_x = train.drop(columns=config.data.target)
    train_y = train[config.data.target]
    train_x = change_numeric2category(train_x, "EstimatedSalary", q=config.data.q)
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed, stratify=train_y
    )

    X_test = test.drop(columns=config.data.target)
    X_test = change_numeric2category(X_test, "EstimatedSalary", q=config.data.q)
    y_test = test[config.data.target]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
