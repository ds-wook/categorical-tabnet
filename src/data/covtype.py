from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def load_covtype_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    data = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    train_x = data.drop(columns=config.data.target)
    train_y = data[config.data.target] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed, stratify=train_y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=config.data.test_size, random_state=config.data.seed, stratify=y_train
    )

    return X_train, X_valid, y_train, y_valid, X_test, y_test
