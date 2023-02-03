from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def load_covtype_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    data = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    train_x = data.drop(columns=config.data.target)
    train_y = data[config.data.target] - 1

    return train_x, train_y
