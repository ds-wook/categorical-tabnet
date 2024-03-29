import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.census import load_census_dataset
from data.covtype import load_covtype_dataset
from data.credit import load_credit_dataset
from data.psychometrics import load_psychometrics_dataset
from data.rossmann import load_rossmann_dataset
from data.shrutime import load_shrutime_dataset
from data.telco import load_telco_dataset


class TabularDataset(Dataset):
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def load_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load train dataset
    """
    match cfg.data.name:
        case "census":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_census_dataset(cfg)
        case "covtype":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_covtype_dataset(cfg)
        case "shrutime":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_shrutime_dataset(cfg)
        case "psychometrics":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_psychometrics_dataset(cfg)
        case "rossmann":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_rossmann_dataset(cfg)
        case "telco":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_telco_dataset(cfg)
        case "credit":
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_credit_dataset(cfg)
        case _:
            raise ValueError(f"Dataset {cfg.data.name} not supported")
        
    return X_train, X_valid, X_test, y_train, y_valid, y_test
