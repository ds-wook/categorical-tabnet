from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_telco_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load telco dataset
    """
    df = pd.read_csv(Path(get_original_cwd()) / config.data.path / config.data.train)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    df["Churn"].replace(to_replace="Yes", value=1, inplace=True)
    df["Churn"].replace(to_replace="No", value=0, inplace=True)

    for col in config.data.cat_features:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    train_y = df[config.data.target]
    train_x = df.drop(columns=[*config.data.drop_features] + [config.data.target])

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=config.data.test_size, random_state=config.data.seed, stratify=train_y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=config.data.test_size, random_state=config.data.seed, stratify=y_train
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
