from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_credit_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    """
    df = pd.read_csv(Path(cfg.data.path) / cfg.data.train)

    for col in cfg.data.cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    attrition_flag_map = {"Existing Customer": 0, "Attrited Customer": 1}

    df["Attrition_Flag"] = df["Attrition_Flag"].map(attrition_flag_map)

    train_x = df.drop(columns=[*cfg.data.drop_features, cfg.data.target])
    train_y = df[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=cfg.data.test_size, random_state=cfg.data.seed, stratify=train_y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=cfg.data.test_size, random_state=cfg.data.seed, stratify=y_train
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
