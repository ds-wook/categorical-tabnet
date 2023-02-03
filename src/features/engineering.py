from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from category_encoders import CatBoostEncoder, OneHotEncoder
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def delete_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Delete features
    """
    df = df.drop(features, axis=1)
    return df


def categorize_train_features(config: DictConfig, train: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder
    label_encoder = LabelEncoder()

    for cat_feature in tqdm(config.data.cat_features):
        train[cat_feature] = label_encoder.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

    return train


def categorize_test_features(config: DictConfig, test: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        test: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder

    for cat_feature in tqdm(config.data.cat_features):
        le_encoder = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        test[cat_feature] = le_encoder.transform(test[cat_feature])

    return test


def catboost_encoder_multiclass(
    config: DictConfig, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = train_y.astype(str)
    enc = OneHotEncoder().fit(y)
    y_onehot = enc.transform(y)
    class_names = y_onehot.columns

    for cat_feature in tqdm(config.features.cat_features):
        train_x[cat_feature] = train_x[cat_feature].astype(str)
        test_x[cat_feature] = test_x[cat_feature].astype(str)

        for class_ in class_names:
            enc = CatBoostEncoder()
            enc.fit(train_x[cat_feature], y_onehot[class_])
            train_x[f"{cat_feature}_{class_}"] = enc.transform(train_x[cat_feature])
            test_x[f"{cat_feature}_{class_}"] = enc.transform(test_x[cat_feature])

    return train_x, test_x
