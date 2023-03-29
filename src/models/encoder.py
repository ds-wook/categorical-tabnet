from typing import List

import pandas as pd
from category_encoders import CatBoostEncoder
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from models.base import BaseEncoder


class CatBoostCategoricalEncoder(BaseEncoder):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.results = {}

    def _fit(self, train_transform: pd.Series, train_y: pd.Series) -> pd.DataFrame:
        # load encoder
        cb_encoder = CatBoostEncoder()
        # fit and transform
        cb_encoder.fit(train_transform, train_y)

        return cb_encoder


def categorize_tabnet_features(config: DictConfig, train: pd.DataFrame) -> tuple[List[int], List[int]]:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    categorical_columns = []
    categorical_dims = {}

    label_encoder = LabelEncoder()

    for cat_feature in tqdm(config.data.cat_features):
        train[cat_feature] = label_encoder.fit_transform(train[cat_feature].values)
        categorical_columns.append(cat_feature)
        categorical_dims[cat_feature] = len(label_encoder.classes_)

    features = [col for col in train.columns if col not in [config.data.target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return cat_idxs, cat_dims
