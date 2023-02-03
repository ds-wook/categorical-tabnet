import pandas as pd
from category_encoders import CatBoostEncoder
from omegaconf import DictConfig

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
