from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from sklearn.neural_network import MLPClassifier, MLPRegressor

from models.base import BaseModel


class MlpClassificationTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> MLPClassifier:
        """method train"""
        if self.config.models.task_type == "binary" or self.config.models.task_type == "multiclass":
            model = MLPClassifier(verbose=True)

        elif self.config.models.task_type == "regression":
            model = MLPRegressor(verbose=True)

        else:
            raise NotImplementedError

        model.fit(X_train, y_train)

        return model
