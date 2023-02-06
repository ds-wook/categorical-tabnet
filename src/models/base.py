from __future__ import annotations

import gc
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseEncoder(metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        self.config = config
        self.path = Path(get_original_cwd()) / config.data.encoder

    @abstractclassmethod
    def _fit(self, train_transform: pd.Series, train_y: pd.Series) -> NoReturn:
        raise NotImplementedError

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series) -> pd.DataFrame:
        """
        Fitting encoding
        Args:
            train_x: train_x
            train_y: train_y
        Returns:
            dataframe
        """
        for cat_feature in tqdm(self.config.data.cat_features):
            # convert to string
            train_x[cat_feature] = train_x[cat_feature].astype(str)

            cb_encoder = self._fit(train_x[cat_feature], train_y)
            train_x[cat_feature] = cb_encoder.transform(train_x[cat_feature])

            self.results[cat_feature] = cb_encoder

        return train_x

    def transform(self, test_x: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            test_x: dataframe
        Returns:
            dataframe
        """
        for cat_feature, encoder in tqdm(self.results.items()):
            test_x[cat_feature] = test_x[cat_feature].astype(str)
            test_x[cat_feature] = encoder.transform(test_x[cat_feature])

        return test_x


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        self.config = config
        self._num_fold_iter = 0
        self.oof_preds = None

    @abstractclassmethod
    def _fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> NoReturn:
        raise NotImplementedError

    def save_model(self, model_path: Path | str, model_name: str) -> BaseModel:
        """
        Save model
        Args:
            model_path: model path
            model_name: model name
        Return:
            Model Result
        """

        with open(model_path / model_name, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

        return

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> BaseModel:
        """
        Train model
        """
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def train_cross_validation(self, train_x: pd.DataFrame, train_y: pd.Series) -> BaseModel:
        models = dict()

        kfold = StratifiedKFold(
            n_splits=self.config.models.n_splits,
            random_state=self.config.data.seed,
            shuffle=True,
        )
        splits = kfold.split(train_x, train_y)
        oof_preds = np.zeros((train_x.shape[0], np.unique(train_y).shape[0]))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            self._num_fold_iter = fold

            x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            x_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            if self.config.log.experiment:
                wandb.init(
                    entity=self.config.log.entity,
                    project=self.config.log.project,
                    name=self.config.log.name + f"-fold-{fold}",
                )

                model = self._fit(x_train, y_train, x_valid, y_valid)

                oof_preds[valid_idx] = (
                    model.predict(x_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(x_valid))
                    if isinstance(model, xgb.Booster)
                    else model.predict_proba(np.array(x_valid))
                    if isinstance(model, TabNetMultiTaskClassifier)
                    else model.predict_proba(x_valid)
                )

                del x_train, y_train, x_valid, y_valid, model
                gc.collect()

                wandb.finish()

            else:
                model = self._fit(x_train, y_train, x_valid, y_valid)

                oof_preds[valid_idx] = (
                    model.predict(x_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(x_valid))
                    if isinstance(model, xgb.Booster)
                    else model.predict_proba(np.array(x_valid))
                    if isinstance(model, TabNetMultiTaskClassifier)
                    else model.predict_proba(x_valid)
                )

                del x_train, y_train, x_valid, y_valid, model
                gc.collect()

        self.oof_preds = oof_preds
        self.result = ModelResult(oof_preds=oof_preds, models=models)

        return self
