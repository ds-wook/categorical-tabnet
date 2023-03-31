from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.config.data.cat_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.config.data.cat_features)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            callbacks=[
                wandb_lgb.wandb_callback(),
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ]
            if self.config.log.experiment
            else [
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ],
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> CatBoostClassifier | CatBoostRegressor:
        train_set = Pool(X_train, y_train)
        valid_set = Pool(X_valid, y_valid)

        if self.config.models.task_type == "binary":
            model = CatBoostClassifier(
                random_state=self.config.models.seed,
                **self.config.models.params,
            )
        elif self.config.models.task_type == "multiclass":
            model = CatBoostClassifier(
                random_state=self.config.models.seed,
                loss_function="MultiClass",
                **self.config.models.params,
            )
        elif self.config.models.task_type == "regression":
            model = CatBoostRegressor(
                random_state=self.config.models.seed,
                **self.config.models.params,
            )
        else:
            raise NotImplementedError

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            callbacks=[wandb_cb.WandbCallback()]
            if self.config.models.params.task_type == "CPU" and self.config.log.experiment
            else None,
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dict(self.config.models.params),
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
            callbacks=[wandb_xgb.WandbCallback()] if self.config.log.experiment else None,
        )

        return model


class RandomForestTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series
    ) -> RandomForestClassifier | RandomForestRegressor:
        if self.config.models.task_type == "binary":
            model = RandomForestClassifier(
                random_state=self.config.models.seed,
                **self.config.models.params,
            )
        elif self.config.models.task_type == "multiclass":
            model = RandomForestClassifier(
                random_state=self.config.models.seed,
                **self.config.models.params,
            )
        elif self.config.models.task_type == "regression":
            model = RandomForestRegressor(
                random_state=self.config.models.seed,
                **self.config.models.params,
            )
        else:
            raise NotImplementedError

        model.fit(x_train, y_train)

        return model
