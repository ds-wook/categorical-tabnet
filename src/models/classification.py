from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import torch
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from models.base import BaseModel


class TabNetClassificationTrainer(BaseModel):
    def __init__(self, config: DictConfig, cat_idxs: list[int] = [], cat_dims: list[int] = []):
        super().__init__(config)
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> TabNetMultiTaskClassifier:
        """method train"""
        if self.config.models.multi_task:
            model = TabNetMultiTaskClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=self.config.models.params.lr),
                scheduler_params={
                    "step_size": self.config.models.params.step_size,
                    "gamma": self.config.models.params.gamma,
                },
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type=self.config.models.params.mask_type,
                n_steps=self.config.models.params.n_steps,
                n_d=self.config.models.params.n_d,
                n_a=self.config.models.params.n_a,
                lambda_sparse=self.config.models.params.lambda_sparse,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                verbose=self.config.models.params.verbose,
            )

            model.fit(
                X_train=X_train.to_numpy(),
                y_train=y_train.to_numpy().reshape(-1, 1),
                eval_set=[
                    (X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)),
                    (X_valid.to_numpy(), y_valid.to_numpy().reshape(-1, 1)),
                ],
                eval_name=["train", "val"],
                eval_metric=["logloss"],
                max_epochs=self.config.models.params.max_epochs,
                patience=self.config.models.params.patience,
                batch_size=self.config.models.params.batch_size,
                virtual_batch_size=self.config.models.params.virtual_batch_size,
                num_workers=self.config.models.params.num_workers,
                drop_last=False,
            )
        else:
            model = TabNetClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=self.config.models.params.lr),
                scheduler_params={
                    "step_size": self.config.models.params.step_size,
                    "gamma": self.config.models.params.gamma,
                },
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type=self.config.models.params.mask_type,
                n_steps=self.config.models.params.n_steps,
                n_d=self.config.models.params.n_d,
                n_a=self.config.models.params.n_a,
                lambda_sparse=self.config.models.params.lambda_sparse,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                verbose=self.config.models.params.verbose,
            )

            model.fit(
                X_train=X_train.to_numpy(),
                y_train=y_train.to_numpy(),
                eval_set=[
                    (X_train.to_numpy(), y_train.to_numpy()),
                    (X_valid.to_numpy(), y_valid.to_numpy()),
                ],
                eval_name=["train", "val"],
                eval_metric=["logloss"],
                max_epochs=self.config.models.params.max_epochs,
                patience=self.config.models.params.patience,
                batch_size=self.config.models.params.batch_size,
                virtual_batch_size=self.config.models.params.virtual_batch_size,
                num_workers=self.config.models.params.num_workers,
                drop_last=False,
            )
        return model


class LightGBMCalssificationTrainer(BaseModel):
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


class CatBoostCalssificationTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> CatBoostClassifier:
        train_set = Pool(X_train, y_train)
        valid_set = Pool(X_valid, y_valid)

        model = CatBoostClassifier(
            random_state=self.config.models.seed,
            task_type=self.config.models.task_type,
            **self.config.models.params,
        )
        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            callbacks=[wandb_cb.WandbCallback()]
            if self.config.models.task_type == "CPU" and self.config.log.experiment
            else None,
        )

        return model


class XGBoostCalssificationTrainer(BaseModel):
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


class MlpClassificationTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> MLPClassifier:
        """method train"""
        model = MLPClassifier(verbose=True)
        model.fit(X_train, y_train)

        return


class RandomForestClassificationTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series
    ) -> RandomForestClassifier:
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        return model
