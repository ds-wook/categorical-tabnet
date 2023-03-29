from __future__ import annotations

import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from models.base import BaseModel


class TabNetTrainer(BaseModel):
    def __init__(self, config: DictConfig, cat_idxs: list[int] = [], cat_dims: list[int] = []):
        super().__init__(config)
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> TabNetClassifier | TabNetRegressor | TabNetMultiTaskClassifier:
        """method train"""
        if self.config.models.task_type == "binary":
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
                seed=self.config.models.params.seed,
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
                eval_name=[*self.config.models.eval_name],
                eval_metric=[*self.config.models.eval_metric],
                max_epochs=self.config.models.params.max_epochs,
                patience=self.config.models.params.patience,
                batch_size=self.config.models.params.batch_size,
                virtual_batch_size=self.config.models.params.virtual_batch_size,
                num_workers=self.config.models.params.num_workers,
                drop_last=False,
            )

            return model

        elif self.config.models.task_type == "multiclass":
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
                seed=self.config.models.params.seed,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                verbose=self.config.models.params.verbose,
            )

        elif self.config.models.task_type == "regression":
            model = TabNetRegressor(
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

        else:
            raise NotImplementedError

        model.fit(
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy().reshape(-1, 1),
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)),
                (X_valid.to_numpy(), y_valid.to_numpy().reshape(-1, 1)),
            ],
            eval_name=[*self.config.models.eval_name],
            eval_metric=[*self.config.models.eval_metric],
            max_epochs=self.config.models.params.max_epochs,
            patience=self.config.models.params.patience,
            batch_size=self.config.models.params.batch_size,
            virtual_batch_size=self.config.models.params.virtual_batch_size,
            num_workers=self.config.models.params.num_workers,
            drop_last=False,
        )

        return model
