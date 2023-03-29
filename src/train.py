from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig

from data.dataset import load_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, RandomForestTrainer, XGBoostTrainer
from models.encoder import CatBoostCategoricalEncoder, categorize_tabnet_features
from models.transformer import TabNetTrainer
from utils.evaluate import evaluate_metrics


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset(cfg)
    cat_idxs, cat_dims = categorize_tabnet_features(cfg, pd.concat([X_train, X_valid, X_test]))

    if cfg.models.working == "xgboost":
        xgb_trainer = XGBoostTrainer(config=cfg)
        xgb_model = xgb_trainer.train(X_train, y_train, X_valid, y_valid)
        y_preds = xgb_model.predict(xgb.DMatrix(X_test, enable_categorical=True))

    elif cfg.models.working == "lightgbm":
        lgb_trainer = LightGBMTrainer(config=cfg)
        lgb_model = lgb_trainer.train(X_train, y_train, X_valid, y_valid)
        y_preds = lgb_model.predict(X_test)

    elif cfg.models.working == "catboost":
        cb_trainer = CatBoostTrainer(config=cfg)
        cb_model = cb_trainer.train(X_train, y_train, X_valid, y_valid)
        y_preds = (
            cb_model.predict_proba(X_test)
            if cfg.models.task_type == "multiclass"
            else cb_model.predict_proba(X_test)[:, 1]
        )

    elif cfg.models.working == "rf":
        rf_trainer = RandomForestTrainer(config=cfg)
        rf_model = rf_trainer.train(X_train, y_train, X_valid, y_valid)
        y_preds = (
            rf_model.predict_proba(X_test)
            if cfg.models.task_type == "multiclass"
            else rf_model.predict_proba(X_test)[:, 1]
        )

    elif cfg.models.working == "tabnet":
        tabnet_trainer = TabNetTrainer(config=cfg, cat_dims=cat_dims, cat_idxs=cat_idxs)

        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)
        tabnet_model.save_model(Path(cfg.models.path) / cfg.models.working / cfg.models.results)
        y_preds = (
            tabnet_model.predict_proba(X_test.to_numpy())[0]
            if cfg.models.task_type == "multiclass"
            else tabnet_model.predict_proba(X_test.to_numpy())[:, 1]
        )

    elif cfg.models.working == "catabnet":
        # catboost encoder
        cb_encoder = CatBoostCategoricalEncoder(config=cfg)
        X_train = cb_encoder.fit(X_train, y_train)
        X_valid = cb_encoder.transform(X_valid)
        X_test = cb_encoder.transform(X_test)

        tabnet_trainer = TabNetTrainer(config=cfg)
        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)

        tabnet_model.save_model(Path(cfg.models.path) / cfg.models.working / cfg.models.results)
        y_preds = (
            tabnet_model.predict_proba(X_test.to_numpy())[0]
            if cfg.models.task_type == "multiclass"
            else tabnet_model.predict_proba(X_test.to_numpy())[:, 1]
        )

    else:
        raise NotImplementedError

    evaluate_metrics(cfg, y_test.to_numpy(), y_preds)


if __name__ == "__main__":
    _main()
