from __future__ import annotations

from pathlib import Path

import hydra
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from data.census import load_census_dataset
from data.covtype import load_covtype_dataset
from models.encoder import CatBoostCategoricalEncoder
from models.regression import (
    CatBoostClassificationTrainer,
    LightGBMClassificationTrainer,
    MlpClassificationTrainer,
    RandomForestClassificationTrainer,
    TabNetClassificationTrainer,
    XGBoostClassificationTrainer,
)
from utils.evaluate import evaluate_metrics


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    X_train, X_valid, y_train, y_valid, X_test, y_test = (
        load_census_dataset(cfg) if cfg.data.name == "census" else load_covtype_dataset(cfg)
    )

    if cfg.models.working == "xgboost":
        xgb_trainer = XGBoostClassificationTrainer(config=cfg)
        xgb_model = xgb_trainer.train(X_train, y_train, X_valid, y_valid)
        xgb_preds = xgb_model.predict(xgb.DMatrix(X_test, enable_categorical=True))
        evaluate_metrics(y_test.to_numpy(), xgb_preds)

    elif cfg.models.working == "lightgbm":
        lgb_trainer = LightGBMClassificationTrainer(config=cfg)
        lgb_model = lgb_trainer.train(X_train, y_train, X_valid, y_valid)
        lgb_preds = lgb_model.predict(X_test)
        evaluate_metrics(y_test.to_numpy(), lgb_preds)

    elif cfg.models.working == "catboost":
        cb_trainer = CatBoostClassificationTrainer(config=cfg)
        cb_model = cb_trainer.train(X_train, y_train, X_valid, y_valid)
        cb_preds = cb_model.predict_proba(X_test) if cfg.models.multi_task else cb_model.predict_proba(X_test)[:, 1]
        evaluate_metrics(y_test.to_numpy(), cb_preds)

    elif cfg.models.working == "rf":
        rf_trainer = RandomForestClassificationTrainer(config=cfg)
        rf_model = rf_trainer.train(X_train, y_train, X_valid, y_valid)
        rf_preds = rf_model.predict_proba(X_test) if cfg.models.multi_task else rf_model.predict_proba(X_test)[:, 1]
        evaluate_metrics(y_test.to_numpy(), rf_preds)

    elif cfg.models.working == "tabnet":
        tabnet_trainer = TabNetClassificationTrainer(config=cfg)
        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)
        tabnet_model.save_model(Path(cfg.models.path) / cfg.models.working / cfg.models.results)
        tabnet_preds = (
            tabnet_model.predict_proba(X_test.to_numpy())[0]
            if cfg.models.multi_task
            else tabnet_model.predict_proba(X_test.to_numpy())[:, 1]
        )
        evaluate_metrics(y_test.to_numpy(), tabnet_preds)

    elif cfg.models.working == "catabnet":
        # catboost encoder
        cb_encoder = CatBoostCategoricalEncoder(config=cfg)
        X_train = cb_encoder.fit(X_train, y_train)
        X_valid = cb_encoder.transform(X_valid)
        X_test = cb_encoder.transform(X_test)

        tabnet_trainer = TabNetClassificationTrainer(config=cfg)
        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)

        tabnet_model.save_model(Path(cfg.models.path) / cfg.models.working / cfg.models.results)
        tabnet_preds = (
            tabnet_model.predict_proba(X_test.to_numpy())[0]
            if cfg.models.multi_task
            else tabnet_model.predict_proba(X_test.to_numpy())[:, 1]
        )
        evaluate_metrics(y_test.to_numpy(), tabnet_preds)

    elif cfg.models.working == "mlp":
        scaler = StandardScaler()
        num_features = [col for col in X_train.columns if col in [*cfg.data.cat_features]]

        for num_feature in num_features:
            X_train[num_feature] = scaler.fit_transform(X_train[num_feature].to_numpy().reshape(-1, 1))
            X_valid[num_feature] = scaler.transform(X_valid[num_feature].to_numpy().reshape(-1, 1))
            X_test[num_feature] = scaler.transform(X_test[num_feature].to_numpy().reshape(-1, 1))

        mlp_trainer = MlpClassificationTrainer(config=cfg)
        mlp_model = mlp_trainer.train(X_train, y_train, X_valid, y_valid)
        mlp_preds = mlp_model.predict_proba(X_test) if cfg.models.multi_task else mlp_model.predict_proba(X_test)[:, 1]
        evaluate_metrics(y_test.to_numpy(), mlp_preds)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
