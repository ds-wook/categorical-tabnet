from __future__ import annotations

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig

from sklearn.preprocessing import StandardScaler

from data.dataset import load_rossmann_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from models.encoder import CatBoostCategoricalEncoder
from models.forest import RandomForestTrainer
from models.nn import MlpTrainer, TabNetRegressionTrainer
from utils.evaluate import evaluate_metrics


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    X_train, X_valid, y_train, y_valid, X_test, y_test = load_rossmann_dataset(cfg)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.18, random_state=2023)

    if cfg.models.working == "xgboost":
        xgb_trainer = XGBoostTrainer(config=cfg)
        xgb_model = xgb_trainer.train(X_train, y_train, X_valid, y_valid)
        xgb_preds = xgb_model.predict(xgb.DMatrix(X_test, enable_categorical=True))

        y_test = np.expm1(y_test)
        xgb_preds = np.expm1(xgb_preds)

        evaluate_metrics(y_test.to_numpy(), xgb_preds)

    elif cfg.models.working == "lightgbm":
        lgb_trainer = LightGBMTrainer(config=cfg)
        lgb_model = lgb_trainer.train(X_train, y_train, X_valid, y_valid)
        lgb_preds = lgb_model.predict(X_test)

        y_test = np.expm1(y_test)
        lgb_preds = np.expm1(lgb_preds)

        evaluate_metrics(y_test.to_numpy(), lgb_preds)

    elif cfg.models.working == "catboost":
        cb_trainer = CatBoostTrainer(config=cfg)
        cb_model = cb_trainer.train(X_train, y_train, X_valid, y_valid)
        cb_preds = cb_model.predict(X_test)

        y_test = np.expm1(y_test)
        cb_preds = np.expm1(cb_preds)

        evaluate_metrics(y_test.to_numpy(), cb_preds)

    elif cfg.models.working == "rf":
        rf_trainer = RandomForestTrainer(config=cfg)
        rf_model = rf_trainer.train(X_train, y_train, X_valid, y_valid)
        rf_preds = rf_model.predict(X_test)

        y_test = np.expm1(y_test)
        rf_preds = np.expm1(rf_preds)

        evaluate_metrics(y_test.to_numpy(), rf_preds)

    elif cfg.models.working == "tabnet":
        tabnet_trainer = TabNetRegressionTrainer(config=cfg)
        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)
        tabnet_preds = tabnet_model.predict(X_test.to_numpy()).flatten()

        y_test = np.expm1(y_test)
        tabnet_preds = np.expm1(tabnet_preds)

        evaluate_metrics(y_test.to_numpy(), tabnet_preds)

    elif cfg.models.working == "catabnet":
        # catboost encoder
        cb_encoder = CatBoostCategoricalEncoder(config=cfg)
        X_train = cb_encoder.fit(X_train, y_train)
        X_valid = cb_encoder.transform(X_valid)
        X_test = cb_encoder.transform(X_test)

        tabnet_trainer = TabNetRegressionTrainer(config=cfg)
        tabnet_model = tabnet_trainer.train(X_train, y_train, X_valid, y_valid)
        tabnet_preds = tabnet_model.predict(X_test.to_numpy()).flatten()

        y_test = np.expm1(y_test)
        tabnet_preds = np.expm1(tabnet_preds)

        evaluate_metrics(y_test.to_numpy(), tabnet_preds)

    elif cfg.models.working == "mlp":
        X_train = pd.get_dummies(X_train)
        X_valid = pd.get_dummies(X_valid)
        X_test = pd.get_dummies(X_test)

        scaler = StandardScaler()
        num_features = [col for col in X_train.columns if col in [*cfg.data.cat_features]]

        for num_feature in num_features:
            X_train[num_feature] = scaler.fit_transform(X_train[num_feature].to_numpy().reshape(-1, 1))
            X_valid[num_feature] = scaler.transform(X_valid[num_feature].to_numpy().reshape(-1, 1))
            X_test[num_feature] = scaler.transform(X_test[num_feature].to_numpy().reshape(-1, 1))

        mlp_trainer = MlpTrainer(config=cfg)
        mlp_model = mlp_trainer.train(X_train, y_train, X_valid, y_valid)
        mlp_preds = mlp_model.predict(X_test)
        y_test = np.expm1(y_test)
        mlp_preds = np.expm1(mlp_preds)
        evaluate_metrics(y_test.to_numpy(), mlp_preds)


if __name__ == "__main__":
    _main()
