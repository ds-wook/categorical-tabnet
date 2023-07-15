from __future__ import annotations

from itertools import chain
from pathlib import Path

import flash
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from flash.tabular import TabularClassificationData, TabularClassifier
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TabularDataset, load_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, RandomForestTrainer, XGBoostTrainer
from models.callback import EarlyStoppingCallback
from models.encoder import CatBoostCategoricalEncoder, categorize_tabnet_features
from models.layer import MlpClassificationTrainer
from models.transformer import TabNetTrainer
from utils.evaluate import acc_calc, evaluate_metrics
from utils.utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
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

    elif cfg.models.working == "mlp":
        # Fixed seed
        seed_everything(cfg.models.seed)
        # catboost encoder
        cb_encoder = CatBoostCategoricalEncoder(config=cfg)
        X_train = cb_encoder.fit(X_train, y_train)
        X_valid = cb_encoder.transform(X_valid)
        X_test = cb_encoder.transform(X_test)
        # eval
        accuracy_stat = {"train": [], "validation": []}
        loss_stat = {"train": [], "validation": []}
        early_stopping_callback = EarlyStoppingCallback(0.001, cfg.models.patience)

        # load dataset
        train_dataset = TabularDataset(
            torch.from_numpy(X_train.to_numpy()).float(), torch.from_numpy(y_train.to_numpy()).long()
        )
        valid_dataset = TabularDataset(
            torch.from_numpy(X_valid.to_numpy()).float(), torch.from_numpy(y_valid.to_numpy()).long()
        )

        # Data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.models.batch_size)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1)

        # model
        num_features = X_train.shape[1]
        num_classes = len(y_train.unique())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MlpClassificationTrainer(num_feature=num_features, num_class=num_classes)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.models.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
        criterion = nn.CrossEntropyLoss()

        for progress in tqdm(range(1, cfg.models.num_epochs + 1), leave=False):
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()

            # We loop over training dataset using batches (we use DataLoader to load data with batches)
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass ->>>>
                y_train_pred = model(X_train_batch)

                # Find Loss and backpropagation of gradients
                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = acc_calc(y_train_pred, y_train_batch)

                # backward <------
                train_loss.backward()

                # Update the parameters (weights and biases)
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            #  Then we validate our model - concept is the same
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in valid_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    y_val_pred = model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = acc_calc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

                # end of validation loop
                early_stopping_callback(val_epoch_loss / len(valid_loader))

                if early_stopping_callback.stop_training:
                    break

                loss_stat["train"].append(train_epoch_loss / len(train_loader))
                loss_stat["validation"].append(val_epoch_loss / len(valid_loader))
                accuracy_stat["train"].append(train_epoch_acc / len(train_loader))
                accuracy_stat["validation"].append(val_epoch_acc / len(valid_loader))

                # 2021.05.17
                # This is a part of NN optimization
                scheduler.step(val_epoch_acc / len(valid_loader))

        y_preds = model(torch.from_numpy(X_test.to_numpy()).float().to(device))
        y_preds = torch.nn.functional.softmax(y_preds, dim=1).cpu().detach().numpy()[:, 1]

    elif cfg.models.working == "tabtransformer":
        df_train = pd.concat([X_train, y_train], axis=1)
        df_valid = pd.concat([X_valid, y_valid], axis=1)

        for col in cfg.data.cat_features:
            df_train[col] = df_train[col].astype(str)
            X_valid[col] = X_valid[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        datamodule = TabularClassificationData.from_data_frame(
            categorical_fields=[*cfg.data.cat_features],
            numerical_fields=[col for col in X_test.columns if col not in cfg.data.cat_features],
            target_fields=cfg.data.target,
            train_data_frame=df_train,
            val_data_frame=df_valid,
            batch_size=cfg.models.params.batch_size,
            predict_data_frame=X_test,
        )

        model = TabularClassifier.from_data(
            datamodule,
            lr_scheduler=("StepLR", {"step_size": 250}),
            backbone=cfg.models.params.backbone,
            optimizer=cfg.models.params.optimizer,
            learning_rate=cfg.models.params.lr,
            out_ff_activation=cfg.models.params.out_ff_activation,
            num_attn_blocks=cfg.models.params.num_attn_blocks,
            attn_dropout=cfg.models.params.attn_dropout,
            ff_dropout=cfg.models.params.ff_dropout,
        )

        trainer = flash.Trainer(
            max_epochs=cfg.models.params.max_epochs,
            gpus=torch.cuda.device_count(),
            logger=CSVLogger(save_dir="log/"),
            accumulate_grad_batches=cfg.models.params.accumulate_grad_batches,
            gradient_clip_val=cfg.models.params.gradient_clip_val,
        )
        trainer.fit(model, datamodule=datamodule)

        datamodule = TabularClassificationData.from_data_frame(
            predict_data_frame=X_test.fillna(0),
            parameters=datamodule.parameters,
            batch_size=1,
        )
        y_preds = trainer.predict(model, datamodule=datamodule, output=cfg.models.output)
        y_preds = np.array(list(chain(*y_preds)))[:, 1]

    else:
        raise NotImplementedError

    evaluate_metrics(cfg, y_test.to_numpy(), y_preds)


if __name__ == "__main__":
    _main()
