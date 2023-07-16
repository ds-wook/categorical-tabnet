from __future__ import annotations

from itertools import chain

import flash
import hydra
import numpy as np
import pandas as pd
import torch
from flash.tabular import TabularClassificationData, TabularClassifier
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger

from data.dataset import load_dataset
from models.encoder import CatBoostCategoricalEncoder
from utils.evaluate import evaluate_metrics


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset(cfg)
    cb_encoder = CatBoostCategoricalEncoder(config=cfg)
    X_train = cb_encoder.fit(X_train, y_train)
    X_valid = cb_encoder.transform(X_valid)
    X_test = cb_encoder.transform(X_test)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_valid = pd.concat([X_valid, y_valid], axis=1)

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

    y_preds = trainer.predict(model, datamodule=datamodule, output="probabilities")
    y_preds = np.array(list(chain(*y_preds)))[:, 1]
    assert len(X_test) == len(y_preds)

    evaluate_metrics(cfg, y_test.to_numpy(), y_preds)


if __name__ == "__main__":
    _main()
