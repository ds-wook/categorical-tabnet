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

    # for col in cfg.data.cat_features:
    #     df_train[col] = df_train[col].astype(str)
    #     X_valid[col] = X_valid[col].astype(str)
    #     X_test[col] = X_test[col].astype(str)

    datamodule = TabularClassificationData.from_data_frame(
        categorical_fields=[*cfg.data.cat_features],
        numerical_fields=[*cfg.data.num_features],
        target_fields=cfg.data.target,
        train_data_frame=df_train,
        val_data_frame=df_valid,
        batch_size=16,
        predict_data_frame=X_test,
    )

    model = TabularClassifier.from_data(
        datamodule,
        backbone="tabtransformer",
        optimizer="adamax",
        learning_rate=0.02,
        lr_scheduler=("StepLR", {"step_size": 250}),
        out_ff_activation="LeakyReLU",
        num_attn_blocks=14,
        attn_dropout=0.2,
        ff_dropout=0.2,
    )

    trainer = flash.Trainer(
        max_epochs=50,
        gpus=torch.cuda.device_count(),
        logger=CSVLogger(save_dir="log/"),
        accumulate_grad_batches=10,
        gradient_clip_val=0.1,
    )
    trainer.fit(model, datamodule=datamodule)

    datamodule = TabularClassificationData.from_data_frame(
        predict_data_frame=X_test.fillna(0),
        parameters=datamodule.parameters,
        batch_size=8,
    )
    y_preds = trainer.predict(model, datamodule=datamodule, output="probabilities")
    y_preds = np.array(list(chain(*y_preds)))[:, 1]
    assert len(X_test) == len(y_preds)

    evaluate_metrics(cfg, y_test.to_numpy(), y_preds)


if __name__ == "__main__":
    _main()
