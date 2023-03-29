import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def to_weight(y: pd.Series) -> np.ndarray:
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1.0 / (y[ind] ** 2)
    return w


def root_mean_squared_percentage_error(y: np.ndarray, yhat: np.ndarray) -> float:
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xgb(yhat: xgb.DMatrix, y: xgb.DMatrix) -> tuple[str, float]:
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


def evaluate_metrics(cfg: DictConfig, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Evaluate metrics
    """
    scores = PrettyTable()

    if cfg.models.task_type == "binary":
        loss = log_loss(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        f1 = f1_score(y_true, y_pred > 0.5, average="macro")
        auc = roc_auc_score(y_true, y_pred)
        scores.field_names = ["LogLoss", "Accuracy", "F-score", "ROC-AUC"]
        scores.add_row(
            [f"{loss:.4f}", f"{accuracy:.4f}", f"{f1:.4f}", f"{auc:.4f}"],
        )

    elif cfg.models.task_type == "multiclass":
        accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average="micro")
        auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
        scores.field_names = ["LogLoss", "Accuracy", "F-score", "ROC-AUC"]
        scores.add_row(
            [f"{loss:.4f}", f"{accuracy:.4f}", f"{f1:.4f}", f"{auc:.4f}"],
        )

    elif cfg.models.task_type == "regression":
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        rmspe = root_mean_squared_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        scores.field_names = ["MAE", "RMSE", "RMSPE", "R2"]
        scores.add_row(
            [f"{mae:.4f}", f"{rmse:.4f}", f"{rmspe:.4f}", f"{r2:.4f}"],
        )

    else:
        raise NotImplementedError

    logging.info(f"\n{scores.get_string()}")
