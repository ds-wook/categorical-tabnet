import logging

import numpy as np
import pandas as pd
import xgboost as xgb
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


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Evaluate metrics
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    rmspe = root_mean_squared_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    scores = PrettyTable()
    scores.field_names = ["MAE", "RMSE", "RMSPE", "R2"]
    scores.add_row(
        [f"{mae:.4f}", f"{rmse:.4f}", f"{rmspe:.4f}", f"{r2:.4f}"],
    )

    logging.info(f"\n{scores.get_string()}")


def evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Evaluate metrics
    """

    loss = log_loss(y_true, y_pred)
    accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))

    if np.unique(y_true).shape[0] == 2:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average="binary")
        auc = roc_auc_score(y_true, y_pred)
    else:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average="micro")
        auc = roc_auc_score(y_true, y_pred, multi_class="ovr")

    scores = PrettyTable()
    scores.field_names = ["LogLoss", "Accuracy", "F-score", "ROC-AUC"]
    scores.add_row(
        [f"{loss:.4f}", f"{accuracy:.4f}", f"{f1:.4f}", f"{auc:.4f}"],
    )

    logging.info(f"\n{scores.get_string()}")
