from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import plotly.express as px

from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetClassifier

from data.dataset import load_dataset


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    _, _, X_test, _, _, _ = load_dataset(cfg)
    # define new model and load save parameters
    model = TabNetClassifier()
    model.load_model(Path(cfg.models.path) / cfg.models.working / f"{cfg.models.results}.zip")
    explain_matrix, masks = model.explain(X_test.to_numpy())
    normalized_explain_matrix = np.divide(explain_matrix, explain_matrix.sum(axis=1).reshape(-1, 1))

    fig = px.imshow(
        normalized_explain_matrix[:200, :],
        labels=dict(x="Features", y="Samples", color="Importance"),
        x=X_test.columns,
        title="Sample wise feature importance (reality is more complex than global feature importance)",
    )
    fig.show()


if __name__ == "__main__":
    _main()
