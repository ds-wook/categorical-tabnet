from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpClassificationTrainer(nn.Module):
    def __init__(self, num_feature: list[str], num_class: int):
        super(MlpClassificationTrainer, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, num_class)

        torch.nn.init.xavier_normal_(self.layer_1.weight)
        torch.nn.init.xavier_normal_(self.layer_2.weight)
        torch.nn.init.xavier_normal_(self.layer_3.weight)
        torch.nn.init.xavier_normal_(self.layer_out.weight)

        self.dropout_1 = nn.Dropout(p=0.3)
        self.dropout_2 = nn.Dropout(p=0.2)

        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.batch_norm_3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.layer_2(x)
        x = self.softmax(x)
        x = self.dropout_2(x)

        x = self.layer_out(x)

        return x
