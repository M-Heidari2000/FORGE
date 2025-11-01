import torch
import torch.nn as nn


class MLPWithValueHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(256, 1)
        self.policy_head = nn.Linear(256, out_features)

    def forward(self, x):
        hidden = self.backbone(x)
        value = self.value_head(hidden)
        logits = self.policy_head(x)

        return logits, value