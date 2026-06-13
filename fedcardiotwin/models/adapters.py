"""The per-patient twin: a tiny residual bottleneck adapter + personal head.

Only these parameters (a few thousand) are updated per patient and, in the
federated setting, only adapters ever travel — which is what makes the
communication-cost plot in the paper come out in our favor by construction.
"""
import copy

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)  # identity at init: twin starts as global
        nn.init.zeros_(self.up.bias)

    def forward(self, feat):
        return feat + self.up(self.act(self.down(feat)))


class PatientTwin(nn.Module):
    """Wraps a frozen global backbone; owns adapter + head copy."""

    def __init__(self, base_model, hidden=64):
        super().__init__()
        self.base = base_model
        self.adapter = BottleneckAdapter(base_model.feature_dim, hidden)
        self.head = copy.deepcopy(base_model.head)

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def forward(self, x):
        with torch.no_grad():
            feat = self.base.forward_features(x)
        return self.head(self.adapter(feat))
