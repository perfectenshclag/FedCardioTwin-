"""1d ResNet (wang-style) baseline: 3 residual stages with kernels 8/5/3."""
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 8, padding=4, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch))
        self.short = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] != x.shape[-1]:  # even-kernel padding adds 1 sample
            out = out[..., :x.shape[-1]]
        return self.act(out + self.short(x))


class ResNet1d(nn.Module):
    def __init__(self, num_classes, in_ch=12, channels=(64, 128, 128), dropout=0.5):
        super().__init__()
        layers, ch = [], in_ch
        for c in channels:
            layers.append(BasicBlock1d(ch, c))
            ch = c
        self.body = nn.Sequential(*layers)
        self.feature_dim = 2 * ch
        from .inception1d import ConcatPool1d
        self.pool = ConcatPool1d()
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim), nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes))

    def forward_features(self, x):
        return self.pool(self.body(x))

    def forward(self, x):
        return self.head(self.forward_features(x))
