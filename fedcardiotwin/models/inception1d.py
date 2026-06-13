"""InceptionTime for 12-lead ECG, pure PyTorch.

Architecture follows the configuration that leads the public PTB-XL
benchmark (Strodthoff et al., IEEE JBHI 2021): bottleneck 32, kernel sizes
39/19/9, depth 6 with a residual shortcut every 3 blocks, concat-pool head.
"""
import torch
import torch.nn as nn


def _conv(in_ch, out_ch, ks):
    return nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2, bias=False)


class SEModule1d(nn.Module):
    """Squeeze-and-excitation channel attention — a consistent gain on
    12-lead ECG models at negligible parameter cost."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=-1))
        return x * w.unsqueeze(-1)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, nf=32, kss=(39, 19, 9), se=True):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, nf, 1, bias=False) if in_ch > 1 else None
        branch_in = nf if self.bottleneck is not None else in_ch
        self.convs = nn.ModuleList([_conv(branch_in, nf, ks) for ks in kss])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, nf, 1, bias=False))
        width = nf * (len(kss) + 1)
        self.bn = nn.BatchNorm1d(width)
        self.act = nn.ReLU(inplace=True)
        self.se = SEModule1d(width) if se else nn.Identity()

    def forward(self, x):
        z = self.bottleneck(x) if self.bottleneck is not None else x
        out = torch.cat([c(z) for c in self.convs] + [self.pool_branch(x)], dim=1)
        return self.se(self.act(self.bn(out)))


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, res, x):
        return self.act(x + self.bn(self.conv(res)))


class ConcatPool1d(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class Inception1d(nn.Module):
    def __init__(self, num_classes, in_ch=12, nf=32, depth=6, kss=(39, 19, 9),
                 head_hidden=128, dropout=0.5, se=True):
        super().__init__()
        width = nf * (len(kss) + 1)
        blocks, shortcuts = [], []
        ch = in_ch
        for d in range(depth):
            blocks.append(InceptionBlock(ch, nf=nf, kss=kss, se=se))
            if d % 3 == 2:
                shortcuts.append(Shortcut(in_ch if d == 2 else width, width))
            ch = width
        self.blocks = nn.ModuleList(blocks)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.pool = ConcatPool1d()
        self.feature_dim = 2 * width
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim), nn.Dropout(dropout / 2),
            nn.Linear(self.feature_dim, head_hidden), nn.ReLU(inplace=True),
            nn.BatchNorm1d(head_hidden), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes))

    def forward_features(self, x):
        res = x
        si = 0
        for d, block in enumerate(self.blocks):
            x = block(x)
            if d % 3 == 2:
                x = self.shortcuts[si](res, x)
                si += 1
                res = x
        return self.pool(x)

    def forward(self, x):
        return self.head(self.forward_features(x))
