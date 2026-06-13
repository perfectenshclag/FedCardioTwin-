"""Experiment configuration. Two presets:
FAST  — pipeline sanity on a subset (minutes on a T4).
FULL  — the paper runs (overnight-class on a T4).
"""
from dataclasses import dataclass, field


@dataclass
class FLConfig:
    rounds: int = 60
    local_epochs: int = 1
    local_only_epochs: int = 20      # for the local-only baseline
    batch_size: int = 128
    lr: float = 1e-3
    prox_mu: float = 0.01            # fedprox
    ditto_lambda: float = 0.1        # ditto prox strength
    server_momentum: float = 0.9     # fedavgm
    eval_every: int = 10             # rounds between val-AUROC history points
                                     # (0 disables; last round always logged)


@dataclass
class CentralConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    mixup_alpha: float = 0.2     # mixup on signals+labels (multilabel-safe)
    ema_decay: float = 0.999     # EMA weights are the evaluated weights
    augment: bool = True         # ECG augmentation pipeline (train only)


@dataclass
class TwinConfig:
    adapter_hidden: int = 64
    replay_size: int = 8
    update_steps: int = 5
    update_lr: float = 1e-3


@dataclass
class ConformalConfig:
    alpha: float = 0.1


@dataclass
class ExperimentConfig:
    clients: tuple = ("CPSC", "CPSC-Extra", "Georgia", "Chapman", "Ningbo", "PTB-XL")
    model: str = "inception1d"
    seeds: tuple = (0, 1, 2)
    fl: FLConfig = field(default_factory=FLConfig)
    central: CentralConfig = field(default_factory=CentralConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)


def fast_preset():
    cfg = ExperimentConfig(seeds=(0,))
    cfg.fl.rounds = 3
    cfg.fl.local_only_epochs = 2
    cfg.central.epochs = 2
    return cfg


def full_preset():
    return ExperimentConfig()
