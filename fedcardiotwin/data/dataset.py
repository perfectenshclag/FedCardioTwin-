"""Datasets over the preprocessed caches.

Training recipe follows the PTB-XL benchmarking setup that tops the public
leaderboard: random 2.5 s crops during training, sliding-window aggregation
at evaluation time (handled in the trainer).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CROP_LEN = 250  # 2.5 s @ 100 Hz


class ECGDataset(Dataset):
    def __init__(self, X, Y, indices, train: bool, crop_len: int = CROP_LEN,
                 augment: bool = True):
        self.X, self.Y = X, Y
        self.indices = np.asarray(indices)
        self.train = train
        self.crop_len = crop_len
        self.augment = None
        if train and augment:
            from .augment import ECGAugment
            self.augment = ECGAugment(p=0.5)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        if self.train:
            start = np.random.randint(0, x.shape[-1] - self.crop_len + 1)
            x = x[:, start:start + self.crop_len]
            if self.augment is not None:
                x = self.augment(x)
        return torch.from_numpy(x.copy()), torch.from_numpy(y)


class ClientData:
    """One hospital client: memmapped cache + deterministic 70/10/20 split."""

    def __init__(self, cache_dir, name, seed=0):
        d = os.path.join(cache_dir, name)
        self.name = name
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(d, "Y.npy"))
        n = len(self.Y)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_tr, n_va = int(0.7 * n), int(0.1 * n)
        self.train_idx = perm[:n_tr]
        self.val_idx = perm[n_tr:n_tr + n_va]
        self.test_idx = perm[n_tr + n_va:]

    def loader(self, split, batch_size, num_workers=2):
        from torch.utils.data import DataLoader
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]
        train = split == "train"
        return DataLoader(ECGDataset(self.X, self.Y, idx, train=train),
                          batch_size=batch_size, shuffle=train,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=train and len(idx) > batch_size)

    def __len__(self):
        return len(self.Y)


def load_clients(cache_dir, names, seed=0):
    out = []
    for n in names:
        if os.path.exists(os.path.join(cache_dir, n, "X.npy")):
            out.append(ClientData(cache_dir, n, seed=seed))
    if not out:
        raise FileNotFoundError(f"No client caches found in {cache_dir}")
    return out


class PTBXLStreams:
    """Track B: chronological per-patient record streams for the twin loop.

    Patients in folds 1-8 train the base model; multi-record patients in
    folds 9-10 are the held-out longitudinal evaluation streams.
    """

    def __init__(self, cache_dir):
        d = os.path.join(cache_dir, "PTBXL_TRACKB")
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(d, "Y.npy"))
        self.meta = pd.read_csv(os.path.join(d, "meta.csv"))
        self.meta["row"] = np.arange(len(self.meta))

    def base_indices(self, folds=range(1, 9)):
        m = self.meta[self.meta.strat_fold.isin(list(folds))]
        return m.row.values

    def eval_streams(self, folds=(9, 10), min_records=2):
        m = self.meta[self.meta.strat_fold.isin(list(folds))]
        streams = []
        for pid, g in m.groupby("patient_id"):
            if len(g) >= min_records:
                g = g.sort_values("date")
                streams.append((int(pid), g.row.values))
        return streams
