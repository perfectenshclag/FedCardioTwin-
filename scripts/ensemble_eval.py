#!/usr/bin/env python3
"""Ensemble evaluation: average sigmoid probabilities of several checkpoints
(multiple seeds and/or architectures) per client. Typically the strongest
single number in the paper.

  python scripts/ensemble_eval.py --ckpts checkpoints/central_seed0.pt \\
      checkpoints/central_seed1.pt checkpoints/central_seed2.pt \\
      --models inception1d inception1d inception1d
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from fedcardiotwin.configs import full_preset
from fedcardiotwin.data.dataset import load_clients
from fedcardiotwin.data.labels import ScoredLabelSpace
from fedcardiotwin.models import build_model
from fedcardiotwin.train.metrics import (macro_auroc, macro_f1,
                                         summarize_clients, tune_thresholds)
from fedcardiotwin.train.trainer import predict
from fedcardiotwin.utils import get_device, get_logger, save_json

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="model name per checkpoint (default: inception1d)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    names = args.models or ["inception1d"] * len(args.ckpts)
    assert len(names) == len(args.ckpts)

    cfg = full_preset()
    device = get_device()
    space = ScoredLabelSpace(args.eval_repo)
    clients = load_clients(args.cache_dir, cfg.clients)

    members = []
    for ckpt, name in zip(args.ckpts, names):
        m = build_model(name, space.num_classes)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        members.append(m)
        log.info(f"loaded {ckpt} ({name})")

    results = {}
    for c in clients:
        pv_list, pt_list = [], []
        yv = yt = None
        for m in members:
            pv, yv = predict(m, c.loader("val", args.batch_size), device)
            pt, yt = predict(m, c.loader("test", args.batch_size), device)
            pv_list.append(pv)
            pt_list.append(pt)
        pv, pt = np.mean(pv_list, axis=0), np.mean(pt_list, axis=0)
        th = tune_thresholds(yv, pv)
        results[c.name] = {"auroc": macro_auroc(yt, pt),
                           "f1": macro_f1(yt, pt, th)}
        log.info(f"{c.name}: AUROC={results[c.name]['auroc']:.4f} "
                 f"F1={results[c.name]['f1']:.4f}")

    out = summarize_clients(results)
    out["members"] = list(zip(args.ckpts, names))
    save_json(out, f"{args.results_dir}/ensemble.json")
    log.info(f"ensemble MEAN AUROC={out['MEAN']['auroc']:.4f}")


if __name__ == "__main__":
    main()
