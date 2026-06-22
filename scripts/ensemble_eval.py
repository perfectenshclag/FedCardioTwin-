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
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="single-model checkpoints to ensemble (e.g. the 3 "
                         "centralized seeds)")
    ap.add_argument("--fed-clients", nargs="+", default=None,
                    help="*_clients.pt deployed-model files to ensemble PER "
                         "CLIENT (e.g. the 3 FedPer seeds). Strongest headline.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model name per checkpoint (default: inception1d)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-name", default="ensemble",
                    help="results/<out-name>.json")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    cfg = full_preset()
    device = get_device()
    space = ScoredLabelSpace(args.eval_repo)
    clients = load_clients(args.cache_dir, cfg.clients)

    results = {}

    if args.fed_clients:
        # Ensemble the per-client deployed models across seeds (e.g. FedPer).
        states = [torch.load(p, map_location="cpu") for p in args.fed_clients]
        # The deployed models are an architecture (inception1d), NOT a strategy
        # name -- guard against passing e.g. --models fedper by mistake.
        VALID = {"inception1d", "resnet1d"}
        model_name = (args.models or [cfg.model])[0]
        if model_name not in VALID:
            log.info(f"--models '{model_name}' is not an architecture; "
                     f"using '{cfg.model}'")
            model_name = cfg.model
        for c in clients:
            pv_list, pt_list = [], []
            yv = yt = None
            for st in states:
                m = build_model(model_name, space.num_classes)
                m.load_state_dict(st[c.name])
                pv, yv = predict(m, c.loader("val", args.batch_size), device)
                pt, yt = predict(m, c.loader("test", args.batch_size), device)
                pv_list.append(pv); pt_list.append(pt)
            pv, pt = np.mean(pv_list, axis=0), np.mean(pt_list, axis=0)
            th = tune_thresholds(yv, pv)
            results[c.name] = {"auroc": macro_auroc(yt, pt),
                               "f1": macro_f1(yt, pt, th)}
            log.info(f"{c.name}: AUROC={results[c.name]['auroc']:.4f} "
                     f"F1={results[c.name]['f1']:.4f}")
        members = args.fed_clients
    else:
        assert args.ckpts, "pass --ckpts or --fed-clients"
        names = args.models or ["inception1d"] * len(args.ckpts)
        assert len(names) == len(args.ckpts)
        members = []
        for ckpt, name in zip(args.ckpts, names):
            m = build_model(name, space.num_classes)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            members.append(m)
            log.info(f"loaded {ckpt} ({name})")
        for c in clients:
            pv_list, pt_list = [], []
            yv = yt = None
            for m in members:
                pv, yv = predict(m, c.loader("val", args.batch_size), device)
                pt, yt = predict(m, c.loader("test", args.batch_size), device)
                pv_list.append(pv); pt_list.append(pt)
            pv, pt = np.mean(pv_list, axis=0), np.mean(pt_list, axis=0)
            th = tune_thresholds(yv, pv)
            results[c.name] = {"auroc": macro_auroc(yt, pt),
                               "f1": macro_f1(yt, pt, th)}
            log.info(f"{c.name}: AUROC={results[c.name]['auroc']:.4f} "
                     f"F1={results[c.name]['f1']:.4f}")
        members = list(zip(args.ckpts, names))

    out = summarize_clients(results)
    out["members"] = members
    save_json(out, f"{args.results_dir}/{args.out_name}.json")
    log.info(f"{args.out_name} MEAN AUROC={out['MEAN']['auroc']:.4f} "
             f"WORST={out['WORST']['auroc']:.4f}")


if __name__ == "__main__":
    main()
