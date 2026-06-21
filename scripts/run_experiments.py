#!/usr/bin/env python3
"""Single entry point for every experiment in the paper.

Examples:
  python scripts/run_experiments.py --stage centralized --preset fast
  python scripts/run_experiments.py --stage federated --strategies fedavg fedbn
  python scripts/run_experiments.py --stage twin
  python scripts/run_experiments.py --stage conformal
Results land in results/<stage>_<strategy>_seed<k>.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from fedcardiotwin.configs import fast_preset, full_preset
from fedcardiotwin.data.dataset import PTBXLStreams, load_clients
from fedcardiotwin.data.labels import ScoredLabelSpace
from fedcardiotwin.fl.engine import run_federated
from fedcardiotwin.models import build_model
from fedcardiotwin.train.trainer import evaluate, predict, train_model
from fedcardiotwin.twin.loop import run_twin_evaluation
from fedcardiotwin.conformal.crc import run_conformal
from fedcardiotwin.utils import get_device, get_logger, save_json, seed_everything

log = get_logger()

STRATEGIES = ["local", "fedavg", "fedprox", "fedbn", "fedavgm", "ditto", "fedper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["centralized", "federated", "twin", "conformal", "loho"])
    ap.add_argument("--preset", default="full", choices=["fast", "full"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    cfg = fast_preset() if args.preset == "fast" else full_preset()
    device = get_device()
    log.info(f"device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.stage in ("centralized", "federated", "conformal"):
        space = ScoredLabelSpace(args.eval_repo)
        clients = load_clients(args.cache_dir, cfg.clients)
        log.info(f"clients: {[(c.name, len(c)) for c in clients]}")

    if args.stage == "centralized":
        # Pooled upper bound: concatenate all clients' train data.
        from torch.utils.data import ConcatDataset, DataLoader
        from fedcardiotwin.data.dataset import ECGDataset
        for seed in cfg.seeds:
            result_path = f"{args.results_dir}/centralized_seed{seed}.json"
            if os.path.exists(result_path):
                log.info(f"[centralized] seed{seed} already done, skipping")
                continue
            seed_everything(seed)
            tr = ConcatDataset([ECGDataset(c.X, c.Y, c.train_idx, True,
                                           augment=cfg.central.augment)
                                for c in clients])
            loader = DataLoader(tr, batch_size=cfg.central.batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
            model = build_model(cfg.model, space.num_classes)
            progress_path = f"{args.ckpt_dir}/central_seed{seed}_progress.pt"
            if os.path.exists(progress_path):
                ck = torch.load(progress_path, map_location="cpu")
                model.load_state_dict(ck["model"])
                log.info(f"[centralized] seed{seed} resumed from epoch {ck['epoch']}")
            train_model(model, loader, device, cfg.central.epochs, lr=cfg.central.lr,
                        mixup_alpha=cfg.central.mixup_alpha,
                        ema_decay=cfg.central.ema_decay, log_fn=log.info,
                        ckpt_path=progress_path, ckpt_every=5)
            res = {}
            for c in clients:
                m, _ = evaluate(model, c.loader("val", cfg.central.batch_size),
                                c.loader("test", cfg.central.batch_size), device)
                res[c.name] = m
            save_json(res, f"{args.results_dir}/centralized_seed{seed}.json")
            torch.save(model.state_dict(), f"{args.ckpt_dir}/central_seed{seed}.pt")
            if os.path.exists(progress_path):
                os.remove(progress_path)

    elif args.stage == "federated":
        for strategy in args.strategies:
            for seed in cfg.seeds:
                result_path = f"{args.results_dir}/fed_{strategy}_seed{seed}.json"
                if os.path.exists(result_path):
                    log.info(f"[federated] {strategy} seed{seed} already done, skipping")
                    continue
                seed_everything(seed)
                progress_path = f"{args.ckpt_dir}/fed_{strategy}_seed{seed}_progress.pt"
                try:
                    res, history, comm_mb, gmodel, deployed = run_federated(
                        clients, space.num_classes, cfg.fl, device,
                        model_name=cfg.model, strategy=strategy, seed=seed,
                        ckpt_path=progress_path)
                except Exception as e:
                    # One strategy/seed diverging or erroring must not abort the
                    # remaining runs in this process. Drop its (possibly poisoned)
                    # progress so a future invocation restarts it cleanly.
                    log.error(f"[federated] {strategy} seed{seed} FAILED: {e}")
                    if os.path.exists(progress_path):
                        os.remove(progress_path)
                    continue
                res["comm_mb_per_round"] = comm_mb
                res["history"] = history  # convergence / comm-budget curves
                save_json(res, f"{args.results_dir}/fed_{strategy}_seed{seed}.json")
                torch.save(gmodel.state_dict(),
                           f"{args.ckpt_dir}/fed_{strategy}_seed{seed}.pt")
                # deployed per-client models: what conformal must evaluate
                torch.save(deployed,
                           f"{args.ckpt_dir}/fed_{strategy}_seed{seed}_clients.pt")
                if os.path.exists(progress_path):
                    os.remove(progress_path)

    elif args.stage == "twin":
        streams_db = PTBXLStreams(args.cache_dir)
        from torch.utils.data import DataLoader
        from fedcardiotwin.data.dataset import ECGDataset
        for seed in cfg.seeds:
            result_path = f"{args.results_dir}/twin_seed{seed}.json"
            if os.path.exists(result_path):
                log.info(f"[twin] seed{seed} already done, skipping")
                continue
            seed_everything(seed)
            base_idx = streams_db.base_indices()
            loader = DataLoader(
                ECGDataset(streams_db.X, streams_db.Y, base_idx, True,
                           augment=cfg.central.augment),
                batch_size=cfg.central.batch_size, shuffle=True,
                num_workers=2, pin_memory=True, drop_last=True)
            base = build_model(cfg.model, streams_db.Y.shape[1])
            progress_path = f"{args.ckpt_dir}/twin_seed{seed}_progress.pt"
            if os.path.exists(progress_path):
                ck = torch.load(progress_path, map_location="cpu")
                base.load_state_dict(ck["model"])
                log.info(f"[twin] seed{seed} resumed from epoch {ck['epoch']}")
            train_model(base, loader, device, cfg.central.epochs,
                        lr=cfg.central.lr, mixup_alpha=cfg.central.mixup_alpha,
                        ema_decay=cfg.central.ema_decay, log_fn=log.info,
                        ckpt_path=progress_path, ckpt_every=5)
            streams = streams_db.eval_streams()
            res, arrays = run_twin_evaluation(base, streams, streams_db.X,
                                              streams_db.Y, device, cfg.twin)
            save_json(res, f"{args.results_dir}/twin_seed{seed}.json")
            np.savez(f"{args.results_dir}/twin_seed{seed}_arrays.npz", **arrays)
            log.info(f"twin seed{seed}: {res}")
            if os.path.exists(progress_path):
                os.remove(progress_path)

    elif args.stage == "conformal":
        # Evaluates the *deployed* per-client models (global shared weights
        # composed with private tensors) — required for fedbn correctness.
        deployed_path = None
        for strat in ("fedbn", "fedavg"):
            p = f"{args.ckpt_dir}/fed_{strat}_seed0_clients.pt"
            if os.path.exists(p):
                deployed_path = p
                break
        if deployed_path is None:
            raise FileNotFoundError("Run --stage federated first "
                                    "(no *_clients.pt checkpoint found).")
        deployed = torch.load(deployed_path, map_location="cpu")
        log.info(f"conformal: using {deployed_path}")
        cal, test = {}, {}
        for c in clients:
            model = build_model(cfg.model, space.num_classes)
            model.load_state_dict(deployed[c.name])
            pv, yv = predict(model, c.loader("val", cfg.fl.batch_size), device)
            pt, yt = predict(model, c.loader("test", cfg.fl.batch_size), device)
            cal[c.name], test[c.name] = (pv, yv), (pt, yt)
        res = run_conformal(cal, test, alpha=cfg.conformal.alpha)
        save_json(res, f"{args.results_dir}/conformal.json")
        log.info(f"conformal: federated lambda={res['federated_lambda']:.3f}")

    elif args.stage == "loho":
        # Leave-one-hospital-out: train fedavg on 5 clients, deploy on the
        # unseen 6th. Thresholds come from the training clients' pooled val
        # (the held-out hospital contributes nothing to model or tuning).
        from fedcardiotwin.train.metrics import (macro_auroc, macro_f1,
                                                 tune_thresholds)
        seed = cfg.seeds[0]
        loho = {}
        for held in list(clients):
            train_clients = [c for c in clients if c.name != held.name]
            seed_everything(seed)
            log.info(f"[loho] holding out {held.name}")
            _, _, _, gmodel, _ = run_federated(
                train_clients, space.num_classes, cfg.fl, device,
                model_name=cfg.model, strategy="fedavg", seed=seed)
            pv_all, yv_all = [], []
            for c in train_clients:
                pv, yv = predict(gmodel, c.loader("val", cfg.fl.batch_size), device)
                pv_all.append(pv)
                yv_all.append(yv)
            th = tune_thresholds(np.concatenate(yv_all), np.concatenate(pv_all))
            pt, yt = predict(gmodel, held.loader("test", cfg.fl.batch_size), device)
            loho[held.name] = {"auroc": macro_auroc(yt, pt),
                               "f1": macro_f1(yt, pt, th)}
            log.info(f"[loho] {held.name}: {loho[held.name]}")
        save_json(loho, f"{args.results_dir}/loho_seed{seed}.json")

    log.info("stage complete.")


if __name__ == "__main__":
    main()
