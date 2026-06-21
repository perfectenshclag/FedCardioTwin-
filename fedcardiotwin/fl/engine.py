"""Serial federated simulation engine (single-GPU friendly).

Strategies: local-only, fedavg, fedprox, fedbn (hospital-adapter tier),
fedavgm (server momentum), ditto and fedper (personalized baselines).
A serial loop is deliberate: it is exactly reproducible, runs anywhere a
notebook runs, and matches how FL papers simulate cross-silo settings.

Outputs per run: final per-client metrics, a per-round validation-AUROC
history (for convergence / accuracy-vs-communication plots), the upload
cost per round in MB, and the *deployed* per-client state dicts (global
shared weights composed with each client's private tensors) so downstream
stages (conformal) evaluate exactly what a hospital would run.
"""
import copy
import os

import torch

from ..models import build_model
from ..train.trainer import evaluate, predict, train_model
from ..train.metrics import macro_auroc, summarize_clients
from ..utils import count_bytes, get_logger

log = get_logger()


def _bn_keys(model):
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            for suffix in ("weight", "bias", "running_mean", "running_var",
                           "num_batches_tracked"):
                keys.add(f"{name}.{suffix}")
    return keys


def _head_keys(model):
    return {k for k in model.state_dict() if k.startswith("head.")}


def _avg_states(states, weights, keys):
    out = {}
    total = sum(weights)
    for k in keys:
        acc = None
        for s, w in zip(states, weights):
            t = s[k].float() * (w / total)
            acc = t if acc is None else acc + t
        out[k] = acc.to(states[0][k].dtype)
    return out


def run_federated(clients, num_classes, cfg, device, model_name="inception1d",
                  strategy="fedavg", seed=0, ckpt_path=None, resume=True):
    """Returns (per-client metrics, history, comm_mb_per_round, global_model,
    deployed_states: {client_name: state_dict}).

    If ckpt_path is given, progress (global model, local/personal states,
    momentum, history, completed round) is saved after every round and
    reloaded on resume — so an interrupted run continues instead of
    restarting from round 0."""
    torch.manual_seed(seed)
    global_model = build_model(model_name, num_classes)
    all_keys = set(global_model.state_dict().keys())
    bn = _bn_keys(global_model)
    head = _head_keys(global_model)

    if strategy == "fedbn":
        shared_keys = all_keys - bn
    elif strategy == "fedper":
        shared_keys = all_keys - head - bn
    else:
        shared_keys = all_keys

    local_states = [copy.deepcopy(global_model.state_dict()) for _ in clients]
    personal = ([copy.deepcopy(global_model.state_dict()) for _ in clients]
                if strategy == "ditto" else None)
    momentum = None
    history = []
    comm_mb = count_bytes(global_model.state_dict(), shared_keys) / 1e6
    n_params = sum(p.numel() for p in global_model.parameters())

    train_loaders = [c.loader("train", cfg.batch_size) for c in clients]
    sizes = [len(c.train_idx) for c in clients]

    rounds = 1 if strategy == "local" else cfg.rounds
    local_epochs = cfg.local_only_epochs if strategy == "local" else cfg.local_epochs

    start_round = 0
    if ckpt_path and resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        global_model.load_state_dict(ck["global_state"])
        local_states = ck["local_states"]
        personal = ck["personal"]
        momentum = ck["momentum"]
        history = ck["history"]
        start_round = ck["round"]
        log.info(f"[{strategy}] resumed from round {start_round}/{rounds}")

    def deployed_state(ci):
        """What client ci actually runs in production for this strategy."""
        if strategy == "local":
            return copy.deepcopy(local_states[ci])
        if strategy == "ditto":
            return copy.deepcopy(personal[ci])
        st = copy.deepcopy(global_model.state_dict())
        if strategy in ("fedbn", "fedper"):
            for k in all_keys - shared_keys:
                st[k] = local_states[ci][k].clone()
        return st

    for rnd in range(start_round, rounds):
        # round-start global parameters: prox target for fedprox AND ditto
        g0 = [p.detach().clone() for p in global_model.parameters()]
        states = []
        for ci, client in enumerate(clients):
            model = build_model(model_name, num_classes)
            st = copy.deepcopy(global_model.state_dict())
            if strategy != "fedavg":
                for k in all_keys - shared_keys:
                    st[k] = local_states[ci][k]
            model.load_state_dict(st)

            train_model(model, train_loaders[ci], device, epochs=local_epochs,
                        lr=cfg.lr,
                        prox_mu=cfg.prox_mu if strategy == "fedprox" else 0.0,
                        global_params=g0 if strategy == "fedprox" else None)

            if strategy == "ditto":  # personal model, prox to round-start global
                pm = build_model(model_name, num_classes)
                pm.load_state_dict(personal[ci])
                train_model(pm, train_loaders[ci], device, epochs=local_epochs,
                            lr=cfg.lr, prox_mu=cfg.ditto_lambda, global_params=g0)
                personal[ci] = {k: v.cpu() for k, v in pm.state_dict().items()}

            local_states[ci] = {k: v.cpu() for k, v in model.state_dict().items()}
            states.append(local_states[ci])

        if strategy != "local":
            avg = _avg_states(states, sizes, shared_keys)
            gst = global_model.state_dict()
            if strategy == "fedavgm":
                if momentum is None:
                    momentum = {k: torch.zeros_like(v.float()) for k, v in avg.items()}
                for k in avg:
                    if not gst[k].dtype.is_floating_point:  # e.g. BN step counts
                        gst[k] = avg[k]
                        continue
                    delta = gst[k].float() - avg[k].float()
                    momentum[k] = cfg.server_momentum * momentum[k] + delta
                    # Server learning rate damps the momentum step. Without it
                    # the steady-state update is delta/(1-momentum) ~= 10x a
                    # FedAvg step, which diverges to NaN. server_lr=(1-momentum)
                    # makes the steady-state step equal a vanilla FedAvg step.
                    gst[k] = (gst[k].float()
                              - cfg.server_lr * momentum[k]).to(gst[k].dtype)
            else:
                for k in avg:
                    gst[k] = avg[k]
            global_model.load_state_dict(gst)

        # periodic validation AUROC -> convergence / comm-budget curves
        if cfg.eval_every and ((rnd + 1) % cfg.eval_every == 0 or rnd == rounds - 1):
            entry = {"round": rnd + 1, "cum_upload_mb": comm_mb * (rnd + 1)}
            for ci, client in enumerate(clients):
                m = build_model(model_name, num_classes)
                m.load_state_dict(deployed_state(ci))
                pv, yv = predict(m, client.loader("val", cfg.batch_size), device)
                entry[client.name] = macro_auroc(yv, pv)
            history.append(entry)
        log.info(f"[{strategy}] round {rnd + 1}/{rounds} done")

        if ckpt_path:
            torch.save({
                "round": rnd + 1,
                "global_state": global_model.state_dict(),
                "local_states": local_states,
                "personal": personal,
                "momentum": momentum,
                "history": history,
            }, ckpt_path)

    # Final evaluation: each client evaluates its deployed model.
    results, deployed = {}, {}
    for ci, client in enumerate(clients):
        model = build_model(model_name, num_classes)
        deployed[client.name] = deployed_state(ci)
        model.load_state_dict(deployed[client.name])
        m, _ = evaluate(model, client.loader("val", cfg.batch_size),
                        client.loader("test", cfg.batch_size), device)
        results[client.name] = m
        log.info(f"[{strategy}] {client.name}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    summary = summarize_clients(results)
    summary["model_params"] = n_params
    return summary, history, comm_mb, global_model, deployed
