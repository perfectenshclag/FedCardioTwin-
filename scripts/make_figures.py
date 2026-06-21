#!/usr/bin/env python3
"""Generate publication-ready figures from results/*.json.

Reads the JSON produced by run_experiments.py / ensemble_eval.py and writes
vector (PDF) + raster (PNG) figures to a figures/ directory:

  fig_main_auroc      per-client AUROC by strategy (grouped bars, mean+-std)
  fig_mean_worst      MEAN vs WORST-client AUROC by strategy (the non-IID story)
  fig_convergence     validation AUROC vs round (FedAvg vs FedPer vs Ditto)
  fig_comm            MEAN AUROC vs cumulative upload MB (accuracy/communication)
  fig_conformal       per-hospital FNR + mean set size, local vs federated
  fig_loho            leave-one-hospital-out AUROC on the unseen hospital

Usage:
  python scripts/make_figures.py --results-dir results --out-dir figures
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLIENTS = ["CPSC", "CPSC-Extra", "Georgia", "Chapman", "Ningbo", "PTB-XL"]
# Display order / labels for strategies
STRATS = ["local", "fedavg", "fedprox", "fedbn", "fedavgm", "ditto", "fedper"]
STRAT_LABEL = {"local": "Local", "fedavg": "FedAvg", "fedprox": "FedProx",
               "fedbn": "FedBN", "fedavgm": "FedAvgM", "ditto": "Ditto",
               "fedper": "FedPer"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def _save(fig, out_dir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load_fed(results_dir):
    """{strategy: {client: (mean_auroc, std_auroc)}} aggregated over seeds."""
    per = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(results_dir, "fed_*_seed*.json")):
        base = os.path.basename(path)
        strat = base[len("fed_"):base.rindex("_seed")]
        d = json.load(open(path))
        for c in CLIENTS:
            if c in d and isinstance(d[c], dict):
                per[strat][c].append(d[c]["auroc"])
    out = {}
    for strat, cd in per.items():
        out[strat] = {c: (float(np.mean(v)), float(np.std(v)))
                      for c, v in cd.items()}
    return out


def fig_main_auroc(fed, out_dir):
    strats = [s for s in STRATS if s in fed]
    x = np.arange(len(CLIENTS))
    w = 0.8 / len(strats)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(strats):
        means = [fed[s].get(c, (np.nan, 0))[0] for c in CLIENTS]
        stds = [fed[s].get(c, (np.nan, 0))[1] for c in CLIENTS]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=2,
               label=STRAT_LABEL[s])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(CLIENTS, rotation=15)
    ax.set_ylabel("Macro-AUROC")
    ax.set_ylim(0.65, 1.0)
    ax.set_title("Per-hospital macro-AUROC by FL strategy (mean ± std, 3 seeds)")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    _save(fig, out_dir, "fig_main_auroc")


def fig_mean_worst(fed, out_dir):
    strats = [s for s in STRATS if s in fed]
    means = [np.mean([fed[s][c][0] for c in CLIENTS if c in fed[s]]) for s in strats]
    worst = [np.min([fed[s][c][0] for c in CLIENTS if c in fed[s]]) for s in strats]
    x = np.arange(len(strats))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, means, 0.4, label="MEAN", color="#4C72B0")
    ax.bar(x + 0.2, worst, 0.4, label="WORST-client", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABEL[s] for s in strats], rotation=15)
    ax.set_ylabel("Macro-AUROC")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Mean vs worst-client AUROC (fairness under non-IID)")
    ax.legend()
    _save(fig, out_dir, "fig_mean_worst")


def _history_mean(path):
    d = json.load(open(path))
    hist = d.get("history") or []
    rounds = [h["round"] for h in hist]
    mean_auroc = [np.mean([h[c] for c in CLIENTS if c in h]) for h in hist]
    cum_mb = [h.get("cum_upload_mb", np.nan) for h in hist]
    return rounds, mean_auroc, cum_mb


def fig_convergence(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for s in ["fedavg", "fedprox", "fedbn", "fedavgm", "ditto", "fedper"]:
        p = os.path.join(results_dir, f"fed_{s}_seed0.json")
        if not os.path.exists(p):
            continue
        rounds, mean_auroc, _ = _history_mean(p)
        if len(rounds) >= 2:  # a single point is a partial/fast run, not a curve
            ax.plot(rounds, mean_auroc, marker="o", ms=3, label=STRAT_LABEL[s])
            plotted = True
    if not plotted:
        plt.close(fig)
        print("  (skip convergence: no history field)")
        return
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean validation macro-AUROC")
    ax.set_title("Convergence (seed 0)")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_dir, "fig_convergence")


def fig_comm(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for s in ["fedavg", "fedper", "ditto", "fedbn"]:
        p = os.path.join(results_dir, f"fed_{s}_seed0.json")
        if not os.path.exists(p):
            continue
        _, mean_auroc, cum_mb = _history_mean(p)
        if len(cum_mb) >= 2 and not np.isnan(cum_mb[0]):
            ax.plot(cum_mb, mean_auroc, marker="s", ms=3, label=STRAT_LABEL[s])
            plotted = True
    if not plotted:
        plt.close(fig)
        print("  (skip comm: no cum_upload_mb)")
        return
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Mean validation macro-AUROC")
    ax.set_title("Accuracy vs communication budget (seed 0)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_comm")


def fig_conformal(results_dir, out_dir):
    p = os.path.join(results_dir, "conformal.json")
    if not os.path.exists(p):
        print("  (skip conformal: no conformal.json)")
        return
    d = json.load(open(p))
    alpha = d.get("alpha", 0.1)
    cl = d["clients"]
    names = [c for c in CLIENTS if c in cl]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    has_pers = all("personalized" in cl[c] for c in names)
    modes = [("federated", "Federated", "#55A868")]
    if has_pers:
        modes.append(("personalized", "Personalized (λ_h)", "#8172B3"))
    nb = len(modes) + 1  # +1 for local
    w = 0.8 / nb
    off = (np.arange(nb) - (nb - 1) / 2) * w

    ax1.bar(x + off[0], [cl[c]["local"]["fnr"] for c in names], w,
            label="Local", color="#4C72B0")
    for i, (key, lab, col) in enumerate(modes):
        ax1.bar(x + off[i + 1], [cl[c][key]["fnr"] for c in names], w,
                label=lab, color=col)
    ax1.axhline(alpha, ls="--", color="k", lw=1, label=f"target α={alpha}")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("False-negative rate"); ax1.set_title("FNR coverage")
    ax1.legend(fontsize=8)

    ax2.bar(x + off[0], [cl[c]["local"]["mean_set_size"] for c in names], w,
            label="Local", color="#4C72B0")
    for i, (key, lab, col) in enumerate(modes):
        ax2.bar(x + off[i + 1], [cl[c][key]["mean_set_size"] for c in names], w,
                label=lab, color=col)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel("Mean prediction-set size"); ax2.set_title("Set size")
    ax2.legend(fontsize=8)
    fig.suptitle("Federated conformal risk control: per-hospital coverage")
    _save(fig, out_dir, "fig_conformal")


def fig_loho(results_dir, out_dir):
    p = os.path.join(results_dir, "loho_seed0.json")
    if not os.path.exists(p):
        print("  (skip loho: no loho_seed0.json)")
        return
    d = json.load(open(p))
    names = [c for c in CLIENTS if c in d]
    auroc = [d[c]["auroc"] for c in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, auroc, color="#8172B3")
    for b, a in zip(bars, auroc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Macro-AUROC on held-out hospital")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Leave-one-hospital-out generalization")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    _save(fig, out_dir, "fig_loho")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.results_dir} -> writing {args.out_dir}")
    fed = load_fed(args.results_dir)
    if fed:
        fig_main_auroc(fed, args.out_dir)
        fig_mean_worst(fed, args.out_dir)
    fig_convergence(args.results_dir, args.out_dir)
    fig_comm(args.results_dir, args.out_dir)
    fig_conformal(args.results_dir, args.out_dir)
    fig_loho(args.results_dir, args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
