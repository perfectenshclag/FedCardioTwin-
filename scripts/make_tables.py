#!/usr/bin/env python3
"""Aggregate results/*.json into the paper's tables: mean ± std over seeds
per client (+ MEAN/WORST), communication cost, and Wilcoxon signed-rank
significance of every federated strategy against the FedAvg baseline
(paired over client × seed AUROCs)."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SKIP_KEYS = {"comm_mb_per_round", "history", "model_params", "members"}


def client_rows(result):
    return [k for k, v in result.items()
            if k not in SKIP_KEYS and isinstance(v, dict) and "auroc" in v]


def main(results_dir="results"):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        m = re.match(r"(.+)_seed(\d+)$", name)
        label = m.group(1) if m else name
        with open(path) as f:
            runs[label].append(json.load(f))

    paired_aurocs = {}  # label -> {(client, seed_idx): auroc}
    for label, results in runs.items():
        if not client_rows(results[0]):
            print(f"\n=== {label} ===")
            print(json.dumps(results[0], indent=2))
            continue
        rows = client_rows(results[0])
        print(f"\n=== {label} (n_seeds={len(results)}) ===")
        print(f"{'client':<12} {'AUROC':<18} {'F1':<18}")
        for row in rows:
            au = [r[row]["auroc"] for r in results if row in r]
            f1 = [r[row]["f1"] for r in results if row in r]
            print(f"{row:<12} {np.mean(au):.4f} ± {np.std(au):.4f}   "
                  f"{np.mean(f1):.4f} ± {np.std(f1):.4f}")
        if "comm_mb_per_round" in results[0]:
            print(f"comm: {results[0]['comm_mb_per_round']:.2f} MB/round (upload)")
        paired_aurocs[label] = {
            (row, si): r[row]["auroc"]
            for si, r in enumerate(results) for row in rows
            if row not in ("MEAN", "WORST")}

    # significance vs fedavg, paired over (client, seed)
    base = paired_aurocs.get("fed_fedavg")
    if base:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            print("\n(scipy unavailable — skipping significance tests)")
            return
        print("\n=== Wilcoxon signed-rank vs FedAvg (paired client x seed AUROC) ===")
        for label, scores in sorted(paired_aurocs.items()):
            if label == "fed_fedavg" or not label.startswith("fed_"):
                continue
            keys = sorted(set(scores) & set(base))
            a = np.array([scores[k] for k in keys])
            b = np.array([base[k] for k in keys])
            if len(keys) < 6 or np.allclose(a, b):
                print(f"{label:<16} n={len(keys)} (insufficient/identical pairs)")
                continue
            stat, p = wilcoxon(a, b)
            direction = "better" if a.mean() > b.mean() else "worse"
            print(f"{label:<16} n={len(keys)} mean diff={a.mean() - b.mean():+.4f} "
                  f"({direction})  p={p:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
