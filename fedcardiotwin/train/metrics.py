"""Paper metrics: macro-AUROC (masking classes absent from a split),
threshold-tuned macro-F1, worst-client aggregation, forgetting (BWT)."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_auroc(y_true, y_prob):
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if 0 < col.sum() < len(col):  # class must appear and not saturate
            aucs.append(roc_auc_score(col, y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def tune_thresholds(y_true, y_prob, grid=None):
    """Per-class threshold maximizing F1 on validation data."""
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    th = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0:
            continue
        scores = [f1_score(col, y_prob[:, c] >= t, zero_division=0) for t in grid]
        th[c] = grid[int(np.argmax(scores))]
    return th


def macro_f1(y_true, y_prob, thresholds):
    return float(f1_score(y_true, y_prob >= thresholds[None, :],
                          average="macro", zero_division=0))


def summarize_clients(per_client: dict):
    """{'client': {'auroc':..}} -> adds mean and worst-client rows."""
    keys = next(iter(per_client.values())).keys()
    out = dict(per_client)
    out["MEAN"] = {k: float(np.nanmean([v[k] for v in per_client.values()])) for k in keys}
    out["WORST"] = {k: float(np.nanmin([v[k] for v in per_client.values()])) for k in keys}
    return out


def forgetting(first_pass: list, second_pass: list):
    """Backward transfer: mean drop on early records after later updates.
    Inputs are per-record correctness/score lists aligned by record."""
    deltas = [b - a for a, b in zip(first_pass, second_pass)]
    return float(np.mean(deltas)) if deltas else 0.0
