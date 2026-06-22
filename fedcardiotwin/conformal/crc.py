"""Federated conformal risk control for multilabel ECG prediction sets.

Per client, conformal risk control (Angelopoulos et al.) picks the smallest
threshold lambda such that the calibrated false-negative rate is <= alpha.
The federated variant averages client risk curves (weighted by calibration
size) on a shared lambda grid, yielding one global lambda with group-level
guarantees — the practical recipe from federated conformal prediction work.

Prediction set for a record = {classes with sigmoid prob >= 1 - lambda}.
Outputs per client: empirical FNR (coverage), mean set size.
"""
import numpy as np

LAMBDA_GRID = np.linspace(0.0, 1.0, 201)


def _fnr_at(probs, labels, lam):
    """Empirical false-negative rate of sets {c: p_c >= 1-lam} per record."""
    keep = probs >= (1.0 - lam)
    pos = labels.sum(axis=1)
    missed = (labels * (~keep)).sum(axis=1)
    mask = pos > 0
    if mask.sum() == 0:
        return 0.0
    return float((missed[mask] / pos[mask]).mean())


def crc_lambda(cal_probs, cal_labels, alpha, grid=LAMBDA_GRID):
    """Smallest lambda with upper-corrected risk <= alpha (B=1 bound)."""
    n = len(cal_probs)
    for lam in grid:
        risk = _fnr_at(cal_probs, cal_labels, lam)
        if (n / (n + 1)) * risk + 1.0 / (n + 1) <= alpha:
            return float(lam)
    return 1.0


def federated_lambda(client_cal, alpha, grid=LAMBDA_GRID):
    """client_cal: list of (probs, labels). Weighted-average risk curve."""
    sizes = np.array([len(p) for p, _ in client_cal], dtype=float)
    w = sizes / sizes.sum()
    n_tot = sizes.sum()
    for lam in grid:
        risk = sum(wi * _fnr_at(p, l, lam) for wi, (p, l) in zip(w, client_cal))
        if (n_tot / (n_tot + 1)) * risk + 1.0 / (n_tot + 1) <= alpha:
            return float(lam)
    return 1.0


def evaluate_sets(test_probs, test_labels, lam):
    keep = test_probs >= (1.0 - lam)
    return {
        "lambda": float(lam),
        "fnr": _fnr_at(test_probs, test_labels, lam),
        "mean_set_size": float(keep.sum(axis=1).mean()),
    }


def run_conformal(client_val, client_test, alpha=0.1, cal_alpha=None):
    """client_val/test: dicts name -> (probs, labels).

    Three modes per client:
      local       — lambda calibrated on the client's own data only.
      federated   — one shared lambda (group-level guarantee; can violate a
                    given hospital, e.g. high-prevalence sites).
      personalized— lambda_h = max(federated lambda, conservative local
                    lambda_h): the federated value acts as a shared floor,
                    raised to each hospital's own CRC threshold calibrated at
                    a tighter cal_alpha (<= alpha). The tighter target leaves
                    finite-sample headroom so even small-calibration sites
                    test under alpha, while larger sites benefit from the
                    federated prior.

    cal_alpha defaults to alpha (no conservatism). Set it below alpha (e.g.
    0.08 for alpha=0.1) to robustly hit FNR_h <= alpha at every hospital.
    """
    cal_alpha = alpha if cal_alpha is None else cal_alpha
    fed_lam = federated_lambda(list(client_val.values()), alpha)
    out = {}
    for name in client_val:
        pv, yv = client_val[name]
        pt, yt = client_test[name]
        local_lam = crc_lambda(pv, yv, alpha)            # reported "local"
        pers_local = crc_lambda(pv, yv, cal_alpha)       # conservative
        pers_lam = max(fed_lam, pers_local)
        out[name] = {
            "local": evaluate_sets(pt, yt, local_lam),
            "federated": evaluate_sets(pt, yt, fed_lam),
            "personalized": evaluate_sets(pt, yt, pers_lam),
        }
    return {"alpha": alpha, "cal_alpha": cal_alpha,
            "federated_lambda": fed_lam, "clients": out}
