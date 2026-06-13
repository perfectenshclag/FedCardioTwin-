"""The digital-twin loop: predict -> compare -> alert -> update.

For each held-out multi-record patient, the twin (frozen global backbone +
per-patient adapter/head) forecasts the patient's label distribution before
each new ECG arrives. Deviation between forecast and the new observation is
the alert score; the adapter then updates on the new record with a replay
buffer to prevent forgetting.

Reported metrics:
  cold vs warm AUROC + personalization gain   (Table 3 main rows)
  per_step curve                              (gain per additional ECG)
  alert_auroc                                 (does the deviation score detect
                                               actual label changes between
                                               consecutive recordings?)
  backward_transfer                           (forgetting on early records)
Raw per-record arrays are returned for post-hoc analyses (e.g. conformal
coverage tracked across twin updates) without re-running.
"""
import numpy as np
import torch
import torch.nn as nn

from ..models.adapters import PatientTwin
from ..train.metrics import macro_auroc


def _js_divergence(p, q, eps=1e-7):
    """Mean Jensen-Shannon divergence across independent Bernoulli classes."""
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)

    def _kl_bern(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    m = 0.5 * (p + q)
    return float(np.mean(0.5 * _kl_bern(p, m) + 0.5 * _kl_bern(q, m)))


@torch.no_grad()
def _probs(model, x, device):
    model.eval()
    return torch.sigmoid(model(x.to(device)).float()).cpu().numpy()[0]


def _update_twin(twin, buffer, device, steps, lr):
    twin.train()
    # Per-patient updates must not re-estimate BN statistics from a handful
    # of records: keep BN layers in eval mode (global population stats).
    for m in twin.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    opt = torch.optim.AdamW(twin.trainable_parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    xs = torch.stack([b[0] for b in buffer]).to(device)
    ys = torch.stack([b[1] for b in buffer]).to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(twin(xs), ys)
        loss.backward()
        opt.step()


def _bce(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    y = y.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _twin_probs_on(twin, row, X, device):
    x = torch.from_numpy(X[row].astype(np.float32)).unsqueeze(0)
    return _probs(twin, x, device)


def run_twin_evaluation(base_model, streams, X, Y, device, cfg):
    """Returns (summary dict, raw arrays dict)."""
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    cold_probs, warm_probs, true_ys = [], [], []
    steps_idx, alert_scores, label_changed = [], [], []
    fgt_first, fgt_second = [], []

    for pid, rows in streams:
        twin = PatientTwin(base_model, hidden=cfg.adapter_hidden).to(device)
        buffer, seen = [], []
        forecast = None
        for t, row in enumerate(rows):
            x = torch.from_numpy(X[row].astype(np.float32)).unsqueeze(0)
            y = torch.from_numpy(Y[row].astype(np.float32))

            p_cold = _probs(base_model, x, device)
            p_warm = _probs(twin, x, device)
            if t > 0:  # records after the first are the evaluation targets
                cold_probs.append(p_cold)
                warm_probs.append(p_warm)
                true_ys.append(y.numpy())
                steps_idx.append(t)
                # alert: distance between the twin's forecast (state after
                # ingesting records 0..t-1) and the new observation
                alert_scores.append(_js_divergence(forecast, p_warm))
                label_changed.append(int((Y[row] != Y[seen[-1]]).any()))
                if t == 1:  # first-pass score on record 0, once per patient
                    fgt_first.append(-_bce(_twin_probs_on(twin, seen[0], X, device),
                                           Y[seen[0]]))

            buffer.append((x[0], y))
            seen.append(row)
            if len(buffer) > cfg.replay_size:
                buffer = buffer[-cfg.replay_size:]
            if cfg.update_steps > 0:
                _update_twin(twin, buffer, device, cfg.update_steps, cfg.update_lr)
            # twin state forecast = post-update expectation on latest record
            forecast = _twin_probs_on(twin, row, X, device)

        if len(seen) > 1:  # after the full stream: re-score the first record
            fgt_second.append(-_bce(_twin_probs_on(twin, seen[0], X, device),
                                    Y[seen[0]]))

    cold_probs = np.asarray(cold_probs)
    warm_probs = np.asarray(warm_probs)
    true_ys = np.asarray(true_ys)
    steps_idx = np.asarray(steps_idx)
    alert_scores = np.asarray(alert_scores)
    label_changed = np.asarray(label_changed)

    # per-step curve: personalization gain per additional ECG ingested
    per_step = {}
    for t in sorted(set(steps_idx.tolist())):
        m = steps_idx == t
        if m.sum() >= 10:  # need enough records for a stable AUROC
            per_step[int(t)] = {
                "n": int(m.sum()),
                "cold_auroc": macro_auroc(true_ys[m], cold_probs[m]),
                "warm_auroc": macro_auroc(true_ys[m], warm_probs[m]),
            }

    # alert quality: does deviation rank actual label changes higher?
    alert_auroc = float("nan")
    if 0 < label_changed.sum() < len(label_changed):
        from sklearn.metrics import roc_auc_score
        alert_auroc = float(roc_auc_score(label_changed, alert_scores))

    n = min(len(fgt_first), len(fgt_second))
    bwt = (float(np.mean(np.asarray(fgt_second[:n]) - np.asarray(fgt_first[:n])))
           if n else 0.0)

    summary = {
        "cold_auroc": macro_auroc(true_ys, cold_probs),
        "warm_auroc": macro_auroc(true_ys, warm_probs),
        "personalization_gain": macro_auroc(true_ys, warm_probs)
                                 - macro_auroc(true_ys, cold_probs),
        "per_step": per_step,
        "alert_auroc": alert_auroc,
        "mean_alert_score": float(alert_scores.mean()) if len(alert_scores) else 0.0,
        "backward_transfer": bwt,
        "n_eval_records": int(len(true_ys)),
        "n_patients": len(streams),
    }
    arrays = {"cold": cold_probs, "warm": warm_probs, "y": true_ys,
              "step": steps_idx, "alert": alert_scores, "changed": label_changed}
    return summary, arrays
