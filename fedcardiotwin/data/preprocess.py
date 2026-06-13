"""One-time preprocessing: raw WFDB records -> cached float16 memmaps.

All sources are harmonized to 12-lead, 100 Hz, 10 s (12 x 1000), bandpass
filtered 0.5-45 Hz and per-lead z-normalized. Caching means training never
touches WFDB again (critical for throughput on Kaggle/Colab disks).
"""
import ast
import glob
import os
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from .labels import PTBXLSuperclassSpace, ScoredLabelSpace
from ..utils import get_logger

log = get_logger()

TARGET_FS = 100
TARGET_LEN = 1000  # 10 s @ 100 Hz
N_LEADS = 12

# CinC-2021 training subfolders -> client names (the 6 majors; tiny
# INCART/PTB sources are excluded by default but supported).
DEFAULT_SOURCES = {
    "cpsc_2018": "CPSC",
    "cpsc_2018_extra": "CPSC-Extra",
    "georgia": "Georgia",
    "chapman_shaoxing": "Chapman",
    "ningbo": "Ningbo",
    "ptb-xl": "PTB-XL",
}

_SOS_CACHE = {}


def _bandpass(x, fs=TARGET_FS, lo=0.5, hi=45.0, order=3):
    key = (fs, lo, hi, order)
    if key not in _SOS_CACHE:
        _SOS_CACHE[key] = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(_SOS_CACHE[key], x, axis=-1)


def standardize_signal(sig, fs):
    """(n_samples, n_leads) raw -> (12, 1000) float32, filtered + z-normed."""
    x = sig.T.astype(np.float64)  # (leads, samples)
    if x.shape[0] < N_LEADS:
        x = np.pad(x, ((0, N_LEADS - x.shape[0]), (0, 0)))
    x = x[:N_LEADS]
    x = np.nan_to_num(x)
    if int(fs) != TARGET_FS:
        frac = Fraction(TARGET_FS, int(fs)).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator, axis=-1)
    if x.shape[-1] >= 2 * TARGET_FS:  # need a minimally sane length to filter
        x = _bandpass(x)
    n = x.shape[-1]
    if n >= TARGET_LEN:  # center crop
        start = (n - TARGET_LEN) // 2
        x = x[:, start:start + TARGET_LEN]
    else:
        x = np.pad(x, ((0, 0), (0, TARGET_LEN - n)))
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    return x.astype(np.float32)


def _read_header_meta(hea_path):
    """Parse fs and Dx codes straight from the .hea text (fast, no wfdb)."""
    with open(hea_path) as f:
        lines = f.read().splitlines()
    fs = float(lines[0].split()[2])
    dx = []
    for line in lines:
        if line.startswith("#") and "Dx" in line:
            dx = line.split(":", 1)[1].strip().split(",")
    return fs, dx


def build_cinc_cache(raw_dir, cache_dir, evaluation_repo_dir,
                     sources=None, max_per_source=None, seed=0):
    """Cache every CinC-2021 source as X.npy (N,12,1000 float16) + Y.npy + manifest."""
    import wfdb

    label_space = ScoredLabelSpace(evaluation_repo_dir)
    sources = sources or DEFAULT_SOURCES
    rng = np.random.RandomState(seed)
    os.makedirs(cache_dir, exist_ok=True)

    for folder, client in sources.items():
        out_dir = os.path.join(cache_dir, client)
        if os.path.exists(os.path.join(out_dir, "X.npy")):
            log.info(f"[cache] {client}: already built, skipping")
            continue
        heas = sorted(glob.glob(os.path.join(raw_dir, folder, "**", "*.hea"),
                                recursive=True))
        if not heas:
            log.info(f"[cache] {client}: no records found under {folder}, skipping")
            continue
        if max_per_source and len(heas) > max_per_source:
            heas = [heas[i] for i in rng.permutation(len(heas))[:max_per_source]]

        xs, ys, names = [], [], []
        for hea in heas:
            rec = hea[:-4]
            try:
                fs, dx = _read_header_meta(hea)
                y = label_space.encode(dx)
                if y is None:
                    continue
                sig, _ = wfdb.rdsamp(rec)
                xs.append(standardize_signal(sig, fs).astype(np.float16))
                ys.append(y)
                names.append(os.path.basename(rec))
            except Exception as e:  # corrupt records exist in the wild
                log.info(f"[cache] skip {rec}: {e}")
        if not xs:
            continue
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), np.stack(xs))
        np.save(os.path.join(out_dir, "Y.npy"), np.asarray(ys, dtype=np.uint8))
        pd.DataFrame({"record": names}).to_csv(
            os.path.join(out_dir, "manifest.csv"), index=False)
        log.info(f"[cache] {client}: {len(xs)} records cached")
    return label_space


def build_ptbxl_cache(ptbxl_dir, cache_dir, sampling_rate=100):
    """Track-B cache: original PTB-XL with patient_id + recording date so the
    twin module can replay multi-record patients chronologically."""
    import wfdb

    space = PTBXLSuperclassSpace(ptbxl_dir)
    db = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    out_dir = os.path.join(cache_dir, "PTBXL_TRACKB")
    if os.path.exists(os.path.join(out_dir, "X.npy")):
        log.info("[cache] PTBXL_TRACKB: already built, skipping")
        return space
    xs, rows = [], []
    for ecg_id, row in db.iterrows():
        y = space.encode(row.scp_codes)
        if y is None:
            continue
        sig, meta = wfdb.rdsamp(os.path.join(ptbxl_dir, row[fname_col]))
        xs.append(standardize_signal(sig, meta["fs"]).astype(np.float16))
        rows.append({"ecg_id": ecg_id, "patient_id": int(row.patient_id),
                     "date": str(row.recording_date), "strat_fold": int(row.strat_fold),
                     "y": y})
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), np.stack(xs))
    np.save(os.path.join(out_dir, "Y.npy"),
            np.asarray([r["y"] for r in rows], dtype=np.uint8))
    pd.DataFrame([{k: v for k, v in r.items() if k != "y"} for r in rows]).to_csv(
        os.path.join(out_dir, "meta.csv"), index=False)
    log.info(f"[cache] PTBXL_TRACKB: {len(xs)} records cached")
    return space
