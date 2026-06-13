#!/usr/bin/env python3
"""Build preprocessed caches from raw downloads (run once after wget).

  python scripts/build_cache.py --raw-dir data/raw/cinc2021 --track a
  python scripts/build_cache.py --ptbxl-dir data/raw/ptbxl --track b
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcardiotwin.data.preprocess import build_cinc_cache, build_ptbxl_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["a", "b"])
    ap.add_argument("--raw-dir", default="data/raw/cinc2021")
    ap.add_argument("--ptbxl-dir", default="data/raw/ptbxl")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--max-per-source", type=int, default=None,
                    help="subsample for quick runs")
    args = ap.parse_args()

    if args.track == "a":
        build_cinc_cache(args.raw_dir, args.cache_dir, args.eval_repo,
                         max_per_source=args.max_per_source)
    else:
        build_ptbxl_cache(args.ptbxl_dir, args.cache_dir)


if __name__ == "__main__":
    main()
