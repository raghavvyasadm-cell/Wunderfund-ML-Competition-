"""Look for missing, infinite or extreme values in the feature columns.

Run from the project root:

    python check_data.py
"""
import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, TARGET_COLUMNS

N_SEQS = 400        # sequences to scan per file
BIG = 1e6           # anything above this counts as "extreme"
SEED = 43


def scan(path, n_seqs):
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(SEED)
    picks = rng.choice(pf.num_row_groups, size=min(n_seqs, pf.num_row_groups), replace=False)

    columns = FEATURE_COLUMNS + list(TARGET_COLUMNS)
    n_nan = np.zeros(len(columns), dtype=np.int64)
    n_inf = np.zeros(len(columns), dtype=np.int64)
    n_big = np.zeros(len(columns), dtype=np.int64)
    lo = np.full(len(columns), np.inf)
    hi = np.full(len(columns), -np.inf)
    rows = 0

    for g in picks:
        table = pf.read_row_group(int(g), columns=columns)
        x = np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float64)
        rows += len(x)
        n_nan += np.isnan(x).sum(axis=0)
        n_inf += np.isinf(x).sum(axis=0)
        finite = np.where(np.isfinite(x), x, np.nan)          # ignore inf/NaN for min and max
        n_big += (np.abs(finite) > BIG).sum(axis=0)
        lo = np.minimum(lo, np.nanmin(finite, axis=0))
        hi = np.maximum(hi, np.nanmax(finite, axis=0))

    print(f"\n=== {path}: {len(picks)} sequences, {rows:,} rows ===")
    problems = [i for i in range(len(columns)) if n_nan[i] or n_inf[i] or n_big[i]]
    if not problems:
        print("  No NaN, infinite or extreme values found.")
    else:
        print(f"  {'column':<10} {'NaN':>10} {'inf':>10} {f'|x|>{BIG:g}':>12} {'min':>12} {'max':>12}")
        for i in problems:
            print(f"  {columns[i]:<10} {n_nan[i]:>10,} {n_inf[i]:>10,} {n_big[i]:>12,} "
                  f"{lo[i]:>12.3g} {hi[i]:>12.3g}")

    widest = np.argsort(-(hi - lo))[:5]
    print("  Widest-ranging columns:")
    for i in widest:
        print(f"    {columns[i]:<10} {lo[i]:>12.3g} .. {hi[i]:>12.3g}")


if __name__ == "__main__":
    scan("datasets/train.parquet", N_SEQS)
    scan("datasets/valid.parquet", N_SEQS)