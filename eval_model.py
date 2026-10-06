import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, weighted_pearson

TUNING_SEQS = 100    # must match VALID_SEQS in train_linear.py
EVAL_SEQS = 300      # how many *different* sequences to score on (None = all the rest)
SEED = 42


def standardise(x, mean, scale, std_clip):
    """Same steps as in training: centre, rescale, clip, remove non-finite values."""
    z = (x - mean) / scale
    z = np.clip(z, -std_clip, std_clip)
    return np.nan_to_num(z, nan=0.0, posinf=std_clip, neginf=-std_clip)


def main():
    model = np.load("linear_model.npz")
    mean, scale = model["mean"].astype(np.float64), model["scale"].astype(np.float64)
    coef, intercept = model["coef"].astype(np.float64), model["intercept"].astype(np.float64)
    std_clip = float(model["std_clip"])
    print(f"Model: alpha={float(model['alpha']):g}, "
          f"clipped targets={bool(model['clip_targets'])}, "
          f"sample weights={bool(model['sample_weight'])}")

    pf = pq.ParquetFile("datasets/valid.parquet")
    rng = np.random.default_rng(SEED)
    tuning = set(rng.choice(pf.num_row_groups, size=TUNING_SEQS, replace=False).tolist())

    # Every sequence the tuning run never saw
    untouched = np.array([g for g in range(pf.num_row_groups) if g not in tuning])
    if EVAL_SEQS is not None:
        untouched = np.random.default_rng(SEED + 1).choice(untouched, size=EVAL_SEQS, replace=False)
    print(f"Scoring on {len(untouched)} held-out validation sequences...")

    y_parts, p_parts = [], []
    for g in untouched:
        table = pf.read_row_group(int(g), columns=(FEATURE_COLUMNS + list(TARGET_COLUMNS)
                                                   + ["need_prediction", "is_scored"]))
        mask = (table["need_prediction"].to_numpy().astype(bool)
                & table["is_scored"].to_numpy().astype(bool))
        if not mask.any():
            continue
        x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[mask]
        y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float64)[mask]
        z = standardise(x, mean, scale, std_clip)
        p_parts.append(z @ coef.T + intercept)      # the whole model: one dot product
        y_parts.append(y)

    y_true, y_pred = np.vstack(y_parts), np.vstack(p_parts)
    if not np.isfinite(y_pred).all():
        print("  WARNING: some predictions are not finite")
    s0 = weighted_pearson(y_true[:, 0], y_pred[:, 0])
    s1 = weighted_pearson(y_true[:, 1], y_pred[:, 1])
    print(f"  scored rows: {len(y_true):,}")
    print(f"  t0: {s0:.4f}   t1: {s1:.4f}")
    print(f"  held-out score: {(s0 + s1) / 2:.4f}      (previous version: 0.5706)")


if __name__ == "__main__":
    main()
