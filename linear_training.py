

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP, METRIC_CLIP, weighted_pearson

# ---------- Settings (change these to experiment) ----------
TRAIN_SEQS = 300            # how many training sequences to sample
ROW_STEP = 20               # keep every 20th row (targets barely change between rows)
VALID_SEQS = 100            # validation sequences used to choose alpha
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]   # regularisation strengths to try
CLIP_TARGETS = True         # experiment 1
USE_SAMPLE_WEIGHT = True    # experiment 2
STD_CLIP = 10.0             # clip standardised features to this many spreads
SEED = 42


def columns_to_array(table, columns):
    """Pull the named columns out of a Parquet table into one float64 grid."""
    return np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float64)


def standardise(x, mean, scale):
    """Centre, rescale, clip extremes, and remove any non-finite values.

    These are exactly the steps solution.py will need at prediction time.
    """
    z = (x - mean) / scale
    z = np.clip(z, -STD_CLIP, STD_CLIP)
    return np.nan_to_num(z, nan=0.0, posinf=STD_CLIP, neginf=-STD_CLIP)


def load_training_sample(path, n_seqs, row_step, seed):
    """Features and targets from a random sample of training sequences."""
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    picks = rng.choice(pf.num_row_groups, size=n_seqs, replace=False)

    x_parts, y_parts = [], []
    for g in picks:
        table = pf.read_row_group(int(g), columns=FEATURE_COLUMNS + list(TARGET_COLUMNS))
        x = columns_to_array(table, FEATURE_COLUMNS)
        y = columns_to_array(table, TARGET_COLUMNS)
        # Skip the warm-up rows, then keep every row_step-th row
        x_parts.append(x[WARMUP::row_step])
        y_parts.append(y[WARMUP::row_step])
    return np.vstack(x_parts), np.vstack(y_parts)


def load_validation_sample(path, n_seqs, seed):
    """Only the rows the competition actually scores: need_prediction AND is_scored."""
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    picks = rng.choice(pf.num_row_groups, size=n_seqs, replace=False)

    x_parts, y_parts = [], []
    for g in picks:
        table = pf.read_row_group(
            int(g), columns=FEATURE_COLUMNS + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"])
        mask = (table["need_prediction"].to_numpy().astype(bool)
                & table["is_scored"].to_numpy().astype(bool))
        if not mask.any():
            continue
        x_parts.append(columns_to_array(table, FEATURE_COLUMNS)[mask])
        y_parts.append(columns_to_array(table, TARGET_COLUMNS)[mask])
    return np.vstack(x_parts), np.vstack(y_parts)


def fit_one_target(z, y_col, alpha, sample_weight):
    """Fit one Ridge model for a single target. Returns (weights, intercept)."""
    model = Ridge(alpha=alpha)
    model.fit(z, y_col, sample_weight=sample_weight)
    return model.coef_, model.intercept_


def main():
    print(f"Loading {TRAIN_SEQS} training sequences (every {ROW_STEP}th row)...")
    x_train, y_train = load_training_sample("datasets/train.parquet", TRAIN_SEQS, ROW_STEP, SEED)
    print(f"  training rows: {x_train.shape[0]:,}   features: {x_train.shape[1]}")

    # Feature statistics come from the TRAINING data only, and are part of the model
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    mean = np.nanmean(finite, axis=0)
    scale = np.nanstd(finite, axis=0)
    scale[scale < 1e-8] = 1.0                    # never divide by (almost) zero
    z_train = standardise(x_train, mean, scale)

    # The metric clips targets; optionally train on the clipped version
    y_clipped = np.clip(y_train, -METRIC_CLIP, METRIC_CLIP)
    y_fit = y_clipped if CLIP_TARGETS else y_train
    extreme = (np.abs(y_train) > METRIC_CLIP).mean()
    print(f"  rows with |target| > {METRIC_CLIP}: {extreme:.2%}")
    print(f"  clip targets: {CLIP_TARGETS}   sample weights: {USE_SAMPLE_WEIGHT}")

    # The metric weights each row by |clipped target|, separately per target
    weights = np.abs(y_clipped) if USE_SAMPLE_WEIGHT else np.ones_like(y_clipped)

    print(f"Loading {VALID_SEQS} validation sequences (scored rows only)...")
    x_valid, y_valid = load_validation_sample("datasets/valid.parquet", VALID_SEQS, SEED)
    z_valid = standardise(x_valid, mean, scale)
    print(f"  scored rows: {x_valid.shape[0]:,}")

    print("\nFitting one model per regularisation strength...")
    print(f"{'alpha':>8}  {'score':>7}  {'t0':>7}  {'t1':>7}  {'biggest weight':>15}")
    results = []
    for alpha in ALPHAS:
        coef = np.zeros((2, x_train.shape[1]))
        intercept = np.zeros(2)
        for k in range(2):      # t0 and t1 are fitted separately: different row weights
            coef[k], intercept[k] = fit_one_target(z_train, y_fit[:, k], alpha, weights[:, k])
        pred = z_valid @ coef.T + intercept
        s0 = weighted_pearson(y_valid[:, 0], pred[:, 0])
        s1 = weighted_pearson(y_valid[:, 1], pred[:, 1])
        score = (s0 + s1) / 2
        results.append((score, alpha, coef, intercept))
        print(f"{alpha:>8g}  {score:>7.4f}  {s0:>7.4f}  {s1:>7.4f}  {np.abs(coef).max():>15.3f}")

    best_score, best_alpha, best_coef, best_intercept = max(results, key=lambda r: r[0])
    print(f"\nBest alpha: {best_alpha:g}")
    print(f"Tuning-sample score: {best_score:.4f}      (GRU baseline: 0.6171)")

    np.savez("linear_model.npz",
             mean=mean.astype(np.float32), scale=scale.astype(np.float32),
             coef=best_coef.astype(np.float32), intercept=best_intercept.astype(np.float32),
             std_clip=STD_CLIP, alpha=best_alpha,
             clip_targets=CLIP_TARGETS, sample_weight=USE_SAMPLE_WEIGHT)
    print("Saved the fitted model to linear_model.npz")

    strength = np.abs(best_coef).mean(axis=0)
    order = np.argsort(-strength)[:10]
    print("\nTop 10 features by weight size:")
    for i in order:
        print(f"  {FEATURE_COLUMNS[i]:>8}  t0: {best_coef[0, i]:+.4f}   t1: {best_coef[1, i]:+.4f}")


if __name__ == "__main__":
    main()
