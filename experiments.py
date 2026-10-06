import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP, METRIC_CLIP, weighted_pearson
from features import FEATURE_VERSION, build_sequence

np.seterr(over="ignore", divide="ignore", invalid="ignore")   # Accelerate noise on Apple Silicon

# ---------- Settings ----------
TRAIN_SEQS = 300
ROW_STEP = 20
TUNING_SEQS = 100      # used to choose alpha
EVAL_SEQS = 300        # held out: used only to report
ALPHAS = [10, 100, 1000, 10000]
STD_CLIP = 10.0
SEED = 42

# name, feature set, clip targets, metric sample weights
EXPERIMENTS = [
    ("A raw, no clip, no weights",  "raw",       False, False),
    ("C raw, weights only",         "raw",       False, True),
    ("D raw, clip + weights",       "raw",       True,  True),
    ("E stateless, clip + weights", "stateless", True,  True),
    ("F stateless, weights only",   "stateless", False, True),
    ("G memory, weights only",      "memory",    False, True),
    ("H memory, clip + weights",    "memory",    True,  True),
]


def columns_to_array(table, columns):
    return np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float64)


def sequence_features(table):
    """Build every feature for one complete sequence, in row order."""
    x = columns_to_array(table, FEATURE_COLUMNS)
    return build_sequence(x)      # (features, stateless_names, memory_names)


def load_training(path):
    pf = pq.ParquetFile(path)
    picks = np.random.default_rng(SEED).choice(pf.num_row_groups, size=TRAIN_SEQS, replace=False)
    xs, ys, names = [], [], None
    for g in picks:
        table = pf.read_row_group(int(g), columns=FEATURE_COLUMNS + list(TARGET_COLUMNS))
        full, stateless_names, mem_names = sequence_features(table)
        names = (stateless_names, mem_names)
        # Features are built on the WHOLE sequence first, so memory features are
        # correct, and only then are rows subsampled.
        xs.append(full[WARMUP::ROW_STEP].astype(np.float32))
        ys.append(columns_to_array(table, TARGET_COLUMNS)[WARMUP::ROW_STEP])
    return np.vstack(xs), np.vstack(ys), names


def load_scored(path, picks):
    """Scored rows only, but features still built on the complete sequence."""
    pf = pq.ParquetFile(path)
    xs, ys = [], []
    for g in picks:
        table = pf.read_row_group(int(g), columns=(FEATURE_COLUMNS + list(TARGET_COLUMNS)
                                                   + ["need_prediction", "is_scored"]))
        mask = (table["need_prediction"].to_numpy().astype(bool)
                & table["is_scored"].to_numpy().astype(bool))
        if not mask.any():
            continue
        full, _, _ = sequence_features(table)
        xs.append(full[mask].astype(np.float32))
        ys.append(columns_to_array(table, TARGET_COLUMNS)[mask])
    return np.vstack(xs), np.vstack(ys)


def standardise(x, mean, scale):
    z = np.clip((x - mean) / scale, -STD_CLIP, STD_CLIP)
    return np.nan_to_num(z, nan=0.0, posinf=STD_CLIP, neginf=-STD_CLIP)


def fit_and_score(x_train, y_train, x_tune, y_tune, x_eval, y_eval, clip, weighted):
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    mean = np.nanmean(finite, axis=0)
    scale = np.nanstd(finite, axis=0)
    scale[scale < 1e-8] = 1.0
    z_train = standardise(x_train, mean, scale)
    z_tune = standardise(x_tune, mean, scale)
    z_eval = standardise(x_eval, mean, scale)

    y_clipped = np.clip(y_train, -METRIC_CLIP, METRIC_CLIP)
    y_fit = y_clipped if clip else y_train
    row_weights = np.abs(y_clipped) if weighted else np.ones_like(y_clipped)

    best = None
    for alpha in ALPHAS:
        coef = np.zeros((2, x_train.shape[1]))
        intercept = np.zeros(2)
        for k in range(2):
            model = Ridge(alpha=alpha)
            model.fit(z_train, y_fit[:, k], sample_weight=row_weights[:, k])
            coef[k], intercept[k] = model.coef_, model.intercept_
        pred = z_tune @ coef.T + intercept
        tune = np.mean([weighted_pearson(y_tune[:, k], pred[:, k]) for k in range(2)])
        if best is None or tune > best[0]:
            best = (tune, alpha, coef, intercept)

    tune, alpha, coef, intercept = best
    pred = z_eval @ coef.T + intercept
    if not np.isfinite(pred).all():
        print("  WARNING: non-finite predictions")
    s0 = weighted_pearson(y_eval[:, 0], pred[:, 0])
    s1 = weighted_pearson(y_eval[:, 1], pred[:, 1])
    return {"tune": tune, "alpha": alpha, "t0": s0, "t1": s1, "held_out": (s0 + s1) / 2,
            "coef": coef, "intercept": intercept, "mean": mean, "scale": scale}


def main():
    print(f"Loading {TRAIN_SEQS} training sequences (every {ROW_STEP}th row)...")
    x_train, y_train, (stateless_names, mem_names) = load_training("datasets/train.parquet")
    n_raw = len(FEATURE_COLUMNS)
    n_stateless = len(stateless_names)
    widths = {"raw": n_raw, "stateless": n_raw + n_stateless,
              "memory": n_raw + n_stateless + len(mem_names)}
    print(f"  training rows: {len(x_train):,}")
    print(f"  {n_raw} raw + {n_stateless} stateless + {len(mem_names)} memory "
          f"= {widths['memory']} features")

    pf = pq.ParquetFile("datasets/valid.parquet")
    tuning_picks = np.random.default_rng(SEED).choice(pf.num_row_groups, size=TUNING_SEQS,
                                                      replace=False)
    rest = np.array([g for g in range(pf.num_row_groups) if g not in set(tuning_picks.tolist())])
    eval_picks = np.random.default_rng(SEED + 1).choice(rest, size=EVAL_SEQS, replace=False)

    print(f"Loading {TUNING_SEQS} tuning and {EVAL_SEQS} held-out sequences...")
    x_tune, y_tune = load_scored("datasets/valid.parquet", tuning_picks)
    x_eval, y_eval = load_scored("datasets/valid.parquet", eval_picks)
    print(f"  tuning rows: {len(x_tune):,}   held-out rows: {len(x_eval):,}")

    print(f"\n{'experiment':<30}{'alpha':>7}{'tuning':>9}{'held-out':>10}{'t0':>8}{'t1':>8}{'vs A':>8}")
    baseline, best_run = None, None
    for name, mode, clip, weighted in EXPERIMENTS:
        w = widths[mode]
        r = fit_and_score(x_train[:, :w], y_train, x_tune[:, :w], y_tune,
                          x_eval[:, :w], y_eval, clip, weighted)
        if baseline is None:
            baseline = r["held_out"]
        print(f"{name:<30}{r['alpha']:>7g}{r['tune']:>9.4f}{r['held_out']:>10.4f}"
              f"{r['t0']:>8.4f}{r['t1']:>8.4f}{r['held_out'] - baseline:>+8.4f}")
        if best_run is None or r["held_out"] > best_run[1]["held_out"]:
            best_run = (name, r, mode)

    name, r, mode = best_run
    print(f"\nBest: {name}   held-out {r['held_out']:.4f}")

    np.savez("linear_model.npz", mean=r["mean"].astype(np.float32),
             scale=r["scale"].astype(np.float32), coef=r["coef"].astype(np.float32),
             intercept=r["intercept"].astype(np.float32), std_clip=STD_CLIP,
             alpha=r["alpha"], mode=mode, experiment=name,
             feature_version=FEATURE_VERSION)
    print("Saved the winning model to linear_model.npz")

    names = list(FEATURE_COLUMNS) + stateless_names + mem_names
    kinds = ["raw"] * n_raw + ["stateless"] * n_stateless + ["memory"] * len(mem_names)
    strength = np.abs(r["coef"]).mean(axis=0)
    order = np.argsort(-strength)[:20]
    print("\nTop 20 features by weight size:")
    for i in order:
        tag = {"raw": " ", "stateless": "*", "memory": "+"}[kinds[i]]
        print(f" {tag} {names[i]:<26} t0: {r['coef'][0, i]:+.4f}   t1: {r['coef'][1, i]:+.4f}")
    print("  (* = stateless feature, + = memory feature)")

    print("\nTotal weight by feature kind:")
    for kind in ("raw", "stateless", "memory"):
        idx = [i for i, k in enumerate(kinds) if k == kind and i < r["coef"].shape[1]]
        if idx:
            print(f"  {kind:<10} {np.abs(r['coef'][:, idx]).sum():.3f}")


if __name__ == "__main__":
    main()