import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from features import FEATURE_VERSION, build_sequence
from utils import FEATURE_COLUMNS, METRIC_CLIP, TARGET_COLUMNS, WARMUP, weighted_pearson

np.seterr(over="ignore", divide="ignore", invalid="ignore")

# ---------- Settings ----------
TRAIN_SEQS = 800          # was 300: the first run overfitted after one epoch
ROW_STEP = 10             # was 20: more rows per sequence
TUNING_SEQS = 100
EVAL_SEQS = 300
STD_CLIP = 10.0
SEED = 42

EPOCHS = 40
BATCH_SIZE = 1024
PATIENCE = 6              # stop if the tuning score has not improved for this many epochs
MODEL_FILE = "mlp_model.onnx"

# Configurations tried in one run, since loading the data is the slow part.
# name, hidden sizes, dropout, learning rate, weight decay
CONFIGS = [
    ("baseline shape",     (128, 64),  0.1, 1e-3, 1e-4),
    ("slower learning",    (128, 64),  0.1, 3e-4, 1e-4),
    ("more regularised",   (128, 64),  0.3, 3e-4, 1e-3),
    ("wider + regularised", (256, 128), 0.3, 3e-4, 1e-3),
]


def columns_to_array(table, columns):
    return np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float64)


def load_training(path):
    pf = pq.ParquetFile(path)
    picks = np.random.default_rng(SEED).choice(pf.num_row_groups, size=TRAIN_SEQS, replace=False)
    xs, ys = [], []
    for g in picks:
        table = pf.read_row_group(int(g), columns=FEATURE_COLUMNS + list(TARGET_COLUMNS))
        full, _, _ = build_sequence(columns_to_array(table, FEATURE_COLUMNS))
        xs.append(full[WARMUP::ROW_STEP].astype(np.float32))
        ys.append(columns_to_array(table, TARGET_COLUMNS)[WARMUP::ROW_STEP].astype(np.float32))
    return np.vstack(xs), np.vstack(ys)


def load_scored(path, picks):
    pf = pq.ParquetFile(path)
    xs, ys = [], []
    for g in picks:
        table = pf.read_row_group(int(g), columns=(FEATURE_COLUMNS + list(TARGET_COLUMNS)
                                                   + ["need_prediction", "is_scored"]))
        mask = (table["need_prediction"].to_numpy().astype(bool)
                & table["is_scored"].to_numpy().astype(bool))
        if not mask.any():
            continue
        full, _, _ = build_sequence(columns_to_array(table, FEATURE_COLUMNS))
        xs.append(full[mask].astype(np.float32))
        ys.append(columns_to_array(table, TARGET_COLUMNS)[mask].astype(np.float32))
    return np.vstack(xs), np.vstack(ys)


class Net(nn.Module):
    """Standardisation is baked into the model, so the exported ONNX file is
    self-contained: raw features in, predictions out."""

    def __init__(self, n_features, mean, scale, hidden, dropout, target_scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))
        self.clip = STD_CLIP
        layers, size = [], n_features
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, 2))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        z = (x - self.mean) / self.scale
        z = torch.clamp(z, -self.clip, self.clip)
        z = torch.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return self.body(z) * self.target_scale     # back to the targets' own scale


def score(model, x, y, batch=65536):
    """Weighted Pearson on scored rows, the competition metric."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch])).numpy())
    pred = np.vstack(out).astype(np.float64)
    s0 = weighted_pearson(y[:, 0], pred[:, 0])
    s1 = weighted_pearson(y[:, 1], pred[:, 1])
    return (s0 + s1) / 2, s0, s1


def train_one(name, hidden, dropout, learning_rate, weight_decay,
              x_train, y_scaled, row_weights, mean, scale, target_scale,
              x_tune, y_tune):
    """Train one configuration, keeping the state that scored best on tuning."""
    torch.manual_seed(SEED)
    model = Net(x_train.shape[1], mean, scale, hidden, dropout, target_scale)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_scaled)
    wt = torch.from_numpy(row_weights)
    generator = torch.Generator().manual_seed(SEED)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}: {' -> '.join(str(h) for h in hidden)}, dropout {dropout}, "
          f"lr {learning_rate:g}, weight decay {weight_decay:g}, {params:,} parameters")
    print(f"{'epoch':>6}{'train loss':>12}{'tuning':>9}{'t0':>8}{'t1':>8}{'time':>7}")

    best = {"score": -1.0, "state": None, "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(xt), generator=generator)
        total, seen = 0.0, 0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            optimiser.zero_grad()
            pred = model(xt[idx]) / model.target_scale        # compare on the trained scale
            # Weighted mean squared error: rows the metric cares about count more
            loss = (wt[idx] * (pred - yt[idx]) ** 2).mean()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        tune, s0, s1 = score(model, x_tune, y_tune)
        print(f"{epoch:>6}{total / seen:>12.5f}{tune:>9.4f}{s0:>8.4f}{s1:>8.4f}"
              f"{time.perf_counter() - start:>6.0f}s")
        if tune > best["score"]:
            best = {"score": tune, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= PATIENCE:
            print(f"  no improvement for {PATIENCE} epochs, stopping")
            break
    model.load_state_dict(best["state"])
    return model, best


def main():
    torch.use_deterministic_algorithms(True)

    print(f"Loading {TRAIN_SEQS} training sequences (every {ROW_STEP}th row)...")
    x_train, y_train = load_training("datasets/train.parquet")
    pf = pq.ParquetFile("datasets/valid.parquet")
    tuning_picks = np.random.default_rng(SEED).choice(pf.num_row_groups, size=TUNING_SEQS,
                                                      replace=False)
    rest = np.array([g for g in range(pf.num_row_groups) if g not in set(tuning_picks.tolist())])
    eval_picks = np.random.default_rng(SEED + 1).choice(rest, size=EVAL_SEQS, replace=False)
    print(f"Loading {TUNING_SEQS} tuning and {EVAL_SEQS} held-out sequences...")
    x_tune, y_tune = load_scored("datasets/valid.parquet", tuning_picks)
    x_eval, y_eval = load_scored("datasets/valid.parquet", eval_picks)
    n_features = x_train.shape[1]
    print(f"  train {len(x_train):,} rows   tuning {len(x_tune):,}   held-out {len(x_eval):,}"
          f"   features {n_features}")

    # Feature statistics from the training data only
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    mean = np.nan_to_num(np.nanmean(finite, axis=0)).astype(np.float32)
    scale = np.nan_to_num(np.nanstd(finite, axis=0)).astype(np.float32)
    scale[scale < 1e-8] = 1.0

    # Targets are trained on a unit scale (easier optimisation); correlation
    # does not care about scale, and the model multiplies it back on the way out.
    target_scale = y_train.std(axis=0).astype(np.float32)
    y_scaled = y_train / target_scale

    # The metric weights each row by |clipped target|: do the same here
    row_weights = np.abs(np.clip(y_train, -METRIC_CLIP, METRIC_CLIP)).astype(np.float32)

    results = []
    for name, hidden, dropout, lr, decay in CONFIGS:
        model, best = train_one(name, hidden, dropout, lr, decay,
                                x_train, y_scaled, row_weights, mean, scale, target_scale,
                                x_tune, y_tune)
        held, s0, s1 = score(model, x_eval, y_eval)
        results.append({"name": name, "model": model, "tune": best["score"],
                        "epoch": best["epoch"], "held": held, "t0": s0, "t1": s1})
        print(f"  best epoch {best['epoch']}, tuning {best['score']:.4f}, "
              f"held-out {held:.4f}")

    print(f"\n{'configuration':<24}{'epoch':>7}{'tuning':>9}{'held-out':>10}{'t0':>8}{'t1':>8}")
    for r in results:
        print(f"{r['name']:<24}{r['epoch']:>7}{r['tune']:>9.4f}{r['held']:>10.4f}"
              f"{r['t0']:>8.4f}{r['t1']:>8.4f}")

    # Choose on the tuning score, report the held-out score: the held-out
    # sequences must not influence the choice.
    chosen = max(results, key=lambda r: r["tune"])
    model = chosen["model"]
    print(f"\nChosen on tuning score: {chosen['name']}")
    print(f"Held-out: {chosen['held']:.4f}  (t0 {chosen['t0']:.4f}, t1 {chosen['t1']:.4f})"
          f"   linear model: 0.5906")
    print(f"Expected leaderboard: about {chosen['held'] - 0.011:.4f}")

    # Export for the submission: raw features in, predictions out
    model.eval()
    torch.onnx.export(model, torch.zeros(1, n_features), MODEL_FILE,
                      input_names=["input"], output_names=["output"],
                      # one row at a time at submission time, many rows when
                      # checking: a dynamic batch size allows both
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=13, dynamo=False)

    # Record which feature set this was trained on, inside the model file
    import onnx
    onnx_model = onnx.load(MODEL_FILE)
    entry = onnx_model.metadata_props.add()
    entry.key, entry.value = "feature_version", FEATURE_VERSION
    entry = onnx_model.metadata_props.add()
    entry.key, entry.value = "n_features", str(n_features)
    onnx.save(onnx_model, MODEL_FILE)

    # Check the exported file gives the same answers as PyTorch
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(MODEL_FILE, opts, providers=["CPUExecutionProvider"])
    sample = x_eval[:2000]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()
    onnx_out = session.run(None, {"input": sample})[0]
    gap = np.abs(torch_out - onnx_out).max()
    print(f"ONNX export matches PyTorch to {gap:.2e}  "
          f"({'OK' if gap < 1e-4 else 'MISMATCH - do not submit'})")
    print(f"Saved {MODEL_FILE}")

    with open("mlp_training.json", "w") as handle:
        json.dump({"configuration": chosen["name"], "held_out": chosen["held"],
                   "t0": chosen["t0"], "t1": chosen["t1"], "tuning": chosen["tune"],
                   "epoch": chosen["epoch"], "train_rows": int(len(x_train)),
                   "feature_version": FEATURE_VERSION, "n_features": n_features}, handle, indent=2)


if __name__ == "__main__":
    main()

