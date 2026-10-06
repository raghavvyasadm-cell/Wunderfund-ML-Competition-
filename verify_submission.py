import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SUBMISSION_DIR = "submission"
VALID_FILE = "datasets/valid.parquet"
N_SEQUENCES = 2            # how many sequences to run through
TEST_ROWS = 39_400_000     # size of the real test set, for the time estimate
REQUIRED = ["solution.py", "features.py"]
MODEL_FILES = ["mlp_model.onnx", "linear_model.npz"]   # at least one of these
ALLOWED_IMPORTS = {"numpy", "onnxruntime", "pyarrow"}
BLOCKED = ["pandas", "sklearn", "scipy", "matplotlib", "torch"]


@dataclass(frozen=True)
class DataPoint:
    """Same shape as the platform's DataPoint, defined here on purpose:
    solution.py must not depend on anything outside the ZIP."""
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load_sequences(path, n):
    pf = pq.ParquetFile(path)
    from utils import FEATURE_COLUMNS      # only used by this test script
    out = []
    for g in range(min(n, pf.num_row_groups)):
        table = pf.read_row_group(g, columns=["seq_ix", "step_in_seq", "need_prediction"]
                                  + FEATURE_COLUMNS)
        x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        out.append((int(table["seq_ix"][0].as_py()),
                    table["need_prediction"].to_numpy().astype(bool), x))
    return out


def run(model, sequences):
    preds, warmup_ok = [], True
    for seq_ix, need, x in sequences:
        for t in range(len(x)):
            value = model.predict(DataPoint(seq_ix, t, bool(need[t]), x[t]))
            if not need[t]:
                warmup_ok = warmup_ok and value is None
            else:
                preds.append(np.asarray(value, dtype=np.float64))
    return np.vstack(preds), warmup_ok


def main():
    folder = Path(SUBMISSION_DIR).resolve()
    print(f"Checking {folder}")
    problems = []

    # 1. Required files present
    for name in REQUIRED:
        exists = (folder / name).exists()
        print(f"  {'OK  ' if exists else 'MISS'} {name}")
        if not exists:
            problems.append(f"{name} is missing")
    models = [m for m in MODEL_FILES if (folder / m).exists()]
    print(f"  {'OK  ' if models else 'MISS'} model file: {', '.join(models) or 'none'}")
    if not models:
        problems.append("no model file")
    if problems:
        print("\nFAILED: " + "; ".join(problems))
        return

    # 2. Import with the unavailable libraries blocked
    for name in BLOCKED:
        sys.modules[name] = None        # any import of these now raises
    sys.path.insert(0, str(folder))
    try:
        solution = importlib.import_module("solution")
    except Exception as exc:
        print(f"\nFAILED to import solution.py without {BLOCKED}: {exc}")
        return
    print(f"  OK   imports cleanly without {', '.join(BLOCKED)}")

    # 3. Run real rows through it
    sequences = load_sequences(VALID_FILE, N_SEQUENCES)
    rows = sum(len(x) for _, _, x in sequences)
    model = solution.PredictionModel()
    start = time.perf_counter()
    preds, warmup_ok = run(model, sequences)
    elapsed = time.perf_counter() - start

    print(f"  {'OK  ' if warmup_ok else 'FAIL'} returns None on warm-up rows")
    print(f"  {'OK  ' if preds.shape[1] == 2 else 'FAIL'} output shape is (2,)")
    finite = np.isfinite(preds).all()
    print(f"  {'OK  ' if finite else 'FAIL'} all {len(preds):,} predictions are finite")
    spread = preds.std(axis=0)
    print(f"  {'OK  ' if spread.min() > 1e-8 else 'FAIL'} predictions vary "
          f"(std t0 {spread[0]:.4f}, t1 {spread[1]:.4f})")

    # 4. Determinism: a fresh model on the same rows must give the same answers
    again, _ = run(solution.PredictionModel(), sequences)
    same = np.array_equal(preds, again)
    print(f"  {'OK  ' if same else 'FAIL'} repeated run gives identical predictions")

    # 5. Speed
    per_row = elapsed / rows
    minutes = TEST_ROWS * per_row / 60
    verdict = "OK  " if minutes < 45 else ("WARN" if minutes < 60 else "FAIL")
    print(f"  {verdict} {per_row * 1e6:.1f} us/row -> about {minutes:.0f} minutes "
          f"for {TEST_ROWS:,} rows (limit 60)")

    zip_path = Path("submission.zip")
    if zip_path.exists():
        mb = zip_path.stat().st_size / 1e6
        print(f"  {'OK  ' if mb < 20 else 'FAIL'} submission.zip is {mb:.2f} MB (limit 20)")

    print("\nAll checks run. Fix anything marked FAIL before uploading.")


if __name__ == "__main__":
    main()
