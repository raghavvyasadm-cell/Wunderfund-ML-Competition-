import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS
from features import build_sequence, StreamingFeatures

SEQ = 0
ROWS = 3000        # checking a few thousand rows is enough


def main():
    table = pq.ParquetFile("datasets/train.parquet").read_row_group(SEQ, columns=FEATURE_COLUMNS)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[:ROWS]

    batch, stateless_names, memory_names_ = build_sequence(x)

    stream = StreamingFeatures()
    rows = [stream.update(x[t]) for t in range(len(x))]
    streamed = np.vstack(rows)

    print(f"batch:     {batch.shape}")
    print(f"streaming: {streamed.shape}")
    print(f"{len(FEATURE_COLUMNS)} raw + {len(stateless_names)} stateless "
          f"+ {len(memory_names_)} memory")

    if batch.shape != streamed.shape:
        print("MISMATCH: different shapes")
        return
    gap = np.abs(batch - streamed).max()
    print(f"largest difference: {gap:.3e}")
    print("PASS: the two agree" if gap < 1e-9 else "FAIL: they disagree")

    if gap >= 1e-9:
        worst = np.unravel_index(np.argmax(np.abs(batch - streamed)), batch.shape)
        names = list(FEATURE_COLUMNS) + stateless_names + memory_names_
        print(f"  worst at row {worst[0]}, feature {names[worst[1]]}")


if __name__ == "__main__":
    main()
    