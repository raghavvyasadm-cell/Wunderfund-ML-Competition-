import numpy as np
import pyarrow.parquet as pq
 
import features as F
from utils import FEATURE_COLUMNS
 
SEQ = 0
ROWS = 3000
 
 
def main():
    print(f"features.py: {F.__file__}")
    print(f"version:     {F.FEATURE_VERSION}")
    print(f"numpy:       {np.__version__}")
 
    table = pq.ParquetFile("datasets/train.parquet").read_row_group(SEQ, columns=FEATURE_COLUMNS)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[:ROWS]
    print(f"input:       {x.shape}, dtype {x.dtype}, "
          f"all finite: {np.isfinite(x).all()}, range {x.min():.3f} .. {x.max():.3f}")
 
    names = list(FEATURE_COLUMNS) + list(F.STATELESS_NAMES) + list(F.MEMORY_NAMES)
 
    # --- 1. Compare the stateless path on its own, with no state involved ---
    batch_stateless, batch_signals = F._core(x)
    row_stateless = np.vstack([F._core_row(x[t])[0] for t in range(len(x))])
    row_signals = np.vstack([F._core_row(x[t])[1] for t in range(len(x))])
    d_state = np.abs(batch_stateless - row_stateless)
    d_sig = np.abs(batch_signals - row_signals)
    print(f"\nstateless features: max diff {np.nanmax(d_state):.3e}")
    print(f"base signals:       max diff {np.nanmax(d_sig):.3e}")
 
    # --- 2. Full comparison ---
    batch, _, _ = F.build_sequence(x)
    stream = F.StreamingFeatures()
    streamed = np.vstack([stream.update(x[t]).copy() for t in range(len(x))])
    diff = np.abs(batch - streamed)
 
    per_column = np.nanmax(diff, axis=0)
    order = np.argsort(-per_column)[:10]
    print("\nWorst 10 columns:")
    for i in order:
        kind = ("raw" if i < 112 else
                "stateless" if i < 112 + len(F.STATELESS_NAMES) else "memory")
        print(f"  {names[i]:<28} {kind:<10} max diff {per_column[i]:.3e}")
 
    worst = np.unravel_index(np.nanargmax(diff), diff.shape)
    row, col = int(worst[0]), int(worst[1])
    print(f"\nWorst cell: row {row}, column {names[col]}")
    print(f"  batch     {batch[row, col]:.10g}")
    print(f"  streaming {streamed[row, col]:.10g}")
 
    # --- 3. The inputs behind the worst row ---
    print(f"\nRow {row} input summary:")
    print(f"  all finite: {np.isfinite(x[row]).all()}, "
          f"range {x[row].min():.4f} .. {x[row].max():.4f}")
    for group in ("i1_bid_v", "i1_ask_v"):
        start, end = F.GROUPS[group]
        values = x[row, start:end]
        print(f"  {group}: sum {values.sum():.10g}  "
              f"reduceat-1d {float(np.add.reduceat(x[row], F._STARTS)[F._IX[group]]):.10g}  "
              f"reduceat-2d {float(np.add.reduceat(x[row:row+1], F._STARTS, axis=1)[0, F._IX[group]]):.10g}")
 
    # --- 4. Does the disagreement depend on where the row sits? ---
    single = F._core_row(x[row])[0]
    in_block = F._core(x[row:row + 1])[0][0]
    bad = np.abs(single - np.asarray(in_block))
    print(f"\nSame row computed alone, batch vs row path: max diff {bad.max():.3e}")
    if bad.max() > 1e-9:
        for j in np.argsort(-bad)[:5]:
            print(f"  {F.STATELESS_NAMES[j]:<28} batch {in_block[j]:.10g}  row {single[j]:.10g}")
 
 
if __name__ == "__main__":
    main()