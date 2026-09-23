import pyarrow.parquet as pq
import matplotlib.pyplot as plt

df = pq.ParquetFile("datasets/train.parquet").read_row_group(0).to_pandas()

# Build lists of column names so we don't have to type them all out
a_cols = [f"a{i}" for i in range(8)]                                                  # a0 ... a7
i0_book = [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)]           # i0 prices + volumes
i1_book = [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]           # i1 prices + volumes

# 1. Basic summary of the a-columns
print(df[a_cols].describe())

# 2. How many distinct values does each a-column take?
print("\nDistinct values per column:")
print(df[a_cols].nunique())

# 3. How often does each column change from one row to the next?
print("\nFraction of rows where the value changes:")
print((df[a_cols + ["i0_p0", "i0_v0", "t0", "t1"]].diff() != 0).mean())

# 4. Your observation: do the a-columns change even when i0's book doesn't?
i0_changed = (df[i0_book].diff() != 0).any(axis=1)
i1_changed = (df[i1_book].diff() != 0).any(axis=1)
a_changed = df[a_cols].diff() != 0

print("\nFraction of rows where i0's book changed at all:", i0_changed.mean())
print("On rows where i0's book did NOT change:")
print("  ...fraction where i1's book changed:", i1_changed[~i0_changed].mean())
print("  ...fraction where each a-column changed:")
print(a_changed[~i0_changed].mean())

# 5. Simple correlation between each a-column and the targets
print("\nCorrelation with targets:")
print(df[a_cols + ["t0", "t1"]].corr()[["t0", "t1"]].round(3))

# 6. Picture: all eight a-columns across the whole sequence
fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
for i, col in enumerate(a_cols):
    axes[i].plot(df["step_in_seq"], df[col], linewidth=0.5)
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()