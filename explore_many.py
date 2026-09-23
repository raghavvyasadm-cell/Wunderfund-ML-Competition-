import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N_SEQS = 500   # how many random sequences to check (change this one number to scale up/down)

pf = pq.ParquetFile("datasets/train.parquet")
a_cols = [f"a{i}" for i in range(8)]
targets = ["t0", "t1", "t_sum", "t_diff"]   # the two real targets plus the two new combinations

# Pick N_SEQS different sequences at random, but reproducibly
rng = np.random.default_rng(seed=42)
picks = rng.choice(pf.num_row_groups, size=N_SEQS, replace=False)

rows = []     # one entry per (sequence, feature) correlation
frames = []   # keep each sequence's data so we can pool them later

for g in picks:
    df = pf.read_row_group(int(g), columns=a_cols + ["t0", "t1"]).to_pandas()

    # NEW: split the targets into their two "ingredients"
    df["t_sum"] = df["t0"] + df["t1"]     # the "together" part
    df["t_diff"] = df["t0"] - df["t1"]    # the "seesaw" part

    frames.append(df)
    corr = df.corr()
    for col in a_cols:
        rows.append({"seq": int(g), "feature": col,
                     "corr_t0": corr.loc[col, "t0"],
                     "corr_t1": corr.loc[col, "t1"],
                     "corr_sum": corr.loc[col, "t_sum"],     # NEW
                     "corr_diff": corr.loc[col, "t_diff"]})  # NEW

results = pd.DataFrame(rows)
corr_cols = ["corr_t0", "corr_t1", "corr_sum", "corr_diff"]

# 1. Average correlation per feature across the sequences, and how much it varies
print(f"Correlation with targets across {N_SEQS} sequences (mean and spread):")
print(results.groupby("feature")[corr_cols].agg(["mean", "std"]).round(3))

# 2. How often is each correlation positive? (consistency check)
for c in corr_cols:
    results[c + "_pos"] = results[c] > 0
print("\nFraction of sequences where the correlation is positive:")
print(results.groupby("feature")[[c + "_pos" for c in corr_cols]].mean().round(2))

# 3. Pool all sequences together
all_df = pd.concat(frames)
print("\nCorrelation between the a-columns themselves:")
print(all_df[a_cols].corr().round(2))
print("\nt0 vs t1 correlation, pooled:", round(all_df["t0"].corr(all_df["t1"]), 3))

# 4. NEW: pooled correlation of each a-column with all four target versions
print("\nPooled correlation of each a-column with the targets:")
print(all_df[a_cols + targets].corr().loc[a_cols, targets].round(3))

# 5. NEW: how big is each ingredient? (spread of each target version, pooled)
print("\nSpread (std) of each target version, pooled:")
print(all_df[targets].std().round(3))