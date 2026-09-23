import pyarrow.parquet as pq
import matplotlib.pyplot as plt

df = pq.ParquetFile("datasets/train.parquet").read_row_group(0).to_pandas()

# 1. Which target values occur, and how often? (top 10)
print(df["t0"].value_counts().head(10))

# 2. What fraction of rows have a target of exactly zero?
print("Fraction t0 == 0:", (df["t0"] == 0).mean())
print("Fraction t1 == 0:", (df["t1"] == 0).mean())

# 3. How often does t0 change from one row to the next?
print("Fraction of rows where t0 changes:", (df["t0"].diff() != 0).mean())

# 4. How common is the suspicious a7 value?
print(df["a7"].value_counts().head(5))

# 5. A picture: one price feature and t0 across the whole sequence
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(df["step_in_seq"], df["i0_p0"])
axes[0].set_title("i0_p0 (a price-like feature)")
axes[1].plot(df["step_in_seq"], df["t0"])
axes[1].set_title("t0 (target)")
plt.tight_layout()
plt.show()