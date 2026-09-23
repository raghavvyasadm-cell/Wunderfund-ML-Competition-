import pyarrow.parquet as pq

# Open the training file (this doesn't load everything into memory)
dataset = pq.ParquetFile("datasets/train.parquet")
print("Number of sequences:", dataset.num_row_groups)

# Load just the first sequence and turn it into a pandas DataFrame
table = dataset.read_row_group(0)
df = table.to_pandas()

print("Shape (rows, columns):", df.shape)
print("Column names:", df.columns.tolist())
print(df.head())       # the first 5 rows
print(df.describe())   # summary statistics for every column