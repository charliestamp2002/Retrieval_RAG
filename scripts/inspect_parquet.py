import pandas as pd

PROCESSED_FILE = "data/processed/msmarco_passages_chunked.parquet"
EMBEDDING_FILE = "data/embeddings/tfidf_meta.parquet"

df = pd.read_parquet(EMBEDDING_FILE)

print(df.head(15))  # First 5 rows
# print(df.info())  # Column names, types, non-null counts
# print(df.shape)   # Number of rows and columns
# print(df.dtypes)  # Data types of each column
# print(df.describe())  # Statistical summary