import pyarrow.parquet as pq
import os

def read_parquet_examples():
    """ Example code snippets for reading parquet files using pyarrow. """
    # Read entire file (only "text" column)
    print(os.listdir("."))
    table = pq.read_table("datasets/base_data/shard_00114.parquet", columns=["text"])
    texts = table["text"].to_pylist()
    print(texts)
    
if __name__ == "__main__":
    read_parquet_examples()