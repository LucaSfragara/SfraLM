import pyarrow.parquet as pq


def read_parquet_examples():
    """ Example code snippets for reading parquet files using pyarrow. """
    # Read entire file (only "text" column)
    table = pq.read_table("./data/base_data/shard_00114.parquet", columns=["text"])
    texts = table["text"].to_pylist()
    print(texts)
    
if __name__ == "__main__":
    read_parquet_examples()