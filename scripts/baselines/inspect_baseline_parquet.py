import pandas as pd
from pathlib import Path

DATA_PROCESSED_DIR = Path("data/processed")

def main(): 

    in_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"
    df = pd.read_parquet(in_path)
    print(df.head(3))

if __name__ == "__main__":  
    main()