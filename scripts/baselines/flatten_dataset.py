"""
This script flattens a nested dataset structure into a single-level dataset.
In this case we are flattening MSmarco dataset into the form: 
QUERY_ID|QUERY|PASSAGE_ID|PASSAGE|IS_SELECTED
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
print(f"Root dir: {ROOT_DIR}")
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_msmarco_passages(max_passages: int = None) -> pd.DataFrame:

    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    print(ds)

    column_names = ds.column_names
    print("Column names in the dataset:", column_names)

    records = []
    total_passages = 0

    for row in tqdm(ds, desc = "flattening passages"): 
        query_id = row["query_id"]
        passages = row["passages"]
        query_text = row["query"]
        for is_selected, passage_text in zip(passages["is_selected"], passages["passage_text"]):
            records.append(
                {"query_id": query_id,
                 "passage_id": total_passages,
                 "query_text": query_text,
                 "passage_text": passage_text,
                 "is_selected": is_selected,
                 "set": "msmarco_train"
                 }
            )
            total_passages += 1
            
            if max_passages is not None and total_passages >= max_passages:
                break
        if max_passages is not None and total_passages >= max_passages:
            break

    df = pd.DataFrame.from_records(records)

    return df




def main(): 
    df = load_msmarco_passages(max_passages=5000)
    out_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved flattened dataset to {out_path}")

if  __name__ == "__main__":
    main()
