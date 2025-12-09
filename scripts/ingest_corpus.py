import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_msmarco_passages(max_passages: int = None) -> pd.DataFrame:

    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # if max_passages is not None:
    #     ds = ds.select(range(max_passages))

    column_names = ds.column_names
    print("Column names in the dataset:", column_names)
    # output of above: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers']

    records = []
    total_passages = 0

    for row in tqdm(ds, desc = 'Flattening Passages'): 

        query_id = row['query_id']
        passages = row['passages']

        texts = passages.get("passage_text", [])
        is_selected_list = passages.get("is_selected", [])
        passage_urls = passages.get("url", [])

        for idx, (p_text, p_sel, p_url) in enumerate(zip(texts, is_selected_list, passage_urls)):
            records.append(
                {
                "query_id": query_id,
                "passage_index": idx,
                "passage_text": p_text,
                "is_selected": p_sel,
                "passage_url": p_url,
                "set": "msmarco_train"
                }
            )

            
            total_passages += 1

            if max_passages is not None and total_passages >= max_passages:
                break 

        if max_passages is not None and total_passages >= max_passages:
                break

    df = pd.DataFrame.from_records(records)
    print(f"Flattened to {len(df):,} passage rows.")

    df = df.reset_index(drop=True)
    df["doc_id"] = df.index.astype(int)

    df = df[["doc_id", "query_id", "passage_index", "passage_text", "passage_url", "is_selected", "set"]]
    return df

def main():

    max_passages_env = os.environ.get("MSMARCO_MAX_PASSAGES", None)
    max_passages = int(max_passages_env) if max_passages_env is not None else 10000

    df = load_msmarco_passages(max_passages=max_passages)
    out_path = DATA_PROCESSED_DIR / "msmarco_passages.parquet"
    print(f"Saving {len(df):,} passages to {out_path}")
    df.to_parquet(out_path, index=False)
    print("Finished saving.")
 
    
if __name__ == "__main__":
    main()




