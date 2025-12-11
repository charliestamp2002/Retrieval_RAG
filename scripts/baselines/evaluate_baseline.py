import pandas as pd
from pathlib import Path   

def build_query_to_passage_relevance(df: pd.DataFrame) -> dict:

    query_to_passages = {}

    for _, row in df.iterrows(): 

        query_id = row["query_id"]
        query_text = row["query_text"]
        passage_id = row["passage_id"]
        passage_text = row["passage_text"]
        is_selected = row["is_selected"]

        if query_text not in query_to_passages:
            query_to_passages[query_text] = {}
            query_to_passages[query_text][passage_text] = is_selected
    
    return query_to_passages

def main(): 

    DATA_PROCESSED_DIR = Path("data/processed")
    in_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"

    df = pd.read_parquet(in_path)
    print(df.columns)
    
    query_to_passages = build_query_to_passage_relevance(df)
    print(dict(list(query_to_passages.items())[:3]))
    # print(list(query_to_passages)[:5])

if __name__ == "__main__":  
    main()



        



