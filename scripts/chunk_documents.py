from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"    

def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:

    words = text.split()
    if not text: 
        return []
    
    chunks = []
    start = 0
    n = len(words)

    while start < n: 
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n: 
            break
        start = max(0, end - overlap)
    
    return chunks


def main(): 
    in_path = ROOT_FILE = DATA_PROCESSED_DIR / "msmarco_passages.parquet"
    out_path = ROOT_FILE = DATA_PROCESSED_DIR / "msmarco_passages_chunked.parquet"
    
    if not in_path.exists(): 
        raise FileNotFoundError(f"Input file not found: {in_path}. Run scripts/ingest_corpus.py first to create it.")
    
    df = pd.read_parquet(in_path)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Passages"):
        doc_id = int(row["doc_id"])
        text = str(row["passage_text"])
        chunks = chunk_text_simple(text)

        for chunk_idx, chunk_text in enumerate(chunks):

            records.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "chunk_text": chunk_text,
                    "query_id": row["query_id"],
                    "is_selected": row["is_selected"],
                    "url": row.get("passage_url", None),
                    "set": row.get("set", "msmarco_train") 
                }
            )

    chunked_df = pd.DataFrame.from_records(records)
    print(f"Chunked to {len(chunked_df):,} passage chunks. Saving to {out_path}...")
    chunked_df.to_parquet(out_path, index=False)
    print("Finished.")


if __name__ == "__main__":
    main()
           