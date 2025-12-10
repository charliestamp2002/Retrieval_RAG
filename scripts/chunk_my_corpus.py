from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
CORPUS_PROCESSED_DIR = ROOT_DIR / "data" / "my_corpus" / "processed"
CORPUS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
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

def main() -> None: 
    in_path = CORPUS_PROCESSED_DIR / "personal_documents.parquet"
    out_path = CORPUS_PROCESSED_DIR / "personal_documents_chunked.parquet"

    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run scripts/ingest_my_corpus.py first."
        )
    
    df = pd.read_parquet(in_path)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        doc_id = int(row["doc_id"])
        text = str(row["raw_text"])
        chunks = chunk_text_simple(text, chunk_size=500, overlap=50)

        for chunk_id, chunk_text in enumerate(chunks): 
            records.append(
                {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "source_path" : row["source_path"],
                "title": row["title"],
                "doc_type": row["doc_type"],
                }
            )
    chunk_df = pd.DataFrame.from_records(records)
    chunk_df.to_parquet(out_path, index=False)
    print(f"Chunked documents saved to {out_path}.")

if __name__ == "__main__":  
    main()





       


    
