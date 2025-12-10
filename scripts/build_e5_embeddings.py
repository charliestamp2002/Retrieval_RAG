from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
print(ROOT_DIR)
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EMB_DIR = ROOT_DIR / "data" / "embeddings"

EMB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/e5-base-v2"


def load_chunked_passages() -> pd.DataFrame: 
    in_path = DATA_PROCESSED_DIR / "msmarco_passages_chunked.parquet"

    if not in_path.exists(): 
        raise FileNotFoundError(f"Chunked passages file not found: {in_path}. Run scripts/chunk_documents.py first to create it.")
    
    df = pd.read_parquet(in_path)

    required_cols = ["doc_id", "chunk_id", "chunk_text", "query_id", "is_selected", "url", "set"]

    for c in required_cols: 
        if c not in df.columns: 
            raise ValueError(f"Missing required column in chunked passages DataFrame: {c}")
        
    return df

def build_e5_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:

    model = SentenceTransformer(MODEL_NAME)

    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc = "Building E5 Embeddings"): 
        batch_texts = texts[i:i+batch_size]
        batch_embs = model.encode(
            batch_texts, 
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        all_embs.append(batch_embs.astype(np.float32))
    
    return np.vstack(all_embs)


def main() -> None: 
    df = load_chunked_passages()
    texts = df["chunk_text"].astype(str).tolist()

    print(f"Encoding {len(texts):,} chunks with E5...")
    embeddings = build_e5_embeddings(texts, batch_size=64)
    print(f"Embeddings shape: {embeddings.shape}")

    emb_path = EMB_DIR / "msmarco_e5_embeddings.npy"
    meta_path = EMB_DIR / "msmarco_e5_meta.parquet"

    print("Saving embeddings to:", emb_path)
    np.save(emb_path, embeddings)

    # Save metadata: doc_id, chunk_id, plus any labels you care about
    meta_cols = ["doc_id", "chunk_id", "query_id", "is_selected", "url", "set"]
    existing_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[existing_cols].reset_index(drop=True)

    print(f"Saving metadata to {meta_path}")
    meta_df.to_parquet(meta_path, index=False)

    print("Done building E5 embeddings.")

if __name__ == "__main__": 
    main()

    
