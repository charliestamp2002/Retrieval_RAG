from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
CORPUS_PROCESSED_DIR = ROOT_DIR / "data" / "my_corpus" / "processed"
EMB_DIR = ROOT_DIR / "data" / "my_corpus" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/e5-base-v2"


def load_chunks() -> pd.DataFrame:
    in_path = CORPUS_PROCESSED_DIR / "personal_documents_chunked.parquet"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run scripts/chunk_my_corpus.py first."
        )
    df = pd.read_parquet(in_path)
    return df


def build_e5_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    print(f"Loading E5 model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding personal chunks"):
        batch = [f"passage: {t}" for t in texts[i : i + batch_size]]
        e = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embs.append(e.astype(np.float32))
    return np.vstack(embs)


def main() -> None:
    df = load_chunks()
    texts = df["chunk_text"].astype(str).tolist()

    embeddings = build_e5_embeddings(texts, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")

    emb_path = EMB_DIR / "personal_e5_embeddings.npy"
    meta_path = EMB_DIR / "personal_e5_meta.parquet"

    np.save(emb_path, embeddings)

    meta_cols = ["doc_id", "chunk_id", "source_path", "title", "doc_type"]
    meta_df = df[meta_cols].reset_index(drop=True)
    meta_df.to_parquet(meta_path, index=False)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved meta to {meta_path}")

if __name__ == "__main__":
    main()