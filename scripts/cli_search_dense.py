#!/usr/bin/env python
"""
Simple CLI to test dense retrieval with E5 + FAISS.

Usage:
    python scripts/cli_search_dense.py "your query here"

If no query is passed, it will prompt for one.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EMB_DIR = ROOT_DIR / "data" / "embeddings"
INDEX_DIR = ROOT_DIR / "data" / "index"

MODEL_NAME = "intfloat/e5-base-v2"


def load_dense_artifacts():
    """
    Load FAISS index + E5 model + metadata and chunked text.
    """
    index_path = INDEX_DIR / "msmarco_e5_faiss.index"
    meta_path = EMB_DIR / "msmarco_e5_meta.parquet"
    chunked_path = DATA_PROCESSED_DIR / "msmarco_passages_chunked.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Run scripts/build_faiss_index.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run scripts/build_e5_embeddings.py first."
        )
    if not chunked_path.exists():
        raise FileNotFoundError(
            f"{chunked_path} not found. Run scripts/chunk_documents.py first."
        )

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"Loading meta from {meta_path}")
    meta_df = pd.read_parquet(meta_path)

    print(f"Loading chunked passages from {chunked_path}")
    chunk_df = pd.read_parquet(chunked_path)

    # Sanity check: same number of rows
    if len(meta_df) != index.ntotal:
        print(
            f"WARNING: meta_df has {len(meta_df)} rows but index.ntotal = {index.ntotal}"
        )

    print(f"Loading E5 model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    return index, model, meta_df, chunk_df


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """
    Encode a query string using E5-base-v2.

    E5 expects query inputs like: "query: <text>".
    Returns a 2D numpy array of shape (1, d) with L2-normalized embeddings.
    """
    text = f"query: {query}"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product ~= cosine
    )
    return emb.astype(np.float32)


def dense_search(
    query: str,
    index: faiss.Index,
    model: SentenceTransformer,
    meta_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run dense retrieval for a single query and return top_k results with metadata.
    """
    print(f"\nEncoding query: {query}")
    q_emb = encode_query(model, query)  # shape (1, d)

    print("Searching FAISS index...")
    scores, idxs = index.search(q_emb, top_k)  # scores: (1, k), idxs: (1, k)
    scores = scores[0]
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []

    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        if idx < 0:
            # FAISS may return -1 for empty index or similar; guard just in case
            continue

        # Row idx should correspond to this embedding
        meta_row = meta_df.iloc[int(idx)]
        chunk_row = chunk_df.iloc[int(idx)]

        result = {
            "rank": rank,
            "score": float(score),
            "doc_id": int(meta_row["doc_id"]),
            "chunk_id": int(meta_row["chunk_id"]),
            "chunk_text": str(chunk_row["chunk_text"]),
            "query_id": meta_row.get("query_id", None),
            "url": meta_row.get("url", None),
            "is_selected": int(meta_row.get("is_selected", 0)),
            "set": meta_row.get("set", None),
        }
        results.append(result)

    return results


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a query: ").strip()
        if not query:
            print("No query provided, exiting.")
            return

    index, model, meta_df, chunk_df = load_dense_artifacts()

    top_k = 5
    results = dense_search(
        query=query,
        index=index,
        model=model,
        meta_df=meta_df,
        chunk_df=chunk_df,
        top_k=top_k,
    )

    print(f"\nTop {top_k} results:")
    for r in results:
        print("=" * 80)
        print(
            f"Rank {r['rank']} | score={r['score']:.4f} "
            f"| doc_id={r['doc_id']} | chunk_id={r['chunk_id']} "
            f"| is_selected={r['is_selected']}"
        )
        print(r["chunk_text"][:400], "...")


if __name__ == "__main__":
    main()
