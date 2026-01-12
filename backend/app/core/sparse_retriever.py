"""
Sparse (TF-IDF) retriever core logic.

This module is intended to be used by the FastAPI layer, e.g.:

    from .sparse_retriever import load_tfidf_index, tfidf_search

    tfidf_index, meta_df = load_tfidf_index(ROOT_DIR)
    results = tfidf_search("some query", tfidf_index, meta_df, top_k=10)
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import pickle

from backend.app.core.config import settings
from backend.app.core.s3_client import load_pickle_from_s3, load_parquet_from_s3


# S3 key mappings for TF-IDF (MSMarco only)
S3_PATHS = {
    "tfidf_index": "embeddings/tfidf_index.pkl",
    "tfidf_meta": "embeddings/tfidf_meta.parquet",
    "chunks_fallback": "processed/msmarco_passages_chunked.parquet",
}

def tokenize(text: str) -> List[str]:
    """
    Naive tokenizer: lowercase + whitespace split.
    TODO: extend with better tokenisation / stopwords if needed.
    """
    return text.lower().split()


# Loading TF-IDF index + metadata

def load_tfidf_index(
    root_dir: Optional[Path] = None,
    storage_mode: Optional[Literal["s3", "local"]] = None,
) -> Tuple[Dict[str, Dict[int, float]], pd.DataFrame]:
    """
    Load TF-IDF index and associated metadata.

    Args:
        root_dir: Project root directory (the one containing `data/`). Required for local mode.
        storage_mode: Where to load data from ("s3" or "local"). Defaults to settings.storage_mode.

    Returns:
        tfidf_index: dict mapping term -> dict[doc_id -> tfidf_weight]
        meta_df: DataFrame with at least columns:
                 ['doc_id', 'chunk_id', 'chunk_text', 'query_id', 'is_selected', 'url', 'set']
    """
    # Use settings default if not specified
    if storage_mode is None:
        storage_mode = settings.storage_mode

    # S3 loading path
    if storage_mode == "s3":
        print("[sparse_retriever] Loading TF-IDF from S3...")

        tfidf_index = load_pickle_from_s3(S3_PATHS["tfidf_index"])

        # Try to load tfidf_meta, fall back to chunked passages
        try:
            meta_df = load_parquet_from_s3(S3_PATHS["tfidf_meta"])
        except Exception:
            print("[S3] tfidf_meta not found, falling back to chunked passages")
            meta_df = load_parquet_from_s3(S3_PATHS["chunks_fallback"])

        return tfidf_index, meta_df

    # Local file loading path (original behavior)
    if root_dir is None:
        raise ValueError("root_dir is required for local storage mode")

    emb_dir = root_dir / "data" / "embeddings"
    processed_dir = root_dir / "data" / "processed"

    tfidf_index_path = emb_dir / "tfidf_index.pkl"
    tfidf_meta_path = emb_dir / "tfidf_meta.parquet"
    chunked_path = processed_dir / "msmarco_passages_chunked.parquet"

    if not tfidf_index_path.exists():
        raise FileNotFoundError(
            f"TF-IDF index file not found: {tfidf_index_path}. "
            f"Run scripts/build_tfidf.py first."
        )

    # Load TF-IDF index
    with open(tfidf_index_path, "rb") as f:
        tfidf_index = pickle.load(f)

    if tfidf_meta_path.exists():
        meta_df = pd.read_parquet(tfidf_meta_path)
    else:
        if not chunked_path.exists():
            raise FileNotFoundError(
                f"Neither {tfidf_meta_path} nor {chunked_path} found. "
                "You need ms_marco_passages_chunked.parquet for metadata."
            )
        meta_df = pd.read_parquet(chunked_path)

    return tfidf_index, meta_df

# Core TF-IDF search

def _build_query_tfidf(
    query: str,
    tfidf_index: Dict[str, Dict[int, float]],
    total_documents: int,
) -> Dict[str, float]:
    """
    Build a sparse TF-IDF vector for the query, as a dict: term -> weight.
    Uses the same IDF formula as build_tfidf.py.
    """
    tokens = tokenize(query)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {}

    # Term frequency in query
    term_counts: Dict[str, int] = {}
    for token in tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    query_tfidf: Dict[str, float] = {}
    for term, count in term_counts.items():
        tf = float(count) / float(n_tokens)
        docs_with_term = len(tfidf_index.get(term, {}))
        if docs_with_term == 0:
            # Term not in any document -> contributes nothing
            continue
        # Same IDF as in build_tfidf_index
        idf = float(np.log(float(total_documents) / (1.0 + float(docs_with_term))))
        query_tfidf[term] = tf * idf

    return query_tfidf


def tfidf_search(
    query: str,
    tfidf_index: Dict[str, Dict[int, float]],
    meta_df: pd.DataFrame,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Perform a TF-IDF search over the corpus.

    Args:
        query: Raw query string.
        tfidf_index: dict(term -> dict[doc_id -> tfidf_weight]).
        meta_df: DataFrame with at least doc_id + chunk_text + metadata.
        top_k: Number of top results to return.

    Returns:
        List of result dicts, sorted by descending score, each like:
        {
            "doc_id": int,
            "score": float,
            "chunk_text": str,
            "chunk_id": int,
            "query_id": ...,
            "url": ...,
            "is_selected": ...,
        }
    """
    # Number of documents (we assume each row in meta_df is one "doc" / chunk)
    total_documents = int(len(meta_df))

    # Build query TF-IDF vector
    query_tfidf = _build_query_tfidf(query, tfidf_index, total_documents)
    if not query_tfidf:
        return []

    # Precompute query magnitude
    query_magnitude_sq = sum(weight * weight for weight in query_tfidf.values())
    query_magnitude = float(np.sqrt(query_magnitude_sq)) if query_magnitude_sq > 0 else 0.0

    if query_magnitude == 0.0:
        return []

    # Collect candidate doc_ids: only docs that share at least one query term
    candidate_doc_ids = set()
    for term in query_tfidf.keys():
        candidate_doc_ids.update(tfidf_index.get(term, {}).keys())

    if not candidate_doc_ids:
        return []

    scores: Dict[int, float] = {}

    # Compute cosine similarity for each candidate doc
    for doc_id in candidate_doc_ids:
        dot_product = 0.0
        doc_magnitude_sq = 0.0

        for term, q_weight in query_tfidf.items():
            doc_weight = tfidf_index.get(term, {}).get(doc_id, 0.0)
            dot_product += q_weight * doc_weight
            doc_magnitude_sq += doc_weight * doc_weight

        doc_magnitude = float(np.sqrt(doc_magnitude_sq)) if doc_magnitude_sq > 0 else 0.0

        if doc_magnitude == 0.0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)

    if not scores:
        return []

    # Select top_k doc_ids
    # Turn dict into list of (doc_id, score), sort descending by score
    sorted_doc_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: List[Dict[str, Any]] = []

    # For each doc_id, pull corresponding row from meta_df and build result dict
    for doc_id, score in sorted_doc_scores:
        # We assume doc_id is unique in meta_df; if not, you may need a different join.
        row = meta_df[meta_df["doc_id"] == doc_id].iloc[0]

        result = {
            "doc_id": int(doc_id),
            "score": float(score),
            "chunk_text": str(row["chunk_text"]),
            "chunk_id": int(row["chunk_id"]),
            "query_id": row.get("query_id", None),
            "url": row.get("url", None),
            "is_selected": int(row.get("is_selected", 0)),
            "set": row.get("set", None),
        }
        results.append(result)

    return results
