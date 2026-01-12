"""
Dense (E5 + FAISS) retriever core logic.

Usage (outside this module), e.g. in eval or FastAPI:

    from app.core.dense_retriever import load_dense_index, dense_search

    index, model, meta_df, chunk_df = load_dense_index(ROOT_DIR)
    results = dense_search("some query", index, model, meta_df, chunk_df, top_k=10)
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Literal, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings
from backend.app.core.s3_client import load_faiss_from_s3, load_parquet_from_s3


# S3 key mappings for each corpus
S3_PATHS = {
    "msmarco": {
        "index": "index/msmarco_e5_faiss.index",
        "meta": "embeddings/msmarco_e5_meta.parquet",
        "chunks": "processed/msmarco_passages_chunked.parquet",
    },
    "my_corpus": {
        "index": "index/my_corpus/personal_e5_faiss.index",
        "meta": "embeddings/my_corpus/personal_e5_meta.parquet",
        "chunks": "processed/my_corpus/personal_documents_chunked.parquet",
    },
}


def load_dense_index(
    root_dir: Optional[Path] = None,
    model_name: str = "intfloat/e5-base-v2",
    corpus: str = "msmarco",
    storage_mode: Optional[Literal["s3", "local"]] = None,
):
    
    """
    Load FAISS index, E5 model, metadata, and chunked text.

    Args:
        root_dir: Project root directory (the one containing `data/`). Required for local mode.
        model_name: Hugging Face / sentence-transformers model name.
        corpus: Which corpus to load ("msmarco" or "my_corpus").
        storage_mode: Where to load data from ("s3" or "local"). Defaults to settings.storage_mode.

    Returns:
        index: FAISS index over passage embeddings.
        model: E5 SentenceTransformer model for encoding queries.
        meta_df: DataFrame loaded from ms_marco_e5_meta.parquet (doc_id, chunk_id, etc.).
        chunk_df: DataFrame loaded from msmarco_passages_chunked.parquet (includes chunk_text).
    """
    # Use settings default if not specified
    if storage_mode is None:
        storage_mode = settings.storage_mode

    # Validate corpus
    if corpus not in S3_PATHS:
        raise ValueError(f"Unknown corpus: {corpus}. Must be one of: {list(S3_PATHS.keys())}")

    # S3 loading path
    if storage_mode == "s3":
        print(f"[dense_retriever] Loading {corpus} from S3...")
        s3_keys = S3_PATHS[corpus]

        index = load_faiss_from_s3(s3_keys["index"])
        meta_df = load_parquet_from_s3(s3_keys["meta"])
        chunk_df = load_parquet_from_s3(s3_keys["chunks"])

        print(f"Loading E5 model: {model_name}")
        model = SentenceTransformer(model_name)

        if len(meta_df) != index.ntotal:
            print(
                f"WARNING: meta_df has {len(meta_df)} rows but index.ntotal = {index.ntotal}."
            )

        return index, model, meta_df, chunk_df

    # Local file loading path (original behavior)
    if root_dir is None:
        raise ValueError("root_dir is required for local storage mode")

    if corpus == "msmarco":
        emb_dir = root_dir / "data" / "embeddings"
        index_dir = root_dir / "data" / "index"
        processed_dir = root_dir / "data" / "processed"

        index_path = index_dir / "msmarco_e5_faiss.index"
        meta_path = emb_dir / "msmarco_e5_meta.parquet"
        chunked_path = processed_dir / "msmarco_passages_chunked.parquet"

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


    elif corpus == "my_corpus":
        emb_dir = root_dir / "data" / "my_corpus" / "embeddings"
        index_dir = root_dir / "data" / "my_corpus" / "index"
        processed_dir = root_dir / "data" / "my_corpus" / "processed"

        index_path = index_dir / "personal_e5_faiss.index"
        meta_path = emb_dir / "personal_e5_meta.parquet"
        chunked_path = processed_dir / "personal_documents_chunked.parquet"

        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run scripts/build_faiss_index_my_corpus.py first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Run scripts/build_e5_embeddings_my_corpus.py first."
            )
        if not chunked_path.exists():
            raise FileNotFoundError(
                f"{chunked_path} not found. Run scripts/chunk_documents_my_corpus.py first."
            )

    else:
        raise ValueError(f"Unknown corpus: {corpus}")


    # emb_dir = root_dir / "data" / "embeddings"
    # index_dir = root_dir / "data" / "index"
    # processed_dir = root_dir / "data" / "processed"

    # index_path = index_dir / "msmarco_e5_faiss.index"
    # meta_path = emb_dir / "msmarco_e5_meta.parquet"
    # chunked_path = processed_dir / "msmarco_passages_chunked.parquet"

    # if not index_path.exists():
    #     raise FileNotFoundError(
    #         f"{index_path} not found. Run scripts/build_faiss_index.py first."
    #     )
    # if not meta_path.exists():
    #     raise FileNotFoundError(
    #         f"{meta_path} not found. Run scripts/build_e5_embeddings.py first."
    #     )
    # if not chunked_path.exists():
    #     raise FileNotFoundError(
    #         f"{chunked_path} not found. Run scripts/chunk_documents.py first."
    #     )

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"Loading dense meta from {meta_path}")
    meta_df = pd.read_parquet(meta_path)

    print(f"Loading chunked passages from {chunked_path}")
    chunk_df = pd.read_parquet(chunked_path)

    if len(meta_df) != index.ntotal:
        print(
            f"WARNING: meta_df has {len(meta_df)} rows but index.ntotal = {index.ntotal}."
        )

    print(f"Loading E5 model: {model_name}")
    model = SentenceTransformer(model_name)

    return index, model, meta_df, chunk_df


def _encode_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """
    Encode a query string using E5.

    E5 expects queries like: "query: <text>".
    Returns a (1, d) float32 numpy array, L2-normalised.
    """
    text = f"query: {query}"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def dense_search(
    query: str,
    index: faiss.Index,
    model: SentenceTransformer,
    meta_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Perform dense (E5 + FAISS) retrieval over the corpus.

    Args:
        query: Raw query string.
        index: FAISS index built over passage embeddings.
        model: E5 SentenceTransformer model.
        meta_df: DataFrame with at least ['doc_id', 'chunk_id', 'query_id', 'is_selected', 'url', 'set'].
        chunk_df: DataFrame with at least ['doc_id', 'chunk_id', 'chunk_text'].
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
    if len(meta_df) == 0:
        return []

    q_emb = _encode_query(model, query)  # shape (1, d)

    # Search
    scores, idxs = index.search(q_emb, top_k)
    scores = scores[0]
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []

    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        if idx < 0:
            continue

        meta_row = meta_df.iloc[int(idx)]
        chunk_row = chunk_df.iloc[int(idx)]

        result = {
            "doc_id": int(meta_row["doc_id"]),
            "score": float(score),
            "chunk_text": str(chunk_row["chunk_text"]),
            "chunk_id": int(meta_row["chunk_id"]),
            "query_id": meta_row.get("query_id", None),
            "url": meta_row.get("url", None),
            "is_selected": int(meta_row.get("is_selected", 0)),
            "set": meta_row.get("set", None),
        }
        results.append(result)

    return results
