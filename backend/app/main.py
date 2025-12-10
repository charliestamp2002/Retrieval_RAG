#  ROUGH SKETCH OF A FASTAPI APPLICATION WITH A TF-IDF SEARCH ENDPOINT
#  NOT FINISHED...

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from backend.app.core.sparse_retriever import load_tfidf_index, tfidf_search
from backend.app.core.dense_retriever import load_dense_index, dense_search

class SearchResult(BaseModel):
    doc_id: int
    chunk_id: int
    score: float
    chunk_text: str
    query_id: Optional[int] = None
    url: Optional[str] = None
    is_selected: Optional[int] = None
    set: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    method: str
    k: int
    results: List[SearchResult]


class CombinedSearchResponse(BaseModel):
    query: str
    k: int
    tfidf_results: List[SearchResult]
    dense_results: List[SearchResult]


app = FastAPI(title="Retrieval RAG API", version="0.1.0")

@app.on_event("startup")
def load_indices_on_startup() -> None:
    """
    Load TF-IDF and dense E5 indices + metadata at app startup.
    """
    root_dir = Path(__file__).resolve().parents[2]  # backend/app -> project root

    # Load sparse / TF-IDF artifacts
    print("[startup] Loading TF-IDF index + metadata...")
    tfidf_index, tfidf_meta_df = load_tfidf_index(root_dir)
    app.state.tfidf_index = tfidf_index
    app.state.tfidf_meta_df = tfidf_meta_df

    # Load dense / E5 + FAISS artifacts
    print("[startup] Loading dense E5 + FAISS index + metadata...")
    dense_index, dense_model, dense_meta_df, dense_chunk_df = load_dense_index(root_dir)
    app.state.dense_index = dense_index
    app.state.dense_model = dense_model
    app.state.dense_meta_df = dense_meta_df
    app.state.dense_chunk_df = dense_chunk_df

    print("[startup] All indices loaded.")

@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """
    Simple healthcheck endpoint.
    """
    return {"status": "ok"}

@app.get("/search", response_model=Any)
def search_endpoint(
    query: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    method: str = Query("dense", regex="^(tfidf|dense|both)$"),
):
    """
    Unified search endpoint.

    Params:
      - query: user query string
      - k: number of results
      - method: "tfidf", "dense", or "both"
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    method = method.lower()

    # Sparse search
    if method == "tfidf":
        tfidf_index = app.state.tfidf_index
        tfidf_meta_df = app.state.tfidf_meta_df

        raw_results = tfidf_search(
            query=query,
            tfidf_index=tfidf_index,
            meta_df=tfidf_meta_df,
            top_k=k,
        )
        results = [SearchResult(**r) for r in raw_results]
        return SearchResponse(query=query, method="tfidf", k=k, results=results)

    # Dense search
    if method == "dense":
        dense_index = app.state.dense_index
        dense_model = app.state.dense_model
        dense_meta_df = app.state.dense_meta_df
        dense_chunk_df = app.state.dense_chunk_df

        raw_results = dense_search(
            query=query,
            index=dense_index,
            model=dense_model,
            meta_df=dense_meta_df,
            chunk_df=dense_chunk_df,
            top_k=k,
        )
        results = [SearchResult(**r) for r in raw_results]
        return SearchResponse(query=query, method="dense", k=k, results=results)

    # Both: sparse + dense
    if method == "both":
        tfidf_index = app.state.tfidf_index
        tfidf_meta_df = app.state.tfidf_meta_df

        dense_index = app.state.dense_index
        dense_model = app.state.dense_model
        dense_meta_df = app.state.dense_meta_df
        dense_chunk_df = app.state.dense_chunk_df

        raw_sparse = tfidf_search(
            query=query,
            tfidf_index=tfidf_index,
            meta_df=tfidf_meta_df,
            top_k=k,
        )
        raw_dense = dense_search(
            query=query,
            index=dense_index,
            model=dense_model,
            meta_df=dense_meta_df,
            chunk_df=dense_chunk_df,
            top_k=k,
        )

        sparse_results = [SearchResult(**r) for r in raw_sparse]
        dense_results = [SearchResult(**r) for r in raw_dense]

        return CombinedSearchResponse(
            query=query,
            k=k,
            tfidf_results=sparse_results,
            dense_results=dense_results,
        )

    # Should not reach here because of regex in Query
    raise HTTPException(status_code=400, detail="Unsupported method.")



# app = FastAPI()

# ROOT_DIR = Path(__file__).resolve().parents[2]
# tfidf_index, tfidf_meta = load_tfidf_index(ROOT_DIR)


# @app.get("/search/tfidf")
# def search_tfidf_endpoint(q: str = Query(..., alias="query"), k: int = 10):
#     results = tfidf_search(q, tfidf_index, tfidf_meta, top_k=k)
#     return {"query": q, "results": results}


