#  ROUGH SKETCH OF A FASTAPI APPLICATION WITH A TF-IDF SEARCH ENDPOINT
#  NOT FINISHED...

from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.core.sparse_retriever import load_tfidf_index, tfidf_search
from backend.app.core.dense_retriever import load_dense_index, dense_search

from backend.app.core.reranker import load_reranker, rerank as rerank_fn
from backend.app.core.rag import build_context_from_chunks, generate_answer_hf

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

class RagChunk(BaseModel): 
    doc_id: int
    chunk_id: int
    score: float
    chunk_text: str
    source_path: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    set: Optional[str] = None

class RagRequest(BaseModel):
    query: str
    k: int = 5
    corpus: Literal["msmarco", "my_corpus"] = "my_corpus"
    method: Literal["dense", "rerank"] = "rerank"

class RagResponse(BaseModel):
    query: str
    corpus: str
    method: str
    k: int
    answer: str
    chunks: List[RagChunk]


app = FastAPI(title="Retrieval RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_indices_on_startup() -> None:
    """
    Load TF-IDF and dense E5 indices + metadata at app startup.
    """
    root_dir = Path(__file__).resolve().parents[2]  # backend/app -> project root

    # Load sparse / TF-IDF artifacts (only MSMarco for now)
    print("[startup] Loading TF-IDF index + metadata...")
    tfidf_index, tfidf_meta_df = load_tfidf_index(root_dir)
    app.state.tfidf_index = tfidf_index
    app.state.tfidf_meta_df = tfidf_meta_df

    # Load dense / E5 + FAISS artifacts (MSMarco)
    print("[startup] Loading dense E5 + FAISS index + metadata (msmarco)...")
    ms_index, ms_model, ms_meta_df, ms_chunk_df = load_dense_index(root_dir, corpus="msmarco")
    app.state.dense_msmarco = (ms_index, ms_model, ms_meta_df, ms_chunk_df)
    # app.state.dense_index = dense_index
    # app.state.dense_model = dense_model
    # app.state.dense_meta_df = dense_meta_df
    # app.state.dense_chunk_df = dense_chunk_df

    # Load dense / E5 + FAISS artifacts (my_corpus)
    print("[startup] Loading dense E5 + FAISS index + metadata (my_corpus)...")
    my_index, my_model, my_meta_df, my_chunk_df = load_dense_index(root_dir, corpus="my_corpus")
    app.state.dense_my_corpus = (my_index, my_model, my_meta_df, my_chunk_df)

    app.state.reranker_model = load_reranker()

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
    corpus: str = Query("my_corpus", regex="^(my_corpus|msmarco)$")
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

        if corpus != "msmarco":
            raise HTTPException(status_code=400, detail="TF-IDF search only supports 'msmarco' corpus.")
        
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
        if corpus == "msmarco":
            app.state.dense_index, app.state.dense_model, app.state.dense_meta_df, app.state.dense_chunk_df = app.state.dense_msmarco
        elif corpus == "my_corpus":
            app.state.dense_index, app.state.dense_model, app.state.dense_meta_df, app.state.dense_chunk_df = app.state.dense_my_corpus
        else: 
            raise HTTPException(status_code=400, detail=f"Unsupported corpus: {corpus}")
        
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

        raw_sparse = tfidf_search(
            query=query,
            tfidf_index=tfidf_index,
            meta_df=tfidf_meta_df,
            top_k=k,
        )

        if corpus == "msmarco":
            app.state.dense_index, app.state.dense_model, app.state.dense_meta_df, app.state.dense_chunk_df = app.state.dense_msmarco
        elif corpus == "my_corpus":
            app.state.dense_index, app.state.dense_model, app.state.dense_meta_df, app.state.dense_chunk_df = app.state.dense_my_corpus
        else: 
            raise HTTPException(status_code=400, detail=f"Unsupported corpus: {corpus}")

        dense_index = app.state.dense_index
        dense_model = app.state.dense_model
        dense_meta_df = app.state.dense_meta_df
        dense_chunk_df = app.state.dense_chunk_df

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

@app.post("/rag", response_model=RagResponse)
def rag_endpoint(body: RagRequest):
    """
    RAG endpoint:
      1. Retrieve chunks using dense or dense+rerank.
      2. Build a context string.
      3. Generate an answer.
    """

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    corpus = body.corpus
    method = body.method
    k = body.k

    # 1) Choose dense artifacts for the corpus
    if corpus == "msmarco":
        dense_index, dense_model, dense_meta_df, dense_chunk_df = app.state.dense_msmarco
    elif corpus == "my_corpus":
        dense_index, dense_model, dense_meta_df, dense_chunk_df = app.state.dense_my_corpus
    else:
        raise HTTPException(status_code=400, detail=f"Unknown corpus: {corpus}")

    # 2) Retrieve candidates
    if method == "dense":
        # directly use dense retrieval
        raw_results = dense_search(
            query=query,
            index=dense_index,
            model=dense_model,
            meta_df=dense_meta_df,
            chunk_df=dense_chunk_df,
            top_k=k,
        )

    elif method == "rerank":
        # dense -> candidate pool, then rerank
        candidate_k = max(k, 50)  # a bit deeper than final k
        dense_candidates = dense_search(
            query=query,
            index=dense_index,
            model=dense_model,
            meta_df=dense_meta_df,
            chunk_df=dense_chunk_df,
            top_k=candidate_k,
        )

        reranker_model = app.state.reranker_model
        raw_results = rerank_fn(
            query=query,
            candidates=dense_candidates,
            model=reranker_model,
            top_k=k,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="RAG currently supports method='dense' or 'rerank' only.",
        )

    # 3) Build context for answer generation
    # Use rerank_score if available, else fall back to score
    for r in raw_results:
        if "rerank_score" in r:
            r["score"] = float(r["rerank_score"])

    context = build_context_from_chunks(raw_results, max_chars=4000)

    # 4) Generate answer (stub for now)
    answer = generate_answer_hf(query, context)

    # 5) Convert to RagChunk models
    rag_chunks: List[RagChunk] = []
    for r in raw_results:
        rag_chunks.append(
            RagChunk(
                doc_id=int(r["doc_id"]),
                chunk_id=int(r["chunk_id"]),
                score=float(r.get("score", 0.0)),
                chunk_text=str(r["chunk_text"]),
                source_path=r.get("source_path"),
                title=r.get("title"),
                url=r.get("url"),
                set=r.get("set"),
            )
        )

    return RagResponse(
        query=query,
        corpus=corpus,
        method=method,
        k=k,
        answer=answer,
        chunks=rag_chunks,
    )




# app = FastAPI()

# ROOT_DIR = Path(__file__).resolve().parents[2]
# tfidf_index, tfidf_meta = load_tfidf_index(ROOT_DIR)


# @app.get("/search/tfidf")
# def search_tfidf_endpoint(q: str = Query(..., alias="query"), k: int = 10):
#     results = tfidf_search(q, tfidf_index, tfidf_meta, top_k=k)
#     return {"query": q, "results": results}


