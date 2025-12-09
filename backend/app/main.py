#  ROUGH SKETCH OF A FASTAPI APPLICATION WITH A TF-IDF SEARCH ENDPOINT
#  NOT FINISHED...

from fastapi import FastAPI, Query
from pathlib import Path

from app.core.sparse_retriever import load_tfidf_index, tfidf_search

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[2]
tfidf_index, tfidf_meta = load_tfidf_index(ROOT_DIR)


@app.get("/search/tfidf")
def search_tfidf_endpoint(q: str = Query(..., alias="query"), k: int = 10):
    results = tfidf_search(q, tfidf_index, tfidf_meta, top_k=k)
    return {"query": q, "results": results}


