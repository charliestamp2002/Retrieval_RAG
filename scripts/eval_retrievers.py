#!/usr/bin/env python
"""
Evaluate sparse (TF-IDF) vs dense (E5 + FAISS) retrievers on MS MARCO subset.

Metrics:
  - Recall@k
  - MRR@k
  - nDCG@k
  - Hit@k
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
print(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

from typing import Dict, Any, List, Set, Tuple

import math
import numpy as np
import pandas as pd
from datasets import load_dataset

from backend.app.core.sparse_retriever import load_tfidf_index, tfidf_search
from backend.app.core.dense_retriever import load_dense_index, dense_search
from backend.app.core.reranker import load_reranker, rerank


def recall_at_k(ranked_doc_ids: List[int], relevant_docs: Set[int], k: int) -> float:
    if not relevant_docs:
        return 0.0
    topk = ranked_doc_ids[:k]
    hits = len(set(topk) & relevant_docs)
    return hits / float(len(relevant_docs))


def reciprocal_rank_at_k(
    ranked_doc_ids: List[int], relevant_docs: Set[int], k: int
) -> float:
    topk = ranked_doc_ids[:k]
    for i, doc_id in enumerate(topk):
        if doc_id in relevant_docs:
            return 1.0 / float(i + 1)  # rank is 1-based
    return 0.0


def hit_at_k(ranked_doc_ids: List[int], relevant_docs: Set[int], k: int) -> float:
    topk = ranked_doc_ids[:k]
    return 1.0 if any(doc_id in relevant_docs for doc_id in topk) else 0.0


def ndcg_at_k(ranked_doc_ids: List[int], relevant_docs: Set[int], k: int) -> float:
    def dcg(rel_list: List[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)  # i starts at 0 -> log2(i+2)
            for i, rel in enumerate(rel_list)
        )

    topk = ranked_doc_ids[:k]
    rel_list = [1 if doc_id in relevant_docs else 0 for doc_id in topk]
    dcg_k = dcg(rel_list)

    ideal_rel_list = sorted(rel_list, reverse=True)
    idcg_k = dcg(ideal_rel_list)

    if idcg_k == 0:
        return 0.0
    return dcg_k / idcg_k


def build_ground_truth(root_dir: Path, max_queries: int = 500) -> Dict[int, Set[int]]:
    """
    Build mapping query_id -> set of relevant doc_ids from flattened passages.

    Only uses up to `max_queries` query_ids for speed.
    """
    processed_dir = root_dir / "data" / "processed"
    flat_path = processed_dir / "msmarco_passages.parquet"

    if not flat_path.exists():
        raise FileNotFoundError(
            f"{flat_path} not found. Run scripts/ingest_corpus.py first."
        )

    df_flat = pd.read_parquet(flat_path)

    rel_df = df_flat[df_flat["is_selected"] == 1]

    gt_all = (
        rel_df.groupby("query_id")["doc_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    # Pick a subset
    query_ids = list(gt_all.keys())[:max_queries]
    gt_subset = {qid: gt_all[qid] for qid in query_ids}

    return gt_subset  # query_id -> set(doc_ids)


def build_query_texts(eval_query_ids: List[int]) -> Dict[int, str]:
    """
    Build mapping query_id -> query text using the HF MS MARCO dataset.
    """

    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    query_dict: Dict[int, str] = {}
    wanted = set(eval_query_ids)

    for row in ds:
        qid = row["query_id"]
        if qid in wanted and qid not in query_dict:
            query_dict[qid] = row["query"]
            if len(query_dict) == len(wanted):
                break

    return query_dict


def dedupe_doc_ids(results: List[Dict[str, Any]]) -> List[int]:
    """
    Convert chunk-level results into a doc_id ranking, deduplicated
    (first occurrence of each doc_id wins).
    """
    seen = set()
    ranked_doc_ids: List[int] = []

    for r in results:
        doc_id = int(r["doc_id"])
        if doc_id not in seen:
            seen.add(doc_id)
            ranked_doc_ids.append(doc_id)

    return ranked_doc_ids

def evaluate_retriever(
    name: str,
    search_fn,
    gt: Dict[int, Set[int]],
    query_texts: Dict[int, str],
    top_k_eval: int = 10,
) -> None:
    """
    Evaluate a retriever over many queries.

    Args:
        name: Name for logging ("TF-IDF" / "Dense-E5" / etc.).
        search_fn: function(query: str, top_k: int) -> List[result dict].
        gt: dict query_id -> set(relevant doc_ids).
        query_texts: dict query_id -> query string.
        top_k_eval: k for metrics.

    Prints average Recall@k, MRR@k, nDCG@k, Hit@k.
    """
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    hits: List[float] = []

    query_ids = [qid for qid in gt.keys() if qid in query_texts]

    for qid in query_ids:
        query = query_texts[qid]
        relevant_docs = gt[qid]

        results = search_fn(query, top_k=100)  # search deeper than k
        if not results:
            # no candidates -> all metrics 0 for this query
            recalls.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)
            hits.append(0.0)
            continue

        ranked_doc_ids = dedupe_doc_ids(results)

        recalls.append(recall_at_k(ranked_doc_ids, relevant_docs, top_k_eval))
        mrrs.append(reciprocal_rank_at_k(ranked_doc_ids, relevant_docs, top_k_eval))
        ndcgs.append(ndcg_at_k(ranked_doc_ids, relevant_docs, top_k_eval))
        hits.append(hit_at_k(ranked_doc_ids, relevant_docs, top_k_eval))

    print(f"\n{name} evaluation over {len(recalls)} queries (k={top_k_eval}):")
    print(f"  Recall@{top_k_eval}: {np.mean(recalls):.4f}")
    print(f"  MRR@{top_k_eval}:    {np.mean(mrrs):.4f}")
    print(f"  nDCG@{top_k_eval}:   {np.mean(ndcgs):.4f}")
    print(f"  Hit@{top_k_eval}:    {np.mean(hits):.4f}")

def main() -> None:

    print("Loading sparse TF-IDF artifacts...")
    tfidf_index, tfidf_meta_df = load_tfidf_index(ROOT_DIR)

    print("Loading dense E5 + FAISS artifacts...")
    dense_index, dense_model, dense_meta_df, dense_chunk_df = load_dense_index(ROOT_DIR, corpus="msmarco")

    print("Building ground truth...")
    gt = build_ground_truth(ROOT_DIR, max_queries=500)
    eval_query_ids = list(gt.keys())

    print("Loading query texts...")
    query_texts = build_query_texts(eval_query_ids)

    def sparse_search_fn(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return tfidf_search(query, tfidf_index, tfidf_meta_df, top_k=top_k)

    def dense_search_fn(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return dense_search(
            query,
            dense_index,
            dense_model,
            dense_meta_df,
            dense_chunk_df,
            top_k=top_k,
        )
    
    reranker_model = load_reranker()
    
    def dense_plus_rerank_search_fn(query: str, top_k: int = 10) -> List[Dict[str, Any]]:

        dense_candidates = dense_search(
            query=query,
            index=dense_index,
            model=dense_model,
            meta_df=dense_meta_df,
            chunk_df=dense_chunk_df,
            top_k=50,  # candidate pool
        )
    
        reranked = rerank(
            query = query,
            candidates = dense_candidates,
            model = reranker_model,
            top_k = top_k,
        )

        return reranked


    k_eval = 10

    evaluate_retriever("TF-IDF (sparse)", sparse_search_fn, gt, query_texts, top_k_eval=k_eval)
    evaluate_retriever("E5 + FAISS (dense)", dense_search_fn, gt, query_texts, top_k_eval=k_eval)
    evaluate_retriever("E5 + Reranker (dense + rerank)", dense_plus_rerank_search_fn, gt, query_texts, top_k_eval=k_eval)

if __name__ == "__main__":
    main()
