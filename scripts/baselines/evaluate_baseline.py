import numpy as np
import pandas as pd
from pathlib import Path 
import pickle  
from calc_similarities import build_tf_vector, build_tfidf_vector, calc_similarities, find_sorted_scores

def build_query_to_passage_relevance(df: pd.DataFrame) -> dict:

    query_to_passages = {}

    for _, row in df.iterrows(): 

        query_id = row["query_id"]
        query_text = row["query_text"]
        passage_id = row["passage_id"]
        passage_text = row["passage_text"]
        is_selected = row["is_selected"]

        if query_id not in query_to_passages:
            query_to_passages[query_id] = {}
        
        query_to_passages[query_id][passage_id] = is_selected
    
    return query_to_passages


def retrieve_passages_for_query(query_id, query_text, tfidf_index, idf_index,top_k=10):
    " Retrieve top k passages for a single query"

    tf_vector = build_tf_vector(query_text)
    query_tfidf_vector = build_tfidf_vector(query_text, tf_vector, idf_index)
    sim_scores = calc_similarities(query_tfidf_vector, tfidf_index)
    sorted_scores = find_sorted_scores(sim_scores, top_k=top_k)

    ranked_passage_ids = [passage_id for passage_id, score in sorted_scores]

    return ranked_passage_ids

def retrieve_all_queries(df, tfidf_index, idf_index, top_k = 10):

    res = {}

    for _, row in df.iterrows():
        query_id = row["query_id"]
        query_text = row["query_text"]  

        ranked_passage_ids = retrieve_passages_for_query(query_id, query_text, tfidf_index, idf_index, top_k=top_k)

        res[query_id] = ranked_passage_ids
    
    return res

    

def calculate_mrr(ranked_passaged_ids, relevant_passages):

    for rank, passaged_id in enumerate(ranked_passaged_ids, start=1):
        if passaged_id in relevant_passages:
            return 1.0 / rank
    return 0.0

def calculate_precision_at_k(ranked_passaged_ids, relevant_passages, k):

    retrieved_at_k = ranked_passaged_ids[:k]
    relevant_retrieved = sum(1 for passage_id in retrieved_at_k if passage_id in relevant_passages)
    precision = relevant_retrieved / k
    return precision

def calculate_recall_at_k(ranked_passaged_ids, relevant_passages, k):

    retrieved_at_k = ranked_passaged_ids[:k]
    total_relevant = len(relevant_passages)
    num_relevant = sum(1 for passage_id in retrieved_at_k if passage_id in relevant_passages)
    recall = num_relevant / total_relevant if total_relevant > 0 else 0.0
    return recall

def calculate_hit_rate_at_k(ranked_passage_ids, relevant_passages, k):
    """Calculate Hit Rate@K for a single query."""
    for pid in ranked_passage_ids[:k]:
        if pid in relevant_passages:
            return 1.0
    return 0.0

def calculate_ndcg(ranked_passaged_ids, relevant_passages, k):

    dcg = 0
    for rank, passage_id in enumerate(ranked_passaged_ids[:k], start=1):
        if passage_id in relevant_passages:
            dcg += 1 / (np.log2(rank + 1))
        
    ideal_dcg = 0

    for i, passage_id in enumerate(relevant_passages[:k], start = 1): 
        ideal_dcg += 1 / np.log2(i + 1)
    
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return ndcg

def calculate_ap(ranked_passage_ids, relevant_passages, k):

    if len(relevant_passages) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for rank, passage_id in enumerate(ranked_passage_ids[:k], 1):
        if passage_id in relevant_passages:
            num_hits += 1
            precision_at_k = num_hits / rank
            score += precision_at_k
    
    return score / len(relevant_passages)

def evaulate_retrieval(results, query_to_passages, k = 10):

    mrr_scores = []
    precision_scores = []
    recall_scores = []
    hit_rates = []
    ap_scores = []
    ndcg_scores = []

    for query_id, ranked_passage_ids in results.items():
        relevant_passages = [pid for pid, is_selected in query_to_passages[query_id].items() if is_selected]

        mrr = calculate_mrr(ranked_passage_ids, relevant_passages)
        precision = calculate_precision_at_k(ranked_passage_ids, relevant_passages, k)
        recall = calculate_recall_at_k(ranked_passage_ids, relevant_passages, k)
        hit_rate = calculate_hit_rate_at_k(ranked_passage_ids, relevant_passages, k)
        ap = calculate_ap(ranked_passage_ids, relevant_passages, k)
        ndcg = calculate_ndcg(ranked_passage_ids, relevant_passages, k)

        mrr_scores.append(mrr)
        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_rates.append(hit_rate)
        ap_scores.append(ap)
        ndcg_scores.append(ndcg)

    metrics = {
        "MRR": sum(mrr_scores) / len(mrr_scores),
        "Precision@K": sum(precision_scores) / len(precision_scores),
        "Recall@K": sum(recall_scores) / len(recall_scores),
        "Hit Rate@K": sum(hit_rates) / len(hit_rates),
        "MAP": sum(ap_scores) / len(ap_scores),
        "NDCG@K": sum(ndcg_scores) / len(ndcg_scores),
    }

    return metrics
        

def main(): 

    DATA_PROCESSED_DIR = Path("data/processed")
    in_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"

    INDEX_DIR = Path("data/index")
    with open(INDEX_DIR / "tfidf_index_baseline.pkl", "rb") as f:
        tfidf_index = pickle.load(f)
    with open(INDEX_DIR / "idf_index_baseline.pkl", "rb") as f:    
        idf_index = pickle.load(f)

    df = pd.read_parquet(in_path)
    
    query_to_passages = build_query_to_passage_relevance(df)

    results = retrieve_all_queries(df, tfidf_index, idf_index, top_k=10)

    metrics = evaulate_retrieval(results, query_to_passages, k=10)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # sample = dict(list(query_to_passages.items())[:1])

    # for query, passages in sample.items():
    #     print(f"Query: {query}")
    #     for passage, is_selected in passages.items():
    #         print(f"  Passage: {passage} | Selected: {is_selected}")
    # # print(list(query_to_passages)[:5])

if __name__ == "__main__":  
    main()



        



