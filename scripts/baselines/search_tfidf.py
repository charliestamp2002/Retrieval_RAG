from build_tfidf import build_inverted_index, build_tfidf_index, tokenize
import pandas as pd
from pathlib import Path
import os
import numpy as np
import pickle


def calc_similarity(query_terms, tfidf_index, total_documents): 
    """
    Calculate cosine similarity between query and all documents
    Returns a dictionary of doc_id: similarity_score
    """

    tokens = tokenize(query_terms)
    term_counts = {}
    n = len(tokens)

    for token in tokens:
        if token not in term_counts: 
            term_counts[token] = 1
        term_counts[token] += 1

    query_tfidf = {}
    for term, count in term_counts.items():
        tf = count / n 
        docs_with_term = len(tfidf_index.get(term, {}))
        if docs_with_term == 0:
            query_tfidf[term] = 0
        else: 
            idf = np.log(total_documents / (1 + docs_with_term))
            query_tfidf[term] = tf * idf

    scores = {}

    query_magnitude = np.sqrt(sum([score ** 2 for score in query_tfidf.values()]))
    all_doc_ids = set()

    for term, _ in tfidf_index.items():
        all_doc_ids.update(tfidf_index[term].keys())


    for doc_id in all_doc_ids: 
        dot_product = 0
        doc_magnitude_sq = 0

        for term, query_score in query_tfidf.items():
            doc_score = tfidf_index.get(term, {}).get(doc_id, 0)
            dot_product += query_score * doc_score
            doc_magnitude_sq += doc_score ** 2
        
        
        doc_magnitude = np.sqrt(doc_magnitude_sq)

        if query_magnitude == 0 or doc_magnitude == 0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
        
    return scores


if __name__ == "__main__": 

    ROOT_DIR = Path(__file__).resolve().parents[2]

    DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    CHUNKED_FILE = DATA_PROCESSED_DIR / "msmarco_passages_chunked.parquet"
    EMB_DIR = ROOT_DIR / "data" / "embeddings"

    df = pd.read_parquet(CHUNKED_FILE)

    inverted_index_path = Path(EMB_DIR) / "inverted_index.pkl"
    tfidf_index_path = Path(EMB_DIR) / "tfidf_index.pkl"

    with open(inverted_index_path, "rb") as f:
        inverted_index = pickle.load(f)
        # inverted_index = pd.read_pickle(f)

    with open(tfidf_index_path, "rb") as f:
        tfidf_index = pickle.load(f)
        # tfidf_index = pd.read_pickle(f)


    # inverted_index = build_inverted_index(df)
    # tfidf_index = build_tfidf_index(inverted_index, df)

    total_documents = len(df)

    sample_query = "The ballad of the white horse"

    similarities = calc_similarity(sample_query, tfidf_index, total_documents)

    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Print top 10 results
    for doc_id, score in sorted_results[:10]:
        row = df[df["doc_id"] == doc_id].iloc[0]
        print("=" * 80)
        print(f"Doc {doc_id} | score={score:.4f}")
        print(row["chunk_text"][:400], "...")




 