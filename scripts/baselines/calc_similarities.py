import pandas as pd
from pathlib import Path
import pickle

from build_indexes import tokenize

def build_tf_vector(query: str): 
    """Calculate term frequency for each term in the query."""
    tokens = query.lower().split()
    term_counts = {}
    for token in tokens:
        if token not in term_counts:
            term_counts[token] = 0
        term_counts[token] += 1
    total_terms = len(tokens)
    tf = {term: count / total_terms for term, count in term_counts.items()}
    return tf

def build_tfidf_vector(query: str, tf: dict, idf_index: dict):

    tfidf_vector = {}

    for term in tf: 
        if term in idf_index:
            idf = idf_index[term]
        else: 
            idf = 0.0
    
        tfidf_vector[term] = tf[term] * idf

    return tfidf_vector

def calc_similarities(query_tfidf_vector: dict, tfidf_index: dict): 

    scores = {}

    for term in query_tfidf_vector: 
        if term in tfidf_index:
            for passage_id, tfidf in tfidf_index[term].items():
                if passage_id not in scores:
                    scores[passage_id]= 0.0
                scores[passage_id] += tfidf * query_tfidf_vector[term]

    passage_magnitudes = {}
    for term, posting_list in tfidf_index.items():
        for passage_id, tfidf in posting_list.items():
            if passage_id in scores:
                if passage_id not in passage_magnitudes:
                    passage_magnitudes[passage_id] = 0.0
                passage_magnitudes[passage_id] += tfidf 

    passage_magnitudes = {pid: mag ** 0.5 for pid, mag in passage_magnitudes.items()}
    query_tfidf_vector_mag = sum([v**2 for v in query_tfidf_vector.values()]) ** 0.5

    cosine_scores = {}
    for passage_id, dot_prod in scores.items():
        passage_vector_mag = passage_magnitudes.get(passage_id, 0.0)
        cosine_scores[passage_id] = dot_prod / (passage_vector_mag * query_tfidf_vector_mag + 1e-10)
    
    return cosine_scores

def find_sorted_scores(cosine_scores: dict, top_k: int = 10):
    sorted_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_k]


def main():
    DATA_PROCESSED_DIR = Path("data/processed")
    in_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"

    df = pd.read_parquet(in_path)
    print(df.columns)

    INDEX_DIR = Path("data/index")

    with open(INDEX_DIR / "tfidf_index_baseline.pkl", "rb") as f:
        tfidf_index = pickle.load(f)
    with open(INDEX_DIR / "idf_index_baseline.pkl", "rb") as f:    
        idf_index = pickle.load(f)

    query = "United States"
    tf_vector = build_tf_vector(query)
    # print(f"TF vector: {tf_vector}")
    query_tfidf_vector = build_tfidf_vector(query, tf_vector, idf_index)
    # print(f"TF-IDF vector: {query_tfidf_vector}")
    similarity_scores = calc_similarities(query_tfidf_vector, tfidf_index)
    # print(f"Similarity scores: {similarity_scores}")
    sorted_similarities_scores = find_sorted_scores(similarity_scores, top_k=10)
    print(f"Top 10 similar passages: {sorted_similarities_scores}")
    for passage_id, score in sorted_similarities_scores: 
        passage_text = df[df["passage_id"] == passage_id]["passage_text"].values[0]
        print(f"Passage ID: {passage_id}, Score: {score:.4f}, Text: {passage_text}")
    

if __name__ == "__main__":
    main()