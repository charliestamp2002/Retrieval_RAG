import os
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

def tokenize(text):
    # A naive tokenizer that lowercases and splits on whitespace (for now)
    return text.lower().split()

def build_inverted_index(df):
    inverted_index = {} 
    for _, row in df.iterrows(): 
        doc_id = row["doc_id"]
        text = row["chunk_text"]
        tokens = tokenize(text)
        unique_tokens = set(tokens)

        term_counts = {}

        for token in tokens:
            if token not in term_counts: 
                term_counts[token] = 1
            else: 
                term_counts[token] += 1
        
        n = len(tokens)
        for term, count in term_counts.items(): 
            tf = count / n
            if term not in inverted_index: 
                inverted_index[term] = {}
            
            inverted_index[term][doc_id] = tf

    return inverted_index

def build_tfidf_index(inverted_index, df): 
    N = len(df)
    tfidf_index = {}

    for term, _ in inverted_index.items():
        total_docs_with_term = len(inverted_index[term])
        idf = np.log(N / (1 + total_docs_with_term))
        tfidf_index[term] = {}
        for doc_id, tf in inverted_index[term].items(): 

            tfidf = tf * idf
            tfidf_index[term][doc_id] = tfidf
    
    return tfidf_index





def main(): 

    ROOT_DIR = Path(__file__).resolve().parents[2]
    print(ROOT_DIR)
    DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"    
    CHUNKED_FILE = DATA_PROCESSED_DIR / "msmarco_passages_chunked.parquet"
    # INVERTED_INDEX_FILE = DATA_PROCESSED_DIR / "msmarco_inverted_index.parquet"
    # TFIDF_INDEX_FILE = DATA_PROCESSED_DIR / "msmarco_tfidf_index.parquet"

    if not CHUNKED_FILE.exists(): 
        raise FileNotFoundError(f"Chunked passages file not found: {CHUNKED_FILE}. Run scripts/chunk_documents.py first to create it.")
    
    df = pd.read_parquet(CHUNKED_FILE)

    meta_cols = ["doc_id", "chunk_id", "chunk_text", "query_id", "is_selected", "url", "set"]
    meta_df = df[meta_cols].reset_index(drop=True)     

    EMB_DIR = ROOT_DIR / "data" / "embeddings"
    meta_df.to_parquet(EMB_DIR / "tfidf_meta.parquet", index=False) 

    inverted_index = build_inverted_index(df)
    # for i, (term, _) in enumerate(inverted_index.items()):
    #     if i > 0:
    #         break
    #     print(f"{term}: {inverted_index[term]}")

    tfidf_index = build_tfidf_index(inverted_index, df)
    # for i, (term, _) in enumerate(tfidf_index.items()):
    #     if i > 0:
    #         break
    #     print(f"{term}: {tfidf_index[term]}")

    EMB_DIR = ROOT_DIR / "data" / "embeddings"
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Save tfidf index
    with open(EMB_DIR / "tfidf_index.pkl", "wb") as f:
        pickle.dump(tfidf_index, f)

    # Save inverted index
    with open(EMB_DIR / "inverted_index.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

   
if __name__ == "__main__":
    main()
   



