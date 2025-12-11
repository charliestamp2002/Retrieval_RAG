import numpy as np
import pandas as pd 
from pathlib import Path
import pickle

def tokenize(text):
    # A naive tokenizer that lowercases and splits on whitespace (for now)
    return text.lower().split()

def build_inverted_index_and_doc_freq(df: pd.DataFrame):

    inverted_index = {}
    doc_freq = {}

    for _, row in df.iterrows(): 
        passage_id = row["passage_id"]
        passage_text = row["passage_text"]
        tokens = tokenize(passage_text)
        
        term_counts = {}
        for token in tokens: 
            if token not in term_counts: 
                term_counts[token] = 0
            term_counts[token] += 1

        passage_length = len(tokens)

        for token, count in term_counts.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][passage_id] = count / passage_length

    for term in inverted_index: 
        doc_freq_term = len(inverted_index[term])
        doc_freq[term] = doc_freq_term

    return inverted_index, doc_freq

def build_idf(doc_freq, df):

    N = len(df)
    idf_index = {}

    for term, freq in doc_freq.items(): 
        idf = 1.0 + np.log((N - freq + 0.5) / (freq + 0.5))
        idf_index[term] = idf
    
    return idf_index

def build_tfidf_index(inverted_index, idf_index): 
    tfidf_index = {}

    for term, posting_list in inverted_index.items():
        idf = idf_index[term]
        tfidf_posting_list = {}

        for passage_id, tf in posting_list.items():
            tfidf_posting_list[passage_id] = tf * idf
        tfidf_index[term] = tfidf_posting_list

    return tfidf_index


def main(): 
    DATA_PROCESSED_DIR = Path("data/processed")
    in_path = DATA_PROCESSED_DIR / "msmarco_passages_flattened_baseline.parquet"

    df = pd.read_parquet(in_path)
    inverted_index, doc_freq = build_inverted_index_and_doc_freq(df)
    idf_index = build_idf(doc_freq, df)
    tfidf_index = build_tfidf_index(inverted_index, idf_index)

    # print(f"inverted index sample: {dict(list(inverted_index.items())[:3])}")
    # print(f"doc freq sample: {dict(list(doc_freq.items())[:3])}")
    # print(f"idf index sample: {dict(list(idf_index.items())[:3])}")

    output_dir = Path("data/index")

    with open(output_dir / "inverted_index_baseline.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    with open(output_dir / "idf_index_baseline.pkl", "wb") as f:
        pickle.dump(idf_index, f)
    
    with open(output_dir / "tfidf_index_baseline.pkl", "wb") as f:
        pickle.dump(tfidf_index, f)
    
    with open(output_dir/ "doc_freq_baseline.pkl", "wb") as f:
        pickle.dump(doc_freq, f)

    print("Saved inverted index, idf index, tfidf index, and doc freq to pickle files.")



if __name__ == "__main__":
    main()
    






