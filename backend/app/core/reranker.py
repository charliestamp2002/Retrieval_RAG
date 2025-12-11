
from typing import Any, Dict, List, Tuple
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """
    Load a cross-encoder reranker model.
    The model takes (query, passage) pairs and outputs a relevance score.
    """
    print(f"[reranker] Loading CrossEncoder model: {model_name}")
    model = CrossEncoder(model_name)
    return model

def rerank(
        query: str,
        candidates: List[Dict[str, Any]],
        model: CrossEncoder,
        top_k: int = 10,
        text_key: str = "chunk_text",
    ) -> List[Dict[str, Any]]:

    """
    Rerank a list of candidate passages using a cross-encoder.

    Args:
        query: The raw query string.
        candidates: List of result dicts (e.g. from dense_search), each must
                    contain a text field (default 'chunk_text').
        model: CrossEncoder model instance.
        top_k: Number of reranked results to return.
        text_key: Key in each candidate dict that contains the passage text.

    Returns:
        A new list of candidate dicts, sorted by reranker score (desc),
        with an added key 'rerank_score'.
    """

    if not candidates: 
        return []
    
    # Prepare (query, passage) pairs for the reranker
    pair_inputs = []

    for c in candidates:
        passage = c.get(text_key, "")
        pair_inputs.append((query, passage))


    #  pred rel. scores: 
    scores = model.predict(pair_inputs)

    # Attach scores to candidates
    reranked = []

    for cand, score in zip(candidates, scores):
        new_cand = dict(cand)
        new_cand["rerank_score"] = float(score)
        reranked.append(new_cand)

    reranked.sort(key = lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]




