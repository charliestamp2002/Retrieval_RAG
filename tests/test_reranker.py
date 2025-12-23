"""
Test suite for the reranker module.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.app.core.reranker import load_reranker, rerank


def test_load_reranker():
    print("\n" + "="*60)
    print("TEST 1: Load Reranker Model")
    print("="*60)

    try:
        model = load_reranker()
        print("Reranker model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Failed to load reranker: {e}")
        raise


def test_rerank_basic(model):
    print("\n" + "="*60)
    print("TEST 2: Basic Reranking")
    print("="*60)

    query = "What is machine learning?"
    candidates = [
        {
            'chunk_text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'doc_id': 1,
            'chunk_id': 0,
            'score': 0.7
        },
        {
            'chunk_text': 'The weather today is sunny with clear skies.',
            'doc_id': 2,
            'chunk_id': 1,
            'score': 0.9
        },
        {
            'chunk_text': 'Deep learning uses neural networks with multiple layers.',
            'doc_id': 3,
            'chunk_id': 2,
            'score': 0.6
        },
    ]

    print(f"Query: {query}")
    print(f"Candidates: {len(candidates)}")
    print("\nBefore reranking:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. Score: {c['score']:.3f} | {c['chunk_text'][:60]}...")

    try:
        results = rerank(query, candidates, model, top_k=3)
        print("\nReranking completed successfully")
        print(f"\nAfter reranking (top-{len(results)}):")
        for i, r in enumerate(results, 1):
            rerank_score = r.get('rerank_score', r.get('score'))
            print(f"  {i}. Rerank Score: {rerank_score:.4f} | {r['chunk_text'][:60]}...")

        assert results[0]['doc_id'] != 2, "Irrelevant document should not be ranked first"
        print("\nRelevance ranking improved")

        return results
    except Exception as e:
        print(f"Reranking failed: {e}")
        raise


def test_rerank_top_k(model):
    print("\n" + "="*60)
    print("TEST 3: Top-K Filtering")
    print("="*60)

    query = "information retrieval"
    candidates = [
        {'chunk_text': f'Document {i} about information retrieval systems.', 'doc_id': i, 'chunk_id': i, 'score': 0.5}
        for i in range(10)
    ]

    top_k = 5
    print(f"Query: {query}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Requesting top_k: {top_k}")

    try:
        results = rerank(query, candidates, model, top_k=top_k)
        print(f"Returned {len(results)} results")

        assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
        print(f"Top-K filtering working correctly")

        return results
    except Exception as e:
        print(f"Top-K test failed: {e}")
        raise


def test_rerank_preserves_metadata(model):
    print("\n" + "="*60)
    print("TEST 4: Metadata Preservation")
    print("="*60)

    query = "neural networks"
    candidates = [
        {
            'chunk_text': 'Neural networks are computational models inspired by the brain.',
            'doc_id': 42,
            'chunk_id': 7,
            'score': 0.8,
            'query_id': 123,
            'url': 'https://example.com/doc',
            'is_selected': 1,
            'set': 'test_set',
            'custom_field': 'custom_value'
        }
    ]

    print(f"Query: {query}")
    print(f"Original metadata fields: {list(candidates[0].keys())}")

    try:
        results = rerank(query, candidates, model, top_k=1)
        result = results[0]

        print(f"Result metadata fields: {list(result.keys())}")

        for key in candidates[0].keys():
            if key != 'score':
                assert key in result, f"Field '{key}' not preserved"
                assert result[key] == candidates[0][key], f"Field '{key}' value changed"

        assert 'rerank_score' in result, "rerank_score not added"

        print("All metadata fields preserved")
        print(f"New field added: rerank_score = {result['rerank_score']:.4f}")

        return results
    except Exception as e:
        print(f"Metadata preservation test failed: {e}")
        raise


def test_rerank_empty_candidates(model):
    print("\n" + "="*60)
    print("TEST 5: Empty Candidates Edge Case")
    print("="*60)

    query = "test query"
    candidates = []

    print(f"Query: {query}")
    print(f"Candidates: {len(candidates)} (empty)")

    try:
        results = rerank(query, candidates, model, top_k=5)
        print(f"Handled empty candidates gracefully")
        print(f"Returned: {len(results)} results")
        assert len(results) == 0, "Should return empty list for empty input"
        return results
    except Exception as e:
        print(f"Empty candidates test failed: {e}")
        raise


def run_all_tests():
    print("\n" + "#"*60)
    print("# RERANKER TEST SUITE")
    print("#"*60)

    try:
        model = test_load_reranker()
        test_rerank_basic(model)
        test_rerank_top_k(model)
        test_rerank_preserves_metadata(model)
        test_rerank_empty_candidates(model)

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED")
        print("#"*60)
        print("\nReranker module is working correctly!")

    except Exception as e:
        print("\n" + "#"*60)
        print("# TESTS FAILED")
        print("#"*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
